import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from body_model.utils import SMPL_JOINTS, SMPLH_PATH
from body_model.body_model import BodyModel
from datasets.amass_utils import data_name_list, data_dim
from humor.utils.config_new import ConfigParser
from humor.utils.torch import load_state
from humor.utils.transforms import convert_to_rotmat, compute_world2aligned_mat, rotation_matrix_to_angle_axis


from .diffusion import create_diffusion
from .diffusion_transformer import DiffusionTransformer
from .motion_v_a_e import MotionVAE

IN_ROT_REPS = ['aa', '6d', 'mat']
OUT_ROT_REPS = ['aa', '6d', '9d']
ROT_REP_SIZE = {
    'aa' : 3,
    '6d' : 6,
    'mat' : 9,
    '9d' : 9
}
NUM_SMPL_JOINTS = len(SMPL_JOINTS)
NUM_BODY_JOINTS = NUM_SMPL_JOINTS - 1 # no root
BETA_SIZE = 16

WORLD2ALIGN_NAME_CACHE = {'root_orient' : None, 'trans' : None, 'joints' : None, 'verts' : None, 'joints_vel' : None, 'verts_vel' : None, 'trans_vel' : None, 'root_orient_vel' : None }


class HumorDiffusionTransformer(nn.Module):

    def __init__(self,  latent_size=48,
                        pose_token_dim=256,
                        diffusion_base_dim=256,
                        nhead=4,
                        num_layers=6,
                        dim_feedforward=1024,
                        dropout=0.1,
                        cond_drop_prob=0.1,
                        cfg_scale=4.0,
                        vae_ckpt=None, # path to the VAE checkpoint
                        vae_cfg=None, # path to the VAE config file
                        in_rot_rep='aa', 
                        out_rot_rep='aa',
                        steps_in=1,
                        output_delta=True, # output change in state from decoder rather than next step directly
                        model_data_config='smpl+joints+contacts',
                        detach_sched_samp=True, # if true, detaches outputs of previous step so gradients don't flow through many steps
                        model_use_smpl_joint_inputs=False, # if true, uses smpl joints rather than regressed joints to input at next step (during rollout and sched samp)
                        model_smpl_batch_size=1 # if using smpl joint inputs this should be batch_size of the smpl model (aka data input to rollout)
                ):
        super(HumorDiffusionTransformer, self).__init__()
        self.ignore_keys = []
        
        if vae_ckpt is None or vae_cfg is None:
            raise Exception('Must provide a VAE checkpoint to load!')

        self.steps_in = steps_in
        self.steps_out = 1
        self.out_step_size = 1
        self.detach_sched_samp = detach_sched_samp
        self.output_delta = output_delta

        if self.steps_out > 1:
            raise NotImplementedError('Only supported single step output currently.')

        if out_rot_rep not in OUT_ROT_REPS:
            raise Exception('Not a valid output rotation representation: %s' % (out_rot_rep))
        if in_rot_rep not in IN_ROT_REPS:
            raise Exception('Not a valid input rotation representation: %s' % (in_rot_rep))
        self.out_rot_rep = out_rot_rep
        self.in_rot_rep = in_rot_rep


        # ---------------------------- Input and Output Dimension Calculation ----------------------------

        # get the list of data names for this config
        self.data_names = data_name_list(model_data_config)
        self.aux_in_data_names = self.aux_out_data_names = None # auxiliary data will be returned as part of the input/output dictionary, but not the actual network input/output tensor
        self.pred_contacts = False
        if model_data_config.find('contacts') >= 0: # network is outputting contact classification as well and need to supervise, but not given as input to net.
            self.data_names.remove('contacts')
            self.aux_out_data_names = ['contacts']
            self.pred_contacts = True

        self.need_trans2joint = 'joints' in self.data_names or 'verts' in self.data_names
        self.model_data_config = model_data_config
        
        # Discuss 1: How to deal with the input and output data dim
        self.input_rot_dim = ROT_REP_SIZE[self.in_rot_rep]
        self.input_dim_list = [data_dim(dname, rot_rep_size=self.input_rot_dim) for dname in self.data_names]
        self.input_data_dim = sum(self.input_dim_list)

        self.output_rot_dim = ROT_REP_SIZE[self.out_rot_rep]
        self.output_dim_list = [data_dim(dname, rot_rep_size=self.output_rot_dim) for dname in self.data_names]
        self.delta_output_dim_list = [data_dim(dname, rot_rep_size=ROT_REP_SIZE['mat']) for dname in self.data_names]

        if self.pred_contacts:
            # account for contact classification output
            self.output_dim_list.append(data_dim('contacts'))
            self.delta_output_dim_list.append(data_dim('contacts'))

        self.output_data_dim = sum(self.output_dim_list)

        self.latent_size = latent_size
        past_data_dim = self.steps_in * self.input_data_dim # previous step pose dim
        t_data_dim = self.steps_out * self.input_data_dim # current step pose dim, normally the same as previous step data dim


        # ----------------------------------- Diffusion Model -------------------------------------------

        self.cfg_scale = cfg_scale
        self.cond_drop_prob = cond_drop_prob
        self.diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        self.diffusion_model = DiffusionTransformer(latent_dim=latent_size,
                                                    pose_token_dim=pose_token_dim,
                                                    pose_dim_list=self.input_dim_list,
                                                    d_model=diffusion_base_dim,
                                                    nhead=nhead,
                                                    num_layers=num_layers,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)

        # --------------------------------- Encoder and Decoder ------------------------------------------
        motion_vae_cfg, _ = ConfigParser(vae_cfg).parse('train')
        
        vae_model_cfg = motion_vae_cfg.model_dict
        diffusion_cfg = {
            'latent_size' : latent_size,
            'in_rot_rep' : in_rot_rep,
            'out_rot_rep' : out_rot_rep,
            'steps_in' : steps_in,
            'latent_size' : latent_size,
            'model_data_config' : model_data_config,
            'model_use_smpl_joint_inputs' : model_use_smpl_joint_inputs
        }
        for k, v in diffusion_cfg.items():
            if v != vae_model_cfg[k]: # check if vae and diffusion are using same config
                raise Exception('Diffusion model and VAE model configs do not match for %s: %s vs %s' % (k, v, vae_model_cfg[k]))
        
        
        self.motion_vae = MotionVAE(**motion_vae_cfg.model_dict, model_smpl_batch_size=model_smpl_batch_size)   
        load_state(vae_ckpt, self.motion_vae)
        for param in self.motion_vae.parameters():
            param.requires_grad = False # freeze the VAE model

        # -------------------------------------- Body Model -----------------------------------------------

        self.use_smpl_joint_inputs = model_use_smpl_joint_inputs
        self.smpl_batch_size = model_smpl_batch_size
        if self.use_smpl_joint_inputs:
            # need a body model to compute the joints after each step.
            print('Using SMPL joints rather than regressed joints as input at each step for roll out and scheduled sampling...')
            male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
            self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=self.smpl_batch_size)
            female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
            self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=self.smpl_batch_size)
            neutral_bm_path = os.path.join(SMPLH_PATH, 'neutral/model.npz')
            self.neutral_bm = BodyModel(bm_path=neutral_bm_path, num_betas=16, batch_size=self.smpl_batch_size)
            self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm, 'neutral' : self.neutral_bm}
            for p in self.male_bm.parameters():
                p.requires_grad = False
            for p in self.female_bm.parameters():
                p.requires_grad = False 
            for p in self.neutral_bm.parameters():
                p.requires_grad = False 
            self.ignore_keys = ['male_bm', 'female_bm', 'neutral_bm']


    def prepare_input(self, data_in, device, data_out=None, return_input_dict=False, return_global_dict=False):
        '''
        Concatenates input and output data as expected by the model.

        Also creates a dictionary of GT outputs for use in computing the loss. And optionally
        a dictionary of inputs.
        '''

        #
        # input data
        #
        in_unnorm_data_list = []
        for k in self.data_names:
            cur_dat = data_in[k].to(device)
            B, T = cur_dat.size(0), cur_dat.size(1)
            cur_unnorm_dat = cur_dat.reshape((B, T, self.steps_in, -1))
            in_unnorm_data_list.append(cur_unnorm_dat)
        x_past = torch.cat(in_unnorm_data_list, axis=3)

        input_dict = None
        if return_input_dict:
            input_dict = {k : v for k, v in zip(self.data_names, in_unnorm_data_list)}

            if self.aux_in_data_names is not None:
                for k in self.aux_in_data_names:
                    cur_dat = data_in[k].to(device)
                    B, T = cur_dat.size(0), cur_dat.size(1)
                    cur_unnorm_dat = cur_dat.reshape((B, T, self.steps_in, -1))
                    input_dict[k] = cur_unnorm_dat

        #
        # output
        #
        if data_out is not None:
            out_unnorm_data_list = []
            for k in self.data_names:
                cur_dat = data_out[k].to(device)
                B, T = cur_dat.size(0), cur_dat.size(1)
                cur_unnorm_dat = cur_dat.reshape((B, T, self.steps_out, -1))
                out_unnorm_data_list.append(cur_unnorm_dat)
            x_t = torch.cat(out_unnorm_data_list, axis=3)
            gt_dict = {k : v for k, v in zip(self.data_names, out_unnorm_data_list)}

            if self.aux_out_data_names is not None:
                for k in self.aux_out_data_names:
                    cur_dat = data_out[k].to(device)
                    B, T = cur_dat.size(0), cur_dat.size(1)
                    cur_unnorm_dat = cur_dat.reshape((B, T, self.steps_out, -1))
                    gt_dict[k] = cur_unnorm_dat

            return_list = [x_past, x_t, gt_dict]
            if return_input_dict:
                return_list.append(input_dict)

            #
            # global
            #
            if return_global_dict:
                global_gt_dict = dict()
                for k in self.data_names:
                    global_k = 'global_' + k
                    cur_dat = data_out[global_k].to(device)
                    B, T = cur_dat.size(0), cur_dat.size(1)
                    # expand each to have steps_out since originally they are just B x T x ... x D
                    cur_dat = cur_dat.reshape((B, T, 1, -1)).expand_as(gt_dict[k])
                    global_gt_dict[k] = cur_dat

                if self.aux_out_data_names is not None:
                    for k in self.aux_out_data_names:
                        global_k = 'global_' + k
                        cur_dat = data_out[global_k].to(device)
                        B, T = cur_dat.size(0), cur_dat.size(1)
                        # expand each to have steps_out since originally they are just B x T x ... x D
                        cur_dat = cur_dat.reshape((B, T, 1, -1)).expand_as(gt_dict[k])
                        global_gt_dict[k] = cur_dat

                return_list.append(global_gt_dict)

            return tuple(return_list)

        else:
            if return_input_dict:
                return x_past, input_dict
            else:
                return x_past


    def split_output(self, decoder_out, convert_rots=True):
        '''
        Given the output of the decoder, splits into each state component.
        Also transform rotation representation to matrices.

        Input:
        - decoder_out  (B x steps_out x D)

        Returns:
        - output dict
        '''
        B = decoder_out.size(0)
        decoder_out = decoder_out.reshape((B, self.steps_out, -1))

        # collect outputs
        name_list = self.data_names
        if self.aux_out_data_names is not None:
            name_list = name_list + self.aux_out_data_names
        idx_list = self.delta_output_dim_list if self.output_delta else self.output_dim_list
        out_dict = dict()
        sidx = 0
        for cur_name, cur_idx in zip(name_list, idx_list):
            eidx = sidx + cur_idx
            out_dict[cur_name] = decoder_out[:,:,sidx:eidx]
            sidx = eidx

        # transform rotations
        if convert_rots and not self.output_delta: # output delta already gives rotmats
            if 'root_orient' in self.data_names:
                out_dict['root_orient'] = convert_to_rotmat(out_dict['root_orient'], rep=self.out_rot_rep)
            if 'pose_body' in self.data_names:
                out_dict['pose_body'] = convert_to_rotmat(out_dict['pose_body'], rep=self.out_rot_rep)

        return out_dict


    def forward(self, x_past, x_t):
        '''
        predict the noise.
        '''

        B, _, D = x_past.size()
        past_in = x_past.reshape((B, -1)) # B x 339
        t_in = x_t.reshape((B, -1)) # B x 339
        
        pred_dict= self.single_step(past_in, t_in) # we pass in the difference, and the current gt pose

        return pred_dict


    def single_step(self, past_in, t_in):
        B = past_in.size(0)
        # ground truth z_t
        z = self.motion_vae.encode(past_in, t_in) # B * latent_size

        noise = torch.randn_like(z).to(z.device) # B * latent_size
        model_kwargs = dict(y=past_in) # pass in the condition
        
        time = torch.randint(0, self.diffusion.num_timesteps, (B,)).to(z.device) # B x 1
        # Discuss 2: calculate loss in latent space? Then how to train the decoder? Calculate two losses and backprop twice?
        z_noise = self.diffusion.q_sample(z, time, noise=noise)
        
        cond_mask = self.sample_cond_drop_mask(B, drop_prob=self.cond_drop_prob, device=z.device) # B x 1
        
        pred_noise = self.diffusion_model(z_noise, past_in, time, cond_mask) # B x latent_size
        
        pred_dict = {
            'pred_noise' : pred_noise,
            'z' : z, # B x latent_size
            'z_noise' : z_noise, # B x latent_size
            'noise' : noise, # B x latent_size
        }
        return pred_dict
    
    def sample_cond_drop_mask(self, batch_size, drop_prob=0.1, device='cuda'):
        """
        Generates a [B] boolean tensor indicating which samples to drop condition.
        """
        return (torch.rand(batch_size, device=device) < drop_prob)



    # def decode(self, z, past_in):
    #     '''
    #     Decodes prediction from the latent transition and past states

    #     Input:
    #     - z       (B x latent_size)
    #     - past_in (B x steps_in*D)

    #     Returns:
    #     - decoder_out (B x steps_out*D)
    #     '''
    #     B = z.size(0)
    #     decoder_in = torch.cat([past_in, z], axis=1)
    #     decoder_out = self.decoder(decoder_in).reshape((B, 1, -1))

    #     if self.output_delta:
    #         # network output is the residual, add to the input to get final output
    #         step_in = past_in.reshape((B, self.steps_in, -1))[:,-1:,:] # most recent input step

    #         final_out_list = []
    #         in_sidx = out_sidx = 0
    #         decode_out_dim_list = self.output_dim_list
    #         if self.pred_contacts:
    #             decode_out_dim_list = decode_out_dim_list[:-1] # do contacts separately
    #         for in_dim_idx, out_dim_idx, data_name in zip(self.input_dim_list, decode_out_dim_list, self.data_names):
    #             in_eidx = in_sidx + in_dim_idx
    #             out_eidx = out_sidx + out_dim_idx

    #             # add residual to input (and transform as necessary for rotations)
    #             in_val = step_in[:,:,in_sidx:in_eidx]
    #             out_val = decoder_out[:,:,out_sidx:out_eidx]
    #             if data_name in ['root_orient', 'pose_body']:
    #                 if self.in_rot_rep != 'mat':
    #                     in_val = convert_to_rotmat(in_val, rep=self.in_rot_rep)
    #                 out_val = convert_to_rotmat(out_val, rep=self.out_rot_rep)

    #                 in_val = in_val.reshape((B, 1, -1, 3, 3))
    #                 out_val = out_val.reshape((B, self.steps_out, -1, 3, 3))

    #                 rot_in = torch.matmul(out_val, in_val).reshape((B, self.steps_out, -1)) # rotate by predicted residual
    #                 final_out_list.append(rot_in)
    #             else:
    #                 final_out_list.append(out_val + in_val)

    #             in_sidx = in_eidx
    #             out_sidx = out_eidx
    #         if self.pred_contacts:
    #             final_out_list.append(decoder_out[:,:,out_sidx:])

    #         decoder_out = torch.cat(final_out_list, dim=2)

    #     decoder_out = decoder_out.reshape((B, -1))

    #     return decoder_out


    # def scheduled_sampling(self, x_past, x_t, init_input_dict, p=0.5, gender=None, betas=None, need_global_out=True):
    #     '''
    #     Given all inputs and ground truth outputs for all steps, roll out model predictions
    #     where at each step use the GT input with prob p, otherwise use own previous output.

    #     Input:
    #     - x_past (B x T x steps_in x D)
    #     - x_t    (B x T x steps_out x D)
    #     - init_input_dict : dictionary of each initial state (B x steps_in x D), rotations should be matrices
    #     - p : probability of using the GT input at each step of the sequence
    #     - gender/betas only required if self.use_smpl_joint_inputs is true (used to decide the SMPL body model)
    #     '''
    #     B, T, S, D = x_past.size()
    #     S_out = x_t.size(2)
    #     J = len(SMPL_JOINTS)
    #     cur_input_dict = init_input_dict # this is the predicted input dict

    #     # initial input must be from GT since we don't have any predictions yet
    #     past_in = x_past[:,0,:,:].reshape((B, -1))
    #     t_in = x_t[:,0,:,:].reshape((B, -1))

    #     global_world2local_rot = torch.eye(3).reshape((1, 1, 3, 3)).expand((B, 1, 3, 3)).to(x_past)
    #     global_world2local_trans = torch.zeros((B, 1, 3)).to(x_past)
    #     trans2joint = torch.zeros((B,1,1,3)).to(x_past)
    #     if self.need_trans2joint:
    #         trans2joint = -torch.cat([cur_input_dict['joints'][:,-1,:2], torch.zeros((B, 1)).to(x_past)], axis=1).reshape((B,1,1,3)) # same for whole sequence
    #     pred_local_seq = []
    #     pred_global_seq = []
    #     for t in range(T):
    #         # sample next step from model
    #         x_pred_dict = self.single_step(past_in, t_in)

    #         # save output
    #         pred_local_seq.append(x_pred_dict)

    #         # output is the actual regressed joints, but input to next step can use smpl joints
    #         x_pred_smpl_joints = None
    #         if self.use_smpl_joint_inputs and gender is not None and betas is not None:
    #             # this assumes the model is actually outputting everything we need to run SMPL
    #             # also assumes single output step
    #             smpl_trans = x_pred_dict['trans'][:,0:1].reshape((B, 3)) # only want immediate next frame
    #             smpl_root_orient = rotation_matrix_to_angle_axis(x_pred_dict['root_orient'][:,0:1].reshape((B,3,3))).reshape((B, 3))
    #             smpl_betas = betas[:,0,:]
    #             smpl_pose_body = rotation_matrix_to_angle_axis(x_pred_dict['pose_body'][:,0:1].reshape((B*(J-1),3,3))).reshape((B, (J-1)*3))

    #             smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body]
    #             # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    #             gender_names = ['male', 'female', 'neutral']
    #             pred_joints = []
    #             prev_nbidx = 0
    #             cat_idx_map = np.ones((B), dtype=np.int)*-1
    #             for gender_name in gender_names:
    #                 gender_idx = np.array(gender) == gender_name
    #                 nbidx = np.sum(gender_idx)

    #                 cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int)
    #                 prev_nbidx += nbidx

    #                 gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

    #                 # need to pad extra frames with zeros in case not as long as expected 
    #                 pad_size = self.smpl_batch_size - nbidx
    #                 if pad_size == B:
    #                     # skip if no frames for this gender
    #                     continue
    #                 pad_list = gender_smpl_vals
    #                 if pad_size < 0:
    #                     raise Exception('SMPL model batch size not large enough to accomodate!')
    #                 elif pad_size > 0:
    #                     pad_list = self.zero_pad_tensors(pad_list, pad_size)
                    
    #                 # reconstruct SMPL
    #                 cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose = pad_list
    #                 bm = self.bm_dict[gender_name]
    #                 pred_body = bm(pose_body=cur_pred_pose, betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
    #                 if pad_size > 0:
    #                     pred_joints.append(pred_body.Jtr[:-pad_size])
    #                 else:
    #                     pred_joints.append(pred_body.Jtr)

    #             # cat all genders and reorder to original batch ordering
    #             x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:,:len(SMPL_JOINTS),:].reshape((B, 1, -1))
    #             x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]

    #         # prepare predicted input to next step in case needed
    #         # update input dict with new frame
    #         del_keys = []
    #         for k in cur_input_dict.keys():
    #             if k in x_pred_dict:
    #                 # drop oldest frame and add new prediction
    #                 keep_frames = cur_input_dict[k][:,1:,:]
    #                 # print(keep_frames.size())
    #                 if k == 'joints' and self.use_smpl_joint_inputs and x_pred_smpl_joints is not None:
    #                     # print('Using SMPL joints rather than regressed joints...')
    #                     if self.detach_sched_samp:
    #                         cur_input_dict[k] = torch.cat([keep_frames, x_pred_smpl_joints.detach()], axis=1)
    #                     else:
    #                         cur_input_dict[k] = torch.cat([keep_frames, x_pred_smpl_joints], axis=1)
    #                 else:
    #                     if self.detach_sched_samp:
    #                         cur_input_dict[k] = torch.cat([keep_frames, x_pred_dict[k][:,0:1,:].detach()], axis=1)
    #                     else:
    #                         cur_input_dict[k] = torch.cat([keep_frames, x_pred_dict[k][:,0:1,:]], axis=1)
    #                 # print(cur_input_dict[k].size())
    #             else:
    #                 del_keys.append(k)
    #         for k in del_keys:
    #             del cur_input_dict[k] # don't need it anymore

    #         # get world2aligned rot and translation
    #         if self.detach_sched_samp:
    #             root_orient_mat = x_pred_dict['root_orient'][:,0,:].reshape((B, 3, 3)).detach()
    #             world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
    #             world2aligned_trans = torch.cat([-x_pred_dict['trans'][:,0,:2].detach(), torch.zeros((B,1)).to(x_past)], axis=1)
    #         else:
    #             root_orient_mat = x_pred_dict['root_orient'][:,0,:].reshape((B, 3, 3))
    #             world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
    #             world2aligned_trans = torch.cat([-x_pred_dict['trans'][:,0,:2], torch.zeros((B,1)).to(x_past)], axis=1)

    #         #
    #         # transform inputs to this local frame for next step
    #         #
    #         cur_input_dict = self.apply_world2local_trans(world2aligned_trans, world2aligned_rot, trans2joint, cur_input_dict, cur_input_dict, invert=False)

    #         # convert rots to correct input format
    #         if self.in_rot_rep == 'aa':
    #             if 'root_orient' in self.data_names:
    #                 cur_input_dict['root_orient'] = rotation_matrix_to_angle_axis(cur_input_dict['root_orient'].reshape((B*S,3,3))).reshape((B, S, 3))
    #             if 'pose_body' in self.data_names:
    #                 cur_input_dict['pose_body'] = rotation_matrix_to_angle_axis(cur_input_dict['pose_body'].reshape((B*S*(J-1),3,3))).reshape((B, S, (J-1)*3))
    #         elif self.in_rot_rep == '6d':
    #             if 'root_orient' in self.data_names:
    #                 cur_input_dict['root_orient'] = cur_input_dict['root_orient'][:,:,:6]
    #             if 'pose_body' in self.data_names:
    #                 cur_input_dict['pose_body'] = cur_input_dict['pose_body'].reshape((B, S, J-1, 9))[:,:,:,:6].reshape((B, S, (J-1)*6))


    #         if need_global_out:
    #             #
    #             # compute current world output and update world2local transform
    #             #
    #             cur_world_dict = dict()
    #             cur_world_dict = self.apply_world2local_trans(global_world2local_trans, global_world2local_rot, trans2joint, x_pred_dict, cur_world_dict, invert=True)

    #             if self.detach_sched_samp:
    #                 global_world2local_trans = torch.cat([-cur_world_dict['trans'][:,0:1,:2].detach(), torch.zeros((B, 1, 1)).to(x_past)], axis=2)
    #             else:
    #                 global_world2local_trans = torch.cat([-cur_world_dict['trans'][:,0:1,:2], torch.zeros((B, 1, 1)).to(x_past)], axis=2)

    #             global_world2local_rot = torch.matmul(global_world2local_rot, world2aligned_rot.reshape((B, 1, 3, 3)))

    #             pred_global_seq.append(cur_world_dict)

    #         if t+1 < T:
    #             # choose whether next step will use GT or predicted inputs and prepare them
    #             if np.random.random_sample() < p:
    #                 # use GT
    #                 past_in = x_past[:,t+1,:,:].reshape((B, -1))
    #             else:
    #                 # cat all inputs together to form past_in
    #                 in_data_list = []
    #                 for k in self.data_names:
    #                     in_data_list.append(cur_input_dict[k])
    #                 past_in = torch.cat(in_data_list, axis=2)
    #                 past_in = past_in.reshape((B, -1))

    #             # GT output is the same no matter what                
    #             t_in = x_t[:,t+1,:,:].reshape((B, -1))

    #     if need_global_out:
    #         # aggregate pred_seq
    #         pred_global_seq_out = dict()
    #         for k in pred_global_seq[0].keys():
    #             if k == 'posterior_distrib' or k == 'prior_distrib':
    #                 m = torch.stack([pred_global_seq[i][k][0] for i in range(len(pred_global_seq))], axis=1)
    #                 v = torch.stack([pred_global_seq[i][k][1] for i in range(len(pred_global_seq))], axis=1)
    #                 pred_global_seq_out[k] = (m, v)
    #             else:
    #                 pred_global_seq_out[k] = torch.stack([pred_global_seq[i][k] for i in range(len(pred_global_seq))], axis=1)

    #      # aggregate pred_seq
    #     pred_local_seq_out = dict()
    #     for k in pred_local_seq[0].keys():
    #         # print(k)
    #         if k == 'posterior_distrib' or k == 'prior_distrib':
    #             m = torch.stack([pred_local_seq[i][k][0] for i in range(len(pred_local_seq))], axis=1)
    #             v = torch.stack([pred_local_seq[i][k][1] for i in range(len(pred_local_seq))], axis=1)
    #             pred_local_seq_out[k] = (m, v)
    #         else:
    #             pred_local_seq_out[k] = torch.stack([pred_local_seq[i][k] for i in range(len(pred_local_seq))], axis=1)

    #     if need_global_out:
    #         return pred_global_seq_out, pred_local_seq_out
    #     else:
    #         return pred_local_seq_out


    # def apply_world2local_trans(self, world2local_trans, world2local_rot, trans2joint, input_dict, output_dict, invert=False):
    #     '''
    #     Applies the given world2local transformation to the data in input_dict and stores the result in output_dict.

    #     If invert is true, applies local2world.

    #     - world2local_trans : B x 3 or B x 1 x 3
    #     - world2local_rot :   B x 3 x 3 or B x 1 x 3 x 3
    #     - trans2joint : B x 1 x 1 x 3
    #     '''
    #     B = world2local_trans.size(0)
    #     world2local_rot = world2local_rot.reshape((B, 1, 3, 3))
    #     world2local_trans = world2local_trans.reshape((B, 1, 3))
    #     trans2joint = trans2joint.reshape((B, 1, 1, 3))
    #     if invert:
    #         local2world_rot = world2local_rot.transpose(3, 2)
    #     for k, v in input_dict.items():
    #         # apply differently depending on which data value it is
    #         if k not in WORLD2ALIGN_NAME_CACHE:
    #             # frame of reference is irrelevant, just copy to output
    #             output_dict[k] = input_dict[k]
    #             continue
            
    #         S = input_dict[k].size(1)
    #         if k in ['root_orient']:
    #             # rot: B x S x 3 x 3 sized rotation matrix input
    #             input_mat = input_dict[k].reshape((B, S, 3, 3)) # make sure not B x S x 9
    #             if invert:
    #                 output_dict[k] = torch.matmul(local2world_rot, input_mat).reshape((B, S, 9))
    #             else:
    #                 output_dict[k] = torch.matmul(world2local_rot, input_mat).reshape((B, S, 9))
    #         elif k in ['trans']: 
    #             # trans + rot : B x S x 3
    #             input_trans = input_dict[k]
    #             if invert:
    #                 output_trans = torch.matmul(local2world_rot, input_trans.reshape((B, S, 3, 1)))[:,:,:,0] 
    #                 output_trans = output_trans - world2local_trans
    #                 output_dict[k] = output_trans
    #             else:
    #                 input_trans = input_trans + world2local_trans
    #                 output_dict[k] = torch.matmul(world2local_rot, input_trans.reshape((B, S, 3, 1)))[:,:,:,0]
    #         elif k in ['joints', 'verts']:
    #             # trans + joint + rot : B x S x J x 3
    #             J = input_dict[k].size(2) // 3
    #             input_pts = input_dict[k].reshape((B, S, J, 3)) 
    #             if invert:
    #                 input_pts = input_pts + trans2joint
    #                 output_pts = torch.matmul(local2world_rot.reshape((B,1,1,3,3)), input_pts.reshape((B, S, J, 3, 1)))[:,:,:,:,0]
    #                 output_pts = output_pts - trans2joint - world2local_trans.reshape((B, 1, 1, 3))
    #                 output_dict[k] = output_pts.reshape((B, S, J*3))
    #             else:
    #                 input_pts = input_pts + world2local_trans.reshape((B, 1, 1, 3)) + trans2joint
    #                 output_pts = torch.matmul(world2local_rot.reshape((B, 1, 1, 3, 3)), input_pts.reshape((B, S, J, 3, 1)))[:,:,:,:,0]
    #                 output_pts = output_pts - trans2joint
    #                 output_dict[k] = output_pts.reshape((B, S, J*3))
    #         elif k in ['joints_vel', 'verts_vel']:
    #             # rot : B x S x J x 3
    #             J = input_dict[k].size(2) // 3
    #             input_pts = input_dict[k].reshape((B, S, J, 3, 1))
    #             if invert:
    #                 outuput_pts = torch.matmul(local2world_rot.reshape((B,1,1,3,3)), input_pts)[:,:,:,:,0]
    #                 output_dict[k] = outuput_pts.reshape((B, S, J*3))
    #             else:
    #                 output_pts = torch.matmul(world2local_rot.reshape((B,1,1,3,3)), input_pts)[:,:,:,:,0]
    #                 output_dict[k] = output_pts.reshape((B, S, J*3))
    #         elif k in ['trans_vel', 'root_orient_vel']:
    #             # rot : B x S x 3
    #             input_pts = input_dict[k].reshape((B, S, 3, 1))
    #             if invert:
    #                 output_dict[k] = torch.matmul(local2world_rot, input_pts)[:,:,:,0]
    #             else:
    #                 output_dict[k] = torch.matmul(world2local_rot, input_pts)[:,:,:,0]
    #         else:
    #             print('Received an unexpected key when transforming world2local: %s!' % (k))
    #             exit()
        
    #     return output_dict


    # def zero_pad_tensors(self, pad_list, pad_size):
    #     '''
    #     Assumes tensors in pad_list are B x D
    #     '''
    #     new_pad_list = []
    #     for pad_idx, pad_tensor in enumerate(pad_list):
    #         padding = torch.zeros((pad_size, pad_tensor.size(1))).to(pad_tensor)
    #         new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
    #     return new_pad_list


    # ------------------------------------ Eval and Test Time Optimization Related Methods ---------------------------------

    # def roll_out(self, x_past, init_input_dict, num_steps, use_mean=False, 
    #                 z_seq=None, return_prior=False, gender=None, betas=None, return_z=False,
    #                 canonicalize_input=False,
    #                 uncanonicalize_output=False):
    #     '''
    #     Given input for first step, roll out using own output the entire time by sampling from the prior.
    #     Returns the global trajectory.

    #     Input:
    #     - x_past (B x steps_in x D_in)
    #     - initial_input_dict : dictionary of each initial state (B x steps_in x D), rotations should be matrices
    #                             (assumes initial state is already in its local coordinate system (translation at [0,0,z] and aligned))
    #     - num_steps : the number of timesteps to roll out
    #     - use_mean : if True, uses the mean of latent distribution instead of sampling
    #     - z_seq : (B x steps_out x D) if given, uses as the latent input to decoder at each step rather than sampling
    #     - return_prior : if True, also returns the output of the conditional prior at each step
    #     -gender : list of e.g. ['male', 'female', etc..] of length B
    #     -betas : B x steps_in x D
    #     -return_z : returns the sampled z sequence in addition to the output
    #     - canonicalize_input : if true, the input initial state is assumed to not be in the local aligned coordinate system. It will be transformed before using.
    #     - uncanonicalize_output : if true and canonicalize_input=True, will transform output back into the input frame rather than return in canonical frame.
    #     Returns: 
    #     - x_pred - dict of (B x num_steps x D_out) for each value. Rotations are all matrices.
    #     '''
    #     assert not return_prior, 'humor diffusion does not support return_prior!'

    #     J = len(SMPL_JOINTS)
    #     cur_input_dict = init_input_dict

    #     # need to transform init input to local frame
    #     world2aligned_rot = world2aligned_trans = None
    #     if canonicalize_input:
    #         B, _, _ = cur_input_dict[list(cur_input_dict.keys())[0]].size()
    #         # must transform initial input into the local frame
    #         # get world2aligned rot and translation
    #         root_orient_mat = cur_input_dict['root_orient']
    #         pose_body_mat = cur_input_dict['pose_body']
    #         if 'root_orient' in self.data_names and self.in_rot_rep != 'mat':
    #             root_orient_mat = convert_to_rotmat(root_orient_mat, rep=self.in_rot_rep)
    #         if 'pose_body' in self.data_names and self.in_rot_rep != 'mat':
    #             pose_body_mat = convert_to_rotmat(pose_body_mat, rep=self.in_rot_rep)

    #         root_orient_mat = root_orient_mat[:,-1].reshape((B, 3, 3))
    #         world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
    #         world2aligned_trans = torch.cat([-cur_input_dict['trans'][:,-1,:2], torch.zeros((B,1)).to(root_orient_mat)], axis=1)

    #         # compute trans2joint
    #         if self.need_trans2joint:
    #             trans2joint = -(cur_input_dict['joints'][:,-1,:2] + world2aligned_trans[:, :2])
    #             trans2joint = torch.cat([trans2joint, torch.zeros((B, 1)).to(trans2joint)], axis=1).reshape((B,1,1,3))

    #         # transform to local frame
    #         cur_input_dict = self.apply_world2local_trans(world2aligned_trans, world2aligned_rot, trans2joint, cur_input_dict, cur_input_dict, invert=False)

    #     # check to make sure we have enough input steps, if not, pad
    #     pad_x_past = x_past is not None and x_past.size(1) < self.steps_in
    #     pad_in_dict = cur_input_dict[list(cur_input_dict.keys())[0]].size(1) < self.steps_in
    #     if pad_x_past:
    #         num_pad_steps = self.steps_in -  x_past.size(1)
    #         cur_padding = torch.zeros((x_past.size(0), num_pad_steps, x_past.size(2))).to(x_past) # assuming all data is B x T x D
    #         x_past = torch.cat([cur_padding, x_past], axis=1)
    #     if pad_in_dict:
    #         for k in cur_input_dict.keys():
    #             cur_in_dat = cur_input_dict[k]
    #             num_pad_steps = self.steps_in - cur_in_dat.size(1)
    #             cur_padding = torch.zeros((cur_in_dat.size(0), num_pad_steps, cur_in_dat.size(2))).to(cur_in_dat) # assuming all data is B x T x D
    #             padded_in_dat = torch.cat([cur_padding, cur_in_dat], axis=1)
    #             cur_input_dict[k] = padded_in_dat

    #     if x_past is None or canonicalize_input:
    #         x_past = [cur_input_dict[k] for k in self.data_names]
    #         x_past = torch.cat(x_past, axis=2)
    #     B, S, D = x_past.size()
    #     past_in = x_past.reshape((B, -1))

    #     global_world2local_rot = torch.eye(3).reshape((1, 1, 3, 3)).expand((B, 1, 3, 3)).to(x_past)
    #     global_world2local_trans = torch.zeros((B, 1, 3)).to(x_past)
    #     if canonicalize_input and uncanonicalize_output:
    #         global_world2local_rot = world2aligned_rot.unsqueeze(1)
    #         global_world2local_trans = world2aligned_trans.unsqueeze(1)
    #     trans2joint = torch.zeros((B,1,1,3)).to(x_past)
    #     if self.need_trans2joint:
    #         trans2joint = -torch.cat([cur_input_dict['joints'][:,-1,:2], torch.zeros((B, 1)).to(x_past)], axis=1).reshape((B,1,1,3)) # same for whole sequence
    #     pred_local_seq = []
    #     pred_global_seq = []
    #     z_out_seq = []
    #     for t in range(num_steps):
    #         x_pred_dict = None
    #         # sample next step
    #         z_in = None
    #         if z_seq is not None:
    #             z_in = z_seq[:,t]
    #         sample_out = self.sample_step(past_in, use_mean=use_mean, z=z_in, return_prior=return_prior, return_z=return_z)
    #         if return_z:
    #             z_out = sample_out['z']
    #             z_out_seq.append(z_out)
    #         decoder_out = sample_out['decoder_out']

    #         # split output predictions and transform out rotations to matrices
    #         x_pred_dict = self.split_output(decoder_out, convert_rots=True)
    #         if self.steps_out > 1:
    #             for k in x_pred_dict.keys():
    #                 # only want immediate next frame prediction
    #                 x_pred_dict[k] = x_pred_dict[k][:,0:1,:]

    #         pred_local_seq.append(x_pred_dict)

    #         # output is the actual regressed joints, but input to next step can use smpl joints
    #         x_pred_smpl_joints = None
    #         if self.use_smpl_joint_inputs and gender is not None and betas is not None:
    #             # this assumes the model is actually outputting everything we need to run SMPL
    #             # also assumes single output step
    #             smpl_trans = x_pred_dict['trans'].reshape((B, 3))
    #             smpl_root_orient = rotation_matrix_to_angle_axis(x_pred_dict['root_orient'].reshape((B,3,3))).reshape((B, 3))
    #             smpl_betas = betas[:,0,:]
    #             smpl_pose_body = rotation_matrix_to_angle_axis(x_pred_dict['pose_body'].reshape((B*(J-1),3,3))).reshape((B, (J-1)*3))

    #             smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body]
    #             # each batch index may be a different gender
    #             gender_names = ['male', 'female', 'neutral']
    #             pred_joints = []
    #             prev_nbidx = 0
    #             cat_idx_map = np.ones((B), dtype=np.int)*-1
    #             for gender_name in gender_names:
    #                 gender_idx = np.array(gender) == gender_name
    #                 nbidx = np.sum(gender_idx)
    #                 cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int)
    #                 prev_nbidx += nbidx

    #                 gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

    #                 # need to pad extra frames with zeros in case not as long as expected 
    #                 pad_size = self.smpl_batch_size - nbidx
    #                 if pad_size == B:
    #                     # skip if no frames for this gender
    #                     continue
    #                 pad_list = gender_smpl_vals
    #                 if pad_size < 0:
    #                     raise Exception('SMPL model batch size not large enough to accomodate!')
    #                 elif pad_size > 0:
    #                     pad_list = self.zero_pad_tensors(pad_list, pad_size)
                    
    #                 # reconstruct SMPL
    #                 cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose = pad_list
    #                 bm = self.bm_dict[gender_name]
    #                 pred_body = bm(pose_body=cur_pred_pose, betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
    #                 if pad_size > 0:
    #                     pred_joints.append(pred_body.Jtr[:-pad_size])
    #                 else:
    #                     pred_joints.append(pred_body.Jtr)

    #             # cat all genders and reorder to original batch ordering
    #             x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:,:len(SMPL_JOINTS),:].reshape((B, 1, -1))
    #             x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]

    #         # prepare input to next step
    #         # update input dict with new frame
    #         del_keys = []
    #         for k in cur_input_dict.keys():
    #             if k in x_pred_dict:
    #                 # drop oldest frame and add new prediction
    #                 keep_frames = cur_input_dict[k][:,1:,:]
    #                 # print(keep_frames.size())

    #                 if k == 'joints' and self.use_smpl_joint_inputs and x_pred_smpl_joints is not None:
    #                     cur_input_dict[k] = torch.cat([keep_frames, x_pred_smpl_joints], axis=1)
    #                 else:
    #                     cur_input_dict[k] = torch.cat([keep_frames, x_pred_dict[k]], axis=1)
    #             else:
    #                 del_keys.append(k)
    #         for k in del_keys:
    #             del cur_input_dict[k]

    #         # get world2aligned rot and translation
    #         root_orient_mat = x_pred_dict['root_orient'][:,0,:].reshape((B, 3, 3))
    #         world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
    #         world2aligned_trans = torch.cat([-x_pred_dict['trans'][:,0,:2], torch.zeros((B,1)).to(x_past)], axis=1)

    #         #
    #         # transform inputs to this local frame (body pose is not affected) for next step
    #         #
    #         cur_input_dict = self.apply_world2local_trans(world2aligned_trans, world2aligned_rot, trans2joint, cur_input_dict, cur_input_dict, invert=False)

    #         # convert rots to correct input format
    #         if self.in_rot_rep == 'aa':
    #             if 'root_orient' in self.data_names:
    #                 cur_input_dict['root_orient'] = rotation_matrix_to_angle_axis(cur_input_dict['root_orient'].reshape((B*S,3,3))).reshape((B, S, 3))
    #             if 'pose_body' in self.data_names:
    #                 cur_input_dict['pose_body'] = rotation_matrix_to_angle_axis(cur_input_dict['pose_body'].reshape((B*S*(J-1),3,3))).reshape((B, S, (J-1)*3))
    #         elif self.in_rot_rep == '6d':
    #             if 'root_orient' in self.data_names:
    #                 cur_input_dict['root_orient'] = cur_input_dict['root_orient'][:,:,:6]
    #             if 'pose_body' in self.data_names:
    #                 cur_input_dict['pose_body'] = cur_input_dict['pose_body'].reshape((B, S, J-1, 9))[:,:,:,:6].reshape((B, S, (J-1)*6))

    #         #
    #         # compute current world output and update world2local transform
    #         #
    #         cur_world_dict = dict()
    #         cur_world_dict = self.apply_world2local_trans(global_world2local_trans, global_world2local_rot, trans2joint, x_pred_dict, cur_world_dict, invert=True)
    #         #
    #         # update world2local transform
    #         #
    #         global_world2local_trans = torch.cat([-cur_world_dict['trans'][:,0:1,:2], torch.zeros((B, 1, 1)).to(x_past)], axis=2)
    #         # print(world2aligned_rot)
    #         global_world2local_rot = torch.matmul(global_world2local_rot, world2aligned_rot.reshape((B, 1, 3, 3)))

    #         pred_global_seq.append(cur_world_dict)

    #         # cat all inputs together to form past_in
    #         in_data_list = []
    #         for k in self.data_names:
    #             in_data_list.append(cur_input_dict[k])
    #         past_in = torch.cat(in_data_list, axis=2)
    #         past_in = past_in.reshape((B, -1))

    #     # aggregate global pred_seq
    #     pred_seq_out = dict()
    #     for k in pred_global_seq[0].keys():
    #         pred_seq_out[k] = torch.cat([pred_global_seq[i][k] for i in range(len(pred_global_seq))], axis=1)

    #     if return_z:
    #         z_out_seq = torch.stack(z_out_seq, dim=1)
    #         pred_seq_out['z'] = z_out_seq
 
    #     return pred_seq_out


    # def sample_step(self, past_in, t_in=None, use_mean=False, z=None, return_z=False):
    #     '''
    #     Given past, samples next future state by sampling from prior or posterior and decoding.
    #     If z (B x D) is not None, uses the given z instead of sampling from posterior or prior

    #     Returns:
    #     - decoder_out : (B x steps_out x D) output of the decoder for the immediate next step
    #     '''
    #     B = past_in.size(0)
    #     past_in = past_in.reshape((B, -1)) # B x 339
    #     z_noise = torch.randn((B, self.z_dim)).to(past_in.device)
    #     assert z_noise.shape[0] == past_in.shape[0], 'z noise batch size must match past_in batch size!'

    #     z_noise = torch.cat([z_noise, z_noise], 0) # duplicate for conditional and unconditional diffusion
    #     past_in_null = past_in # TODO: Not sure here. What should be our null class? Here null class is the copy of real class, so actually no unconditional sampling
    #     past_in = torch.cat([past_in, past_in_null], 0) # duplicate for conditional and unconditional diffusion
    #     model_kwargs = dict(cond=past_in, cfg_scale=self.cfg_scale)
        
    #     z_hat = self.diffusion.p_sample_loop(
    #         self.diffusion_model.forward_with_cfg, z_noise.shape, z,  clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=past_in.device
    #     )
    #     z_hat, _ = z_hat.chunk(2, dim=0)  # Remove null class samples

    #     # decode to get next step
    #     decoder_out = self.decode(z_hat, past_in)
    #     decoder_out = decoder_out.reshape((B, self.steps_out, -1)) # B x steps_out x D_out

    #     out_dict = {'decoder_out' : decoder_out}
    #     if return_z:
    #         out_dict['z'] = z_hat
        
    #     return out_dict


    # def infer_global_seq(self, global_seq, full_forward_pass=False):
    #     '''
    #     Given a sequence of global states, formats it (transform each step into local frame and makde B x steps_in x D)
    #     and runs inference (compute prior/posterior of z for the sequence).

    #     If full_forward_pass is true, does an entire forward pass at each step rather than just inference.

    #     Rotations should be in in_rot_rep format.
    #     '''
    #     # used to compute output zero padding
    #     needed_future_steps = (self.steps_out-1)*self.out_step_size
        
    #     latent_states = []
    #     pred_dict_seq = []
    #     B, T, _ = global_seq[list(global_seq.keys())[0]].size()
    #     J = len(SMPL_JOINTS)
    #     trans2joint = None
    #     for t in range(T-1):
    #         # get world2aligned rot and translation
    #         world2aligned_rot = world2aligned_trans = None

    #         root_orient_mat = global_seq['root_orient'][:,t,:].reshape((B, 3, 3))
    #         world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
    #         world2aligned_trans = torch.cat([-global_seq['trans'][:,t,:2], torch.zeros((B,1)).to(root_orient_mat)], axis=1)

    #         # compute trans2joint at first step
    #         if t == 0 and self.need_trans2joint:
    #             trans2joint = -(global_seq['joints'][:,t,:2] + world2aligned_trans[:, :2]) # we cannot make the assumption that the first frame is already canonical
    #             trans2joint = torch.cat([trans2joint, torch.zeros((B, 1)).to(trans2joint)], axis=1).reshape((B,1,1,3))

    #         # get current window
    #         cur_data_dict = dict()
    #         for k in global_seq.keys():
    #             # get in steps
    #             in_sidx = max(0, t-self.steps_in+1)
    #             cur_in_seq = global_seq[k][:,in_sidx:(t+1),:]
    #             if cur_in_seq.size(1) < self.steps_in:
    #                 # must zero pad front
    #                 num_pad_steps = self.steps_in - cur_in_seq.size(1)
    #                 cur_padding = torch.zeros((cur_in_seq.size(0), num_pad_steps, cur_in_seq.size(2))).to(cur_in_seq) # assuming all data is B x T x D
    #                 cur_in_seq = torch.cat([cur_padding, cur_in_seq], axis=1)

    #             # get out steps
    #             cur_out_seq = global_seq[k][:,(t+1):(t+2 + needed_future_steps):self.out_step_size]
    #             if cur_out_seq.size(1) < self.steps_out:
    #                 # zero pad
    #                 num_pad_steps = self.steps_out - cur_out_seq.size(1)
    #                 cur_padding = torch.zeros_like(cur_out_seq[:,0])
    #                 cur_padding = torch.stack([cur_padding]*num_pad_steps, axis=1)
    #                 cur_out_seq = torch.cat([cur_out_seq, cur_padding], axis=1)
    #             cur_data_dict[k] = torch.cat([cur_in_seq, cur_out_seq], axis=1)

    #         # transform to local frame
    #         cur_data_dict = self.apply_world2local_trans(world2aligned_trans, world2aligned_rot, trans2joint, cur_data_dict, cur_data_dict, invert=False)

    #         # create x_past and x_t
    #         # cat all inputs together to form past_in
    #         in_data_list = []
    #         for k in self.data_names:
    #             in_data_list.append(cur_data_dict[k][:,:self.steps_in,:])
    #         x_past = torch.cat(in_data_list, axis=2)
    #         # cat all outputs together to form x_t
    #         out_data_list = []
    #         for k in self.data_names:
    #             out_data_list.append(cur_data_dict[k][:,self.steps_in:,:])
    #         x_t = torch.cat(out_data_list, axis=2)

    #         if full_forward_pass:
    #             x_pred_dict = self(x_past, x_t)
    #             pred_dict_seq.append(x_pred_dict)
    #         else:
    #             # perform inference
    #             z = self.infer(x_past, x_t)
    #             # save z
    #             latent_states.append(z)

    #     if full_forward_pass:
    #         # pred_dict_seq
    #         pred_seq_out = dict()
    #         for k in pred_dict_seq[0].keys():
    #             # print(k)
    #             if k == 'posterior_distrib' or k == 'prior_distrib':
    #                 m = torch.stack([pred_dict_seq[i][k][0] for i in range(len(pred_dict_seq))], axis=1)
    #                 v = torch.stack([pred_dict_seq[i][k][1] for i in range(len(pred_dict_seq))], axis=1)
    #                 pred_seq_out[k] = (m, v)
    #             else:
    #                 pred_seq_out[k] = torch.stack([pred_dict_seq[i][k] for i in range(len(pred_dict_seq))], axis=1)

    #         return pred_seq_out
    #     else:
    #         latent_states = torch.stack(latent_states, axis=1)

    #         return latent_states


    # def infer(self, x_past, x_t):
    #     '''
    #     Inference (compute prior and posterior distribution of z) for a batch of single steps.
    #     NOTE: must do processing before passing in to ensure correct format that this function expects.
        
    #     Input:
    #     - x_past (B x steps_in x D)
    #     - x_t    (B x steps_out x D)

    #     Returns:
    #     - prior_distrib (mu, var)
    #     - posterior_distrib (mu, var)
    #     '''

    #     B, _, D = x_past.size()
    #     past_in = x_past.reshape((B, -1))
    #     t_in = x_t.reshape((B, -1))

    #     return self.infer_step(past_in, t_in)


    # def infer_step(self, past_in, t_in):
    #     '''
    #     single step that computes both prior and posterior for training. Samples from posterior
    #     '''
    #     B = past_in.size(0)
    #     # use past and future to encode latent transition
    #     z = self.encoder(past_in, t_in)

    #     return z