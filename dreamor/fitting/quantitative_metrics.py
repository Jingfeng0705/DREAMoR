import os, sys
import numpy as np
import torch
import csv, random
from tqdm import tqdm

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from datasets.amass_utils import CONTACT_INDS
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS
from fitting.fitting_utils import perspective_projection, bdot, compute_plane_intersection, load_res, prep_res, run_smpl
from fitting.eval_utils import quant_eval_3d, SMPL_SIZES, GRND_PEN_THRESH_LIST, get_grnd_pen_key, AMASS_EVAL_BLACKLIST
from utils.config import SplitLineParser
from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues
from utils.logging import mkdir
from body_model.body_model import BodyModel

J_BODY = len(SMPL_JOINTS)-1 # no root

GT_RES_NAME = 'gt_results'
PRED_RES_NAME = 'output_results'
STAGES_RES_NAMES = ['stage1_results', 'stage2_results', 'stage3_init_results']
OBS_NAME = 'observations'
FPS = 30

GRND_PEN_THRESH_LIST = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
IMW, IMH = 1920, 1080 # all data
DATA_FPS = 30.0
DATA_h = 1.0 / DATA_FPS

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    parser.add_argument('--results', type=str, required=True, help='Path to the results directory to run eval on. Should be a directory of directories - one for each sequence.')
    parser.add_argument('--out', type=str, required=True, help='Path to save evaluation results/visualizations to.')

    # quant eval and options
    parser.add_argument('--quant-stages', dest='quant_stages', action='store_true', help="If given, runs quantitative evaluation on all stages rather than just final.")
    parser.set_defaults(quant_stages=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args

def compute_subset_smpl_joint_err(eval_pred_joints, eval_gt_joints, subset='ee'):
    '''
    Compute SMPL joint position errors betwen pred_joints and gt_joints for the given subject.
    Assumed size of B x 22 x 3
    Options:
    - ee : end-effectors. hands, toebase, and ankle
    - legs : knees, angles, and toes
    '''
    subset_inds = None
    if subset == 'ee':
        subset_inds = [SMPL_JOINTS['leftFoot'], SMPL_JOINTS['rightFoot'],
                       SMPL_JOINTS['leftToeBase'], SMPL_JOINTS['rightToeBase'],
                       SMPL_JOINTS['leftHand'], SMPL_JOINTS['rightHand']]
    elif subset == 'legs':
        subset_inds = [SMPL_JOINTS['leftFoot'], SMPL_JOINTS['rightFoot'],
                       SMPL_JOINTS['leftToeBase'], SMPL_JOINTS['rightToeBase'],
                       SMPL_JOINTS['leftLeg'], SMPL_JOINTS['rightLeg']]
    else:
        print('Unrecognized joint subset!')
        exit()
    
    joint_err = torch.norm(eval_pred_joints[:, subset_inds] - eval_gt_joints[:, subset_inds], dim=-1)
    return joint_err

def compute_joint_accel(joint_seq):
    ''' Magnitude of joint accelerations for joint_seq : T x J x 3 '''
    joint_accel = joint_seq[:-2] - (2*joint_seq[1:-1]) + joint_seq[2:]
    joint_accel = joint_accel / ((DATA_h**2))
    joint_accel_mag = torch.norm(joint_accel, dim=-1)
    return joint_accel, joint_accel_mag

def compute_toe_floor_pen(joint_seq, floor_plane, thresh_list=[0.0]):
    '''
    Given SMPL body joints sequence and the floor plane, computes number of times
    the toes penetrate the floor and the total number of frames.

    - thresh_list : compute the penetration ratio for each threshold in cm in this list

    Returns:
    - list of num_penetrations for each threshold, the number of total frames, and penetration distance at threshold 0.0
    '''
    toe_joints = joint_seq[:,[SMPL_JOINTS['leftToeBase'], SMPL_JOINTS['rightToeBase']], :]
    toe_joints = toe_joints.reshape((-1, 3))
    floor_normal = floor_plane[:3].reshape((1, 3))
    floor_normal = floor_normal / torch.norm(floor_normal, dim=-1, keepdim=True)
    floor_normal = floor_normal.expand_as(toe_joints)

    _, s = compute_plane_intersection(toe_joints, -floor_normal, floor_plane.reshape((1, 4)).expand((toe_joints.size(0), 4)))

    num_pen_list = torch.zeros((len(thresh_list))).to(torch.int).to(joint_seq.device)
    for thresh_idx, pen_thresh in enumerate(thresh_list):
        num_pen_thresh = torch.sum(s < -pen_thresh)
        num_pen_list[thresh_idx] = num_pen_thresh

    num_tot = s.size(0)

    pen_dist = torch.Tensor(np.array((0)))
    if torch.sum(s < 0) > 0:
        pen_dist = -s[s < 0] # distance of penetration at threshold of 0

    return num_pen_list, num_tot, pen_dist

def get_grnd_pen_key(thresh):
    return 'ground_pen@%0.2f' % (thresh)

def quant_eval_3d(eval_dict, pred_data, gt_data, obs_data):
    # get positional errors for each modality
    for modality in ['joints3d', 'verts3d', 'mesh3d']:
        eval_pred = pred_data[modality]
        eval_gt = gt_data[modality]

        # all positional errors
        pos_err_all = torch.norm(eval_pred - eval_gt, dim=-1).detach().cpu().numpy()
        eval_dict[modality + '_all'].append(pos_err_all)

        # ee and legs
        if modality == 'joints3d':
            joints3d_ee = compute_subset_smpl_joint_err(eval_pred, eval_gt, subset='ee').detach().cpu().numpy()
            joints3d_legs = compute_subset_smpl_joint_err(eval_pred, eval_gt, subset='legs').detach().cpu().numpy()
            eval_dict['joints3d_ee'].append(joints3d_ee)
            eval_dict['joints3d_legs'].append(joints3d_legs)

        # split by occluded/visible if this was the observed modality
        if modality in obs_data:
            # visible data (based on observed)
            eval_obs = obs_data[modality]
            invis_mask = torch.isinf(eval_obs)
            vis_mask = torch.logical_not(invis_mask) # T x N x 3
            num_invis_pts = torch.sum(invis_mask[:,:,0])
            num_vis_pts = torch.sum(vis_mask[:,:,0])

            if num_vis_pts > 0:
                pred_vis = eval_pred[vis_mask].reshape((num_vis_pts, 3))
                gt_vis = eval_gt[vis_mask].reshape((num_vis_pts, 3))
                vis_err = torch.norm(pred_vis - gt_vis, dim=-1).detach().cpu().numpy()
                eval_dict[modality + '_vis'].append(vis_err)
            else:
                eval_dict[modality + '_vis'].append(np.zeros((0)))

            # invisible data 
            if num_invis_pts > 0:
                pred_invis = eval_pred[invis_mask].reshape((num_invis_pts, 3))
                gt_invis = eval_gt[invis_mask].reshape((num_invis_pts, 3))
                invis_err = torch.norm(pred_invis - gt_invis, dim=-1).detach().cpu().numpy()
                eval_dict[modality + '_occ'].append(invis_err)
            else:
                eval_dict[modality + '_occ'].append(np.zeros((0)))

    # per-joint acceleration mag
    pred_joint_accel, pred_joint_accel_mag = compute_joint_accel(pred_data['joints3d'])
    eval_dict['accel_mag'].append(pred_joint_accel_mag.detach().cpu().numpy())

    # toe-floor penetration
    floor_plane = torch.zeros((4)).to(pred_data['joints3d'])
    floor_plane[2] = 1.0
    num_pen_list, num_tot, pen_dist = compute_toe_floor_pen(pred_data['joints3d'], floor_plane, thresh_list=GRND_PEN_THRESH_LIST)
    eval_dict['ground_pen_dist'].append(pen_dist.detach().cpu().numpy())
    for thresh_idx, pen_thresh in enumerate(GRND_PEN_THRESH_LIST):
        cur_pen_key = get_grnd_pen_key(pen_thresh)
        eval_dict[cur_pen_key].append(num_pen_list[thresh_idx].detach().cpu().item())
        eval_dict[cur_pen_key + '_cnt'].append(num_tot)

    # contact classification (output number correct and total frame cnt)
    pred_contacts = pred_data['contacts'][:,CONTACT_INDS] # only compare joints for which the prior is trained
    gt_contacts = gt_data['contacts'][:,CONTACT_INDS]
    num_correct = np.sum((pred_contacts - gt_contacts) == 0)
    total_cnt = pred_contacts.shape[0]*pred_contacts.shape[1]
    eval_dict['contact_acc'].append(num_correct)
    eval_dict['contact_acc_cnt'].append(total_cnt)
    
def main(args):
    # print(args)
    mkdir(args.out)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    quant_out_path = args.out
    quant_res_dict = None; obs_mod = None
    seq_name_list = []; result_dir = args.results
    
    data_gt = np.load(os.path.join(result_dir, 'gt.npz'))
    # #batch_size * #seqs * #seq_len(always 1) * shape
    observed_data, gt_data = data_gt
    data_output = np.load(os.path.join(result_dir, 'output.npz'))
    _, output_data = data_output
    batch_size = gt_data['betas'].shape[0]
    seq_len = gt_data['betas'].shape[1]
    num_betas = gt_data['betas'].shape[2]
    
    for idx in tqdm(range(seq_len)):
        gt_data_idx = {k: torch.Tensor(gt_data[k][:, idx]).to(device) for k in gt_data.files}
        output_data_idx = {k: torch.Tensor(output_data[k][:, idx]).to(device) for k in output_data.files}
        # Process ground-truth & prediction results
        gt_res = prep_res(gt_data_idx, device, batch_size)
        pred_res = prep_res(output_data_idx, device, batch_size)
        # Convert SMPL parameters to 3D joints / verts
        gt_bm = BodyModel(bm_path='../body_models/smplh/neutral/model.npz',
                        num_betas=num_betas, batch_size=batch_size).to(device)
        pred_bm = BodyModel(bm_path='../body_models/smplh/neutral/model.npz',
                            num_betas=num_betas, batch_size=batch_size).to(device)
        gt_body   = run_smpl(gt_res, gt_bm)
        pred_body = run_smpl(pred_res, pred_bm)
        # ========== Quantitative Evaluation ==========
        # Load observed data (3D keypoints, 2D joints, or point clouds)
        obs_dict = observed_data[:][idx]
        obs_data = {k: torch.Tensor(obs_dict[k]).to(device) for k in obs_dict.files}
        # Initialize quant_res_dict on first sequence
        if quant_res_dict is None:
            quant_res_dict = {}
            obs_mod = list(obs_data.keys())[0]
            for name in [PRED_RES_NAME] + (STAGES_RES_NAMES if args.quant_stages else []):
                quant_res_dict[name] = {}
                # Prepare empty lists for each metric
                for m in ['joints3d_all', 'joints3d_ee', 'joints3d_legs', 'verts3d_all', 'mesh3d_all',
                            'contact_acc', 'accel_mag', 'ground_pen_dist']:
                    quant_res_dict[name][m] = []
                # Ground-penetration counts for each threshold
                for thr in GRND_PEN_THRESH_LIST:
                    key = get_grnd_pen_key(thr)
                    quant_res_dict[name][key]     = []
                    quant_res_dict[name][key+'_cnt'] = []
        
        # Prepare GT eval dict for quant_eval_3d
        gt_eval = {
            'joints3d': gt_body.Jtr[:, :len(SMPL_JOINTS)],         # all joints
            'verts3d' : gt_body.v[:, KEYPT_VERTS],                # keypoint vertices
            'mesh3d'  : gt_body.v,                                # full mesh
            'contacts': gt_res.get('contacts', None)               # ground contacts
        }

        # Evaluate final prediction and optionally stages
        to_eval = [(pred_body, PRED_RES_NAME, pred_res)]
        if args.quant_stages:
            for stage in STAGES_RES_NAMES:
                stage_res = load_res(result_dir, stage + '.npz')
                stage_body = run_smpl(prep_res(stage_res, device, gt_res['trans'].shape[0]), pred_bm)
                to_eval.append((stage_body, stage, stage_res))

        for body, name, res in to_eval:
            pred_eval = {
                'joints3d': body.Jtr[:, :len(SMPL_JOINTS)],
                'verts3d' : body.v[:, KEYPT_VERTS],
                'mesh3d'  : body.v,
                'contacts': res.get('contacts', gt_res['contacts'])
            }
            # quant_eval_3d appends the following metrics into quant_res_dict[name]:
            #   - joint errors (all, end-effectors, legs)
            #   - vertex errors
            #   - mesh errors
            #   - contact classification accuracy
            #   - acceleration magnitude (smoothness)
            #   - ground penetration distance & counts per threshold
            quant_eval_3d(quant_res_dict[name], pred_eval, gt_eval, obs_data)
    
    # evals that need to be summed/meaned over all entries
    agg_vals = ['joints3d_all', 'joints3d_ee', 'joints3d_legs', 'verts3d_all', 'mesh3d_all']
    agg_vals += [obs_mod + '_vis', obs_mod + '_occ']
    agg_vals += ['accel_mag', 'ground_pen_dist']
    frac_vals = ['contact_acc']
    for pen_thresh in GRND_PEN_THRESH_LIST:
        frac_vals.append(get_grnd_pen_key(pen_thresh))

    supp_vals = ['ground_pen_dist_normalized', 'ground_pen_mean_agg_frac'] # values computed after from everything else

    per_seq_out_vals = agg_vals + frac_vals
    out_vals = agg_vals + frac_vals + supp_vals

    eval_names_out = []
    eval_means_out = []
    eval_maxs_out = []
    eval_meds_out = []
    for eval_name, eval_res in quant_res_dict.items():
        eval_names_out.append(eval_name)

        # agg evals
        agg_per_seq_means = []
        agg_all_means = []
        agg_all_stds = []
        agg_all_meds = []
        agg_all_maxs = []
        agg_all_mins = []
        for agg_val in agg_vals:
            per_seq_val = eval_res[agg_val]
            per_seq_means = []
            for x in per_seq_val:
                seq_mean = np.mean(x) if x.shape[0] > 0 else -1
                per_seq_means.append(seq_mean)
            agg_per_seq_means.append(per_seq_means)

            all_val = np.concatenate(eval_res[agg_val], axis=0)
            if all_val.shape[0] == 0:
                dummy_res = -1
                agg_all_means.append(dummy_res)
                agg_all_stds.append(dummy_res)
                agg_all_meds.append(dummy_res)
                agg_all_maxs.append(dummy_res)
                agg_all_mins.append(dummy_res)
            else:
                all_mean = np.mean(all_val)
                agg_all_means.append(all_mean)
                all_std = np.std(all_val)
                agg_all_stds.append(all_std)
                all_median = np.median(all_val)
                agg_all_meds.append(all_median)
                all_max = np.amax(all_val)
                agg_all_maxs.append(all_max)
                all_min = np.amin(all_val)
                agg_all_mins.append(all_min)

        # fraction evals
        for frac_val in frac_vals:
            frac_val_cnt = frac_val + '_cnt'
            per_seq_val = eval_res[frac_val]
            per_seq_cnt = eval_res[frac_val_cnt]
            per_seq_means = [float(seq_val) / seq_cnt for seq_val, seq_cnt in zip(per_seq_val, per_seq_cnt)]
            agg_per_seq_means.append(per_seq_means)

            all_val = np.array(eval_res[frac_val], dtype=float)
            all_cnt = np.array(eval_res[frac_val_cnt], dtype=float)
            all_mean = np.sum(all_val) / np.sum(all_cnt)
            agg_all_means.append(all_mean)
            agg_all_stds.append(-1)
            agg_all_meds.append(-1)
            agg_all_maxs.append(-1)
            agg_all_mins.append(-1)

        # supplemental values
        ground_pen_dist_normalized_mean = agg_all_means[out_vals.index('ground_pen_dist')]*agg_all_means[out_vals.index(get_grnd_pen_key(0.0))]
        agg_all_means.append(ground_pen_dist_normalized_mean)
        ground_pen_dist_normalized_med = agg_all_meds[out_vals.index('ground_pen_dist')]*agg_all_means[out_vals.index(get_grnd_pen_key(0.0))]
        agg_all_meds.append(ground_pen_dist_normalized_med)
        ground_pen_frac_sum = 0.0
        for pen_thresh in GRND_PEN_THRESH_LIST:
            ground_pen_frac_sum += agg_all_means[out_vals.index(get_grnd_pen_key(pen_thresh))]
        ground_pen_mean_agg_frac = ground_pen_frac_sum / len(GRND_PEN_THRESH_LIST)
        agg_all_means.append(ground_pen_mean_agg_frac)
        agg_all_meds.append(-1)

        agg_all_stds.extend([-1, -1])
        agg_all_maxs.extend([-1, -1])
        agg_all_mins.extend([-1, -1])

        # save
        eval_means_out.append(agg_all_means)
        eval_maxs_out.append(agg_all_maxs)
        eval_meds_out.append(agg_all_meds)


        stage_out_path = os.path.join(quant_out_path, eval_name + '_per_seq_mean.csv')
        with open(stage_out_path, 'w') as f:
            csvw = csv.writer(f, delimiter=',')
            # write heading
            csvw.writerow(['seq_name'] + per_seq_out_vals)
            # write data
            for j, seq_name in enumerate(seq_name_list):
                cur_row = [agg_per_seq_means[vidx][j] for vidx in range(len(per_seq_out_vals))] 
                csvw.writerow([seq_name] + cur_row)

        stats_name_list = ['mean', 'std', 'median', 'max', 'min']
        stats_list = [agg_all_means, agg_all_stds, agg_all_meds, agg_all_maxs, agg_all_mins]
        for stat_name, stat_data in zip(stats_name_list, stats_list):
            agg_out_path = os.path.join(quant_out_path, eval_name + '_agg_%s.csv' % (stat_name))
            with open(agg_out_path, 'w') as f:
                csvw = csv.writer(f, delimiter=',')
                # write heading
                csvw.writerow(out_vals)
                # write data
                csvw.writerow(stat_data)

    #  one file that saves all means together for easy compare
    stats_name_list = ['mean', 'max', 'median']
    stats_list = [eval_means_out, eval_maxs_out, eval_meds_out]
    for stat_name, stat_data in zip(stats_name_list, stats_list):
        compare_out_path = os.path.join(quant_out_path, 'compare_%s.csv' % (stat_name))
        with open(compare_out_path, 'w') as f:
            csvw = csv.writer(f, delimiter=',')
            # write heading
            csvw.writerow(['method'] + out_vals)
            # write data
            for eval_name_out, eval_stat_out in zip(eval_names_out, stat_data):
                csvw.writerow([eval_name_out] + eval_stat_out)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    main(args)