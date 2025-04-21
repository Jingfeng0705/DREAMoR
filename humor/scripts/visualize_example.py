import os
import numpy as np
import torch
import yaml
import argparse
import cv2
from humor.datasets.amass_discrete_dataset import AmassDiscreteDataset
from torch.utils.data import Dataset, DataLoader
from humor.body_model.body_model import BodyModel
from humor.utils.transforms import matrot2axisangle
from humor.datasets.amass_utils import CONTACT_INDS, NUM_BODY_JOINTS
from humor.viz.utils import viz_smpl_seq
from humor.body_model.utils import SMPL_JOINTS

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    male_bm_path   = os.path.join(args.smplh_path, 'male/model.npz')
    female_bm_path = os.path.join(args.smplh_path, 'female/model.npz')

    slice_size = args.data_steps_in + args.data_steps_out
    
    male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=slice_size).to(device)
    male_bm_world = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=args.sample_num_frames+1).to(device)
    male_bm_global = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=args.sample_num_frames, use_vtx_selector=False).to(device)

    female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=slice_size).to(device)
    female_bm_world = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=args.sample_num_frames+1).to(device)
    female_bm_global = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=args.sample_num_frames, use_vtx_selector=False).to(device)

    dataset = AmassDiscreteDataset(
        split='train',
        data_paths=args.data_path,
        split_by=args.split_by,
        sample_num_frames=args.sample_num_frames,
        step_frames_in=args.data_steps_in,
        step_frames_out=args.data_steps_out,
        data_rot_rep=args.data_rot_rep,
        data_return_config=args.data_return_config,
    )

    loader = DataLoader(dataset, 
                        batch_size=1,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                        worker_init_fn=lambda _: np.random.seed()) 

    assert args.batch_size == 1, "Batch size be 1 for visualization"
    data_in, data_out, meta = next(iter(loader))

    betas = meta['betas'].to(device)

    root_orient = data_out['root_orient']
    root_orient = matrot2axisangle(root_orient.numpy().reshape((args.batch_size, args.sample_num_frames, 9))).reshape((args.batch_size, args.sample_num_frames, 3))
    root_orient = torch.Tensor(root_orient).to(device)

    pose_body = data_out['pose_body']
    pose_body = matrot2axisangle(pose_body.numpy().reshape((args.batch_size* args.sample_num_frames, NUM_BODY_JOINTS, 9))).reshape((args.batch_size, args.sample_num_frames, NUM_BODY_JOINTS*3))
    pose_body = torch.Tensor(pose_body).to(device)

    trans = data_out['trans'].to(device)
    joints = data_out['joints'].to(device)
    joints_vel = data_out['joints_vel'].to(device)
    trans_vel = data_out['trans_vel'].to(device)


    bm_global = male_bm_global if meta['gender'][0] == 'male' else female_bm_global
    body = bm_global(pose_body=pose_body[0], 
                    pose_hand=None, 
                    betas=betas[0,0].reshape((1, -1)).expand((args.sample_num_frames, 16)), 
                    root_orient=root_orient[0], 
                    trans=trans[0].squeeze(1),)


    viz_contacts = None
    contacts = data_out['contacts'][0].squeeze(1)
    viz_contacts = torch.zeros((args.batch_size, args.sample_num_frames, len(SMPL_JOINTS))).to(contacts)
    viz_contacts[:,:,CONTACT_INDS] = contacts

        
    viz_smpl_seq(body, imw=1080, imh=1080, fps=30,
                render_body=True, render_joints=True, render_skeleton=False, render_ground=True,
                contacts=viz_contacts[0],
                joints_seq=joints[0], use_offscreen=True)
    output_folder = f"./render_out"
    # make a video with all the frames
    assert(os.path.exists(output_folder)), f"Output folder {output_folder} does not exist"
    frame_files = sorted(os.listdir(output_folder))
    frames = []
    for frame_file in frame_files:
        if frame_file.endswith(".png"):
            frame_path = os.path.join(output_folder, frame_file)
            img = cv2.imread(frame_path)
            frames.append(img)
    
    if frames != []:
        height, width, _ = frames[0].shape
        output_path = os.path.join(output_folder, "video.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config_path)
    args.data_path = config['data_path']
    args.split_by = config['split_by']
    args.sample_num_frames = config['sample_num_frames']
    args.data_steps_in = config['data_steps_in']
    args.data_steps_out = config['data_steps_out']
    args.data_rot_rep = config['data_rot_rep']
    args.data_return_config = config['data_return_config']
    args.batch_size = config['batch_size']
    args.smplh_path = config['smplh_path']
    main(args)
    