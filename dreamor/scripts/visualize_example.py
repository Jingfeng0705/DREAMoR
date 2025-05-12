import os
import numpy as np
import torch
import argparse
import cv2

from torch.utils.data import DataLoader
from dreamor.datasets.amass_discrete_dataset import AmassDiscreteDataset
from dreamor.body_model.body_model import BodyModel
from dreamor.utils.transforms import matrot2axisangle
from dreamor.datasets.amass_utils import CONTACT_INDS, NUM_BODY_JOINTS
from dreamor.viz.utils import viz_smpl_seq
from dreamor.body_model.utils import SMPL_JOINTS
from dreamor.utils.config_new import ConfigParser

def main(args, app_transform):
    base_args = getattr(args, "base_dict")
    dataset_args = getattr(args, "dataset_dict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    male_bm_path = os.path.join('./body_models/smplh', 'male/model.npz')
    female_bm_path = os.path.join('./body_models/smplh', 'female/model.npz')

    slice_size = dataset_args['step_frames_in'] + dataset_args['step_frames_out']

    # Body models
    male_bm_global = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=dataset_args['sample_num_frames'], use_vtx_selector=False).to(device)
    female_bm_global = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=dataset_args['sample_num_frames'], use_vtx_selector=False).to(device)

    # Dataset and Loader
    dataset = AmassDiscreteDataset(
        split='test',
        data_paths=dataset_args['data_paths'],
        split_by=dataset_args['split_by'],
        sample_num_frames=dataset_args['sample_num_frames'],
        step_frames_in=dataset_args['step_frames_in'],
        step_frames_out=dataset_args['step_frames_out'],
        data_rot_rep=dataset_args['data_rot_rep'],
        data_return_config=dataset_args['data_return_config'],
    )

    loader = DataLoader(dataset, 
                        batch_size=base_args.get('batch_size', 1),
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                        worker_init_fn=lambda _: np.random.seed())

    batch = next(iter(loader))
    data_in, data_out, meta = batch

    batch_size = data_in['trans'].shape[0]
    print(f"Visualizing {batch_size} sequences...")

    # Preparing data
    betas = meta['betas'].to(device)

    root_orient = matrot2axisangle(data_out['root_orient'].numpy().reshape((batch_size, dataset_args['sample_num_frames'], 9))).reshape((batch_size, dataset_args['sample_num_frames'], 3))
    root_orient = torch.Tensor(root_orient).to(device)

    pose_body = matrot2axisangle(data_out['pose_body'].numpy().reshape((batch_size * dataset_args['sample_num_frames'], NUM_BODY_JOINTS, 9))).reshape((batch_size, dataset_args['sample_num_frames'], NUM_BODY_JOINTS * 3))
    pose_body = torch.Tensor(pose_body).to(device)

    trans = data_out['trans'].to(device)
    joints = data_out['joints'].to(device)

    contacts = data_out['contacts'].squeeze(2).to(device)
    viz_contacts = torch.zeros((batch_size, dataset_args['sample_num_frames'], len(SMPL_JOINTS)), device=device)
    viz_contacts[:,:,CONTACT_INDS] = contacts

    # Output setup
    output_folder = base_args['out']
    input_folder = './render_out'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(batch_size):
        bm_global = male_bm_global if meta['gender'][i] == 'male' else female_bm_global

        body = bm_global(
            pose_body=pose_body[i],
            pose_hand=None,
            betas=betas[i,0].reshape((1, -1)).expand((dataset_args['sample_num_frames'], 16)),
            root_orient=root_orient[i],
            trans=trans[i].squeeze(1)
        )
        CAM_OFFSET = [0.0, 2.25, 0.9]
        
        viz_smpl_seq(
            body,
            imw=1280, imh=720, fps=30,
            render_body=True,
            render_joints=True,
            render_skeleton=False,
            render_ground=True,
            contacts=viz_contacts[i],
            joints_seq=joints[i],
            use_offscreen=True,
            RGBA=True,
            follow_camera=True,
            cam_offset=CAM_OFFSET,
            joint_color=[ 0.0, 1.0, 0.0 ],
            point_color=[0.0, 0.0, 1.0],
            skel_color=[0.5, 0.5, 0.5],
            joint_rad=0.02,
            point_rad=0.02
        )

        # Save video
        save_video(input_folder, output_folder, f"video_{i}.mp4")

    print(f"All videos saved to {output_folder}")


def save_video(input_folder, output_folder, video_filename):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    frames = [cv2.imread(os.path.join(input_folder, f)) for f in frame_files]

    if frames:
        height, width, _ = frames[0].shape
        output_path = os.path.join(output_folder, video_filename)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Video saved to {output_path}")

    # Clean up frames
    for f in frame_files:
        os.remove(os.path.join(input_folder, f))
    if os.path.exists(input_folder):
        os.rmdir(input_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file')
    parser.add_argument('--app_transform', type=bool, default=False, help='Apply app transform')
    args = parser.parse_args()

    config_parser = ConfigParser(args.config_path)
    args_obj, _ = config_parser.parse('train')

    main(args_obj, args.app_transform)
