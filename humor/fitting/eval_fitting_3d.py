'''
Viz and evaluation for 3D AMASS tasks.
'''

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import csv, random

import numpy as np

import torch
import torch.nn as nn

from utils.config import SplitLineParser
from utils.logging import mkdir

from fitting.fitting_utils import load_res, prep_res, run_smpl
from fitting.eval_utils import quant_eval_3d, SMPL_SIZES, GRND_PEN_THRESH_LIST, get_grnd_pen_key, AMASS_EVAL_BLACKLIST

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS

from viz.utils import viz_smpl_seq, create_video

J_BODY = len(SMPL_JOINTS)-1 # no root

GT_RES_NAME = 'gt_results'
PRED_RES_NAME = 'stage3_results'
STAGES_RES_NAMES = ['stage1_results', 'stage2_results', 'stage3_init_results']
OBS_NAME = 'observations'
FPS = 30

# rendering options
IMW, IMH = 1280, 720
CAM_OFFSET = [0.0, 2.25, 0.9]

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)
    # Arguments for quantitative evaluation
    parser.add_argument('--quant', dest='run_quant_eval', action='store_true', help="If given, runs quantitative evaluation and saves the results.")
    parser.add_argument('--quant-stages', dest='quant_stages', action='store_true', help="If given, runs quantitative evaluation on all stages rather than just final.")
    # Arguments for qualitative evaluation
    parser.add_argument('--qual', dest='run_qual_eval', action='store_true', help="If given, runs qualitative (visualization) evaluation and saves the results.")
    parser.add_argument('--viz-stages', dest='viz_stages', action='store_true', help="If given, visualized intermediate optimization stages.")
    parser.add_argument('--viz-observation', dest='viz_observation', action='store_true', help="If given, visualizes observations on bodies (e.g. verts).")
    parser.add_argument('--viz-contacts', dest='viz_contacts', action='store_true', help="Render predicted contacts on the joints")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles eval ordering")

    return parser.parse_known_args(argv)


def main(args):
    print(args)
    mkdir(args.out)

    quant_out_path = qual_out_path = None
    if args.run_quant_eval:
        quant_out_path = os.path.join(args.out, 'eval_quant')
    if args.run_qual_eval:
        qual_out_path = os.path.join(args.out, 'eval_qual')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Gather sequence directories
    all_result_dirs = [os.path.join(args.results, f) for f in sorted(os.listdir(args.results)) if f[0] != '.']
    if args.shuffle:
        random.seed(0)
        random.shuffle(all_result_dirs)

    quant_res_dict = None
    obs_mod = None
    seq_name_list = []
    body_model_dict = {}

    for ridx, result_dir in enumerate(all_result_dirs):
        seq_name = os.path.basename(result_dir)
        print(f'Evaluating {seq_name}... {ridx+1}/{len(all_result_dirs)}')

        # Load ground-truth
        gt_res = load_res(result_dir, GT_RES_NAME + '.npz')
        gt_res = prep_res(gt_res, device, gt_res['trans'].shape[0])

        # Load prediction
        pred_res = load_res(result_dir, PRED_RES_NAME + '.npz')
        pred_res = prep_res(pred_res, device, gt_res['trans'].shape[0])

        # Build SMPL models
        with open(os.path.join(result_dir, 'meta.txt')) as f:
            optim_path = f.readline().split()[1]
            gt_path    = f.readline().split()[1]
        if gt_path not in body_model_dict:
            body_model_dict[gt_path] = BodyModel(bm_path=gt_path,
                                                num_betas=gt_res['betas'].shape[0],
                                                batch_size=gt_res['trans'].shape[0]).to(device)
        if optim_path not in body_model_dict:
            body_model_dict[optim_path] = BodyModel(bm_path=optim_path,
                                                    num_betas=pred_res['betas'].shape[0],
                                                    batch_size=gt_res['trans'].shape[0]).to(device)
        gt_bm   = body_model_dict[gt_path]
        pred_bm = body_model_dict[optim_path]

        # Convert SMPL parameters to 3D joints / verts
        gt_body   = run_smpl(gt_res, gt_bm)
        pred_body = run_smpl(pred_res, pred_bm)

        # ========== Quantitative Evaluation ==========
        if args.run_quant_eval:
            # Load observed data (3D keypoints, 2D joints, or point clouds)
            obs_dict = np.load(os.path.join(result_dir, OBS_NAME + '.npz'))
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

        # ========== Qualitative Evaluation ==========
        if args.run_qual_eval:
            out_dir = os.path.join(qual_out_path, seq_name)
            mkdir(out_dir)

            # Visualize final prediction
            viz_smpl_seq(
                pred_body,
                imw=IMW, imh=IMH, fps=FPS,
                render_body=True,
                render_joints=args.viz_observation,
                render_skeleton=args.viz_observation,
                render_ground=True,
                contacts=pred_res.get('contacts') if args.viz_contacts else None,
                joints_seq=obs_data.get('joints3d'),
                points_seq=obs_data.get('verts3d'),
                out_path=os.path.join(out_dir, 'humor')
            )
            create_video(os.path.join(out_dir, 'humor/frame_%08d.png'), os.path.join(out_dir, 'humor.mp4'), FPS)

            # Optionally visualize intermediate stage
            if args.viz_stages:
                stage2 = run_smpl(prep_res(load_res(result_dir, 'stage2_results.npz'), device, gt_res['trans'].shape[0]), pred_bm)
                viz_smpl_seq(
                    stage2,
                    imw=IMW, imh=IMH, fps=FPS,
                    render_body=True,
                    render_joints=args.viz_observation,
                    render_skeleton=args.viz_observation,
                    render_ground=True,
                    out_path=os.path.join(out_dir, 'stage2')
                )
                create_video(os.path.join(out_dir, 'stage2/frame_%08d.png'), os.path.join(out_dir, 'stage2.mp4'), FPS)

            # Visualize GT + observations only
            viz_smpl_seq(
                gt_body,
                imw=IMW, imh=IMH, fps=FPS,
                render_body=True,
                render_joints=args.viz_observation,
                render_skeleton=args.viz_observation,
                render_ground=True,
                out_path=os.path.join(out_dir, 'gt_obs_only')
            )
            create_video(os.path.join(out_dir, 'gt_obs_only/frame_%08d.png'), os.path.join(out_dir, 'gt_obs_only.mp4'), FPS)

    # After loop: write quantitative CSVs (means, medians, etc.) under eval_quant/
    if args.run_quant_eval:
        mkdir(quant_out_path)
        # ... (aggregation & CSV writing) ...
        # Comments:
        # The code below collects sequence-wise and global statistics for each metric:
        #   * joints3d_all, joints3d_ee, joints3d_legs, verts3d_all, mesh3d_all
        #   * contact_acc (fraction correct)
        #   * accel_mag (smoothness)
        #   * ground_pen_dist and per-threshold ground_pen_xxx
        # It then saves:
        #   - per-sequence mean CSV for each method
        #   - overall mean, std, median, max, min CSVs
        #   - a comparison CSV stacking all methods side-by-side.

if __name__=='__main__':
    args, _ = parse_args(sys.argv[1:])
    main(args)
