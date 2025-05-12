import os
import json
import pickle as pkl
import re
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    SoftPhongShader,
)
from pytorch3d.utils import cameras_from_opencv_projection

from dreamor.body_model.body_model import BodyModel

# --------------------------------------------------------------------------------------
# Configuration ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def load_config(config_path: str):
    """Load YAML and return a SimpleNamespace (dot‑access)."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    defaults = {
        "render_width": 1920,
        "render_height": 1080,
        "start": 0,
        "end": -1,
        "no_motion": False,
        "save_dir": "visualizations",
        "data_root": "data/",
        "num_betas": 16,
        "overlay_alpha":  0.8,    
        "white_thresh" :250 
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # Build derived paths --------------------------------------------------------------
    date = cfg["motion"].split("/")[0]
    motion_base = os.path.join(cfg["data_root"], "ground_truth", cfg["motion"])

    # video path + starting frame (optional, but handy)
    video_base = os.path.dirname(motion_base.replace("ground_truth", "video_release"))
    video_path = os.path.join(video_base, "data2.mp4")
    match = re.search(r"gt_\d+_(\d+)_\d+\.pkl", motion_base)
    frame_start = int(match.group(1)) if match else 0
    save_video_path = os.path.join( cfg["save_dir"], "render.mp4")
    cfg.update(
        {
            "motion": motion_base,
            "extrin": os.path.join(cfg["data_root"], "calibrations", date, "extrin.json"),
            "intrin": os.path.join(cfg["data_root"], "calibrations", date, "intrin.json"),
            "save_motion_dir": os.path.join(
                cfg["save_dir"], os.path.dirname(cfg["motion"]).replace("ground_truth", ""), "motion"
            ),
            "video_path": video_path,
            "video_start_frame": frame_start,
            "save_video_path": save_video_path,
            "overlay_frames": cfg["overlay_frames"],
        }
    )

    return SimpleNamespace(**cfg)


# --------------------------------------------------------------------------------------
# Device ------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)


# --------------------------------------------------------------------------------------
# Camera utils ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def load_cameras(args):
    with open(args.extrin, "r") as f:
        extrin = json.load(f)
    with open(args.intrin, "r") as f:
        intrin = json.load(f)["color"]

    R = np.array(extrin["rotation"]).reshape((1, 3, 3))
    T = np.array(extrin["translation"]).reshape((1, 3))
    K = np.array(
        [intrin["fx"], 0, intrin["cx"], 0, intrin["fy"], intrin["cy"], 0, 0, 1]
    ).reshape(3, 3)

    K /= 3840 / args.render_width  # scale from 4 K capture to output size
    K[2, 2] = 1

    cams = cameras_from_opencv_projection(
        torch.from_numpy(R).float().to(device),
        torch.from_numpy(T).float().to(device),
        torch.from_numpy(K).unsqueeze(0).float().to(device),
        torch.tensor([args.render_height, args.render_width]).unsqueeze(0).to(device),
    )
    return cams


# --------------------------------------------------------------------------------------
# Rendering ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def make_renderer(cameras, image_size):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=10,
        max_faces_per_bin=100_000,
    )
    shader = SoftPhongShader(device=device, cameras=cameras)
    return MeshRenderer(MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader)


# --------------------------------------------------------------------------------------
# Human‑only rendering ----------------------------------------------------------------
# --------------------------------------------------------------------------------------

def render_human_sequence(args, smplh):
    with open(args.motion, "rb") as f:
        gt = pkl.load(f)

    cameras = load_cameras(args)
    renderer = make_renderer(cameras, [args.render_height, args.render_width])

    human_rgb = torch.ones(6890, 3, device=device)
    textures = TexturesVertex([human_rgb])

    end_frame = gt["smplPose"].shape[0] if args.end == -1 else args.end + 1
    os.makedirs(args.save_motion_dir, exist_ok=True)

    for idx in range(args.start, end_frame):
        pose = torch.as_tensor(gt["smplPose"][idx]).unsqueeze(0).float().to(device)
        hand = torch.as_tensor(gt["smplHandPose"][idx]).unsqueeze(0).float().to(device)
        betas = torch.as_tensor(gt["smplShape"][idx]).unsqueeze(0).float().to(device)
        trans = torch.as_tensor(gt["smplTrans"][idx]).unsqueeze(0).float().to(device)

        smpl_out = smplh(
            betas=betas,
            root_orient=pose[:, :3],
            pose_body=pose[:, 3:66],
            pose_hand=hand,
            trans=trans,
        )

        mesh = Meshes([smpl_out.v.squeeze(0)], [smpl_out.f], textures)
        img = renderer(mesh)
        cv2.imwrite(
            os.path.join(args.save_motion_dir, f"{idx}.png"),
            255 * img[0, :, :, :-1].cpu().numpy(),
        )

def _plot_subplot(ax, x, y, bounds, labels, mark_every):
    colors = ["red", "green", "blue"]
    for i in range(3):
        ax.plot(
            x,
            y[i],
            color=colors[i],
            label=labels[i],
            marker="o",
            mfc="w",
            ms=2.5,
            markevery=mark_every,
        )
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bounds[0] - 0.05, bounds[1] + 0.05)
    ax.grid(True, linestyle="--")
    ax.legend(framealpha=0.5)
# --------------------------------------------------------------------------------------
# Render and Overlay Together ---------------------------------------------------------
# --------------------------------------------------------------------------------------

def render_and_overlay_video(args, smplh):
    import tqdm

    with open(args.motion, "rb") as f:
        gt = pkl.load(f)

    cameras = load_cameras(args)
    renderer = make_renderer(cameras, [args.render_height, args.render_width])

    human_rgb = torch.ones(6890, 3, device=device)
    textures = TexturesVertex([human_rgb])

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.video_start_frame + args.start)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if not ret:
        raise IOError("Unable to read first frame from video")
    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.save_video_path), exist_ok=True)
    writer = cv2.VideoWriter(args.save_video_path, fourcc, fps, (w, h))

    total_frames = gt["smplPose"].shape[0]
    end_frame = min(total_frames, args.overlay_frames) if args.end == -1 else min(args.end + 1, args.overlay_frames)

    for idx in tqdm.tqdm(range(args.start, end_frame)):
        if idx > args.start:
            ret, frame = cap.read()
            if not ret:
                break

        # Render human
        pose = torch.as_tensor(gt["smplPose"][idx]).unsqueeze(0).float().to(device)
        hand = torch.as_tensor(gt["smplHandPose"][idx]).unsqueeze(0).float().to(device)
        betas = torch.as_tensor(gt["smplShape"][idx]).unsqueeze(0).float().to(device)
        trans = torch.as_tensor(gt["smplTrans"][idx]).unsqueeze(0).float().to(device)

        smpl_out = smplh(
            betas=betas,
            root_orient=pose[:, :3],
            pose_body=pose[:, 3:66],
            pose_hand=hand,
            trans=trans,
        )

        mesh = Meshes([smpl_out.v.squeeze(0)], [smpl_out.f], textures)
        render_img = renderer(mesh)

        overlay = (255 * render_img[0, :, :, :-1].cpu().numpy()).astype(np.uint8)
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

        # Simple white threshold mask
        mask = np.any(overlay < args.white_thresh, axis=2)

        # Blend
        blended = frame.copy()
        blended[mask] = (
            args.overlay_alpha * overlay[mask].astype(np.float32)
            + (1 - args.overlay_alpha) * frame[mask].astype(np.float32)
        ).astype(np.uint8)

        writer.write(blended)

    writer.release()
    cap.release()
    print(f"Overlay video saved to {args.save_video_path}")
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render human mesh over video directly")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML file")
    cfg_path = parser.parse_args().config

    args = load_config(cfg_path)
    if args.render_width / args.render_height != 16 / 9:
        raise ValueError("Render resolution must be 16:9")

    smplh = BodyModel(args.smpl_model_path, num_betas=args.num_betas).to(device)
    if not args.no_motion:
        render_and_overlay_video(args, smplh)

if __name__ == "__main__":
    main()
