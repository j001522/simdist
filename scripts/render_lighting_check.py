"""Render a couple of frames from the SIMPLIFIED RGB data-collection env.

Sanity check for scene lighting after making the base skyLight Nucleus-independent
(uwlab rl_state_cfg.py). Mirrors scripts/generate_data.py's launch + env, applies
manip_simplify.simplify_events (so randomize_sky_light is nulled, exactly as during
collection), steps a few frames with zero actions, and dumps the three camera views
as PNGs for visual inspection. NOT a data run -- no policies/critic loaded.
"""

print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os

import numpy as np
import torch
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnv

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.data_collection_rgb_cfg import (
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
)
from simdist.rl.manip_simplify import simplify_events

OUT_DIR = "/shared/giacomo/simdist/lighting_check"
NUM_ENVS = 2
SETTLE_STEPS = 30  # let physics settle + cameras populate before grabbing frames
CAMS = ("front_rgb", "side_rgb", "wrist_rgb")


def save_frame(img_uint8: torch.Tensor, path: str) -> None:
    """img_uint8: [H, W, 3] uint8 on device -> PNG."""
    arr = img_uint8.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 42

    # Same simplification the recorder applies (config/generate_data.yaml).
    simplify_events(
        cfg.events,
        dataset_dir="/shared/giacomo/experiments/ood_insertion/resets",
        reset_types=("GraspedVertHigh",),
        probs=(1.0,),
    )
    print("[lighting_check] simplify_events applied (randomize_sky_light nulled)")

    env = ManagerBasedRLEnv(cfg)
    action_dim = env.action_manager.total_action_dim
    print(f"[lighting_check] action_dim={action_dim}, num_envs={NUM_ENVS}")

    obs, _ = env.reset()
    zero_action = torch.zeros((NUM_ENVS, action_dim), device=env.device)
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(zero_action)

    dc = obs["data_collection"]
    print(f"[lighting_check] data_collection keys: {list(dc.keys())}")
    for cam in CAMS:
        imgs = dc[cam]  # [N, H, W, 3] uint8
        print(f"  {cam}: shape={tuple(imgs.shape)} dtype={imgs.dtype} "
              f"mean={imgs.float().mean().item():.1f} max={imgs.max().item()}")
        for env_i in range(NUM_ENVS):
            out = os.path.join(OUT_DIR, f"env{env_i}_{cam}.png")
            save_frame(imgs[env_i], out)
            print(f"    saved {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
