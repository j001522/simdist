"""Roll out ONE expert episode in the simplified RGB env and save it as video.

Diagnostic for the lighting/background question: a settled zero-action frame is
not representative, so here we drive the state expert (iteration 1700) through a
full 16 s episode and record what the three cameras actually see over the
trajectory. Reuses Ur5eRecordEnvCfg (which adds the state `policy` obs group the
expert consumes and the 16 s horizon) but strips the hdf5 recorders -- we only
want frames, not a dataset. simplify_events is applied so conditions match
collection (DR off, GraspedVertHigh bank, randomize_sky_light nulled).

Outputs to /shared/giacomo/simdist/lighting_check/:
  * expert_episode_panel.mp4   front | side | wrist, per step
  * expert_ep_{front,side,wrist}_{start,mid,end}.png  stills for quick inspection
"""

print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from simdist.rl.ur5e_omnireset import Ur5eRecordEnvCfg
from simdist.rl.manip_simplify import simplify_events
from simdist.utils.torch import get_actor_critic_from_iteration

OUT_DIR = "/shared/giacomo/simdist/lighting_check"
RL_RUN = "omnireset_2026-06-21_17-10-39"
EXPERT_ITER = 1700
MAX_STEPS = 200  # 16 s / 0.1 s = 160; cap a bit above in case of no truncation
CAMS = ("front_rgb", "side_rgb", "wrist_rgb")
SCRATCH = "/tmp/claude-1001/-shared-giacomo/cba41a65-e6fc-43f8-a6f7-b0f66b6e3ae0/scratchpad"


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grab(obs) -> dict:
    """obs['data_collection'][cam] -> [H,W,3] uint8 numpy for env 0."""
    dc = obs["data_collection"]
    return {cam: dc[cam][0].detach().cpu().numpy().astype(np.uint8) for cam in CAMS}


def main() -> None:
    # mode: "simplified" (default) applies simplify_events; "full" leaves the env
    # at the real data-collection distribution (all DR on, 4-way reset mix).
    mode = sys.argv[1] if len(sys.argv) > 1 else "simplified"
    assert mode in ("simplified", "full", "keepcam"), f"unknown mode {mode!r}"
    tag = f"_{mode}"
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Ur5eRecordEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 42
    if mode in ("simplified", "keepcam"):
        # keepcam: apply simplify but RESTORE the camera-pose/focal events afterward,
        # to test whether nulling those events is what breaks the wrist view.
        cam_events = (
            "randomize_front_camera", "randomize_front_camera_focal_length",
            "randomize_side_camera", "randomize_side_camera_focal_length",
            "randomize_wrist_camera", "randomize_wrist_camera_focal_length",
        )
        stash = {n: getattr(cfg.events, n, None) for n in cam_events} if mode == "keepcam" else {}
        simplify_events(
            cfg.events,
            dataset_dir="/shared/giacomo/experiments/ood_insertion/resets",
            reset_types=("GraspedVertHigh",),
            probs=(1.0,),
        )
        for n, term in stash.items():
            setattr(cfg.events, n, term)
    # Strip dataset recorders: we only want frames, not an hdf5.
    cfg.recorders = RecorderManagerBaseCfg()
    cfg.recorders.dataset_export_dir_path = SCRATCH
    cfg.recorders.dataset_filename = "unused"
    # DIAGNOSTIC ONLY: null the terminations that otherwise cut the episode short
    # so we can watch the full trajectory. Does NOT change the real collection config.
    cfg.terminations.corrupted_camera = None
    cfg.terminations.abnormal_robot = None
    print(f"[expert_ep] mode={mode}; recorders stripped; corrupted_camera+abnormal_robot nulled (diag)")

    env = ManagerBasedRLEnv(cfg)
    expert, _ = get_actor_critic_from_iteration(RL_RUN, EXPERT_ITER, device)
    print(f"[expert_ep] expert {EXPERT_ITER} loaded")

    obs, _ = env.reset()
    frames = {cam: [] for cam in CAMS}
    panel = []
    std_trace = {cam: [] for cam in CAMS}
    corrupt_steps = 0
    step = 0
    done = False
    while not done and step < MAX_STEPS:
        with torch.no_grad():
            action = expert(obs["policy"])
        obs, _, terminated, truncated, _ = env.step(action)
        g = grab(obs)
        any_corrupt = False
        for cam in CAMS:
            frames[cam].append(g[cam])
            s = float(g[cam].astype(np.float32).std())
            std_trace[cam].append(s)
            any_corrupt |= s < 10.0
        corrupt_steps += int(any_corrupt)
        row = np.hstack([label(g[cam].copy(), f"{cam.replace('_rgb','')} s{std_trace[cam][-1]:.0f}") for cam in CAMS])
        panel.append(row)
        # NOTE: terminations are diagnostic-nulled, so this only ends on time_out.
        done = bool(terminated[0].item() or truncated[0].item())
        step += 1

    lines = [f"episode ran {step} steps (done={done}); "
             f"{corrupt_steps}/{step} steps would trip corrupted_camera (std<10)"]
    for cam in CAMS:
        st = np.array(std_trace[cam])
        lines.append(f"  {cam}: std min/mean/max = {st.min():.1f}/{st.mean():.1f}/{st.max():.1f}; "
                     f"steps<10 = {(st < 10).sum()}/{len(st)}")
    summary = f"mode={mode}\n" + "\n".join(lines)
    print("[expert_ep]\n" + summary, flush=True)
    with open(os.path.join(OUT_DIR, f"expert_episode_stats{tag}.txt"), "w") as f:
        f.write(summary + "\n")

    panel_path = os.path.join(OUT_DIR, f"expert_episode_panel{tag}.mp4")
    imageio.mimsave(panel_path, panel, fps=10, macro_block_size=None)
    print(f"[expert_ep] saved {panel_path} ({len(panel)} frames)")

    # Stills at start / mid / end for quick inspection.
    idxs = {"start": 0, "mid": step // 2, "end": step - 1}
    for cam in CAMS:
        for pos, i in idxs.items():
            p = os.path.join(OUT_DIR, f"expert_ep{tag}_{cam.replace('_rgb','')}_{pos}.png")
            imageio.imwrite(p, frames[cam][i])
    print(f"[expert_ep] saved stills for {list(idxs)}")

    # Per-camera brightness trace (mean over episode) to quantify.
    for cam in CAMS:
        arr = np.stack(frames[cam]).astype(np.float32)
        print(f"  {cam}: ep-mean={arr.mean():.1f} first={arr[0].mean():.1f} last={arr[-1].mean():.1f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
