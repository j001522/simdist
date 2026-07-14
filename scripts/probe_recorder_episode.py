"""Roll out a chosen checkpoint in the DATA-RECORDER env and save panel videos.

Visual-inspection tool for "how does the expert (or any checkpoint) behave when
the SimDist DataRecorder rolls it out?" -- the env where insertion success is
0% across every collected dataset (see memory: recorder-zero-insertion).

Faithful to `DataRecorder`: same `Ur5eRecordEnvCfg`, same
`expert_policy(obs["policy"])` action path (no extra scaling), optional
`simplify_events` so conditions match a simplified collection. Differences are
inspection-only: num_envs=1, clean expert (NO action corruption), hdf5 recorders
stripped, and the short-circuit terminations nulled so the FULL trajectory plays
(toggle with --keep-terminations).

Each frame overlays the peg->hole distance and a SUCCESS flag read from the
env's live `progress_context` reward term (the SAME quantity the RL success
uses). NOTE: do NOT reconstruct this from the recorded
`insertive_asset_in_receptive_asset_frame` obs -- that obs is the RAW
root-to-root transform and sits ~1.48 cm (the hole `assembled_offset`) above the
success frame, so a correctly SEATED peg reads ~1.48 cm there, not 0.

Examples:
  # one simplified episode with the default expert (iter 1700)
  isaac-python scripts/probe_recorder_episode.py
  # three episodes of a mid checkpoint, full (un-simplified) distribution
  isaac-python scripts/probe_recorder_episode.py --iter 1000 --episodes 3 --mode full

Outputs to <out>/ (default simdist/lighting_check/recorder_probe):
  * iter{ITER}_{mode}_panel.mp4               front | side | wrist, all episodes
  * iter{ITER}_{mode}_ep{k}_{start,mid,end}.png   panel-row stills per episode
  * iter{ITER}_{mode}_stats.txt               per-episode min-distance / success
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--iter", type=int, default=1700, help="checkpoint iteration to roll out")
parser.add_argument("--episodes", type=int, default=1, help="number of episodes to record")
parser.add_argument("--mode", choices=("simplified", "full"), default="simplified",
                    help="simplified: DR off + single reset bank; full: real collection distribution")
parser.add_argument("--rl-run", default="omnireset_2026-06-21_17-10-39", help="RL run name under checkpoints/rl")
parser.add_argument("--reset-dir", default="/shared/giacomo/experiments/ood_insertion/resets",
                    help="reset-bank dir for --mode simplified")
parser.add_argument("--reset-types", nargs="+", default=["GraspedVertHigh"],
                    help="reset bank(s) for --mode simplified")
parser.add_argument("--out", default="/shared/giacomo/simdist/lighting_check/recorder_probe", help="output dir")
parser.add_argument("--max-steps", type=int, default=200, help="per-episode step cap (horizon is ~160)")
parser.add_argument("--fps", type=int, default=10, help="panel video fps")
parser.add_argument("--seed", type=int, default=42, help="env seed")
parser.add_argument("--keep-terminations", action="store_true",
                    help="respect corrupted_camera/abnormal_robot terminations instead of nulling them")
parser.add_argument("--action", choices=("eval", "train"), default="eval",
                    help="arm OSC action scale. eval = record default (RelativeOSCEvalAction, z-scale 0.002); "
                         "train = the scale the expert was trained/evaluated at (RelativeOSCAction, z-scale 0.02). "
                         "Use 'train' to test whether the eval z-scale is why the expert fails to seat.")
parser.add_argument("--null-arm-dr", action="store_true",
                    help="remove randomize_arm_sysid + randomize_osc_gains (match the ideal Stage-1 actuator "
                         "the expert trained on, which the working State env uses).")
args = parser.parse_args()

print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from simdist.rl.ur5e_omnireset import Ur5eRecordEnvCfg
from simdist.rl.manip_simplify import simplify_events
from simdist.utils.torch import get_actor_critic_from_iteration

CAMS = ("front_rgb", "side_rgb", "wrist_rgb")
SCRATCH = "/tmp/claude-1001/-shared-giacomo/a8bc9a0d-2f2d-47b1-98ed-458a6c146da4/scratchpad"


def label(img, text, y=18):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grab_cams(obs):
    """obs['data_collection'][cam] -> [H,W,3] uint8 numpy for env 0."""
    dc = obs["data_collection"]
    return {cam: dc[cam][0].detach().cpu().numpy().astype(np.uint8) for cam in CAMS}


def insertion_metrics(progress):
    """(xyz_dist_m, euler_xy_rad, success) from the live progress_context term (env 0)."""
    return (float(progress.xyz_distance[0]),
            float(progress.euler_xy_distance[0]),
            bool(progress.success[0]))


def build_env():
    cfg = Ur5eRecordEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    if args.mode == "simplified":
        simplify_events(
            cfg.events,
            dataset_dir=args.reset_dir,
            reset_types=tuple(args.reset_types),
            probs=tuple(1.0 / len(args.reset_types) for _ in args.reset_types),
            pin_arm_nominal=not args.null_arm_dr,
        )
    if args.null_arm_dr:
        # Remove sysid + OSC-gain randomization entirely so the arm uses the ideal
        # actuator the Stage-1 expert trained on (the State env has neither event).
        for name in ("randomize_arm_sysid", "randomize_osc_gains"):
            if getattr(cfg.events, name, None) is not None:
                setattr(cfg.events, name, None)
        print("[probe] null-arm-dr: randomize_arm_sysid + randomize_osc_gains removed")
    # Strip dataset recorders: we only want frames, not an hdf5.
    cfg.recorders = RecorderManagerBaseCfg()
    cfg.recorders.dataset_export_dir_path = SCRATCH
    cfg.recorders.dataset_filename = "unused"
    if args.action == "train":
        # Swap the record env's high-Kp/small-scale eval action for the base OSC
        # action the expert was trained + evaluated at (z-scale 0.02 vs 0.002).
        from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (
            Ur5eRobotiq2f85RelativeOSCAction,
        )
        cfg.actions = Ur5eRobotiq2f85RelativeOSCAction()
        print("[probe] action=train: using RelativeOSCAction (base scale, z=0.02)")
    if not args.keep_terminations:
        # INSPECTION ONLY: let the full trajectory play; does not change collection config.
        cfg.terminations.corrupted_camera = None
        cfg.terminations.abnormal_robot = None
    return ManagerBasedRLEnv(cfg)


def main():
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = build_env()
    progress = env.reward_manager.get_term_cfg("progress_context").func  # ProgressContext
    policy, _ = get_actor_critic_from_iteration(args.rl_run, args.iter, device)
    print(f"[probe] iter={args.iter} mode={args.mode} episodes={args.episodes} "
          f"terminations={'kept' if args.keep_terminations else 'nulled'}")

    tag = f"iter{args.iter}_{args.mode}_act-{args.action}"
    panel = []            # concatenated across all episodes
    stats_lines = []

    for k in range(args.episodes):
        obs, _ = env.reset()
        ep_rows, ep_dists = [], []
        act_mag, joint_speed = [], []   # policy-idle vs arm-stuck diagnostic
        prev_qpos = obs["data_collection"]["arm_joint_pos"][0].detach().cpu().numpy()
        succeeded = False
        step, done = 0, False
        while not done and step < args.max_steps:
            with torch.no_grad():
                action = policy(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(action)
            cams = grab_cams(obs)
            xyz, _euler, succ = insertion_metrics(progress)
            succeeded |= succ
            ep_dists.append(xyz)
            # |arm action| (6 OSC dims) and how far the 6 arm joints actually moved this step
            a = action[0, :6].detach().cpu().numpy()
            qpos = obs["data_collection"]["arm_joint_pos"][0].detach().cpu().numpy()
            act_mag.append(float(np.abs(a).mean()))
            joint_speed.append(float(np.linalg.norm(qpos - prev_qpos)))
            prev_qpos = qpos
            dist_txt = f"d={xyz * 100:5.1f}cm |a|={act_mag[-1]:.2f} dq={joint_speed[-1]:.3f}{'  INSERTED' if succ else ''}"
            row = np.hstack([label(cams[c].copy(), c.replace("_rgb", "")) for c in CAMS])
            row = label(row, f"iter{args.iter} ep{k + 1}/{args.episodes} t{step:3d}  {dist_txt}",
                        y=row.shape[0] - 8)
            ep_rows.append(row)
            done = bool(terminated[0].item() or truncated[0].item())
            step += 1

        panel.extend(ep_rows)
        # per-episode stills at start / mid / end
        idxs = {"start": 0, "mid": step // 2, "end": step - 1}
        for pos, i in idxs.items():
            imageio.imwrite(os.path.join(args.out, f"{tag}_ep{k + 1}_{pos}.png"), ep_rows[i])
        # split into thirds so we can SEE where/whether it goes idle
        am, js = np.array(act_mag), np.array(joint_speed)
        n = max(len(am) // 3, 1)
        thirds = lambda v: f"[{v[:n].mean():.3f} {v[n:2*n].mean():.3f} {v[2*n:].mean():.3f}]"
        line = (f"ep {k + 1}: steps={step} min_dist={min(ep_dists) * 100:.2f}cm "
                f"final={ep_dists[-1] * 100:.2f}cm success={succeeded} | "
                f"|action| thirds={thirds(am)} jointspeed thirds={thirds(js)}")
        stats_lines.append(line)
        print("[probe] " + line, flush=True)

    panel_path = os.path.join(args.out, f"{tag}_panel.mp4")
    imageio.mimsave(panel_path, panel, fps=args.fps, macro_block_size=None)
    print(f"[probe] saved {panel_path} ({len(panel)} frames)")

    header = (f"checkpoint iter {args.iter}  run {args.rl_run}  mode {args.mode}  "
              f"episodes {args.episodes}\n"
              f"distance/success from live progress_context term (true success frame)\n")
    with open(os.path.join(args.out, f"{tag}_stats.txt"), "w") as f:
        f.write(header + "\n".join(stats_lines) + "\n")
    print("[probe]\n" + header + "\n".join(stats_lines))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
