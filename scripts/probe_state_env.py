"""Control experiment: run the EXPORTED expert jit in the in-distribution STATE
env (the same task family the expert was trained + evaluated on, ~0.90 success)
and log action magnitude / joint motion / peg->hole distance.

Purpose: the exported policy saturates (|action|~6, arm frozen) in the RGB record
env. This isolates whether that is (a) an EXPORT bug -- the jit is broken, so it
fails even here in-distribution -- or (b) a record-ENV bug -- it inserts fine here
but not in the record env. `play.py` works, but play.py loads the FULL rsl_rl
checkpoint; this loads the SAME exported jit the DataRecorder uses.

  isaac-python scripts/probe_state_env.py --iter 1700 --episodes 3
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=1700)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--rl-run", default="omnireset_2026-06-21_17-10-39")
parser.add_argument("--max-steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print("Starting Isaac Sim")
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import (
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,  # State-Play-v0: implicit actuator, base gains, no sysid DR
)
from simdist.utils.torch import get_actor_critic_from_iteration


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Ur5eRobotiq2f85RelCartesianOSCEvalCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg)
    progress = env.reward_manager.get_term_cfg("progress_context").func
    robot = env.scene["robot"]
    policy, _ = get_actor_critic_from_iteration(args.rl_run, args.iter, device)
    print(f"[state-probe] iter={args.iter} env=State-Play (in-distribution)")

    for k in range(args.episodes):
        obs, _ = env.reset()
        dists, act_mag, joint_speed = [], [], []
        prev_q = robot.data.joint_pos[0, :6].detach().cpu().numpy()
        succeeded = False
        step, done = 0, False
        while not done and step < args.max_steps:
            with torch.no_grad():
                action = policy(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(action)
            dists.append(float(progress.xyz_distance[0]))
            succeeded |= bool(progress.success[0])
            a = action[0, :6].detach().cpu().numpy()
            q = robot.data.joint_pos[0, :6].detach().cpu().numpy()
            act_mag.append(float(np.abs(a).mean()))
            joint_speed.append(float(np.linalg.norm(q - prev_q)))
            prev_q = q
            done = bool(terminated[0].item() or truncated[0].item())
            step += 1
        am, js = np.array(act_mag), np.array(joint_speed)
        n = max(len(am) // 3, 1)
        thirds = lambda v: f"[{v[:n].mean():.3f} {v[n:2*n].mean():.3f} {v[2*n:].mean():.3f}]"
        print(f"[state-probe] ep {k + 1}: steps={step} min_dist={min(dists) * 100:.2f}cm "
              f"final={dists[-1] * 100:.2f}cm success={succeeded} | "
              f"|action| thirds={thirds(am)} jointspeed thirds={thirds(js)}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
