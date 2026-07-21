"""Closed-loop MPPI planning on the UR5e peg-insertion env.

Manipulation analog of ``simulate_go2.py``. Runs the world model + MPPI planner in
the same Isaac env the dataset was recorded from (``Ur5eRecordEnvCfg``), with the
recorders stripped -- we plan, we do not record.

Three things differ from the Go2 harness beyond the obvious obs/action shapes:

  * ``cmd_dim == 0``: peg insertion has no goal command, so the controller is fed a
    zero-width command vector. ControllerBase tiles it to (T+1, 0) and the
    manipulation model never reads it.
  * ``extero_obs`` is a (n_cam, 224, 224, 3) uint8 image stack, not a flat vector.
    Images are passed through raw -- the encoder's ResNet does ImageNet
    normalization internally (``resnet.imagenet_normalize`` divides by 255).
  * there is no zero-action settling phase after reset. Go2 applies zero actions for
    ``reset_steps`` to let the robot stabilize; here action dim 6 is the gripper, so
    a zero action would command the gripper open and drop the peg.

Cameras require ``--enable_cameras``, which AppLauncher is given below.

NOTE: this single-process script CANNOT run on the DGX Spark -- Isaac's python has
no JAX, and the GPU JAX stack needs numpy>=2 while IsaacLab needs numpy<2, so they
cannot share one interpreter. On the Spark use the 2-process harness instead:

    scripts/remote/run_ur5e.sh model.checkpoint=<run> --headless

This file remains the reference for any environment where one python has both
(and is the exact loop scripts/remote/simulate_ur5e_remote.py mirrors).
"""

import argparse
import sys
import os

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # Keep only unknown args for hydra
    return args


args = parse_args()

# Isaac and JAX share the GPU: cap JAX so the planner does not starve the renderer.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

from isaaclab.app import AppLauncher

# enable_cameras is mandatory: without it the RGB annotators never render and the
# data_collection obs group comes back empty.
app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from simdist.utils import paths, model as model_utils, config
from simdist.control.controller_base import ControllerInput
from simdist.control.mppi import MppiController
from simdist.rl.ur5e_omnireset import Ur5eRecordEnvCfg
from simdist.rl.manip_simplify import simplify_events

OBS_GROUP = "data_collection"  # the vision obs group the world model was trained on


class Ur5eSim:
    def __init__(self, cfg: dict):
        print("Loaded config:")
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        self.num_episodes = cfg["sim"]["num_episodes"]
        self.max_steps = cfg["sim"]["max_steps"]

        # create controller
        ckpt_dir = os.path.join(
            paths.get_model_checkpoints_dir(), self.cfg["model"]["checkpoint"]
        )
        model, model_cfg, _ = model_utils.load_model_from_ckpt(
            ckpt_dir, self.cfg["model"]["step"]
        )
        self.controller = MppiController(model, model_cfg, self.cfg["control"])
        sys_cfg = model_cfg["system"]
        self.proprio_obs_names = config.proprio_obs_names_from_sys_config(sys_cfg)
        self.extero_obs_names = config.extero_obs_names_from_sys_config(sys_cfg)
        self.act_dim = config.action_dim_from_sys_config(sys_cfg)
        assert config.cmd_dim_from_sys_config(sys_cfg) == 0, (
            "simulate_ur5e assumes cmd_dim == 0; got "
            f"{config.cmd_dim_from_sys_config(sys_cfg)}"
        )

        # create the env
        self.env = self.build_env()
        # live insertion metrics (same term probe_recorder_episode.py reads)
        self.progress = self.env.reward_manager.get_term_cfg("progress_context").func

        self.obs_dict, _ = self.env.reset()

        # set up the controller. cmd is zero-width for manipulation.
        self.zero_cmd = np.zeros((0,), dtype=np.float32)
        self.zero_action = np.zeros((self.act_dim,), dtype=np.float32)
        x = self.get_controller_input(self.zero_action)
        self.controller.initialize(x, self.zero_cmd)
        self.controller.reset(x, self.zero_cmd)

        self.pbar = tqdm(desc="Simulation", unit="step", total=self.num_episodes * self.max_steps)

        # wandb
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = (
            f"{date_time}"
            if cfg["wandb"]["run_name"] is None
            else cfg["wandb"]["run_name"]
        )
        if self.cfg["wandb"]["log"]:
            wandb.init(
                project="tyswy_ur5e_sim",
                entity=self.cfg["wandb"]["entity"],
                name=self.run_name,
                config=cfg,
            )

        self.episode_stats = []

    def build_env(self):
        env_cfg = Ur5eRecordEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.seed = self.cfg["sim"]["seed"]

        if self.cfg["sim"]["simplified"]:
            reset_types = tuple(self.cfg["sim"]["reset_types"])
            simplify_events(
                env_cfg.events,
                dataset_dir=self.cfg["sim"]["reset_dir"],
                reset_types=reset_types,
                probs=tuple(1.0 / len(reset_types) for _ in reset_types),
            )

        # We plan, we do not record: drop the recorder terms (they would try to write
        # an hdf5 and query a critic this env has no checkpoint for).
        env_cfg.recorders = RecorderManagerBaseCfg()

        return ManagerBasedRLEnv(env_cfg)

    def get_obs(self):
        """proprio (proprio_dim,) float32 and images (n_cam, 224, 224, 3) uint8."""
        obs = self.obs_dict[OBS_GROUP]
        proprio_obs = np.concatenate(
            [obs[name][0].detach().cpu().numpy().ravel() for name in self.proprio_obs_names]
        ).astype(np.float32)
        # camera order must match extero_obs.types in the system config -- that is the
        # order the encoder's per-camera ResNet features are concatenated in.
        extero_obs = np.stack(
            [obs[name][0].detach().cpu().numpy().astype(np.uint8) for name in self.extero_obs_names]
        )
        return proprio_obs, extero_obs

    def get_controller_input(self, prev_action: np.ndarray) -> ControllerInput:
        proprio_obs, extero_obs = self.get_obs()
        return {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "prev_action": prev_action,
        }

    def insertion_metrics(self):
        return (
            float(self.progress.xyz_distance[0]),
            float(self.progress.success[0]),
        )

    def run(self):
        for ep in range(self.num_episodes):
            if not simulation_app.is_running():
                break
            self.run_episode(ep)

        self.logging()
        self.env.close()
        simulation_app.close()

    def run_episode(self, ep: int):
        self.obs_dict, _ = self.env.reset()
        # refill the controller history with the post-reset observation
        last_action = self.zero_action.copy()
        self.controller.reset(self.get_controller_input(last_action), self.zero_cmd)

        succeeded = False
        min_dist = np.inf
        total_reward = 0.0
        step = 0
        done = False

        while not done and step < self.max_steps and simulation_app.is_running():
            self.controller.update(self.get_controller_input(last_action))

            ctrl_out = self.controller.run_control()
            action = ctrl_out["actions"][0]  # first action of the planned sequence
            last_action = action

            action_torch = torch.from_numpy(np.asarray(action, dtype=np.float32))
            action_torch = action_torch.unsqueeze(0).to(self.env.device)
            self.obs_dict, reward, terminated, truncated, _ = self.env.step(action_torch)

            dist, succ = self.insertion_metrics()
            succeeded |= bool(succ)
            min_dist = min(min_dist, dist)
            total_reward += float(reward[0].item())

            done = bool(terminated[0].item() or truncated[0].item())
            step += 1
            self.pbar.update(1)

        stats = {
            "episode": ep,
            "steps": step,
            "success": succeeded,
            "min_dist_cm": min_dist * 100.0,
            "total_reward": total_reward,
        }
        self.episode_stats.append(stats)
        print(
            f"[sim] ep {ep + 1}/{self.num_episodes}: steps={step} "
            f"success={succeeded} min_dist={min_dist * 100:.2f}cm "
            f"reward={total_reward:.2f}",
            flush=True,
        )

    def logging(self):
        n = len(self.episode_stats)
        if n == 0:
            return
        metrics = {
            "success_rate": sum(s["success"] for s in self.episode_stats) / n,
            "mean_min_dist_cm": float(np.mean([s["min_dist_cm"] for s in self.episode_stats])),
            "mean_reward": float(np.mean([s["total_reward"] for s in self.episode_stats])),
            "mean_episode_length": float(np.mean([s["steps"] for s in self.episode_stats])),
            "num_episodes": n,
        }
        print(metrics)

        if self.cfg["wandb"]["log"]:
            wandb.log(metrics)
            wandb.finish()


@hydra.main(**paths.get_simulate_ur5e_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    ur5e_sim = Ur5eSim(dict_cfg)
    ur5e_sim.run()


if __name__ == "__main__":
    main()
