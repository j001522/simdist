"""UR5e peg-insertion closed-loop MPPI, with the planner in a SEPARATE process —
Process A of the 2-process Spark loop.

This is `scripts/simulate_ur5e.py` with the in-process `MppiController` replaced by
the same `RemoteController` pattern `simulate_go2_remote.py` uses: reset/update/
run_control are forwarded over a localhost socket to `mppi_server.py` (Process B,
JAX/GPU). Required on the DGX Spark because IsaacLab needs numpy<2 while the GPU
JAX stack needs numpy>=2, so they cannot share one interpreter.

This process imports ONLY jax-free simdist modules (rl.ur5e_omnireset,
rl.manip_simplify, utils.config, utils.paths); all JAX lives in the server.

Run via `run_ur5e.sh`, or manually (after starting mppi_server.py):
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
      <isaac-python.sh> simulate_ur5e_remote.py --mppi-port 5599 --enable_cameras \
        model.checkpoint=wm_manip_5hz_ln [hydra overrides]
"""
import argparse
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDIST_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _SIMDIST_ROOT)
import ipc  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mppi-host", default="127.0.0.1")
parser.add_argument("--mppi-port", type=int, default=5599)
parser.add_argument("--video", action="store_true",
                    help="Record a video offscreen (works headless / no display).")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown  # keep only unknown args for hydra

# The world model is vision-based: the cameras must render or obs["data_collection"]
# comes back empty. Unlike Go2 this is required even without --video.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
from tqdm import tqdm  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg  # noqa: E402

from simdist.utils import paths, config  # noqa: E402  (jax-free)
from simdist.rl.ur5e_omnireset import Ur5eRecordEnvCfg  # noqa: E402  (jax-free)
from simdist.rl.manip_simplify import simplify_events  # noqa: E402  (jax-free)

OBS_GROUP = "data_collection"  # the vision obs group the world model was trained on


class RemoteController:
    """Mirrors the ControllerBase interface used by the sim loop, forwarding each
    call to the MPPI server over a socket. Holds no JAX state itself."""

    def __init__(self, host, port, ckpt_dir, ctrl_cfg):
        self.sock = socket.create_connection((host, port))
        self._call({"op": "build", "ckpt_dir": ckpt_dir, "ctrl_cfg": ctrl_cfg})

    def _call(self, msg):
        ipc.send_msg(self.sock, msg)
        rep = ipc.recv_msg(self.sock)
        if rep is None or not rep.get("ok"):
            raise RuntimeError(f"mppi_server error: {rep}")
        return rep

    def initialize(self, x, cmd):
        self._call({"op": "initialize", "x": x, "cmd": cmd})

    def reset(self, x, cmd):
        self._call({"op": "reset", "x": x, "cmd": cmd})

    def update(self, x):
        self._call({"op": "update", "x": x})

    def run_control(self):
        return self._call({"op": "run_control"})["out"]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Ur5eSim:
    def __init__(self, cfg: dict, mppi_host: str, mppi_port: int):
        print("Loaded config:")
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        self.num_episodes = cfg["sim"]["num_episodes"]
        self.max_steps = cfg["sim"]["max_steps"]

        # locate the checkpoint + read its model config (jax-free; the server builds
        # the actual JAX model from the same file)
        ckpt_dir = os.path.join(
            paths.get_model_checkpoints_dir(), self.cfg["model"]["checkpoint"]
        )
        model_cfg = OmegaConf.to_container(
            OmegaConf.load(os.path.join(ckpt_dir, paths.get_model_config_filename())),
            resolve=True,
        )
        sys_cfg = model_cfg["system"]
        self.proprio_obs_names = config.proprio_obs_names_from_sys_config(sys_cfg)
        self.extero_obs_names = config.extero_obs_names_from_sys_config(sys_cfg)
        self.act_dim = config.action_dim_from_sys_config(sys_cfg)
        cmd_dim = config.cmd_dim_from_sys_config(sys_cfg)
        assert cmd_dim == 0, f"simulate_ur5e assumes cmd_dim == 0; got {cmd_dim}"

        # create the remote controller (connects to the MPPI server)
        self.controller = RemoteController(
            mppi_host, mppi_port, ckpt_dir, self.cfg["control"]
        )

        self.raw_env, self.env = self.build_env()
        # live insertion metrics (same term probe_recorder_episode.py reads).
        # Read managers off raw_env: with --video, self.env is a RecordVideo wrapper.
        self.progress = self.raw_env.reward_manager.get_term_cfg("progress_context").func
        self.device = self.raw_env.device

        self.obs_dict, _ = self.env.reset()

        # cmd is zero-width for manipulation
        self.zero_cmd = np.zeros((0,), dtype=np.float32)
        self.zero_action = np.zeros((self.act_dim,), dtype=np.float32)
        x = self.get_controller_input(self.zero_action)
        self.controller.initialize(x, self.zero_cmd)
        self.controller.reset(x, self.zero_cmd)

        self.pbar = tqdm(desc="Simulation", unit="step",
                         total=self.num_episodes * self.max_steps)
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

        # offscreen render + RecordVideo wrapper if --video
        render_mode = "rgb_array" if args.video else None
        raw_env = ManagerBasedRLEnv(env_cfg, render_mode=render_mode)
        env = raw_env
        if args.video:
            vdir = args.video_dir or os.path.join(os.getcwd(), "videos")
            os.makedirs(vdir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                raw_env,
                video_folder=vdir,
                step_trigger=lambda s: s == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
            print(f"[video] recording {args.video_length} steps to {vdir}", flush=True)
        return raw_env, env

    def get_obs(self):
        """proprio (proprio_dim,) float32 and images (n_cam, 224, 224, 3) uint8."""
        obs = self.obs_dict[OBS_GROUP]
        proprio_obs = np.concatenate(
            [obs[n][0].detach().cpu().numpy().ravel() for n in self.proprio_obs_names]
        ).astype(np.float32)
        # camera order must match extero_obs.types in the system config -- that is the
        # order the encoder concatenates its per-camera ResNet features in.
        extero_obs = np.stack(
            [obs[n][0].detach().cpu().numpy().astype(np.uint8)
             for n in self.extero_obs_names]
        )
        return proprio_obs, extero_obs

    def get_controller_input(self, prev_action: np.ndarray):
        proprio_obs, extero_obs = self.get_obs()
        return {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "prev_action": prev_action,
        }

    def insertion_metrics(self):
        return float(self.progress.xyz_distance[0]), bool(self.progress.success[0])

    def run(self):
        for ep in range(self.num_episodes):
            if not simulation_app.is_running():
                break
            self.run_episode(ep)
        self.logging()
        self.controller.close()
        self.env.close()
        simulation_app.close()

    def run_episode(self, ep: int):
        self.obs_dict, _ = self.env.reset()
        last_action = self.zero_action.copy()
        # refill the controller history with the post-reset observation
        self.controller.reset(self.get_controller_input(last_action), self.zero_cmd)

        succeeded = False
        min_dist = np.inf
        total_reward = 0.0
        grip = []
        step = 0
        done = False

        while not done and step < self.max_steps and simulation_app.is_running():
            self.controller.update(self.get_controller_input(last_action))

            action = self.controller.run_control()["actions"][0]  # first planned step
            last_action = action
            grip.append(float(action[6]))

            action_torch = torch.from_numpy(np.asarray(action, dtype=np.float32))
            action_torch = action_torch.unsqueeze(0).to(self.device)
            self.obs_dict, reward, terminated, truncated, _ = self.env.step(action_torch)

            dist, succ = self.insertion_metrics()
            succeeded |= succ
            min_dist = min(min_dist, dist)
            total_reward += float(reward[0].item())

            done = bool(terminated[0].item() or truncated[0].item())
            step += 1
            self.pbar.update(1)

        # The gripper action is thresholded at 0 by BinaryJointAction (<0 = close).
        # MPPI's mean is zero-initialized, i.e. it sits exactly on that boundary,
        # while the expert commands ~-2.3 -- so track the sign, not just the value.
        g = np.array(grip) if grip else np.zeros(1)
        stats = {
            "episode": ep, "steps": step, "success": succeeded,
            "min_dist_cm": min_dist * 100.0, "total_reward": total_reward,
            "grip_mean": float(g.mean()), "grip_frac_close": float((g < 0).mean()),
        }
        self.episode_stats.append(stats)
        print(
            f"[sim] ep {ep + 1}/{self.num_episodes}: steps={step} "
            f"success={succeeded} min_dist={min_dist * 100:.2f}cm "
            f"reward={total_reward:.2f} | gripper mean={g.mean():+.2f} "
            f"closed={100 * (g < 0).mean():.0f}% (expert: -2.28, 94%)",
            flush=True,
        )

    def logging(self):
        n = len(self.episode_stats)
        if n == 0:
            return
        mean = lambda k: float(np.mean([s[k] for s in self.episode_stats]))  # noqa: E731
        metrics = {
            "success_rate": sum(s["success"] for s in self.episode_stats) / n,
            "mean_min_dist_cm": mean("min_dist_cm"),
            "mean_reward": mean("total_reward"),
            "mean_episode_length": mean("steps"),
            "gripper_frac_close": mean("grip_frac_close"),
            "num_episodes": n,
        }
        print(metrics)


@hydra.main(**paths.get_simulate_ur5e_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    sim = Ur5eSim(dict_cfg, args.mppi_host, args.mppi_port)
    sim.run()


if __name__ == "__main__":
    main()
