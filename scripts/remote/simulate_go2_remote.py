"""Go2 closed-loop simulation with the world-model MPPI controller running in a
SEPARATE process — Process A of the 2-process Spark loop.

This is `scripts/simulate_go2.py` with the in-process `MppiController` replaced by
a thin `RemoteController` that forwards reset/update/run_control over a localhost
socket to `mppi_server.py` (Process B, JAX/GPU). The split is required on the DGX
Spark because Isaac/IsaacLab need numpy<2 while the GPU JAX stack needs numpy>=2,
so they cannot share one Python.

This process imports ONLY jax-free simdist modules (rl.go2, utils.config,
utils.paths); all JAX/flax lives in the server.

Run via `run.sh`, or manually (after starting mppi_server.py):
    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
      <isaac-python.sh> simulate_go2_remote.py --mppi-port 5599 [hydra overrides]
"""
import argparse
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root (contains the `simdist` package); simdist is not pip-installed in the
# Isaac python on purpose (its deps pull jax/numpy>=2), so add it to the path.
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
parser.add_argument("--video_length", type=int, default=200,
                    help="Number of steps to record.")
parser.add_argument("--video_dir", default=None,
                    help="Output dir for videos (default: <cwd>/videos).")
# adds --headless, --enable_cameras, --device, etc.
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown  # keep only unknown args for hydra
if args.video:
    args.enable_cameras = True  # offscreen rendering needs cameras

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
from tqdm import tqdm  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from simdist.utils import paths, config  # noqa: E402  (jax-free)
from simdist.rl.go2 import Go2SimEnvCfg  # noqa: E402  (jax-free)


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

    def set_fut_cmd(self, fut_cmd):
        self._call({"op": "set_fut_cmd", "fut_cmd": fut_cmd})

    def update(self, x):
        self._call({"op": "update", "x": x})

    def run_control(self):
        return self._call({"op": "run_control"})["out"]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Go2Sim:
    def __init__(self, cfg: dict, mppi_host: str, mppi_port: int):
        print("Loaded config:")
        print(OmegaConf.to_yaml(cfg))
        self.cfg = cfg
        self.total_steps = cfg["sim"]["total_steps"]
        self.reset_steps = cfg["sim"]["reset_steps"]

        # locate checkpoint dir + read its model config (jax-free; the server
        # builds the actual JAX model from the same file)
        ckpt_dir = os.path.join(
            paths.get_model_checkpoints_dir(), self.cfg["model"]["checkpoint"]
        )
        model_cfg = OmegaConf.to_container(
            OmegaConf.load(
                os.path.join(ckpt_dir, paths.get_model_config_filename())
            ),
            resolve=True,
        )
        self.proprio_obs_names = config.proprio_obs_names_from_sys_config(
            model_cfg["system"]
        )

        # create the remote controller (connects to the MPPI server)
        self.controller = RemoteController(
            mppi_host, mppi_port, ckpt_dir, self.cfg["control"]
        )

        # create env cfg
        env_cfg = Go2SimEnvCfg()

        # Disable domain-randomization events that wrap now-class-based IsaacLab
        # mdp terms (randomize_actuator_gains / randomize_joint_parameters) as if
        # they were plain functions — incompatible with our IsaacLab 2.3.2 and
        # irrelevant to this visualisation.
        for _ev in ("joint_stiffness_and_damping", "joint_friction"):
            if getattr(env_cfg.events, _ev, None) is not None:
                setattr(env_cfg.events, _ev, None)

        # set command in env
        env_cfg.commands.base_velocity.forward_vel = cfg["task"]["forward_vel"]
        env_cfg.commands.base_velocity.kx = cfg["task"]["kx"]
        env_cfg.commands.base_velocity.ky = cfg["task"]["ky"]
        env_cfg.commands.base_velocity.k_heading = cfg["task"]["k_heading"]

        # set up env
        terr = cfg["sim"]["terrain"]
        diff = cfg["sim"]["terrain_difficulty"]
        env_cfg.scene.terrain.terrain_generator.seed = cfg["sim"]["seed"]
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.size = (15.0, 15.0)
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (diff, diff)
        terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {terr: terrains[terr]}
        env_cfg.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        fric = cfg["sim"]["friction"]
        env_cfg.events.physics_material.params["static_friction_range"] = (fric, fric)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (fric, fric)
        rest = cfg["sim"]["restitution"]
        env_cfg.events.physics_material.params["restitution_range"] = (rest, rest)
        mass = cfg["sim"]["add_mass"]
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (mass, mass)

        # create the env (offscreen render + RecordVideo wrapper if --video)
        render_mode = "rgb_array" if args.video else None
        self.raw_env = ManagerBasedRLEnv(env_cfg, render_mode=render_mode)
        self.env = self.raw_env
        if args.video:
            vdir = args.video_dir or os.path.join(os.getcwd(), "videos")
            os.makedirs(vdir, exist_ok=True)
            self.env = gym.wrappers.RecordVideo(
                self.raw_env,
                video_folder=vdir,
                step_trigger=lambda s: s == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
            print(f"[video] recording {args.video_length} steps to {vdir}",
                  flush=True)
        self.cmd_manager = self.raw_env.command_manager
        self.obs_dict, _ = self.env.reset()
        proprio_obs, height_scan = self.get_obs(self.obs_dict)

        # set up the controller
        self.zero_cmd = np.zeros((3,))
        self.zero_action = np.zeros((12,))
        x = self.make_controller_input(proprio_obs, height_scan, self.zero_action)
        self.controller.initialize(x, self.zero_cmd)
        self.controller.reset(x, self.zero_cmd)
        self.controller.set_fut_cmd(self.zero_cmd)

        self.pbar = tqdm(desc="Simulation", unit="step", total=self.total_steps)

        # state
        self.steps = 0
        self.total_step_count = 0
        self.last_action = self.zero_action.copy()
        self.total_reward = 0
        self.episode_terminated = False
        self.episode_length = 0

    def run(self):
        while simulation_app.is_running():
            was_reset = self.step()
            if self.total_step_count >= self.total_steps:
                print("Total steps reached. Exiting simulation.")
                break
            if self.cfg["sim"]["kill_on_reset"] and was_reset:
                print("Episode terminated. Exiting simulation.")
                break
        self.logging()
        self.controller.close()
        self.env.close()
        simulation_app.close()

    def get_obs(self, obs_dict):
        proprio_obs = np.concatenate(
            [
                obs_dict["obs"][name].squeeze().cpu().numpy()
                for name in self.proprio_obs_names
            ]
        )
        height_scan = obs_dict["obs"]["height_scan"].squeeze().cpu().numpy()
        return proprio_obs, height_scan

    def make_controller_input(self, proprio_obs, extero_obs, prev_action):
        return {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "prev_action": prev_action,
        }

    def step(self):
        resetting = self.steps < self.reset_steps
        if resetting:
            cmd = self.zero_cmd
        else:
            cmd = self.cmd_manager.get_command("base_velocity").cpu().numpy()[0]
        self.controller.set_fut_cmd(cmd)

        proprio_obs, height_scan = self.get_obs(self.obs_dict)
        self.controller.update(
            self.make_controller_input(proprio_obs, height_scan, self.last_action)
        )

        ctrl_out = self.controller.run_control()
        action = ctrl_out["actions"][0]
        if resetting:
            action = self.zero_action
        self.last_action = action

        action_torch = self.action_to_torch(action)
        self.obs_dict, reward, reset = self.env.step(action_torch)[0:3]
        self.pbar.update(1)
        self.steps += 1
        self.total_step_count += 1
        if not self.episode_terminated:
            self.total_reward += reward[0].item()
            self.episode_length += 1
        if reset[0]:
            self.reset()
        return reset[0]

    def action_to_torch(self, action):
        return torch.from_numpy(np.asarray(action)).unsqueeze(0).to(
            self.raw_env.device)

    def reset(self):
        self.episode_terminated = True
        self.steps = 0
        proprio_obs, height_scan = self.get_obs(self.obs_dict)
        x = self.make_controller_input(proprio_obs, height_scan, self.zero_action)
        self.controller.reset(x, self.zero_cmd)

    def logging(self):
        denom = max(self.episode_length - self.reset_steps, 1)
        print({
            "total_reward": self.total_reward,
            "reward_per_step": self.total_reward / denom,
            "total_steps": self.total_step_count,
            "episode_length": self.episode_length,
        })


@hydra.main(**paths.get_simulate_go2_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    go2_sim = Go2Sim(dict_cfg, args.mppi_host, args.mppi_port)
    go2_sim.run()


if __name__ == "__main__":
    main()
