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
import datetime
import json
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
# Control experiments: drive the env with the DATA-COLLECTION expert instead of MPPI.
#   --expert              : run the expert directly in-process (no server) -> tests the
#                           env + reset config (the insertion ceiling).
#   --expert --via-server : round-trip the expert action through the server's echo op ->
#                           tests the 2-process IPC plumbing with known-good actions.
parser.add_argument("--expert", action="store_true",
                    help="Drive the env with the data-collection expert, not MPPI.")
parser.add_argument("--expert-run", default="omnireset_2026-07-01_16-31-09",
                    help="rsl_rl run dir holding the expert checkpoint (the expert that "
                         "generated the current dataset: eval success 0.89 at iter 1600).")
parser.add_argument("--expert-iter", type=int, default=1600,
                    help="Expert checkpoint iteration.")
parser.add_argument("--via-server", action="store_true",
                    help="With --expert: send each action through the server echo op "
                         "(IPC-plumbing test). Without it the expert runs in-process.")
# Checkpoint sweep: evaluate every step in the run dir in one process. The run is not
# monotonic across steps, so picking a checkpoint by "latest" is unsound -- this is how
# you pick one by measured planning performance instead.
parser.add_argument("--sweep", action="store_true",
                    help="Evaluate every checkpoint step in the run dir, not just one.")
parser.add_argument("--sweep-episodes", type=int, default=5,
                    help="Episodes per checkpoint (default 5).")
# Comma-separated, not nargs="*": a greedy list would swallow the trailing hydra
# overrides (they do not start with "-") and then fail to int() them.
parser.add_argument("--sweep-steps", default=None,
                    help="Restrict the sweep to these steps, comma-separated "
                         "(e.g. --sweep-steps 38000,40000). Default: all in the dir.")
parser.add_argument("--sweep-out", default=None,
                    help="JSON report path (default: <ckpt_dir>/mppi_sweep.json). "
                         "The figure is written alongside it as .png.")
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


def plot_sweep(report: dict, png_path: str):
    """Success rate and mean reward against checkpoint step, one figure."""
    import matplotlib
    matplotlib.use("Agg")  # no display: this runs headless
    import matplotlib.pyplot as plt

    rows = report["steps"]
    xs = [r["step"] for r in rows]
    n = report["episodes_per_step"]

    fig, (ax_sr, ax_rw) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    ax_sr.plot(xs, [r["success_rate"] for r in rows], marker="o", color="tab:blue")
    ax_sr.set_ylabel("success rate")
    # With n episodes the rate can only land on multiples of 1/n, so pin the axis to
    # [0, 1] -- autoscale on a flat-zero sweep is meaningless and reads as noise.
    ax_sr.set_ylim(-0.05, 1.05)
    ax_sr.grid(alpha=0.3)

    ax_rw.plot(xs, [r["mean_reward"] for r in rows], marker="o", color="tab:orange")
    ax_rw.set_ylabel("mean episode reward")
    ax_rw.set_xlabel("checkpoint step")
    ax_rw.grid(alpha=0.3)

    ax_sr.set_title(f"{report['checkpoint']} — MPPI, {n} episodes per checkpoint")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


class RemoteController:
    """Mirrors the ControllerBase interface used by the sim loop, forwarding each
    call to the MPPI server over a socket. Holds no JAX state itself."""

    def __init__(self, host, port, ckpt_dir, ctrl_cfg, build=True):
        self.sock = socket.create_connection((host, port))
        if build:
            self.build(ckpt_dir, ctrl_cfg)

    def build(self, ckpt_dir, ctrl_cfg, ckpt_step=None):
        """(Re)build the server-side controller, optionally pinning a checkpoint step.

        The sweep calls this once per checkpoint so both processes stay up across the
        whole run -- rebuilding costs a model load plus a jit warm, against ~2 min of
        Isaac startup for a fresh process."""
        msg = {"op": "build", "ckpt_dir": ckpt_dir, "ctrl_cfg": ctrl_cfg}
        if ckpt_step is not None:
            msg["ckpt_step"] = ckpt_step
        self._call(msg)

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

    def echo(self, actions):
        """Round-trip an action list through the server unchanged (IPC test)."""
        return self._call({"op": "echo", "actions": actions})["out"]["actions"]

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
        self.ckpt_dir = ckpt_dir
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

        # create the remote controller (connects to the MPPI server). Skipped entirely
        # when running the expert in-process; echo-only (no model build) when the expert
        # is routed through the server for the IPC-plumbing test.
        self.controller = None
        if not args.expert:
            # In sweep mode the per-checkpoint build happens in run_sweep, so skip the
            # default build here rather than loading a model we immediately discard.
            self.controller = RemoteController(
                mppi_host, mppi_port, ckpt_dir, self.cfg["control"],
                build=not args.sweep,
            )
        elif args.via_server:
            self.controller = RemoteController(
                mppi_host, mppi_port, ckpt_dir, None, build=False
            )

        self.raw_env, self.env = self.build_env()
        # live insertion metrics (same term probe_recorder_episode.py reads).
        # Read managers off raw_env: with --video, self.env is a RecordVideo wrapper.
        self.progress = self.raw_env.reward_manager.get_term_cfg("progress_context").func
        self.device = self.raw_env.device

        self.obs_dict, _ = self.env.reset()

        # load the data-collection expert (Process A has torch; consumes the privileged
        # "policy" obs group, same as generate_data / render_expert_episode).
        self.expert = None
        if args.expert:
            from simdist.utils.torch import get_actor_critic_from_iteration
            self.expert, _ = get_actor_critic_from_iteration(
                args.expert_run, args.expert_iter, self.device
            )
            mode = "via server echo" if args.via_server else "direct in-process"
            print(f"[expert] loaded {args.expert_run} iter {args.expert_iter} "
                  f"({mode})", flush=True)

        # cmd is zero-width for manipulation
        self.zero_cmd = np.zeros((0,), dtype=np.float32)
        self.zero_action = np.zeros((self.act_dim,), dtype=np.float32)
        if self.expert is None and not args.sweep:
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
        if args.sweep:
            self.run_sweep()
        else:
            for ep in range(self.num_episodes):
                if not simulation_app.is_running():
                    break
                self.run_episode(ep)
            self.logging()
        if self.controller is not None:
            self.controller.close()
        self.env.close()
        simulation_app.close()

    def run_episode(self, ep: int):
        self.obs_dict, _ = self.env.reset()
        last_action = self.zero_action.copy()
        # refill the controller history with the post-reset observation
        if self.expert is None:
            self.controller.reset(self.get_controller_input(last_action), self.zero_cmd)

        succeeded = False
        min_dist = np.inf
        total_reward = 0.0
        grip = []
        step = 0
        done = False

        while not done and step < self.max_steps and simulation_app.is_running():
            if self.expert is not None:
                # data-collection expert: consumes privileged "policy" obs, no scaling
                with torch.no_grad():
                    action = (self.expert(self.obs_dict["policy"])[0]
                              .detach().cpu().numpy().astype(np.float32))
                if args.via_server:
                    # same serialize/send/recv path MPPI actions take, model bypassed
                    action = np.asarray(self.controller.echo(action.tolist()),
                                        dtype=np.float32)
            else:
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
            return None
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
        return metrics

    @staticmethod
    def discover_steps(ckpt_dir: str):
        """Checkpoint step dirs, ascending. Orbax writes one numeric dir per step
        alongside model_config.yaml, so numeric-and-directory is the whole test."""
        return sorted(
            int(d) for d in os.listdir(ckpt_dir)
            if d.isdigit() and os.path.isdir(os.path.join(ckpt_dir, d))
        )

    def run_sweep(self):
        steps = self.discover_steps(self.ckpt_dir)
        if args.sweep_steps:
            wanted = {int(s) for s in args.sweep_steps.split(",") if s.strip()}
            missing = wanted - set(steps)
            if missing:
                raise SystemExit(f"no such checkpoint step(s): {sorted(missing)}")
            steps = [s for s in steps if s in wanted]
        if not steps:
            raise SystemExit(f"no checkpoint step dirs under {self.ckpt_dir}")

        n = args.sweep_episodes
        self.num_episodes = n  # so run_episode's per-episode print reads correctly
        out_path = args.sweep_out or os.path.join(self.ckpt_dir, "mppi_sweep.json")
        report = {
            "checkpoint": self.cfg["model"]["checkpoint"],
            "checkpoint_dir": self.ckpt_dir,
            "episodes_per_step": n,
            "max_steps": self.max_steps,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "sim": self.cfg["sim"],
            "control": self.cfg["control"],
            "steps": [],
        }
        print(f"[sweep] {len(steps)} checkpoints x {n} episodes -> {out_path}",
              flush=True)

        self.pbar.close()
        self.pbar = tqdm(desc="Sweep", unit="step",
                         total=len(steps) * n * self.max_steps)
        for step in steps:
            if not simulation_app.is_running():
                print(f"[sweep] app closed, stopping before step {step}", flush=True)
                break
            print(f"\n[sweep] ===== checkpoint step {step} =====", flush=True)
            self.controller.build(self.ckpt_dir, self.cfg["control"], step)
            # A rebuild returns a fresh MppiController: its ring buffer and jit caches
            # have to be primed again before run_episode's reset.
            x = self.get_controller_input(self.zero_action)
            self.controller.initialize(x, self.zero_cmd)

            self.episode_stats = []
            for ep in range(n):
                if not simulation_app.is_running():
                    break
                self.run_episode(ep)
            metrics = self.logging()
            if metrics is not None:
                report["steps"].append({"step": step, **metrics})
                report["episodes"] = report.get("episodes", {})
                report["episodes"][str(step)] = list(self.episode_stats)
                # Rewrite after every checkpoint: a sweep is long and a kill partway
                # should still leave a usable report.
                with open(out_path, "w") as f:
                    json.dump(report, f, indent=2)

        print(f"[sweep] wrote {out_path}", flush=True)
        if report["steps"]:
            png = os.path.splitext(out_path)[0] + ".png"
            plot_sweep(report, png)
            print(f"[sweep] wrote {png}", flush=True)
        return report


@hydra.main(**paths.get_simulate_ur5e_hydra_config())
def main(cfg: DictConfig):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    sim = Ur5eSim(dict_cfg, args.mppi_host, args.mppi_port)
    sim.run()


if __name__ == "__main__":
    main()
