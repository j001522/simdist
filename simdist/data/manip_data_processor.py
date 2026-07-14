"""Manipulation data processor (UR5e peg insertion).

Go2's ``DataProcessor`` assumes flat-vector extero and loads episodes via Isaac's
``HDF5DatasetFileHandler`` (which cannot load our vlen-JPEG image fields). This
processor instead:

  * reads the raw robomimic-style HDF5 DIRECTLY with h5py -> **no Isaac import**, so
    process_data runs in the pure-JAX ``simdist-jax`` env (see manipulation_port.md 3.0);
  * selects proprio = ``arm_joint_pos`` (6) and extero = the 3 RGB cameras by name,
    ignoring the privileged obs also present in the file (schema in port doc 3.0.0);
  * stores images as **JPEG passthrough** (bytes copied, not decoded/scaled) into a
    per-camera ``images.hdf5`` -- decoded lazily in the dataloader. Proprio / actions /
    rewards / values are scaled as in Go2; images are NEVER scaled here (the encoder
    applies ImageNet normalization);
  * keeps the same global-concatenation + ``start_idxs`` layout as Go2 so the rest of
    the pipeline (dataset windowing) is unchanged apart from the image path.

TODO(verify-on-data): field names / dtypes are from the inspected schema (port doc
3.0.0); run once on a small slice of the real HDF5 to confirm h5py reads + vlen
passthrough round-trip before a full run.
"""

import json
import os

import h5py
import numpy as np
import yaml

from simdist.data import DATA_KEY
from simdist.data.data_processor import _H5Appender, _RunningStats
from simdist.data.jpeg_io import is_jpeg_dataset
from simdist.modeling import types
from simdist.utils import config as config_utils
from simdist.utils import io as io_utils
from simdist.utils import paths


class _JpegAppender:
    """Growable per-camera vlen-uint8 (JPEG bytes) HDF5 store, one dataset per camera."""

    def __init__(self, path: str, camera_names: list[str]):
        self.f = h5py.File(path, "w")
        self.camera_names = camera_names
        self.dsets = {
            name: self.f.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.uint8),
            )
            for name in camera_names
        }

    def append(self, name: str, frames, image_shape) -> None:
        """Append a demo's frames (iterable of JPEG byte arrays) for one camera."""
        dset = self.dsets[name]
        n = dset.shape[0]
        dset.resize(n + len(frames), axis=0)
        for i, buf in enumerate(frames):
            dset[n + i] = np.asarray(buf, dtype=np.uint8)
        # record frame shape once (identical across frames/demos)
        if "image_shape" not in dset.attrs:
            dset.attrs["image_shape"] = np.asarray(image_shape, dtype=np.int64)
            dset.attrs["jpeg"] = True

    def __len__(self) -> int:
        return self.dsets[self.camera_names[0]].shape[0]

    def close(self) -> None:
        self.f.flush()
        self.f.close()


def _clean_float(arr: np.ndarray) -> np.ndarray:
    """float32 with NaN/Inf -> 0 (mirrors data_processor._to_valid_numpy for numpy)."""
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class ManipulationDataProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.sys = cfg["system"]
        self.H = cfg["history_length"]
        self.T = cfg["prediction_length"]
        self.beg_padding = cfg["beg_padding"]
        self.end_padding = cfg["end_padding"]

        self.proprio_obs_dim = config_utils.proprio_obs_dim_from_sys_config(self.sys)
        self.action_dim = config_utils.action_dim_from_sys_config(self.sys)
        self.cmd_dim = config_utils.cmd_dim_from_sys_config(self.sys)
        self.proprio_obs_names = config_utils.proprio_obs_names_from_sys_config(self.sys)
        self.extero_obs_names = config_utils.extero_obs_names_from_sys_config(self.sys)
        self.image_shapes = config_utils.extero_obs_image_shapes_from_sys_config(self.sys)
        assert len(self.image_shapes) == len(self.extero_obs_names), (
            "manip processor expects every extero obs to be an image (list dim); "
            f"got names={self.extero_obs_names} shapes={self.image_shapes}"
        )

        # Images are NOT scaled here -> no extero tracker. commands is zero-width for
        # manip (cmd_dim=0), tracked only to keep the ScalerParams schema populated.
        self.stats_trackers = {
            "proprio_obs": _RunningStats(self.proprio_obs_dim),
            "actions": _RunningStats(self.action_dim),
            "commands": _RunningStats(self.cmd_dim),
            "rewards": _RunningStats(0),
            "values": _RunningStats(0),
        }
        self.exp_pol_flags_tracker = _RunningStats(0)

    def run(self):
        raw_data_path = paths.get_raw_data_path(self.dataset_name)
        processed_data_dir = paths.get_processed_data_dir(
            self.dataset_name, self.sys["name"], self.H, self.T
        )
        os.makedirs(processed_data_dir, exist_ok=True)
        with open(os.path.join(processed_data_dir, "system.yaml"), "w") as f:
            yaml.dump(self.sys, f, default_flow_style=False)

        min_len = self.H + self.T + self.beg_padding + self.end_padding

        start_idxs_h5 = _H5Appender(
            paths.get_start_idxs_path(processed_data_dir), DATA_KEY, 0, dtype=np.int64
        )
        proprio_obs_h5 = _H5Appender(
            paths.get_proprio_obs_path(processed_data_dir), DATA_KEY, self.proprio_obs_dim
        )
        images_h5 = _JpegAppender(
            paths.get_images_path(processed_data_dir), self.extero_obs_names
        )
        acts_h5 = _H5Appender(
            paths.get_actions_path(processed_data_dir), DATA_KEY, self.action_dim
        )
        cmd_h5 = _H5Appender(
            paths.get_commands_path(processed_data_dir), DATA_KEY, self.cmd_dim
        )
        rewards_h5 = _H5Appender(paths.get_rewards_path(processed_data_dir), DATA_KEY, 0)
        values_h5 = _H5Appender(paths.get_values_path(processed_data_dir), DATA_KEY, 0)
        exp_pol_flags_h5 = _H5Appender(
            paths.get_expert_policy_flags_path(processed_data_dir),
            DATA_KEY,
            0,
            dtype=np.bool_,
        )

        all_ep_lens = []
        ep_rewards = []
        n_skipped_short = 0
        start = 0

        raw = h5py.File(raw_data_path, "r")
        data_grp = raw[DATA_KEY]
        # NOT contiguous / not zero-padded -> index by key (port doc 3.0.0).
        demo_names = list(data_grp.keys())

        for demo_name in demo_names:
            g = data_grp[demo_name]
            ep_len = g["actions"].shape[0]
            all_ep_lens.append(ep_len)
            if ep_len < min_len:
                n_skipped_short += 1
                continue

            obs = g["obs"]

            # proprio (arm_joint_pos) -> (T, proprio_dim), scaled
            proprio_np = np.concatenate(
                [np.asarray(obs[name]) for name in self.proprio_obs_names], axis=-1
            )
            proprio_np = _clean_float(proprio_np)
            proprio_obs_h5.append(proprio_np)
            self.stats_trackers["proprio_obs"].update(proprio_np)

            # extero images -> JPEG passthrough per camera, no scaling
            for name in self.extero_obs_names:
                cam_dset = obs[name]
                assert is_jpeg_dataset(cam_dset), (
                    f"{demo_name}/{name} is not a JPEG-tagged dataset; "
                    "manip processor expects vlen-JPEG images"
                )
                frames = list(cam_dset)  # list of uint8 byte arrays, length T
                images_h5.append(name, frames, cam_dset.attrs["image_shape"])

            acts_np = _clean_float(g["actions"])
            acts_h5.append(acts_np)
            self.stats_trackers["actions"].update(acts_np)

            # commands: zero-width for manip (no goal command); placeholder for schema
            cmd_np = np.zeros((ep_len,), dtype=np.float32)
            cmd_h5.append(cmd_np)
            self.stats_trackers["commands"].update(cmd_np)

            rewards_np = _clean_float(g["reward"])
            ep_rewards.append(float(np.sum(rewards_np)))
            rewards_h5.append(rewards_np)
            self.stats_trackers["rewards"].update(rewards_np)

            values_np = _clean_float(g["value"])
            values_h5.append(values_np)
            self.stats_trackers["values"].update(values_np)

            exp_pol_flags_np = np.asarray(g["expert_policy_flag"], dtype=np.bool_)
            exp_pol_flags_h5.append(exp_pol_flags_np)
            self.exp_pol_flags_tracker.update(exp_pol_flags_np.astype(np.float32))

            ep_start_idxs = np.arange(
                start + self.beg_padding,
                start + ep_len - (self.H + self.T + self.end_padding),
                dtype=np.int64,
            )
            start_idxs_h5.append(ep_start_idxs)
            start += ep_len

        raw.close()

        # ScalerParams TypedDict requires an extero_obs entry; images are not scaled, so
        # write an identity placeholder (the manip model's scaler_params_mapping omits
        # extero_obs, so this is never applied). See manipulation_port.md 3.3.
        scaler_params: types.ScalerParams = {
            k: tracker.finalize() for k, tracker in self.stats_trackers.items()
        }
        scaler_params["extero_obs"] = {"mean": [0.0], "std": [1.0]}
        io_utils.save_scaler_params(scaler_params, processed_data_dir)

        dataset_metrics = {
            "num_episodes": len(all_ep_lens),
            "num_episodes_kept": len(all_ep_lens) - n_skipped_short,
            "num_episodes_skipped_short": n_skipped_short,
            "min_len_threshold": min_len,
            "avg_episode_length": float(np.mean(all_ep_lens)) if all_ep_lens else 0.0,
            "num_trajectories": len(start_idxs_h5),
            "avg_reward_per_episode": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "avg_reward_per_step": self.stats_trackers["rewards"].finalize()["mean"],
            "exp_actions_to_total_actions_ratio": self.exp_pol_flags_tracker.finalize()[
                "mean"
            ],
        }
        with open(os.path.join(processed_data_dir, "dataset_metrics.json"), "w") as f:
            json.dump(dataset_metrics, f, indent=4)

        start_idxs_h5.close()
        proprio_obs_h5.close()
        images_h5.close()
        acts_h5.close()
        cmd_h5.close()
        rewards_h5.close()
        values_h5.close()
        exp_pol_flags_h5.close()
