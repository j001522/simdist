from typing import TypedDict, Any
import os

from torch.utils.data import Dataset
import cv2
import h5py
import numpy as np

from simdist.modeling import types
from simdist.utils import paths, config, registry
from simdist.data import DATA_KEY
from simdist.data import jpeg_io


_DATASET_REGISTRY: registry.Registry["DatasetBase"] = registry.Registry("Dataset")


def register_dataset(name: str):
    return _DATASET_REGISTRY.register(name)


def get_dataset(cfg: dict) -> "DatasetBase":
    dataset_name = cfg["model"]["dataset"]["type"]
    return _DATASET_REGISTRY.create(dataset_name, cfg)


class DatasetItem(TypedDict):
    model_in: types.ModelInputs
    labels: types.TrainingLabels
    metadata: dict[str, Any]


class DatasetBatch(TypedDict):
    model_in: types.ModelInputs
    labels: types.TrainingLabels


class DatasetBase(Dataset):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._eval_mode = False
        self._data_dir = ""

    def eval(self) -> None:
        self._eval_mode = True

    def train(self) -> None:
        self._eval_mode = False

    def __len__(self) -> int:
        raise NotImplementedError("This method must be implemented.")

    def __getitem__(self, idx: int) -> DatasetItem:
        raise NotImplementedError("This method must be implemented.")

    @property
    def data_dir(self) -> str:
        return self._data_dir


@register_dataset("world_model")
class WorldModelDatasetBase(DatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        dataset_name = cfg["data"]["dataset_name"]
        sys_name = cfg["system"]["name"]
        self.H = config.history_length_from_config(cfg)
        self.T = config.prediction_length_from_config(cfg)

        self._files: dict[str, h5py.File] = {}
        self._data: dict[str, h5py.Dataset] = {}

        # Load data
        self._data_dir = paths.get_processed_data_dir(
            dataset_name, sys_name, self.H, self.T
        )
        if not os.path.exists(self._data_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self._data_dir}' does not exist."
                "Did you remember to run scripts/process_data.py?"
            )
        self._load_files(self._data_dir)
        self._load_data()

    def __len__(self) -> int:
        return len(self._data["start_idxs"])

    def __getitem__(self, idx: int) -> DatasetItem:
        t_H = self._data["start_idxs"][idx]
        item = self.get_item_by_t_H(t_H)
        if not self._eval_mode:
            item = self.data_augmentations(item)
        return item

    def get_item_by_t_H(self, t_H: int) -> DatasetItem:
        # metadata to be passed to subclasses
        t = t_H + self.H
        t_T = t + self.T
        metadata = {
            "t_H": t_H,
            "t": t,
            "t_T": t_T,
        }

        # inputs
        proprio_obs_hist: np.ndarray = self._data["proprio_obs"][t_H : t + 1]
        extero_obs: np.ndarray = self._data["extero_obs"][t]
        acts_hist: np.ndarray = self._data["actions"][t_H:t]
        fut_acts: np.ndarray = self._data["actions"][t:t_T]
        fut_cmds: np.ndarray = self._data["commands"][t : t_T + 1]
        model_inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": proprio_obs_hist,
            "extero_obs": extero_obs,
            "acts_hist": acts_hist,
            "fut_acts": fut_acts,
            "fut_cmds": fut_cmds,
        }

        # labels
        proprio_obs: np.ndarray = self._data["proprio_obs"][t + 1 : t_T + 1]
        extero_obs: np.ndarray = self._data["extero_obs"][t + 1 : t_T + 1]
        rewards: np.ndarray = self._data["rewards"][t:t_T]
        values: np.ndarray = self._data["values"][t + 1 : t_T + 1]
        actions: np.ndarray = self._data["actions"][t:t_T]
        exp_pol_flags: np.ndarray = self._data["exp_pol_flags"][t:t_T]
        labels: types.WorldModelSchema.Labels = {
            "proprio_obs": proprio_obs,
            "extero_obs": extero_obs,
            "rewards": rewards,
            "values": values,
            "actions": actions,
            "exp_pol_flags": exp_pol_flags,
        }

        item: DatasetItem = {
            "model_in": model_inputs,
            "labels": labels,
            "metadata": metadata,
        }
        return item

    def data_augmentations(self, item: DatasetItem) -> DatasetItem:
        return item

    def _load_files(self, data_dir: str) -> None:
        self._files["start_idxs"] = h5py.File(paths.get_start_idxs_path(data_dir))
        self._files["proprio_obs"] = h5py.File(paths.get_proprio_obs_path(data_dir))
        self._files["extero_obs"] = h5py.File(paths.get_extero_obs_path(data_dir))
        self._files["actions"] = h5py.File(paths.get_actions_path(data_dir))
        self._files["commands"] = h5py.File(paths.get_commands_path(data_dir))
        self._files["rewards"] = h5py.File(paths.get_rewards_path(data_dir))
        self._files["values"] = h5py.File(paths.get_values_path(data_dir))
        self._files["exp_pol_flags"] = h5py.File(
            paths.get_expert_policy_flags_path(data_dir)
        )

    def _load_data(self) -> None:
        for key, file in self._files.items():
            self._data[key] = file[DATA_KEY]

    @staticmethod
    def get_dummy_item(cfg: dict) -> DatasetItem:
        sys_cfg = cfg["system"]
        proprio_obs_dim = config.proprio_obs_dim_from_sys_config(sys_cfg)
        extero_obs_dim = config.extero_obs_dim_from_sys_config(sys_cfg)
        act_dim = config.action_dim_from_sys_config(sys_cfg)
        cmd_dim = config.cmd_dim_from_sys_config(sys_cfg)
        H = cfg["model"]["dataset"]["history_length"]
        T = cfg["model"]["dataset"]["prediction_length"]

        rng = np.random.default_rng()

        inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": rng.standard_normal((H + 1, proprio_obs_dim)),
            "extero_obs": rng.standard_normal((extero_obs_dim,)),
            "acts_hist": rng.standard_normal((H, act_dim)),
            "fut_acts": rng.standard_normal((T, act_dim)),
            "fut_cmds": rng.standard_normal((T + 1, cmd_dim)),
        }

        labels: types.WorldModelSchema.Labels = {
            "proprio_obs": rng.standard_normal((T, proprio_obs_dim)),
            "extero_obs": rng.standard_normal((T, extero_obs_dim)),
            "rewards": rng.standard_normal((T,)),
            "values": rng.standard_normal((T,)),
            "actions": rng.standard_normal((T, act_dim)),
            "exp_pol_flags": np.ones((T,)),
        }

        return {
            "model_in": inputs,
            "labels": labels,
            "metadata": {"dummy": True},
        }


@register_dataset("quadruped_world_model")
class QuadrupedWorldModelDataset(WorldModelDatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        sys_cfg = self.cfg["system"]
        aug_cfg = self.cfg["model"]["dataset"]["augmentations"]
        self.add_noise = aug_cfg["add_noise"]

        if not self.add_noise:
            return

        # Determine noise to add to proprioceptive observations
        proprio_ob_dims = [ob["dim"] for ob in sys_cfg["proprio_obs"]["types"]]
        self.proprio_obs_dim = sum(proprio_ob_dims)
        proprio_obs_noises = [ob["noise"] for ob in sys_cfg["proprio_obs"]["types"]]
        self.proprio_obs_noise_stds = []
        for dim, noise in zip(proprio_ob_dims, proprio_obs_noises):
            self.proprio_obs_noise_stds.extend([noise] * dim)
        self.proprio_obs_noise_stds = np.array(self.proprio_obs_noise_stds)

        # Determine noise to add to the height_scan
        self.height_noise = None
        for ob in sys_cfg["extero_obs"]["types"]:
            if ob["name"] == "height_scan":
                self.height_noise = ob["noise"]
                break
        if self.height_noise is None:
            raise ValueError("Height scan noise not found")

    def data_augmentations(self, item: DatasetItem) -> DatasetItem:
        if not self.add_noise:
            return item

        # Apply noise to proprioceptive observations
        item["model_in"]["proprio_obs_hist"] = _apply_noise(
            item["model_in"]["proprio_obs_hist"], self.proprio_obs_noise_stds
        )

        # Apply noise to height scan
        item["model_in"]["extero_obs"] = _apply_noise(
            item["model_in"]["extero_obs"], self.height_noise
        )

        return item


def _apply_noise(arr: np.ndarray, std: float | np.ndarray) -> np.ndarray:
    noise = np.random.randn(*arr.shape)
    return arr + noise * std


@register_dataset("manipulation_world_model")
class ManipulationWorldModelDataset(WorldModelDatasetBase):
    """UR5e peg-insertion dataset (paper app:manip).

    Same global-concat + ``start_idxs`` windowing as the Go2 dataset, but the extero
    field is images rather than a flat vector:
      * ``images.hdf5`` holds one vlen-JPEG dataset per camera (keyed by camera name),
        not a single ``extero_obs.hdf5`` keyed by ``DATA_KEY`` -> override the file /
        data loaders and decode the needed frames lazily in ``get_item_by_t_H``;
      * the model input ``extero_obs`` is the LATEST frame-set ``(n_cam, H, W, 3)`` uint8;
        the label ``extero_obs`` is the future window ``(T, n_cam, H, W, 3)`` uint8 used
        for the latent-dynamics target ``sg(E(o_{t+1:t+T}))``;
      * augmentations: gaussian proprio noise (as Go2) plus image augs (crop/jitter/blur)
        applied to the INPUT frame-set only. The future/target frames are left clean so
        the regression target is not corrupted (see manipulation_port.md 3.2/3.3).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        sys_cfg = self.cfg["system"]
        self.extero_obs_names = config.extero_obs_names_from_sys_config(sys_cfg)
        self.n_cam = len(self.extero_obs_names)

        aug_cfg = self.cfg["model"]["dataset"]["augmentations"]
        self.add_noise = bool(aug_cfg.get("add_noise", False))
        self.image_aug_cfg = aug_cfg.get("image") or {}

        self.proprio_obs_noise_stds = None
        if self.add_noise:
            proprio_ob_dims = [ob["dim"] for ob in sys_cfg["proprio_obs"]["types"]]
            proprio_obs_noises = [ob["noise"] for ob in sys_cfg["proprio_obs"]["types"]]
            stds = []
            for dim, noise in zip(proprio_ob_dims, proprio_obs_noises):
                stds.extend([noise] * dim)
            self.proprio_obs_noise_stds = np.array(stds)

    def _load_files(self, data_dir: str) -> None:
        # Same as the base loader, but the extero field is the per-camera image store.
        self._files["start_idxs"] = h5py.File(paths.get_start_idxs_path(data_dir))
        self._files["proprio_obs"] = h5py.File(paths.get_proprio_obs_path(data_dir))
        self._files["images"] = h5py.File(paths.get_images_path(data_dir))
        self._files["actions"] = h5py.File(paths.get_actions_path(data_dir))
        self._files["commands"] = h5py.File(paths.get_commands_path(data_dir))
        self._files["rewards"] = h5py.File(paths.get_rewards_path(data_dir))
        self._files["values"] = h5py.File(paths.get_values_path(data_dir))
        self._files["exp_pol_flags"] = h5py.File(
            paths.get_expert_policy_flags_path(data_dir)
        )

    def _load_data(self) -> None:
        cam_names = config.extero_obs_names_from_sys_config(self.cfg["system"])
        for key, file in self._files.items():
            if key == "images":
                # one vlen-JPEG dataset per camera, in config order
                self._data["images"] = {name: file[name] for name in cam_names}
            else:
                self._data[key] = file[DATA_KEY]

    def _decode_frames(self, indices) -> np.ndarray:
        """Decode global-step ``indices`` for every camera -> ``(len, n_cam, H, W, 3)``."""
        per_cam = [
            jpeg_io.decode_jpeg_frames(self._data["images"][name], indices)
            for name in self.extero_obs_names
        ]
        return np.stack(per_cam, axis=1)

    def get_item_by_t_H(self, t_H: int) -> DatasetItem:
        t = t_H + self.H
        t_T = t + self.T
        metadata = {"t_H": int(t_H), "t": int(t), "t_T": int(t_T)}

        # inputs
        proprio_obs_hist: np.ndarray = self._data["proprio_obs"][t_H : t + 1]
        extero_obs: np.ndarray = self._decode_frames([t])[0]  # (n_cam, H, W, 3)
        acts_hist: np.ndarray = self._data["actions"][t_H:t]
        fut_acts: np.ndarray = self._data["actions"][t:t_T]
        fut_cmds: np.ndarray = self._data["commands"][t : t_T + 1]
        model_inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": proprio_obs_hist,
            "extero_obs": extero_obs,
            "acts_hist": acts_hist,
            "fut_acts": fut_acts,
            "fut_cmds": fut_cmds,
        }

        # labels
        proprio_obs: np.ndarray = self._data["proprio_obs"][t + 1 : t_T + 1]
        fut_extero_obs: np.ndarray = self._decode_frames(
            range(t + 1, t_T + 1)
        )  # (T, n_cam, H, W, 3)
        rewards: np.ndarray = self._data["rewards"][t:t_T]
        values: np.ndarray = self._data["values"][t + 1 : t_T + 1]
        actions: np.ndarray = self._data["actions"][t:t_T]
        exp_pol_flags: np.ndarray = self._data["exp_pol_flags"][t:t_T]
        labels: types.WorldModelSchema.Labels = {
            "proprio_obs": proprio_obs,
            "extero_obs": fut_extero_obs,
            "rewards": rewards,
            "values": values,
            "actions": actions,
            "exp_pol_flags": exp_pol_flags,
        }

        return {"model_in": model_inputs, "labels": labels, "metadata": metadata}

    def data_augmentations(self, item: DatasetItem) -> DatasetItem:
        if self.add_noise and self.proprio_obs_noise_stds is not None:
            item["model_in"]["proprio_obs_hist"] = _apply_noise(
                item["model_in"]["proprio_obs_hist"], self.proprio_obs_noise_stds
            )
        if self.image_aug_cfg:
            imgs = item["model_in"]["extero_obs"]  # (n_cam, H, W, 3) uint8
            out = np.empty_like(imgs)
            for i in range(imgs.shape[0]):
                out[i] = _augment_image(imgs[i], self.image_aug_cfg)
            item["model_in"]["extero_obs"] = out
        return item

    @staticmethod
    def get_dummy_item(cfg: dict) -> DatasetItem:
        sys_cfg = cfg["system"]
        proprio_obs_dim = config.proprio_obs_dim_from_sys_config(sys_cfg)
        act_dim = config.action_dim_from_sys_config(sys_cfg)
        cmd_dim = config.cmd_dim_from_sys_config(sys_cfg)
        image_shapes = config.extero_obs_image_shapes_from_sys_config(sys_cfg)
        n_cam = len(image_shapes)
        h, w, c = image_shapes[0]
        H = cfg["model"]["dataset"]["history_length"]
        T = cfg["model"]["dataset"]["prediction_length"]

        rng = np.random.default_rng()

        inputs: types.WorldModelSchema.Inputs = {
            "proprio_obs_hist": rng.standard_normal((H + 1, proprio_obs_dim)),
            "extero_obs": rng.integers(0, 256, (n_cam, h, w, c), dtype=np.uint8),
            "acts_hist": rng.standard_normal((H, act_dim)),
            "fut_acts": rng.standard_normal((T, act_dim)),
            "fut_cmds": rng.standard_normal((T + 1, cmd_dim)),
        }
        labels: types.WorldModelSchema.Labels = {
            "proprio_obs": rng.standard_normal((T, proprio_obs_dim)),
            "extero_obs": rng.integers(0, 256, (T, n_cam, h, w, c), dtype=np.uint8),
            "rewards": rng.standard_normal((T,)),
            "values": rng.standard_normal((T,)),
            "actions": rng.standard_normal((T, act_dim)),
            "exp_pol_flags": np.ones((T,)),
        }
        return {"model_in": inputs, "labels": labels, "metadata": {"dummy": True}}


def _augment_image(img: np.ndarray, aug_cfg: dict) -> np.ndarray:
    """Apply the configured image augmentations to a single (H, W, 3) uint8 frame."""
    if aug_cfg.get("random_crop"):
        img = _random_resized_crop(img)
    if aug_cfg.get("color_jitter"):
        img = _color_jitter(img)
    if aug_cfg.get("gaussian_blur"):
        img = _gaussian_blur(img)
    return img


def _random_resized_crop(
    img: np.ndarray, scale: tuple[float, float] = (0.85, 1.0)
) -> np.ndarray:
    h, w = img.shape[:2]
    s = np.random.uniform(*scale)
    ch, cw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)
    crop = img[top : top + ch, left : left + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _color_jitter(
    img: np.ndarray, brightness: float = 0.2, contrast: float = 0.2
) -> np.ndarray:
    b = 1.0 + np.random.uniform(-brightness, brightness)
    c = 1.0 + np.random.uniform(-contrast, contrast)
    mean = float(img.mean())
    out = (img.astype(np.float32) * b - mean) * c + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def _gaussian_blur(
    img: np.ndarray, p: float = 0.5, max_sigma: float = 1.5
) -> np.ndarray:
    if np.random.rand() > p:
        return img
    sigma = float(np.random.uniform(0.1, max_sigma))
    return cv2.GaussianBlur(img, (0, 0), sigma)
