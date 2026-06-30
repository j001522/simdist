"""HDF5 dataset handler that JPEG-encodes image observations on write.

Camera frames dominate manipulation dataset size (3x 224x224x3 uint8 = 441 KB/step
raw, ~165 KB gzip). Storing them as JPEG (~10-15 KB/frame at Q92) shrinks the dataset
~10x with negligible quality loss -- the encoder is an ImageNet ResNet (ImageNet is
JPEG) and training applies heavy visual augmentation, and real-robot data is JPEG too.

Image fields (4D ``(T, H, W, C)`` uint8) are stored as a variable-length array of
JPEG byte strings, tagged with ``attrs["jpeg"]`` and the original frame shape. Every
other field is written exactly as the base handler does. Use ``decode_jpeg_dataset``
on the read side (process_data) to recover ``(T, H, W, C)`` uint8.
"""

import cv2
import h5py
import numpy as np

from isaaclab.utils.datasets import EpisodeData
from isaaclab.utils.datasets.hdf5_dataset_file_handler import HDF5DatasetFileHandler

JPEG_QUALITY = 92


def _is_image(arr: np.ndarray) -> bool:
    return arr.ndim == 4 and arr.shape[-1] in (1, 3) and arr.dtype == np.uint8


def _write_jpeg_field(group: h5py.Group, key: str, value: np.ndarray) -> None:
    """Store (T, H, W, C) uint8 as a length-T vlen array of JPEG byte strings."""
    num_frames = value.shape[0]
    dset = group.create_dataset(key, shape=(num_frames,), dtype=h5py.vlen_dtype(np.uint8))
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    for t in range(num_frames):
        ok, buf = cv2.imencode(".jpg", value[t], params)
        if not ok:
            raise RuntimeError(f"JPEG encode failed for '{key}' frame {t}")
        dset[t] = buf.reshape(-1)
    dset.attrs["jpeg"] = True
    dset.attrs["image_shape"] = np.asarray(value.shape[1:], dtype=np.int64)


def decode_jpeg_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Recover a JPEG-encoded vlen dataset to a ``(T, H, W, C)`` uint8 array."""
    num_frames = dset.shape[0]
    h, w, c = (int(x) for x in dset.attrs["image_shape"])
    out = np.empty((num_frames, h, w, c), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if c == 3 else cv2.IMREAD_GRAYSCALE
    for t in range(num_frames):
        img = cv2.imdecode(np.asarray(dset[t], dtype=np.uint8), flag)
        out[t] = img.reshape(h, w, c)
    return out


def is_jpeg_dataset(dset: h5py.Dataset) -> bool:
    return bool(dset.attrs.get("jpeg", False))


class JpegHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """HDF5 handler that JPEG-encodes (T, H, W, C) uint8 image fields on write."""

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        episode_group_name = (
            f"demo_{demo_id}" if demo_id is not None else f"demo_{self._demo_count}"
        )
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the dataset")
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0
        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
                return
            arr = value.cpu().numpy()
            if _is_image(arr):
                _write_jpeg_field(group, key, arr)
            else:
                group.create_dataset(key, data=arr, compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]
        if demo_id is None:
            self._demo_count += 1
