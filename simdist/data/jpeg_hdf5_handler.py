"""HDF5 dataset handler that JPEG-encodes image observations on write.

Camera frames dominate manipulation dataset size (3x 224x224x3 uint8 = 441 KB/step
raw, ~165 KB gzip). Storing them as JPEG (~10-15 KB/frame at Q92) shrinks the dataset
~10x with negligible quality loss -- the encoder is an ImageNet ResNet (ImageNet is
JPEG) and training applies heavy visual augmentation, and real-robot data is JPEG too.

Image fields (4D ``(T, H, W, C)`` uint8) are stored as a variable-length array of
JPEG byte strings, tagged with ``attrs["jpeg"]`` and the original frame shape. Every
other field is written exactly as the base handler does. Use ``decode_jpeg_dataset``
(in ``jpeg_io``) on the read side (process_data / dataloader) to recover the frames.

Only this *write* handler needs Isaac (base class). The Isaac-free read/encode
helpers live in ``jpeg_io`` so the pure-JAX training env can import them.
"""

from isaaclab.utils.datasets import EpisodeData
from isaaclab.utils.datasets.hdf5_dataset_file_handler import HDF5DatasetFileHandler

from simdist.data.jpeg_io import (  # noqa: F401  (re-exported for back-compat)
    JPEG_QUALITY,
    _is_image,
    _write_jpeg_field,
    decode_jpeg_dataset,
    is_jpeg_dataset,
)


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
