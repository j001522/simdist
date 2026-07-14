"""Isaac-free JPEG read/encode helpers for image observations in HDF5.

Split out of ``jpeg_hdf5_handler.py`` so the read side (used by ``process_data``
and the world-model dataloader) does not import ``isaaclab``. Only the *write*
handler class in ``jpeg_hdf5_handler.py`` needs Isaac (its base class); everything
here depends on ``cv2``/``h5py``/``numpy`` only and is safe to import from the
pure-JAX training env.

Image fields (4D ``(T, H, W, C)`` uint8) are stored as a variable-length array of
JPEG byte strings, tagged with ``attrs["jpeg"]`` and the original frame shape. Use
``decode_jpeg_dataset`` on the read side to recover ``(T, H, W, C)`` uint8.
"""

import cv2
import h5py
import numpy as np

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


def decode_jpeg_frames(dset: h5py.Dataset, indices) -> np.ndarray:
    """Decode a subset of frames from a JPEG vlen dataset -> ``(len(indices), H, W, C)``
    uint8.

    Like ``decode_jpeg_dataset`` but for an arbitrary index subset: the world-model
    dataloader only needs the latest frame plus a short future window, not the whole
    global-concat store.
    """
    h, w, c = (int(x) for x in dset.attrs["image_shape"])
    flag = cv2.IMREAD_COLOR if c == 3 else cv2.IMREAD_GRAYSCALE
    idxs = list(indices)
    out = np.empty((len(idxs), h, w, c), dtype=np.uint8)
    for i, t in enumerate(idxs):
        img = cv2.imdecode(np.asarray(dset[int(t)], dtype=np.uint8), flag)
        out[i] = img.reshape(h, w, c)
    return out


def is_jpeg_dataset(dset: h5py.Dataset) -> bool:
    return bool(dset.attrs.get("jpeg", False))
