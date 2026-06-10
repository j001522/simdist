"""Tiny length-prefixed IPC over a localhost TCP socket.

Bridges the two processes of the DGX-Spark inference loop, which cannot share
one Python because of the numpy ABI wall (Isaac/IsaacLab need numpy<2, the GPU
JAX stack needs numpy>=2). Process A (Isaac, numpy 1.26) sends per-step
observations; Process B (JAX/MPPI, numpy 2.x, GPU) returns actions.

The two processes run DIFFERENT numpy versions, so numpy arrays must NOT be
pickled directly: numpy 2.x pickles reference ``numpy._core`` (renamed from
``numpy.core`` in 1.x) and fail to unpickle under 1.26. Instead, arrays are
encoded as (raw C bytes, dtype string, shape) — a version-agnostic payload —
and reconstructed with the *local* numpy on each side. jax arrays and numpy
scalars are coerced to numpy first. Everything else is plain pickle.
"""
import pickle
import socket
import struct

import numpy as np

_ND = "__nd__"


def _encode(obj):
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_encode(v) for v in obj)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {_ND: (arr.tobytes(), str(arr.dtype), tuple(arr.shape))}
    # jax arrays / numpy scalars / other array-likes -> portable numpy bytes
    if hasattr(obj, "__array__") and not isinstance(obj, (str, bytes, bytearray)):
        arr = np.ascontiguousarray(np.asarray(obj))
        return {_ND: (arr.tobytes(), str(arr.dtype), tuple(arr.shape))}
    return obj


def _decode(obj):
    if isinstance(obj, dict):
        if _ND in obj and len(obj) == 1:
            data, dtype, shape = obj[_ND]
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_decode(v) for v in obj)
    return obj


def _recvn(conn: socket.socket, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def send_msg(conn: socket.socket, obj) -> None:
    data = pickle.dumps(_encode(obj), protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!Q", len(data)) + data)


def recv_msg(conn: socket.socket):
    header = _recvn(conn, 8)
    if header is None:
        return None
    (n,) = struct.unpack("!Q", header)
    payload = _recvn(conn, n)
    if payload is None:
        return None
    return _decode(pickle.loads(payload))
