"""Tiny length-prefixed pickle IPC over a localhost TCP socket.

Bridges the two processes of the DGX-Spark inference loop, which cannot share
one Python because of the numpy ABI wall (Isaac/IsaacLab need numpy<2, the GPU
JAX stack needs numpy>=2). Process A (Isaac, numpy 1.26) sends per-step
observations; Process B (JAX/MPPI, numpy 2.x, GPU) returns actions. Payloads are
small numpy arrays (state vectors / height scan / action sequence), so plain
pickle over a stream socket is more than fast enough at the ~5 Hz control rate.
"""
import pickle
import socket
import struct


def _recvn(conn: socket.socket, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def send_msg(conn: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!Q", len(data)) + data)


def recv_msg(conn: socket.socket):
    header = _recvn(conn, 8)
    if header is None:
        return None
    (n,) = struct.unpack("!Q", header)
    payload = _recvn(conn, n)
    if payload is None:
        return None
    return pickle.loads(payload)
