"""MPPI / world-model inference server — Process B of the 2-process Spark loop.

Runs in the JAX venv (jax 0.10.1 / flax 0.12.7 / numpy 2.x, GPU). Holds the real
`MppiController`; receives ControllerInput + command dicts from the Isaac process
over a localhost socket and returns action sequences. See `simulate_go2_remote.py`
for Process A and `ipc.py` for the wire protocol.

Weights are NOT restored from the checkpoint (fresh init): this is a
plumbing / visualisation harness, so only the architecture and observation/action
dims (read from `model_config.yaml`) matter. Swap in a real load for evaluation.

Run:
    /shared/giacomo/jax_spark_test/bin/python mppi_server.py --port 5599 \
        --simdist-dir /shared/giacomo/simdist
"""
import argparse
import os
import socket
import sys
import traceback
import types as _t
import importlib.machinery as _machinery

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ipc  # noqa: E402


def _install_stubs() -> None:
    """`simdist.data.dataset` pulls torch + h5py for a type only (DatasetBatch),
    never used on the inference path. Stub them so the JAX side stays lean."""
    for n in ("torch", "torch.utils", "torch.utils.data"):
        mod = sys.modules.setdefault(n, _t.ModuleType(n))
        # transformers (needed by the DINOv2 backbone) probes for torch with
        # importlib.util.find_spec, which raises ValueError on a module whose
        # __spec__ is None. Give the stub a spec so the probe returns cleanly and
        # transformers concludes torch is simply unavailable.
        mod.__spec__ = _machinery.ModuleSpec(n, None)
    d = sys.modules["torch.utils.data"]
    d.Dataset = type("Dataset", (), {})
    d.DataLoader = object
    d.random_split = None
    d.get_worker_info = None
    sys.modules["torch"].utils = sys.modules["torch.utils"]
    sys.modules["torch.utils"].data = d
    ds = _t.ModuleType("simdist.data.dataset")
    ds.DatasetBatch = type("DatasetBatch", (dict,), {})
    sys.modules["simdist.data.dataset"] = ds


def build_controller(simdist_dir: str, ckpt_dir: str, ctrl_cfg: dict,
                     restore: bool = True, ckpt_step: int | None = None):
    sys.path.insert(0, simdist_dir)
    import flax.nnx as nnx
    from omegaconf import OmegaConf

    from simdist.modeling import models
    from simdist.utils import model as mu, paths
    from simdist.control.mppi import MppiController

    if restore:
        # tolerant restore (handles old-flax rng-state nesting); loads the
        # trained weights + scaler params so the policy actually controls.
        # ckpt_step=None loads the latest step; pass a step to pin a specific
        # checkpoint (the run is not monotonic across steps -- pick by the
        # planning probe, not by latest).
        model, model_cfg, step = mu.load_model_from_ckpt(ckpt_dir, ckpt_step)
        print(f"[mppi_server] restored trained weights (step {step})", flush=True)
    else:
        model_cfg = OmegaConf.to_container(
            OmegaConf.load(os.path.join(ckpt_dir,
                                        paths.get_model_config_filename())),
            resolve=True,
        )
        model = models.get_model(model_cfg, mu.make_dummy_scaler_params(model_cfg),
                                 nnx.Rngs(0))
        print("[mppi_server] fresh-init weights (no restore)", flush=True)
    # The debug Intermediates are written under trainer.py's nnx.jit, where the module
    # is an nnx argument. MPPI's jit closes over the model instead, so the same writes
    # raise TraceContextError. Nothing reads them at inference.
    model.collect_debug_stats = False
    if getattr(model, "encoder", None) is not None:
        model.encoder.collect_debug_stats = False

    controller = MppiController(model, model_cfg, ctrl_cfg)
    return controller


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--simdist-dir", default="/shared/giacomo/simdist")
    ap.add_argument("--fresh", action="store_true",
                    help="Skip checkpoint restore (random-init weights).")
    ap.add_argument("--ckpt-step", type=int, default=None,
                    help="Pin a specific checkpoint step (default: latest).")
    args = ap.parse_args()

    # Bind the port BEFORE the slow jax import so readiness probes / the client
    # can connect immediately (the connection just queues until we accept).
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[mppi_server] listening on {args.host}:{args.port}", flush=True)

    _install_stubs()
    import jax
    print(f"[mppi_server] jax {jax.__version__} devices={jax.devices()}", flush=True)

    state = {"controller": None}

    def handle(msg: dict) -> dict:
        op = msg["op"]
        if op == "build":
            state["controller"] = build_controller(
                args.simdist_dir, msg["ckpt_dir"], msg["ctrl_cfg"],
                restore=not args.fresh, ckpt_step=args.ckpt_step)
            print("[mppi_server] controller built", flush=True)
            return {"ok": True}
        if op == "echo":
            # IPC-plumbing test: round-trip an externally-computed action (e.g. the
            # expert) through the same serialize/send/recv path MPPI actions take, with
            # no model involved. Confirms the split process doesn't corrupt actions.
            return {"ok": True, "out": {"actions": msg["actions"]}}
        c = state["controller"]
        if c is None:
            return {"ok": False, "err": "controller not built"}
        if op == "initialize":
            c.initialize(msg["x"], msg["cmd"])
            return {"ok": True}
        if op == "reset":
            c.reset(msg["x"], msg["cmd"])
            return {"ok": True}
        if op == "set_fut_cmd":
            c.set_fut_cmd(msg["fut_cmd"])
            return {"ok": True}
        if op == "update":
            c.update(msg["x"])
            return {"ok": True}
        if op == "run_control":
            return {"ok": True, "out": c.run_control()}
        return {"ok": False, "err": f"unknown op {op}"}

    # Serve sequential clients (the controller state persists across
    # connections), so a readiness probe that connects and immediately
    # disconnects does not kill the server.
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[mppi_server] client connected: {addr}", flush=True)
            try:
                while True:
                    msg = ipc.recv_msg(conn)
                    if msg is None:
                        break
                    try:
                        rep = handle(msg)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        rep = {"ok": False, "err": repr(e)}
                    ipc.send_msg(conn, rep)
            finally:
                conn.close()
            print("[mppi_server] client disconnected; awaiting next", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


if __name__ == "__main__":
    main()
