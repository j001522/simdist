"""Probe the trained WM's heads against real dataset states (JAX venv, no Isaac).

Answers three questions about a manipulation WM checkpoint, using raw demos from the
fixed dataset as ground truth:
  1. VALUE discrimination: does pred V rank seated > near > mid > far like the
     recorded critic labels do? (the agreed pass/fail before trusting MPPI evals)
  2. REWARD head sanity: pred 5-step reward sum vs recorded.
  3. GRIPPER counterfactual: the MPPI return (sum gamma^k r_k + gamma^(T+1) V_T)
     for the recorded arm actions with gripper pinned CLOSED (-2.5) vs OPEN (+2.5).
     If open >= closed away from the hole, MPPI will open the gripper and drop the
     peg -- exactly the observed failure.
  Also reports the base-policy head's gripper output (what frozen_action_dims pins to).

Run (JAX venv python, from anywhere):
  python scripts/probe_wm_heads.py \
        --ckpt wm_manip_24708772 --step 66000
"""

import argparse
import io as _io
import os
import sys
import types as _t

import numpy as np

_SIMDIST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SIMDIST)

# Same stub as scripts/remote/mppi_server.py: simdist.data.dataset pulls torch for a
# type only (DatasetBatch), never used on the inference path.
for _n in ("torch", "torch.utils", "torch.utils.data"):
    sys.modules.setdefault(_n, _t.ModuleType(_n))
_d = sys.modules["torch.utils.data"]
_d.Dataset = type("Dataset", (), {})
_d.DataLoader = object
_d.random_split = None
_d.get_worker_info = None
sys.modules["torch"].utils = sys.modules["torch.utils"]
sys.modules["torch.utils"].data = _d
_ds = _t.ModuleType("simdist.data.dataset")
_ds.DatasetBatch = type("DatasetBatch", (dict,), {})
sys.modules["simdist.data.dataset"] = _ds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", default="wm_manip_24708772")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--dataset", default="2026-07-16_14-49-21")
parser.add_argument("--demos", type=int, default=200)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import h5py
import jax.numpy as jnp
from PIL import Image

from simdist.utils.model import load_model_from_ckpt

CAMS = ("front_rgb", "side_rgb", "wrist_rgb")  # config order (system/ur5e.yaml)
H, T = 5, 5
GAMMA = 0.99  # control/mppi_manip.yaml discount
GRIP = 6  # gripper action dim

ckpt_dir = os.path.join(_SIMDIST, "checkpoints", "models", args.ckpt)
model, model_cfg, step = load_model_from_ckpt(ckpt_dir, args.step)
print(f"[probe] loaded {args.ckpt} step {step}", flush=True)
assert model_cfg["model"]["dataset"]["history_length"] == H
assert model_cfg["model"]["dataset"]["prediction_length"] == T


def decode_frame(jpeg_bytes) -> np.ndarray:
    """Match the training path: raw JPEGs were written by cv2.imencode from RGB
    arrays (channels stored swapped), and read back with cv2.imdecode. PIL returns
    the semantic (swapped) channels, so reverse to recover the recorded array."""
    img = np.asarray(Image.open(_io.BytesIO(np.asarray(jpeg_bytes, dtype=np.uint8).tobytes())))
    return img[..., ::-1]


path = os.path.join(_SIMDIST, "datasets", "sim", args.dataset, "raw_data.hdf5")
f = h5py.File(path, "r")
data = f["data"]
names = list(data.keys())
rng = np.random.default_rng(args.seed)
sel = rng.choice(names, size=min(args.demos, len(names)), replace=False)

BINS = [
    ("seated <2cm", 0.0, 0.02),
    ("near 2-5cm", 0.02, 0.05),
    ("mid 5-10cm", 0.05, 0.10),
    ("far >10cm", 0.10, 9.0),
]

samples = []  # (demo, t, dist, expert_flag)
for name in sel:
    g = data[name]
    rel = g["obs/insertive_asset_in_receptive_asset_frame"][:]
    dist = np.linalg.norm(rel[:, :3], axis=1)
    n = len(dist)
    if n < H + T + 2:
        continue
    fl = g["expert_policy_flag"][:]
    # up to one sample per bin per demo, to balance bins
    for _, lo, hi in BINS:
        ts = [t for t in range(H, n - T - 1) if lo <= dist[t] < hi]
        if ts:
            t = int(rng.choice(ts))
            samples.append((name, t, float(dist[t]), bool(fl[t])))

print(f"[probe] {len(samples)} samples from {len(sel)} demos", flush=True)


def build_inputs(batch):
    xs = {"proprio_obs_hist": [], "extero_obs": [], "acts_hist": [], "fut_acts": [],
          "fut_cmds": []}
    labels = {"value_T": [], "rew_sum": [], "act0": []}
    for name, t, _, _ in batch:
        g = data[name]
        xs["proprio_obs_hist"].append(g["obs/arm_joint_pos"][t - H : t + 1])
        xs["acts_hist"].append(g["actions"][t - H : t])
        xs["fut_acts"].append(g["actions"][t : t + T])
        xs["fut_cmds"].append(np.zeros((T + 1, 0), dtype=np.float32))
        frames = np.stack([decode_frame(g[f"obs/{c}"][t]) for c in CAMS])
        xs["extero_obs"].append(frames)
        labels["value_T"].append(g["value"][t + T])
        labels["rew_sum"].append(g["reward"][t : t + T].sum())
        labels["act0"].append(g["actions"][t])
    xs = {k: jnp.asarray(np.stack(v)) for k, v in xs.items()}
    labels = {k: np.asarray(v) for k, v in labels.items()}
    return xs, labels


rows = []
for i in range(0, len(samples), args.batch):
    chunk = samples[i : i + args.batch]
    xs, labels = build_inputs(chunk)
    y = model.inference(xs)
    fut = np.asarray(xs["fut_acts"])

    closed = fut.copy()
    closed[:, :, GRIP] = -2.5
    y_closed = model.inference({**xs, "fut_acts": jnp.asarray(closed)})
    opened = fut.copy()
    opened[:, :, GRIP] = +2.5
    y_open = model.inference({**xs, "fut_acts": jnp.asarray(opened)})

    disc = GAMMA ** np.arange(T)
    fdisc = GAMMA ** (T + 1)

    def ret(yy):
        r = np.asarray(yy["rewards"]).reshape(len(chunk), T)
        v = np.asarray(yy["values"]).reshape(len(chunk), T)[:, -1]
        return (r * disc).sum(1) + fdisc * v

    pv = np.asarray(y["values"]).reshape(len(chunk), T)[:, -1]
    pr = np.asarray(y["rewards"]).reshape(len(chunk), T).sum(1)
    pa = np.asarray(y["actions"]).reshape(len(chunk), T, -1)[:, 0]
    r_rec, r_cl, r_op = ret(y), ret(y_closed), ret(y_open)

    for j, (name, t, d, fl) in enumerate(chunk):
        rows.append(dict(dist=d, expert=fl,
                         v_pred=float(pv[j]), v_label=float(labels["value_T"][j]),
                         r_pred=float(pr[j]), r_label=float(labels["rew_sum"][j]),
                         grip_pred=float(pa[j, GRIP]), grip_rec=float(labels["act0"][j, GRIP]),
                         ret_rec=float(r_rec[j]), ret_closed=float(r_cl[j]),
                         ret_open=float(r_op[j])))
    if (i // args.batch) % 10 == 0:
        print(f"[probe] {i + len(chunk)}/{len(samples)}", flush=True)

dist = np.array([r["dist"] for r in rows])
vp = np.array([r["v_pred"] for r in rows]); vl = np.array([r["v_label"] for r in rows])
rp = np.array([r["r_pred"] for r in rows]); rl = np.array([r["r_label"] for r in rows])
gp = np.array([r["grip_pred"] for r in rows])
ex = np.array([r["expert"] for r in rows])
rc = np.array([r["ret_closed"] for r in rows]); ro = np.array([r["ret_open"] for r in rows])

print("\n=== value head ===")
print(f"corr(v_pred, v_label) = {np.corrcoef(vp, vl)[0,1]:+.3f}   "
      f"corr(v_pred, dist) = {np.corrcoef(vp, dist)[0,1]:+.3f}   "
      f"(labels: corr(v_label, dist) = {np.corrcoef(vl, dist)[0,1]:+.3f})")
print(f"{'bin':<14}{'n':>5}{'v_pred':>10}{'v_label':>10}{'r_pred':>10}{'r_label':>10}")
for lab, lo, hi in BINS:
    m = (dist >= lo) & (dist < hi)
    if m.sum() == 0:
        continue
    print(f"{lab:<14}{m.sum():>5}{vp[m].mean():>10.2f}{vl[m].mean():>10.2f}"
          f"{rp[m].mean():>10.2f}{rl[m].mean():>10.2f}")

print("\n=== gripper: MPPI return closed(-2.5) vs open(+2.5) ===")
print(f"{'bin':<14}{'n':>5}{'ret_closed':>12}{'ret_open':>12}{'delta':>9}{'open wins':>10}")
for lab, lo, hi in BINS:
    m = (dist >= lo) & (dist < hi)
    if m.sum() == 0:
        continue
    dlt = rc[m] - ro[m]
    print(f"{lab:<14}{m.sum():>5}{rc[m].mean():>12.2f}{ro[m].mean():>12.2f}"
          f"{dlt.mean():>9.2f}{(dlt < 0).mean():>10.2f}")

print("\n=== base-policy head, gripper dim (first predicted action) ===")
print(f"{'bin':<14}{'n':>5}{'mean pred':>11}{'frac<0':>8}   (expert emits ~-2.28, "
      f"93% of expert-flag steps <0)")
for lab, lo, hi in BINS:
    m = (dist >= lo) & (dist < hi) & ex
    if m.sum() == 0:
        continue
    print(f"{lab:<14}{m.sum():>5}{gp[m].mean():>11.2f}{(gp[m] < 0).mean():>8.2f}")
print(f"\n[probe] reward corr(pred, label) = {np.corrcoef(rp, rl)[0,1]:+.3f}")
