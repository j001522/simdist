"""Is the WM value channel laterally blind? (JAX venv, no Isaac)

Observed failure with the gripper frozen closed: MPPI descends confidently but at
the wrong lateral spot ("inserts into the table"). In the data descent and
approach are confounded (the expert only descends over the hole), so the value
head may have learned V ~ f(height) with no lateral term.

Test A (state discrimination, real frames): states in a fixed height band,
  split by lateral offset |xy|. Compare pred V(s_{t+1}) (values[:,0]) against the
  recorded critic label value[t+1] (the ceiling: the critic sees privileged state).
    pred flat across |xy| while label falls  -> value head learned the height
                                                shortcut (data gap / head failure)
    pred falls with |xy| like the label      -> head fine on real frames; blame the
                                                dynamics rollout (planner evaluates
                                                predicted latents, not real frames)

Test B (action counterfactual): expert-flag states at 3-10cm; MPPI return for the
  recorded 5-step actions vs the same actions with lateral dims (0,1 = dx,dy)
  negated. Ground truth: recorded must win (it leads to insertion).

Run (JAX venv python, from anywhere):
  python scripts/probe_wm_lateral.py \
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

# Same stub as scripts/remote/mppi_server.py (torch imported for a type only).
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
parser.add_argument("--per-cell", type=int, default=60)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import h5py
import jax.numpy as jnp
from PIL import Image

from simdist.utils.model import load_model_from_ckpt

CAMS = ("front_rgb", "side_rgb", "wrist_rgb")
H, T = 5, 5
GAMMA = 0.99
LAT = (0, 1)  # dx, dy action dims

ckpt_dir = os.path.join(_SIMDIST, "checkpoints", "models", args.ckpt)
model, model_cfg, step = load_model_from_ckpt(ckpt_dir, args.step)
print(f"[probe] loaded {args.ckpt} step {step}", flush=True)


def decode_frame(jpeg_bytes) -> np.ndarray:
    img = np.asarray(Image.open(_io.BytesIO(np.asarray(jpeg_bytes, dtype=np.uint8).tobytes())))
    return img[..., ::-1]


# cell -> (z_lo, z_hi, xy_lo, xy_hi); z relative to the seated plateau at 0.0148
CELLS = {
    "lowz_aligned": (0.020, 0.060, 0.000, 0.010),
    "lowz_mislat":  (0.020, 0.060, 0.020, 0.050),
    "lowz_farlat":  (0.020, 0.060, 0.050, 9.000),
    "midz_aligned": (0.060, 0.120, 0.000, 0.010),
    "midz_mislat":  (0.060, 0.120, 0.020, 0.050),
    "midz_farlat":  (0.060, 0.120, 0.050, 9.000),
}

path = os.path.join(_SIMDIST, "datasets", "sim", args.dataset, "raw_data.hdf5")
f = h5py.File(path, "r")
data = f["data"]
names = list(data.keys())
rng = np.random.default_rng(args.seed)
rng.shuffle(names)

cell_samples = {c: [] for c in CELLS}
approach_samples = []  # for test B
for name in names:
    if all(len(v) >= args.per_cell for v in cell_samples.values()) and \
            len(approach_samples) >= 4 * args.per_cell:
        break
    g = data[name]
    rel = g["obs/insertive_asset_in_receptive_asset_frame"][:, :3]
    n = len(rel)
    if n < H + T + 2:
        continue
    z = np.abs(rel[:, 2])
    xy = np.linalg.norm(rel[:, :2], axis=1)
    dist = np.linalg.norm(rel, axis=1)
    fl = g["expert_policy_flag"][:]
    for c, (zlo, zhi, xlo, xhi) in CELLS.items():
        if len(cell_samples[c]) >= args.per_cell:
            continue
        ts = [t for t in range(H, n - T - 1)
              if zlo <= z[t] < zhi and xlo <= xy[t] < xhi]
        if ts:
            t = int(rng.choice(ts))
            cell_samples[c].append((name, t, float(xy[t]), float(z[t])))
    if len(approach_samples) < 4 * args.per_cell:
        ts = [t for t in range(H, n - T - 1)
              if 0.03 <= dist[t] < 0.10 and fl[t]]
        if ts:
            t = int(rng.choice(ts))
            approach_samples.append((name, t, float(dist[t]), 0.0))

for c, v in cell_samples.items():
    print(f"[probe] cell {c}: {len(v)} samples", flush=True)
print(f"[probe] approach (test B): {len(approach_samples)} samples", flush=True)


def build_inputs(batch):
    xs = {"proprio_obs_hist": [], "extero_obs": [], "acts_hist": [], "fut_acts": [],
          "fut_cmds": []}
    labels = []
    for name, t, _, _ in batch:
        g = data[name]
        xs["proprio_obs_hist"].append(g["obs/arm_joint_pos"][t - H : t + 1])
        xs["acts_hist"].append(g["actions"][t - H : t])
        xs["fut_acts"].append(g["actions"][t : t + T])
        xs["fut_cmds"].append(np.zeros((T + 1, 0), dtype=np.float32))
        xs["extero_obs"].append(np.stack([decode_frame(g[f"obs/{c}"][t]) for c in CAMS]))
        labels.append(g["value"][t + 1])
    return {k: jnp.asarray(np.stack(v)) for k, v in xs.items()}, np.asarray(labels)


def run_batches(samples, alt_fut=None):
    """Returns v_pred (=values[:,0]), v_label, and MPPI returns for recorded and
    (optionally) altered fut_acts."""
    vps, vls, rets, rets_alt = [], [], [], []
    disc = GAMMA ** np.arange(T)
    fdisc = GAMMA ** (T + 1)
    for i in range(0, len(samples), args.batch):
        chunk = samples[i : i + args.batch]
        xs, vl = build_inputs(chunk)
        y = model.inference(xs)
        v = np.asarray(y["values"]).reshape(len(chunk), T)
        r = np.asarray(y["rewards"]).reshape(len(chunk), T)
        vps.append(v[:, 0]); vls.append(vl)
        rets.append((r * disc).sum(1) + fdisc * v[:, -1])
        if alt_fut is not None:
            fut = np.asarray(xs["fut_acts"]).copy()
            fut = alt_fut(fut)
            y2 = model.inference({**xs, "fut_acts": jnp.asarray(fut)})
            v2 = np.asarray(y2["values"]).reshape(len(chunk), T)
            r2 = np.asarray(y2["rewards"]).reshape(len(chunk), T)
            rets_alt.append((r2 * disc).sum(1) + fdisc * v2[:, -1])
    out = [np.concatenate(vps), np.concatenate(vls), np.concatenate(rets)]
    out.append(np.concatenate(rets_alt) if rets_alt else None)
    return out


print("\n=== A: value vs lateral offset at fixed height (real frames) ===")
print(f"{'cell':<15}{'n':>4}{'|xy| mean':>10}{'v_pred':>9}{'v_label':>9}")
for c in CELLS:
    s = cell_samples[c]
    if not s:
        continue
    vp, vl, _, _ = run_batches(s)
    xy = np.mean([x[2] for x in s])
    print(f"{c:<15}{len(s):>4}{xy:>10.3f}{vp.mean():>9.2f}{vl.mean():>9.2f}", flush=True)

print("\n=== B: recorded approach actions vs counterfactuals (expert, 3-10cm) ===")

VARIANTS = {
    "neg_lateral (dx,dy)": lambda fut: _set(fut, LAT, -1),
    "neg_all_arm (0-5)": lambda fut: _set(fut, tuple(range(6)), -1),
    "zero_arm (0-5)": lambda fut: _set(fut, tuple(range(6)), 0),
    "random_arm N(0,3)": lambda fut: _rand(fut),
    # constant pushes (~1 sigma of the raw action envelope): what an MPPI drift
    # candidate looks like; a healthy model must price lateral pushes like
    # vertical ones (test A says ~0.6 value/cm of lateral offset)
    "push dx +2.5": lambda fut: _push(fut, 0, +2.5),
    "push dx -2.5": lambda fut: _push(fut, 0, -2.5),
    "push dy +2.5": lambda fut: _push(fut, 1, +2.5),
    "push dz +2.5 (up)": lambda fut: _push(fut, 2, +2.5),
    "push dz -2.5 (down)": lambda fut: _push(fut, 2, -2.5),
}


def _set(fut, dims, mult):
    fut[:, :, dims] = mult * fut[:, :, dims]
    return fut


def _rand(fut):
    r = np.random.default_rng(1)
    fut[:, :, :6] = r.normal(0, 3.0, fut[:, :, :6].shape)
    return fut


def _push(fut, dim, val):
    fut[:, :, dim] = fut[:, :, dim] + val
    return fut


for label, fn in VARIANTS.items():
    vp, vl, ret_rec, ret_alt = run_batches(approach_samples, alt_fut=fn)
    d = ret_rec - ret_alt
    print(f"{label:<22} ret_recorded {ret_rec.mean():.2f}  ret_alt {ret_alt.mean():.2f}  "
          f"delta {d.mean():+.3f}  recorded wins {np.mean(d > 0):.2f}", flush=True)
