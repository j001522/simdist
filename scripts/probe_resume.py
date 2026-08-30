"""Verify that a checkpoint restores COMPLETELY, i.e. that resuming from it would
continue the run rather than silently start from a partly-random model.

Run 24591803-era `load_model_from_ckpt` had a fallback path that copied only the
leaves whose path matched, leaving the rest at their `nnx.Rngs(0)` init. The loader
now raises on a skipped leaf, but that only covers leaves the fallback *knows* it
dropped. This probe rebuilds the model through the real loader and then checks that
what came back is actually the trained model: no leaf still sitting at its random
init, and real movement between consecutive checkpoints.

    python scripts/probe_resume.py --run wm_dinoft_cls_l64_26096400
    python scripts/probe_resume.py --run <name> --step 72000

Exit code 0 = restore is complete (loader took the strict path, nothing left at init).
CPU is enough (no training happens here): JAX_PLATFORMS=cpu.
"""

import argparse
import os
import sys

import numpy as np
import jax
import orbax.checkpoint as ocp
import flax.nnx as nnx

from simdist.modeling import models
from simdist.utils import model as model_utils, paths


def flatten(tree, prefix=()):
    """{'a': {'b': arr}} -> {('a','b'): arr}."""
    out = {}
    for k, v in tree.items():
        path = prefix + (str(k),)
        if isinstance(v, dict):
            out.update(flatten(v, path))
        else:
            out[path] = v
    return out


def as_float_array(v):
    """np view of a leaf, or None for the leaves that carry no learned value
    (rng keys — which refuse conversion outright — counters, integer state)."""
    if jax.dtypes.issubdtype(getattr(v, "dtype", None), jax.dtypes.prng_key):
        return None
    a = np.asarray(v)
    return a if a.dtype.kind in "fc" and a.size else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint dir name under checkpoints/models")
    ap.add_argument("--step", type=int, default=None, help="step to load (default: latest)")
    args = ap.parse_args()

    ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), args.run)
    print(f"ckpt_dir = {ckpt_dir}")

    # 1. the real resume path
    model, model_cfg, step = model_utils.load_model_from_ckpt(ckpt_dir, args.step)
    print(f"loaded step = {step}")
    m = model_cfg["model"]
    print(f"  model={m.get('name', m.get('_target_', '?'))} latent_dim={m.get('latent_dim')}")
    print(f"  dataset={model_cfg.get('data', {}).get('dataset_name')}")
    print(f"  trainable params = {model_utils.count_params(model, nnx.Param)}")

    _, model_state, _ = nnx.split(model, nnx.Not(nnx.Intermediate), ...)
    restored = flatten(model_state.to_pure_dict())

    # A strict `StandardRestore(item=tree)` (the path the loader took, since it printed
    # no fallback warning) already refuses to return unless the on-disk tree and the
    # freshly-built model tree have exactly the same structure -- no disk-only leaf is
    # dropped and no model leaf is left un-overwritten. A raw target-less restore cannot
    # be used as an independent cross-check here: without a target tree orbax reads the
    # saved sharding, which names cuda:0 and fails anywhere but on a GPU.
    #
    # What is still worth proving on CPU is that the restored values are TRAINED values:
    # (a) they differ from the `nnx.Rngs(0)` init the model was built with, and
    # (b) they differ from -- but stay close to -- the previous checkpoint.
    #
    # "Equal to the init" only proves a leaf was never restored if the init is RANDOM.
    # A DINOv2 run's frozen backbone is loaded from assets/*.npz inside `get_model`, and
    # zero/one-initialised leaves are deterministic too -- all of those legitimately come
    # back equal. Build the init twice under different seeds: a leaf that differs between
    # the two is random-initialised, and matching it exactly is the failure we are after.
    dummy = model_utils.make_dummy_scaler_params(model_cfg)
    init, init_alt = [
        flatten(nnx.split(
            models.get_model(model_cfg, dummy, nnx.Rngs(seed)),
            nnx.Not(nnx.Intermediate), ...,
        )[1].to_pure_dict())
        for seed in (0, 1234)
    ]

    unrestored, deterministic, n_leaves = [], 0, 0
    for path, v in restored.items():
        a = as_float_array(v)
        b, c = as_float_array(init.get(path)), as_float_array(init_alt.get(path))
        if a is None or b is None or c is None or a.shape != b.shape:
            continue
        n_leaves += 1
        if np.array_equal(b, c):
            deterministic += 1  # pretrained npz / zeros / ones: uninformative here
        elif np.array_equal(a, b):
            unrestored.append(path)

    print(f"\nfloat leaves: {n_leaves} total, {deterministic} deterministic-init "
          f"(pretrained/zeros — cannot be told apart from a restore)")
    print(f"  random-init leaves still bit-identical to their init: {len(unrestored)}")
    for path in unrestored[:12]:
        print(f"    ! UNRESTORED (random init): {'/'.join(path)}")

    ok = not unrestored

    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as mngr:
        steps = sorted(mngr.all_steps())
    prev = [s for s in steps if s < step]
    if prev:
        prev_model, _, prev_step = model_utils.load_model_from_ckpt(ckpt_dir, prev[-1])
        _, prev_state, _ = nnx.split(prev_model, nnx.Not(nnx.Intermediate), ...)
        prev_flat = flatten(prev_state.to_pure_dict())
        deltas, groups = [], {}
        for path, v in restored.items():
            a = as_float_array(v)
            b = as_float_array(prev_flat.get(path))
            if a is None or b is None or a.shape != b.shape:
                continue
            d = float(np.max(np.abs(a - b)))
            deltas.append(d)
            # group by module (dinov2 blocks by index, everything else by its head node)
            key = "/".join(path[:6]) if path[:2] == ("encoder", "dinov2") else path[0]
            n_moved, n_tot = groups.get(key, (0, 0))
            groups[key] = (n_moved + (d > 0), n_tot + 1)
        moved = sum(d > 0 for d in deltas)
        print(f"\nvs previous checkpoint (step {prev_step}): {moved}/{len(deltas)} leaves moved, "
              f"max|delta|={max(deltas):.3e}")
        frozen = [k for k, (m, _) in sorted(groups.items()) if m == 0]
        print(f"  modules that did not move at all ({len(frozen)}): "
              + (", ".join(frozen[:8]) + (" ..." if len(frozen) > 8 else "") if frozen else "none"))
        if moved == 0:
            print("    ! two consecutive checkpoints are identical -- training was not progressing")
            ok = False

    print("\nRESULT:", "COMPLETE RESTORE" if ok else "INCOMPLETE / MISMATCHED RESTORE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
