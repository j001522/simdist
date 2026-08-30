"""Merge segmented generate_data HDF5s into a single raw_data.hdf5.

Companion to scripts/collect_chunked.sh. Collection is split into several short
segments so one OOM cannot destroy a 15h run (2026-07-25: a 256-env
stop_on_success run was OOM-killed at 25%, taking the tmux server with it).
This script glues the segments back into ONE file so nothing downstream --
process_data.py, dataset_report.py, the scaler, training -- has to learn about
chunking.

The merge is faithful and cheap: `data/` carries only `env_args` (str) and
`total` (int64); each demo carries `num_samples` (int64) and `success` (bool).
Demos are copied with h5py Group.copy, i.e. at the HDF5 object level, so filters
(gzip on the numeric datasets) and the variable-length image encoding are
preserved and nothing is deserialized into RAM.

Usage:
    isaac-python scripts/merge_datasets.py --dry-run \
        --out datasets/sim/<name>/raw_data.hdf5 --inputs seg0.hdf5 seg1.hdf5 ...

`--drop-short 20` removes the per-segment desync artifacts: the recorder seeds
episode_length_buf randomly, so each env's FIRST episode starts mid-episode and
can be shorter than min_episode_length. That costs ~num_envs episodes per
segment (~1% over 4 segments vs ~0.26% for a single run). Pass the run's
min_episode_length to drop them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import h5py
import yaml


def _demo_keys(group):
    """Demo groups sort as demo_0, demo_1, demo_10 -- order by the integer."""
    return sorted(group.keys(), key=lambda k: int(k.split("_")[1]))


def _num_envs(env_args: str):
    return json.loads(env_args)["sim_args"]["num_envs"]


def _plan(inputs, drop_short, allow_env_mismatch=False):
    """Validate every input up front; never start writing a doomed merge."""
    env_args = None
    plan = []
    for path in inputs:
        if not os.path.isfile(path):
            sys.exit(f"missing input: {path}")
        with h5py.File(path, "r") as f:
            if "data" not in f:
                sys.exit(f"{path}: no /data group")
            data = f["data"]
            ea = data.attrs["env_args"]
            if env_args is None:
                env_args = ea
            elif _num_envs(ea) != _num_envs(env_args):
                msg = (
                    f"num_envs mismatch: {path} has {_num_envs(ea)}, "
                    f"expected {_num_envs(env_args)}"
                )
                if not allow_env_mismatch:
                    sys.exit(f"{msg} -- segments are not comparable")
                # Cross-RUN merges (not chunk segments of one run) legitimately differ
                # here: num_envs only sets how many episodes are collected in parallel,
                # not what an episode contains. Everything downstream reads per-demo
                # groups. Pass --allow-env-mismatch when you mean it.
                print(f"  WARNING: {msg} -- allowed explicitly.")
            keys = _demo_keys(data)
            kept = [k for k in keys if int(data[k].attrs["num_samples"]) >= drop_short]
            steps = sum(int(data[k].attrs["num_samples"]) for k in kept)
            succ = sum(1 for k in kept if bool(data[k].attrs.get("success", False)))
            plan.append({"path": path, "keys": kept, "steps": steps, "succ": succ,
                         "dropped": len(keys) - len(kept)})
            print(
                f"  {os.path.basename(os.path.dirname(path)):24s} "
                f"{len(keys):6d} demos -> keep {len(kept):6d} "
                f"(drop {len(keys)-len(kept):4d}), {steps:8d} steps, "
                f"success {succ/max(len(kept),1):.1%}"
            )
    return env_args, plan


def _merged_record_yaml(inputs, out_dir):
    """Copy segment 0's record.yaml with the segments' `steps` summed.

    `steps` is the PER-ENV step budget passed to generate_data (total timesteps
    = steps * envs), so the merged value is the sum of the segments' `steps`,
    NOT the summed episode lengths. A single un-chunked run of the same size
    would have recorded exactly this, so downstream readers see what they
    expect. Per-segment seeds are kept alongside rather than overwriting the
    scalar `seed` field.
    """
    src = os.path.join(os.path.dirname(inputs[0]), "record.yaml")
    if not os.path.isfile(src):
        print("  (no record.yaml in segment 0 -- skipping)")
        return
    with open(src) as f:
        rec = yaml.safe_load(f)

    seeds, steps = [], 0
    for p in inputs:
        rp = os.path.join(os.path.dirname(p), "record.yaml")
        if os.path.isfile(rp):
            with open(rp) as f:
                seg = yaml.safe_load(f)
            seeds.append(seg.get("seed"))
            steps += int(seg.get("steps", 0))
    if len(set(seeds)) != len(seeds):
        print(f"  WARNING: segments share seeds {seeds} -- duplicate data?")
    rec["steps"] = steps
    rec["merged_from"] = [os.path.dirname(p) for p in inputs]
    rec["merged_seeds"] = seeds

    # Segment 0's reset_types describe segment 0 only. On a cross-run merge the inputs
    # deliberately differ (e.g. GraspedAboveTable + GraspedNearHole) and copying one of
    # them silently mislabels two thirds of the file, so record the union and keep the
    # per-input lists alongside it.
    per_input = []
    for p in inputs:
        rp = os.path.join(os.path.dirname(p), "record.yaml")
        seg = {}
        if os.path.isfile(rp):
            with open(rp) as f:
                seg = yaml.safe_load(f) or {}
        per_input.append(list((seg.get("simplify") or {}).get("reset_types") or []))
    if len({tuple(x) for x in per_input}) > 1:
        union = list(dict.fromkeys(t for lst in per_input for t in lst))
        rec.setdefault("simplify", {})["reset_types"] = union
        rec["merged_reset_types"] = per_input
        print(f"  reset_types differ across inputs -> union {union}")

    dst = os.path.join(out_dir, "record.yaml")
    with open(dst, "w") as f:
        yaml.dump(rec, f, default_flow_style=False)
    print(f"  wrote {dst} (steps={steps}, seeds={seeds})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination raw_data.hdf5")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="segment raw_data.hdf5 paths, in collection order")
    ap.add_argument("--drop-short", type=int, default=0, metavar="N",
                    help="drop demos with num_samples < N (pass min_episode_length)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report the merge without writing")
    ap.add_argument("--allow-env-mismatch", action="store_true",
                    help="permit inputs recorded with different num_envs (cross-RUN "
                         "merge, e.g. two collection campaigns with different reset "
                         "distributions). Everything else must still match.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.dry_run:
        sys.exit(f"refusing to overwrite existing {args.out}")

    print(f"Planning merge of {len(args.inputs)} segments:")
    env_args, plan = _plan(args.inputs, args.drop_short, args.allow_env_mismatch)

    total_demos = sum(len(p["keys"]) for p in plan)
    total_steps = sum(p["steps"] for p in plan)
    total_succ = sum(p["succ"] for p in plan)
    total_drop = sum(p["dropped"] for p in plan)
    in_bytes = sum(os.path.getsize(p["path"]) for p in plan)
    print(
        f"\nMERGED: {total_demos} demos, {total_steps} steps, "
        f"success {total_succ/max(total_demos,1):.1%}, dropped {total_drop} short\n"
        f"        num_envs={_num_envs(env_args)}, ~{in_bytes/1e9:.1f} GB -> {args.out}"
    )

    # statvfs needs an existing path, so resolve to the nearest existing ancestor
    # (the output dir usually does not exist yet on a fresh merge).
    probe = os.path.dirname(os.path.abspath(args.out))
    while probe and not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    if free < in_bytes * 1.05:
        sys.exit(f"insufficient space: need ~{in_bytes/1e9:.1f} GB, have {free/1e9:.1f} GB")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    idx = 0
    written = 0
    with h5py.File(args.out, "w") as fo:
        gd = fo.create_group("data")
        for seg in plan:
            with h5py.File(seg["path"], "r") as fi:
                di = fi["data"]
                for k in seg["keys"]:
                    # HDF5-level copy: carries attrs, filters and vlen images,
                    # and streams rather than loading the episode into RAM.
                    di.copy(k, gd, name=f"demo_{idx}")
                    written += int(di[k].attrs["num_samples"])
                    idx += 1
            print(f"  merged {os.path.basename(os.path.dirname(seg['path']))} "
                  f"-> {idx} demos, {written} steps")
        gd.attrs["env_args"] = env_args
        gd.attrs["total"] = written

    assert written == total_steps, f"step count drift: {written} != {total_steps}"
    _merged_record_yaml(args.inputs, out_dir)
    print(f"\ndone: {idx} demos, {written} steps -> {args.out} "
          f"({os.path.getsize(args.out)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
