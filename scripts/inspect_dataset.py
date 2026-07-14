"""Lightweight, read-only inspector for robomimic-style HDF5 datasets (raw_data.hdf5).

Reads only group/dataset metadata (.attrs, shape, dtype) for schema/stats -- never
loads bulk array data -- so it stays fast even on multi-GB files. Frame extraction
is the one mode that decodes actual pixel data, and only for a single demo.

Usage:
    python scripts/inspect_dataset.py schema  <path/to/raw_data.hdf5>
    python scripts/inspect_dataset.py stats    <path/to/raw_data.hdf5>
    python scripts/inspect_dataset.py frames   <path/to/raw_data.hdf5> --out-dir DIR [--demo demo_123]
"""

import argparse
import io
import json
import statistics
import sys

import h5py
import numpy as np


def _jsonable(v):
    return v.tolist() if hasattr(v, "tolist") else v


def _describe_group(g):
    out = {}
    for k in g.keys():
        item = g[k]
        if isinstance(item, h5py.Group):
            out[k] = {"__group__": _describe_group(item)}
        else:
            entry = {
                "shape": item.shape,
                "dtype": str(item.dtype),
                "chunks": item.chunks,
                "compression": item.compression,
            }
            if item.attrs:
                entry["attrs"] = {ak: _jsonable(av) for ak, av in item.attrs.items()}
            out[k] = entry
    return out


def _keyset(struct, prefix=""):
    keys = set()
    for k, v in struct.items():
        p = f"{prefix}/{k}"
        if "__group__" in v:
            keys |= _keyset(v["__group__"], p)
        else:
            keys.add(p)
    return keys


def cmd_schema(args):
    with h5py.File(args.path, "r") as f:
        data_grp = f["data"]
        root_attrs = {k: _jsonable(v) for k, v in data_grp.attrs.items()}
        demo_names = list(data_grp.keys())
        n_demos = len(demo_names)

        import random

        random.seed(0)
        sample_idx = sorted(
            set([0, 1, n_demos - 1] + random.sample(range(n_demos), min(args.sample, n_demos)))
        )
        sample_names = [demo_names[i] for i in sample_idx]

        per_demo = {}
        for name in sample_names:
            g = data_grp[name]
            per_demo[name] = {
                "attrs": {k: _jsonable(v) for k, v in g.attrs.items()},
                "structure": _describe_group(g),
            }

        keysets = [_keyset(per_demo[n]["structure"]) for n in sample_names]
        common = set.intersection(*keysets) if keysets else set()
        union = set.union(*keysets) if keysets else set()

        result = {
            "file": args.path,
            "num_demos_total": n_demos,
            "data_group_attrs": root_attrs,
            "schema_consistent_across_sampled_demos": common == union,
            "keys_differing_if_any": sorted(union - common),
            "sampled_demo_names": sample_names,
            "representative_demo_structure": per_demo[sample_names[0]],
        }
        print(json.dumps(result, indent=2, default=str))


def cmd_stats(args):
    with h5py.File(args.path, "r") as f:
        data_grp = f["data"]
        demo_names = list(data_grp.keys())
        n = len(demo_names)
        num_samples = []
        successes = 0
        for name in demo_names:
            g = data_grp[name]
            num_samples.append(int(g.attrs.get("num_samples", -1)))
            if g.attrs.get("success", False):
                successes += 1

        len1 = sum(1 for x in num_samples if x == 1)
        result = {
            "num_demos": n,
            "success_count": successes,
            "success_rate": successes / n,
            "num_samples_min": min(num_samples),
            "num_samples_max": max(num_samples),
            "num_samples_mean": sum(num_samples) / n,
            "num_samples_median": statistics.median(num_samples),
            "demos_with_num_samples_eq_1": len1,
            "demos_with_num_samples_eq_1_frac": len1 / n,
            "sum_num_samples": sum(num_samples),
        }
        print(json.dumps(result, indent=2))


def cmd_frames(args):
    from PIL import Image

    with h5py.File(args.path, "r") as f:
        data_grp = f["data"]
        if args.demo is not None:
            target = args.demo
        else:
            target = next(
                (
                    name
                    for name in data_grp.keys()
                    if int(data_grp[name].attrs.get("num_samples", 0)) == args.target_length
                ),
                list(data_grp.keys())[0],
            )

        g = data_grp[target]
        print(f"Using demo: {target}, num_samples={g.attrs.get('num_samples')}, success={g.attrs.get('success')}")

        cams = [k for k in g["obs"].keys() if g[f"obs/{k}"].attrs.get("jpeg", False)]
        for cam in cams:
            dset = g[f"obs/{cam}"]
            T = dset.shape[0]
            frames = []
            for t in range(T):
                jpeg_bytes = dset[t]
                if isinstance(jpeg_bytes, np.ndarray):
                    jpeg_bytes = jpeg_bytes.tobytes()
                frames.append(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))

            idxs = np.linspace(0, T - 1, min(8, T)).astype(int)
            thumbs = [frames[i].resize((160, 160)) for i in idxs]
            sheet = Image.new("RGB", (160 * len(thumbs), 160))
            for i, th in enumerate(thumbs):
                sheet.paste(th, (i * 160, 0))
            sheet.save(f"{args.out_dir}/{target}_{cam}_contactsheet.png")

            frames[0].save(
                f"{args.out_dir}/{target}_{cam}_replay.gif",
                save_all=True,
                append_images=frames[1:],
                duration=1000 // 12,
                loop=0,
            )
            print(f"  {cam}: saved contact sheet + {T}-frame gif")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_schema = sub.add_parser("schema", help="Print group/dataset layout + schema-consistency check")
    p_schema.add_argument("path")
    p_schema.add_argument("--sample", type=int, default=20, help="number of random demos to cross-check")
    p_schema.set_defaults(func=cmd_schema)

    p_stats = sub.add_parser("stats", help="Print dataset-wide stats (all demos, attrs only)")
    p_stats.add_argument("path")
    p_stats.set_defaults(func=cmd_stats)

    p_frames = sub.add_parser("frames", help="Decode JPEG obs fields of one demo into contact sheets + GIFs")
    p_frames.add_argument("path")
    p_frames.add_argument("--out-dir", required=True)
    p_frames.add_argument("--demo", default=None, help="demo group name, e.g. demo_123 (default: first match of --target-length)")
    p_frames.add_argument("--target-length", type=int, default=160, help="num_samples to look for when --demo is unset")
    p_frames.set_defaults(func=cmd_frames)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
