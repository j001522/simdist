"""Visualise the state-space coverage of a recorded dataset (raw_data.hdf5).

Every recorded timestep of every demo is binned into a 2D log-density map, so darker /
hotter cells = more frequently visited. This is a qualitative "where has the dataset
actually been?" inspection tool.

Three stages, because raw_data.hdf5 is huge (~73 GB for the merged set) and walking its
~100k demo groups costs minutes, while replotting should be instant:

    extract   stream the low-dimensional obs (never touches the JPEG images) into a small
              ``.npz`` cache (~200 MB for 2.2M transitions).
    plot      2x2 density figure: end effector and plug, both in the robot root frame and
              on shared axis limits, projected onto (x, z) and (x, y).
    overlay   the same density, but projected through the front camera and alpha-blended
              onto a real recorded frame of the scene. One image for the end effector,
              one for the plug.

Usage:
    # one-off cache build (~15-30 min on the merged dataset; run via dataset_coverage.sbatch)
    python scripts/plot_dataset_coverage.py extract \
        datasets/sim/2026-07-27_merged/raw_data.hdf5 \
        -o outputs/coverage/merged.npz --workers 16

    python scripts/plot_dataset_coverage.py plot outputs/coverage/merged.npz \
        -o outputs/coverage/merged.png

    python scripts/plot_dataset_coverage.py overlay outputs/coverage/merged.npz \
        --raw datasets/sim/2026-07-27_merged/raw_data.hdf5 \
        -o outputs/coverage/merged_scene.png
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import numpy as np

# obs key -> short name; all recorded as [x, y, z, rx, ry, rz] (axis-angle)
OBS_KEYS = {
    "ee": "end_effector_pose",
    "peg": "insertive_asset_pose",
    "sock": "receptive_asset_pose",
    "rel": "insertive_asset_in_receptive_asset_frame",
}

# Camera rig, mirrored from the data-collection env config:
#   UWLab/.../omnireset/config/ur5e_robotiq_2f85/data_collection_rgb_cfg.py
# The prims are parented to the Robot prim, so these offsets are already expressed in the
# robot root frame -- the same frame as the poses in the cache. No world transform needed.
# NOTE the env randomizes camera pose (+-5 cm, +-2 deg) and focal length (11.2-15.2 mm) on
# every reset and does not record the sampled values, so a projection onto one recorded
# frame carries that jitter. Measured, it is +-3.5 px RMS at 320 px wide -- small, and
# zero-mean, so it blurs a single frame's alignment but does not bias the aggregate.
#
# ``pos_correction`` is an EMPIRICAL calibration, not a config value. Projecting with the
# nominal pos put every object ~10 px too far right (see cmd_check). Fitting a camera
# translation against 1402 green-blob correspondences gives +3.8 cm along the camera's own
# x axis and drops the reprojection RMS from 10.58 px to 3.50 px -- which is the per-episode
# randomization scatter, i.e. the noise floor. The residual is a pure translation: it scales
# as 1/depth (du*depth is flat across depth bins while du varies 30%), and adding focal
# length to the fit does not improve the RMS at all, so it is not a zoom or principal-point
# error. The underlying cause is most likely a frame mismatch between the camera prim's
# parent (the Robot prim) and the root body that end_effector_pose is expressed in; that has
# not been confirmed in Isaac, so treat this as a measured correction. Pass
# --no-cam-correction to project from the nominal pose instead. Only fitted for the front
# camera; the side camera has no correspondences to fit against.
CAMERAS = {
    "front": {
        "pos": (1.0770121, -0.1679045, 0.4486344),
        "rot": (0.70564552, 0.46613815, 0.25072644, 0.47107948),  # (w, x, y, z), opengl
        "focal": 13.20,
        "pos_correction": (0.0093, 0.0372, 0.0043),  # root frame; == +3.78 cm on camera x
    },
    "side": {
        "pos": (0.8323904, 0.5877843, 0.2805111),
        "rot": (0.29008842, 0.22122445, 0.51336143, 0.77676798),
        "focal": 20.10,
    },
}
APERTURE = 20.955  # PinholeCameraCfg.horizontal_aperture default [mm]
SENSOR_WH = (320, 240)  # TiledCameraCfg width, height -- the intrinsics live in this space
STORED_WH = (224, 224)  # process_image() resizes to this (plain bilinear, no crop)


# --------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------


def _read_chunk(args) -> dict:
    """Worker: read the low-dim obs of a list of demo names from an independent handle."""
    path, names = args
    out = {k: [] for k in OBS_KEYS}
    out["reward"] = []
    out["expert"] = []
    out["ep_len"] = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        for name in names:
            grp = data[name]
            obs = grp["obs"]
            n = obs[OBS_KEYS["ee"]].shape[0]
            for short, key in OBS_KEYS.items():
                out[short].append(np.asarray(obs[key][:], dtype=np.float32))
            out["reward"].append(np.asarray(grp["reward"][:], dtype=np.float32))
            out["expert"].append(np.asarray(grp["expert_policy_flag"][:], dtype=bool))
            out["ep_len"].append(n)
    return {k: (np.concatenate(v) if k != "ep_len" else np.asarray(v, np.int32)) for k, v in out.items()}


def cmd_extract(args) -> None:
    path = str(Path(args.path).resolve())

    t0 = time.time()
    with h5py.File(path, "r") as f:
        grp = f["data"]
        n_demos = len(grp)
        if args.list_names:
            # Enumerating ~100k link names costs many minutes on GPFS -- only do it when
            # the demo naming is not the standard contiguous "demo_<i>".
            names = sorted(grp.keys(), key=lambda s: int(s.rsplit("_", 1)[-1]))
        else:
            names = [f"demo_{i}" for i in range(n_demos)]
            if names[-1] not in grp:
                raise SystemExit(
                    f"demo names are not contiguous demo_0..demo_{n_demos - 1}; rerun with --list-names"
                )
    print(f"[extract] {len(names)} demos resolved in {time.time() - t0:.1f}s", flush=True)

    if args.demo_stride > 1:
        names = names[:: args.demo_stride]
    if args.max_demos:
        names = names[: args.max_demos]
    print(f"[extract] reading {len(names)} demos with {args.workers} worker(s)", flush=True)

    chunks = [names[i : i + args.chunk_size] for i in range(0, len(names), args.chunk_size)]
    jobs = [(path, c) for c in chunks]

    t0 = time.time()
    results = []
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap(_read_chunk, jobs)):
                results.append(res)
                _progress(i + 1, len(jobs), t0)
    else:
        for i, job in enumerate(jobs):
            results.append(_read_chunk(job))
            _progress(i + 1, len(jobs), t0)
    print(flush=True)

    cache = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    ep_len = cache.pop("ep_len")
    cache["ep_len"] = ep_len
    cache["ep_start"] = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
    cache["demo_id"] = np.repeat(np.arange(len(ep_len), dtype=np.int32), ep_len)
    cache["t"] = np.concatenate([np.arange(n, dtype=np.int32) for n in ep_len])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache)
    n = len(cache["reward"])
    print(f"[extract] {n} transitions from {len(ep_len)} demos -> {out} "
          f"({out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")


def _progress(done: int, total: int, t0: float) -> None:
    if done % 10 and done != total:
        return
    el = time.time() - t0
    eta = el / done * (total - done)
    print(f"\r[extract] {done}/{total} chunks  {el:.0f}s elapsed  {eta:.0f}s eta", end="", flush=True)


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues, batched. aa: (N, 3) -> (N, 3, 3)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = np.where(small, 0.0, aa / np.where(small, 1.0, theta))
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = np.cos(theta[:, 0]), np.sin(theta[:, 0])
    C = 1.0 - c
    return np.stack(
        [
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def quat_to_matrix(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def compose(parent_pose: np.ndarray, child_pos: np.ndarray) -> np.ndarray:
    """Express ``child_pos`` (given in the parent's frame) in the parent's parent frame."""
    R = axis_angle_to_matrix(parent_pose[:, 3:6])
    return np.einsum("nij,nj->ni", R, child_pos) + parent_pose[:, 0:3]


def bodies_in_root(cache) -> dict:
    """The two things we plot, both in the robot root frame."""
    ee_pose = cache["ee"]
    return {
        "ee": ("End effector", ee_pose[:, :3]),
        "plug": ("Plug", compose(ee_pose, cache["peg"][:, :3])),
    }


def project_to_image(pts_root: np.ndarray, cam: dict, focal: float | None = None,
                     correct: bool = True):
    """Robot-root-frame points -> pixel coords in the native 320x240 sensor image.

    Everything downstream works in sensor space; the stored 224x224 frames are un-stretched
    back to 320x240 for display, which is what keeps the scene from looking squashed.

    Returns ``(uv, valid)`` where uv is (N, 2) in sensor pixels and valid marks the points
    in front of the camera.
    """
    R = quat_to_matrix(cam["rot"])  # camera axes -> root frame
    pos = np.asarray(cam["pos"], dtype=np.float64)
    if correct:
        pos = pos + np.asarray(cam.get("pos_correction", (0.0, 0.0, 0.0)), dtype=np.float64)
    d = pts_root - pos
    p_cam = d @ R  # == R.T @ d, i.e. root -> camera

    # OpenGL convention: camera looks down -Z, +X right, +Y up
    depth = -p_cam[:, 2]
    valid = depth > 1e-3
    depth = np.where(valid, depth, 1.0)

    w, h = SENSOR_WH
    f = cam["focal"] if focal is None else focal
    fx = fy = f / APERTURE * w
    u = fx * p_cam[:, 0] / depth + w / 2.0
    v = -fy * p_cam[:, 1] / depth + h / 2.0
    return np.stack([u, v], axis=1), valid


# --------------------------------------------------------------------------------------
# plot: 2x2 density figure
# --------------------------------------------------------------------------------------


def _shared_limits(all_pts: np.ndarray, clip: float):
    """One common span for every axis so all four panels are directly comparable."""
    lo = np.percentile(all_pts, clip, axis=0)
    hi = np.percentile(all_pts, 100 - clip, axis=0)
    span = float(np.max(hi - lo)) * 1.08
    centre = (lo + hi) / 2.0
    return centre, span


def _density(x, y, rng, bins, smooth):
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter

        H = gaussian_filter(H, sigma=smooth, mode="constant") * (2 * np.pi * smooth**2)
    return H, xe, ye


def cmd_plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    c = np.load(args.cache)
    bodies = bodies_in_root(c)
    n = len(c["ee"])

    centre, span = _shared_limits(np.concatenate(list(p for _, p in bodies.values())), args.clip)
    axis_rng = [(centre[k] - span / 2, centre[k] + span / 2) for k in range(3)]
    # two projections are enough to cover all three axes
    cols = [(0, 2, "x", "z"), (0, 1, "x", "y")]

    grids = {}
    vmax = 0.0
    for key, (_, pts) in bodies.items():
        for i, j, _, _ in cols:
            H, xe, ye = _density(pts[:, i], pts[:, j], [axis_rng[i], axis_rng[j]], args.bins, args.smooth)
            grids[key, i, j] = (H, xe, ye)
            vmax = max(vmax, H.max())

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6), constrained_layout=True)
    norm = LogNorm(vmin=0.5, vmax=vmax)  # shared across all panels

    for r, (key, (label, _)) in enumerate(bodies.items()):
        for col, (i, j, xl, yl) in enumerate(cols):
            ax = axes[r][col]
            H, xe, ye = grids[key, i, j]
            im = ax.imshow(
                np.ma.masked_where(H < 0.5, H).T,
                origin="lower",
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                cmap=args.cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xlabel(f"{xl} [m]", fontsize=9)
            ax.set_ylabel(f"{yl} [m]", fontsize=9)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_title(f"{label} — {xl}{yl}", fontsize=11, loc="left", fontweight="bold")
            else:
                ax.set_title(f"{xl}{yl}", fontsize=11, loc="left")

    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.015)
    cb.set_label("visits per bin", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Dataset coverage in the robot root frame — {Path(args.cache).stem}\n"
        f"{len(c['ep_len']):,} demos · {n:,} transitions · shared axis limits and colour scale",
        fontsize=11,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"[plot] wrote {out}")


# --------------------------------------------------------------------------------------
# overlay: density projected onto a recorded camera frame
# --------------------------------------------------------------------------------------


def _load_background(raw_path: str, demo: str, frame: int, camera: str, upscale: int = 4):
    """Decode one recorded frame and undo the 320x240 -> 224x224 squash.

    process_image() stretched the sensor image to a square with a plain bilinear resize, so
    displaying it as-is distorts the scene. Resizing back to the sensor aspect (and past it,
    for a crisper render) restores the true geometry and matches the projection.
    """
    import cv2

    with h5py.File(raw_path, "r") as f:
        dset = f["data"][demo]["obs"][f"{camera}_rgb"]
        h, w, ch = (int(x) for x in dset.attrs["image_shape"])
        # cv2.imencode/imdecode round-trip the array as stored, so no BGR swap is needed
        img = cv2.imdecode(np.asarray(dset[frame], dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img.reshape(h, w, ch)
    return cv2.resize(
        img, (SENSOR_WH[0] * upscale, SENSOR_WH[1] * upscale), interpolation=cv2.INTER_LANCZOS4
    )


def cmd_overlay(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    c = np.load(args.cache)
    cam = CAMERAS[args.camera]
    bg = _load_background(args.raw, args.bg_demo, args.bg_frame, args.camera, args.upscale)
    iw, ih = SENSOR_WH  # all coordinates live in sensor space; bg is just rendered into it
    print(f"[overlay] background {args.bg_demo}[{args.bg_frame}] {args.camera}_rgb "
          f"-> {bg.shape[1]}x{bg.shape[0]} in a {iw}x{ih} sensor frame")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for key, (label, pts) in bodies_in_root(c).items():
        uv, valid = project_to_image(pts.astype(np.float64), cam, args.focal,
                                     correct=not args.no_cam_correction)
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < iw) & (uv[:, 1] >= 0) & (uv[:, 1] < ih)
        frac = inside.mean()
        print(f"[overlay] {key}: {frac:.1%} of {len(pts):,} points project inside the frame")
        if frac < 0.05:
            print(f"[overlay] WARNING: almost nothing lands in frame -- check the camera convention")

        bins = [args.bins, max(1, round(args.bins * ih / iw))]
        H, _, _ = _density(
            uv[inside, 0], uv[inside, 1], [[0, iw], [0, ih]], bins, args.smooth
        )
        H = H.T
        norm = LogNorm(vmin=args.floor, vmax=H.max())

        # per-pixel alpha ramped with the (log) density: sparse cells stay translucent so
        # the scene shows through, the hot core is close to opaque
        rgba = plt.get_cmap(args.cmap)(norm(np.ma.masked_where(H < args.floor, H)))
        ramp = np.clip(norm(np.maximum(H, args.floor)).filled(0.0), 0.0, 1.0)
        rgba[..., 3] = np.where(H < args.floor, 0.0, args.alpha * (0.25 + 0.75 * ramp))

        fig, ax = plt.subplots(figsize=(8.4, 8.4 * ih / iw), constrained_layout=True)
        ax.imshow(bg, extent=[0, iw, ih, 0], interpolation="lanczos")
        ax.imshow(rgba, extent=[0, iw, ih, 0], interpolation="bilinear")
        im = plt.cm.ScalarMappable(norm=norm, cmap=args.cmap)
        ax.set_xlim(0, iw)
        ax.set_ylim(ih, 0)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("visits per pixel bin", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(
            f"{label} coverage — {len(c['ep_len']):,} demos, {len(pts):,} transitions\n"
            f"{args.camera} camera, {'nominal' if args.no_cam_correction else 'measured'} "
            f"calibration; background is one recorded frame",
            fontsize=10,
        )
        path = out.with_name(f"{out.stem}_{key}{out.suffix}")
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"[overlay] wrote {path}")


# --------------------------------------------------------------------------------------
# check: does the projection actually land on the objects?
# --------------------------------------------------------------------------------------


def cmd_check(args) -> None:
    """Contact sheet projecting each frame's OWN 3D poses onto that same frame.

    Markers on the objects => the projection is right and any offset in the overlay is the
    per-episode camera jitter. Markers consistently off in one direction => a bug.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam = CAMERAS[args.camera]
    demos = [f"demo_{i}" for i in args.demos]
    ncol = 4
    nrow = int(np.ceil(len(demos) / ncol))
    iw, ih = SENSOR_WH

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow), constrained_layout=True)
    with h5py.File(args.raw, "r") as f:
        for ax, name in zip(np.atleast_1d(axes).ravel(), demos):
            obs = f["data"][name]["obs"]
            t = min(args.frame, obs["end_effector_pose"].shape[0] - 1)
            ee = np.asarray(obs["end_effector_pose"][t], np.float64)[None]
            marks = {
                "wrist_3": (ee[:, :3], "#00e5ff"),
                "plug": (compose(ee, np.asarray(obs["insertive_asset_pose"][t], np.float64)[None, :3]), "#39ff14"),
                "socket": (compose(ee, np.asarray(obs["receptive_asset_pose"][t], np.float64)[None, :3]), "#ff1744"),
            }
            ax.imshow(
                _load_background(args.raw, name, t, args.camera, args.upscale),
                extent=[0, iw, ih, 0],
                interpolation="lanczos",
            )
            for lab, (pt, col) in marks.items():
                uv, _ = project_to_image(pt, cam, args.focal,
                                         correct=not args.no_cam_correction)
                ax.plot(uv[0, 0], uv[0, 1], "o", ms=10, mfc="none", mew=2.0, color=col, label=lab)
            ax.set_xlim(0, iw)
            ax.set_ylim(ih, 0)
            ax.set_title(f"{name} t={t}", fontsize=8)
            ax.axis("off")
    np.atleast_1d(axes).ravel()[0].legend(fontsize=7, loc="lower left")
    cal = "nominal calibration" if args.no_cam_correction else "measured calibration"
    fig.suptitle(f"Projection check — {args.camera} camera, {cal}", fontsize=12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"[check] wrote {out}")


# --------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="stream low-dim obs from raw_data.hdf5 into an .npz cache")
    e.add_argument("path", help="path to raw_data.hdf5")
    e.add_argument("-o", "--out", required=True, help="output .npz cache")
    e.add_argument("--max-demos", type=int, default=0, help="0 = all")
    e.add_argument("--demo-stride", type=int, default=1, help="take every Nth demo")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--chunk-size", type=int, default=200, help="demos per worker task")
    e.add_argument("--list-names", action="store_true",
                   help="enumerate demo names instead of assuming demo_0..demo_N-1 (slow)")
    e.set_defaults(func=cmd_extract)

    q = sub.add_parser("plot", help="2x2 density figure in the robot root frame")
    q.add_argument("cache")
    q.add_argument("-o", "--out", required=True)
    q.add_argument("--bins", type=int, default=400)
    q.add_argument("--cmap", default="magma_r", help="light->dark sequential colormap")
    q.add_argument("--clip", type=float, default=0.1, help="percentile clipped off each axis end")
    q.add_argument("--smooth", type=float, default=1.0, help="gaussian blur sigma in bins; 0 disables")
    q.add_argument("--dpi", type=int, default=200)
    q.set_defaults(func=cmd_plot)

    o = sub.add_parser("overlay", help="density projected onto a recorded camera frame")
    o.add_argument("cache")
    o.add_argument("--raw", required=True, help="raw_data.hdf5 to pull the background frame from")
    o.add_argument("-o", "--out", required=True, help="_ee and _plug are appended to the stem")
    o.add_argument("--camera", default="front", choices=sorted(CAMERAS))
    o.add_argument("--bg-demo", default="demo_0")
    o.add_argument("--bg-frame", type=int, default=0)
    o.add_argument("--focal", type=float, default=None, help="override the nominal focal length [mm]")
    o.add_argument("--bins", type=int, default=224, help="pixel bins across the image")
    o.add_argument("--cmap", default="inferno", help="dark->bright colormap, readable over a photo")
    o.add_argument("--alpha", type=float, default=0.75)
    o.add_argument("--floor", type=float, default=1.0, help="hide bins below this count")
    o.add_argument("--smooth", type=float, default=1.0)
    o.add_argument("--upscale", type=int, default=4, help="background resample factor for a crisper render")
    o.add_argument("--no-cam-correction", action="store_true",
                   help="project from the nominal camera pose, without the measured offset")
    o.add_argument("--dpi", type=int, default=200)
    o.set_defaults(func=cmd_overlay)

    k = sub.add_parser("check", help="verify the projection lands on the objects")
    k.add_argument("--raw", required=True)
    k.add_argument("-o", "--out", required=True)
    k.add_argument("--camera", default="front", choices=sorted(CAMERAS))
    k.add_argument("--demos", type=int, nargs="+", default=[0, 5, 11, 23, 40, 77, 101, 250, 512, 1024, 2048, 4096])
    k.add_argument("--frame", type=int, default=10)
    k.add_argument("--focal", type=float, default=None)
    k.add_argument("--upscale", type=int, default=4)
    k.add_argument("--no-cam-correction", action="store_true")
    k.add_argument("--dpi", type=int, default=130)
    k.set_defaults(func=cmd_check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
