"""Decode a world-model rollout into a ground-truth vs prediction contact sheet.

Takes ONE dataset window (H frames of context, T future steps), runs the frozen world
model's dynamics on the recorded future actions, decodes the predicted latents with a
trained debug pixel decoder, and writes a sheet whose top row is the ground-truth
future and whose bottom row is the decoded prediction, one column per rollout step:

    front_rgb GT    [  t  ][ t+1 ][ t+2 ][ t+3 ][ t+4 ][ t+5 ]
    front_rgb PRED  [  t  ][ t+1 ][ t+2 ][ t+3 ][ t+4 ][ t+5 ]

The manipulation system has three cameras and a single fused z, so by default every
camera gets its own GT/PRED pair of rows (``--cameras front_rgb`` for the plain 2x6).

Every column is ONE decoder forward on ONE latent -- the decoder has no time axis and
cannot tell which step it is drawing. Column t is ``decoder(E(o_t))``: the same
renderer applied to an encoded latent instead of a predicted one, which makes it the
decoder's ceiling. Whatever is already wrong at t is a limit of the encoder+decoder
pair; whatever degrades from t to t+5 is the dynamics drifting. Without that column a
blurry t+5 is ambiguous, so it is on by default (``--no-encoded`` to drop it).

Expect blur: pooling discards the patch grid and z is a 256-d bottleneck over three
cameras plus proprio (see pixel_decoder.py). Arm pose and peg position are the signal;
texture is not.

The world model, its dataset and H/T are all read from the decoder checkpoint's own
``decoder_config.yaml``, so the pairing can never be mismatched.

Run (JAX conda env, NOT the apptainer; needs a GPU for the encoder forward):

    PY=/projects/0/prjs0951/conda_envs/simdist-jax/bin/python
    $PY scripts/render_wm_rollout.py -d dec_wm_resnet_l256_25083707_25354721 --index 0

    # a few random windows, one camera, plus the encoded reference column
    $PY scripts/render_wm_rollout.py -d dec_wm_resnet_l256_25083707_25354721 \
        --num-windows 4 --cameras wrist_rgb

    # same window through all three decoders, to compare representations
    for D in dec_wm_resnet_l256_25083707_25354721 \
             dec_wm_dino_l256_25083733_25354721 \
             dec_wm_dinoft_cls_l256_25321112_25354721; do
        $PY scripts/render_wm_rollout.py -d $D --index 12345
    done
"""

import argparse
import copy
import os
import sys

_SIMDIST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SIMDIST)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-d",
        "--decoder-run",
        required=True,
        help="Decoder run name under checkpoints/decoders/, or a path to one.",
    )
    p.add_argument(
        "--decoder-step",
        type=int,
        default=None,
        help="Decoder checkpoint step to load (default: latest).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset name override (default: the one the world model was trained on).",
    )
    p.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help="Window index/indices into the dataset's start_idxs.",
    )
    p.add_argument(
        "--t-H",
        type=int,
        nargs="+",
        default=None,
        help="Raw global start step(s) instead of --index; the window is t_H..t_H+H+T.",
    )
    p.add_argument(
        "--num-windows",
        type=int,
        default=1,
        help="Number of random windows to render when neither --index nor --t-H given.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for random window picks.")
    p.add_argument(
        "--cameras",
        default="all",
        help="Comma-separated camera names (or 'all'), e.g. front_rgb,wrist_rgb.",
    )
    p.add_argument(
        "--steps",
        default=None,
        help="Comma-separated rollout horizons to show, 1-based (default: all T).",
    )
    p.add_argument(
        "--no-encoded",
        action="store_true",
        help="Drop the leading o_t / decoder(E(o_t)) reference column.",
    )
    p.add_argument(
        "--cell-size",
        type=int,
        default=None,
        help="Resize each panel to this many pixels (default: native resolution).",
    )
    p.add_argument("--no-labels", action="store_true", help="Write the bare grid.")
    p.add_argument(
        "--save-npz",
        action="store_true",
        help="Also dump the raw ground-truth/prediction arrays next to the PNG.",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output file (single window) or directory. Default: "
        "checkpoints/decoders/<run>/rollouts/.",
    )
    return p.parse_args()


args = parse_args()

import cv2
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from simdist.data.dataset import get_dataset
from simdist.modeling.decoder_trainer import dataset_cfg_from_world_model
from simdist.utils import config as config_utils
from simdist.utils import model as model_utils
from simdist.utils import paths

# Label band and separator geometry, in pixels of the final sheet.
_TITLE_H = 34
_HEADER_H = 26
_GUTTER_W = 132
_GAP = 2  # between panels, so adjacent frames do not read as one image
_BLOCK_GAP = 10  # between one camera's GT/PRED pair and the next camera's
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    ckpt_dir = args.decoder_run
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = os.path.join(paths.get_decoder_checkpoints_dir(), args.decoder_run)
    dec_cfg = model_utils.read_decoder_config(ckpt_dir)

    # ---- the world model this decoder was trained against --------------------------
    wm_run = dec_cfg["world_model"]["run_name"]
    wm_step = dec_cfg["world_model"].get("resolved_step", dec_cfg["world_model"]["step"])
    model, wm_cfg, wm_step = model_utils.load_model_from_ckpt(
        os.path.join(paths.get_model_checkpoints_dir(), wm_run), wm_step
    )
    latent_dim = wm_cfg["model"]["latent_dim"]
    image_shapes = config_utils.extero_obs_image_shapes_from_sys_config(wm_cfg["system"])
    cam_names = config_utils.extero_obs_names_from_sys_config(wm_cfg["system"])
    img_h, img_w, _ = image_shapes[0]
    T = config_utils.prediction_length_from_config(wm_cfg)
    print(f"World model '{wm_run}' @ step {wm_step}: latent_dim={latent_dim} "
          f"cams={cam_names} image={img_h}x{img_w} H/T={wm_cfg['model']['dataset']['history_length']}/{T}")

    decoder, dec_cfg, dec_step = model_utils.load_decoder_from_ckpt(
        ckpt_dir, latent_dim, len(cam_names), img_h, args.decoder_step
    )
    print(f"Decoder '{os.path.basename(ckpt_dir)}' @ step {dec_step}")

    # ---- dataset: the decoder run's own preprocessing (augmentation off) ------------
    ds_run_cfg = copy.deepcopy(dec_cfg)
    if args.dataset:
        ds_run_cfg["data"]["dataset_name"] = args.dataset
    dataset = get_dataset(dataset_cfg_from_world_model(ds_run_cfg, wm_cfg))
    dataset.eval()
    print(f"Dataset {dataset.data_dir} ({len(dataset)} windows)")

    cams = _resolve_cameras(args.cameras, cam_names)
    steps = _resolve_steps(args.steps, T)

    # ---- windows --------------------------------------------------------------------
    if args.t_H is not None:
        items = [dataset.get_item_by_t_H(t) for t in args.t_H]
        tags = [f"tH{t}" for t in args.t_H]
    else:
        idxs = args.index
        if idxs is None:
            rng = np.random.default_rng(args.seed)
            idxs = rng.choice(len(dataset), size=args.num_windows, replace=False).tolist()
        items = [dataset[int(i)] for i in idxs]
        tags = [f"idx{int(i)}" for i in idxs]

    batch = {
        part: {k: np.stack([it[part][k] for it in items]) for k in items[0][part]}
        for part in ("model_in", "labels")
    }
    batch = model_utils.dataset_batch_to_jax(batch)

    # ---- frozen forward + decode -----------------------------------------------------
    @nnx.jit
    def rollout(model, decoder, x):
        """Encode once, roll the dynamics on the recorded actions, decode both.

        Mirrors decoder_trainer.encode_batch: the encoding is reused for the rollout
        rather than letting ``inference`` re-run the backbone.
        """
        encoding = model.encode_context(x)
        z = encoding["latent"]
        z_pred = model.inference_from_encoding(x, encoding)["latents"]
        return decoder(z), decoder(z_pred)

    recon_enc, recon_pred = rollout(model, decoder, batch["model_in"])
    recon_enc = np.asarray(jnp.clip(recon_enc, 0.0, 1.0))  # (B, n_cam, H, W, C)
    recon_pred = np.asarray(jnp.clip(recon_pred, 0.0, 1.0))  # (B, T, n_cam, H, W, C)
    gt_now = np.asarray(batch["model_in"]["extero_obs"]) / 255.0
    gt_fut = np.asarray(batch["labels"]["extero_obs"]) / 255.0

    # ---- write ------------------------------------------------------------------------
    out_dir, out_file = _resolve_out(args.out, ckpt_dir, len(items))
    os.makedirs(out_dir, exist_ok=True)
    ds_name = os.path.basename(os.path.dirname(dataset.data_dir))

    for b, (item, tag) in enumerate(zip(items, tags)):
        title = (
            f"{os.path.basename(ckpt_dir)} @ {dec_step} | wm {wm_run} @ {wm_step} | "
            f"{ds_name} {tag} t={item['metadata']['t']}"
        )
        sheet = _compose(
            gt_now[b],
            recon_enc[b],
            gt_fut[b],
            recon_pred[b],
            cams=cams,
            cam_names=cam_names,
            steps=steps,
            include_encoded=not args.no_encoded,
            cell=args.cell_size,
            labels=not args.no_labels,
            title=title,
        )
        path = out_file or os.path.join(out_dir, f"rollout_{ds_name}_{tag}.png")
        cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"  {path}")
        _print_mse(gt_fut[b], recon_pred[b], cams, cam_names, steps)
        if args.save_npz:
            npz = os.path.splitext(path)[0] + ".npz"
            np.savez_compressed(
                npz,
                gt_now=gt_now[b],
                recon_encoded=recon_enc[b],
                gt_future=gt_fut[b],
                recon_rollout=recon_pred[b],
                cameras=np.array(cam_names),
                t=item["metadata"]["t"],
            )
            print(f"  {npz}")


def _resolve_cameras(spec: str, cam_names: list[str]) -> list[int]:
    if spec == "all":
        return list(range(len(cam_names)))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
        elif tok in cam_names:
            out.append(cam_names.index(tok))
        else:
            raise ValueError(f"unknown camera {tok!r}; have {cam_names}")
    return out


def _resolve_steps(spec: str | None, T: int) -> list[int]:
    """1-based rollout horizons to display."""
    if spec is None:
        return list(range(1, T + 1))
    steps = [int(tok) for tok in spec.split(",")]
    bad = [k for k in steps if not 1 <= k <= T]
    if bad:
        raise ValueError(f"steps {bad} outside the model's horizon 1..{T}")
    return steps


def _resolve_out(out: str | None, ckpt_dir: str, n_windows: int):
    """(directory, explicit file or None). A file is only honoured for one window."""
    if out is None:
        return os.path.join(ckpt_dir, "rollouts"), None
    if os.path.splitext(out)[1] and n_windows == 1:
        return os.path.dirname(os.path.abspath(out)), out
    return out, None


def _compose(
    gt_now: np.ndarray,
    recon_enc: np.ndarray,
    gt_fut: np.ndarray,
    recon_pred: np.ndarray,
    cams: list[int],
    cam_names: list[str],
    steps: list[int],
    include_encoded: bool,
    cell: int | None,
    labels: bool,
    title: str,
) -> np.ndarray:
    """One camera per GT/PRED row pair, one column per rollout step.

    ``gt_now``/``recon_enc`` are (n_cam, H, W, C); ``gt_fut``/``recon_pred`` are
    (T, n_cam, H, W, C). Values in [0, 1]; the sheet comes back as uint8 RGB.
    """
    col_labels = (["t (enc)"] if include_encoded else []) + [f"t+{k}" for k in steps]

    rows, row_labels, mse_cells = [], [], []
    for c in cams:
        gt_row = ([gt_now[c]] if include_encoded else []) + [
            gt_fut[k - 1, c] for k in steps
        ]
        pred_row = ([recon_enc[c]] if include_encoded else []) + [
            recon_pred[k - 1, c] for k in steps
        ]
        rows.append([_cell(x, cell) for x in gt_row])
        rows.append([_cell(x, cell) for x in pred_row])
        row_labels += [f"{cam_names[c]} GT", f"{cam_names[c]} PRED"]
        mse_cells.append([None] * len(gt_row))
        mse_cells.append(
            [float(np.mean((p - g) ** 2)) for p, g in zip(pred_row, gt_row)]
        )

    h, w = rows[0][0].shape[:2]
    n_rows, n_cols = len(rows), len(rows[0])
    off_x = _GUTTER_W if labels else 0
    off_y = (_TITLE_H + _HEADER_H) if labels else 0
    # A PRED row sits directly under its own GT row; the wider gap separates cameras.
    ys = []
    y = off_y
    for r in range(n_rows):
        y += 0 if r == 0 else (_BLOCK_GAP if r % 2 == 0 else _GAP)
        ys.append(y)
        y += h
    xs = [off_x + c * (w + _GAP) for c in range(n_cols)]

    sheet = np.zeros((y, xs[-1] + w, 3), np.uint8)
    sheet[:] = 24  # dark ground, so white labels read and panel edges stay visible

    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            sheet[ys[r] : ys[r] + h, xs[c] : xs[c] + w] = img
            if labels and mse_cells[r][c] is not None:
                _text(sheet, f"mse {mse_cells[r][c]:.4f}", (xs[c] + 6, ys[r] + h - 8), 0.42)

    if not labels:
        return sheet

    _text(sheet, title, (8, 22), 0.5)
    for c, lab in enumerate(col_labels):
        _text(sheet, lab, (xs[c] + 6, _TITLE_H + 18), 0.55)
    for r, lab in enumerate(row_labels):
        _text(sheet, lab, (8, ys[r] + h // 2), 0.45)
    return sheet


def _cell(img: np.ndarray, cell: int | None) -> np.ndarray:
    """Float image in [0, 1] -> uint8 RGB panel, optionally resized."""
    out = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if cell is not None and cell != out.shape[0]:
        out = cv2.resize(out, (cell, cell), interpolation=cv2.INTER_AREA)
    return out


def _text(canvas: np.ndarray, s: str, org: tuple[int, int], scale: float) -> None:
    """White text with a dark outline, so it stays legible over any panel."""
    cv2.putText(canvas, s, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, s, org, _FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _print_mse(gt_fut, recon_pred, cams, cam_names, steps) -> None:
    """Per-camera, per-horizon reconstruction MSE -- the numbers behind the picture."""
    header = "    " + "".join(f"{'t+' + str(k):>10}" for k in steps)
    print(f"    per-step MSE (images in [0,1])\n{header}")
    for c in cams:
        vals = [float(np.mean((recon_pred[k - 1, c] - gt_fut[k - 1, c]) ** 2)) for k in steps]
        print(f"    {cam_names[c]:<12}" + "".join(f"{v:>10.5f}" for v in vals))


if __name__ == "__main__":
    main()
