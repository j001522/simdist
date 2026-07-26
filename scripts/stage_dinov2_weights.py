"""Stage frozen DINOv2 weights for simdist/modeling/dinov2.py.

RUN THIS ON A LOGIN NODE. Snellius compute nodes have no outbound network, so a training
job cannot reach HuggingFace (the same failure that broke OmniReset's runtime kinematics
download). This writes assets/<variant>-<img_size>.npz once; jobs then read it locally.

Two things happen here that cannot happen at train time:

1. pt -> flax conversion (needs torch, which is CPU-only in this env -- fine, one pass).
2. The position embedding is interpolated from the checkpoint's 518px grid to our input
   resolution using TORCH's interpolate_pos_encoding, and baked into the params. This is
   a bug workaround, not an optimisation: FlaxDinov2's own interpolate_pos_encoding
   reshapes with the input batch size instead of 1 and therefore only works at batch
   size 1. Baking it makes num_patches == num_positions so that branch early-returns.
   See the module docstring in simdist/modeling/dinov2.py.

    python scripts/stage_dinov2_weights.py                          # small @ 224
    python scripts/stage_dinov2_weights.py --variant dinov2-base
    python scripts/stage_dinov2_weights.py --img-size 252
"""

import argparse
import os

import numpy as np

from simdist.modeling.dinov2 import KNOWN_VARIANTS, load_staged_dinov2, save_staged_dinov2
from simdist.utils import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="dinov2-small", choices=KNOWN_VARIANTS)
    ap.add_argument("--img-size", type=int, default=224,
                    help="input resolution the position embedding is baked for")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import Dinov2Config, Dinov2Model, FlaxDinov2Model

    if not transformers.__version__.startswith("4.48"):
        raise SystemExit(
            f"transformers {transformers.__version__} found, but Flax support was removed "
            "in v5 (modeling_flax_dinov2.py is gone on main). Pin transformers==4.48.*"
        )

    repo = f"facebook/{args.variant}"
    assets = paths.get_assets_dir()
    os.makedirs(assets, exist_ok=True)
    out = args.out or os.path.join(assets, f"{args.variant}-{args.img_size}.npz")

    print(f"loading {repo} (pt -> flax conversion)")
    flax_model = FlaxDinov2Model.from_pretrained(repo, from_pt=True)
    pt_model = Dinov2Model.from_pretrained(repo).eval()
    cfg = Dinov2Config.from_pretrained(repo)

    if args.img_size % cfg.patch_size:
        raise SystemExit(
            f"--img-size {args.img_size} is not divisible by patch_size {cfg.patch_size}"
        )
    grid = args.img_size // cfg.patch_size
    n_tokens = grid * grid + 1

    print(f"baking position embedding: {cfg.image_size}px -> {args.img_size}px "
          f"({grid}x{grid} patches, {n_tokens} tokens) via torch")
    with torch.no_grad():
        dummy = torch.zeros(1, n_tokens, cfg.hidden_size)
        baked = pt_model.embeddings.interpolate_pos_encoding(
            dummy, args.img_size, args.img_size
        ).numpy()
    if baked.shape != (1, n_tokens, cfg.hidden_size):
        raise SystemExit(f"unexpected baked position embedding shape {baked.shape}")

    params = flax_model.params
    params = params.unfreeze() if hasattr(params, "unfreeze") else dict(params)
    params["embeddings"] = dict(params["embeddings"])
    params["embeddings"]["position_embeddings"] = baked

    # Declaring image_size = our resolution is what makes FlaxDinov2's buggy
    # interpolate_pos_encoding branch early-return.
    cfg_dict = cfg.to_dict()
    cfg_dict["image_size"] = args.img_size

    n = save_staged_dinov2(out, cfg_dict, params)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {n:,} params)")

    # ---- verify the artefact round-trips and still matches the torch reference
    print("verifying staged file against the torch reference ...")
    from simdist.modeling.dinov2 import DINOv2Backbone

    backbone = DINOv2Backbone(load_staged_dinov2(out), img_size=args.img_size)
    rng = np.random.default_rng(0)
    px = rng.standard_normal((4, args.img_size, args.img_size, 3)).astype(np.float32)
    got = np.asarray(backbone(px))  # NHWC in, pooled out

    with torch.no_grad():
        ref = pt_model(torch.from_numpy(px.transpose(0, 3, 1, 2))).last_hidden_state.numpy()
    ref_pooled = np.concatenate([ref[:, 0], ref[:, 1:].mean(axis=1)], axis=-1)

    rel = np.abs(got - ref_pooled).max() / np.abs(ref_pooled).max()
    print(f"  pooled {got.shape} vs torch {ref_pooled.shape}: relative diff {rel:.2e}")
    if rel > 1e-4:
        raise SystemExit(f"  DIVERGENT from the torch reference (rel {rel:.2e}) -- not usable")
    print("  OK")


if __name__ == "__main__":
    main()
