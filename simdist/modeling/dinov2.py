"""Frozen DINOv2 image backbone: a thin nnx wrapper around transformers' FlaxDinov2Model.

Drop-in replacement for ResNet18Backbone in ManipulationEncoder -- same call signature and
the same "(..., H, W, 3) -> (..., out_features)" contract, so encode_latent changes only in
which backbone it holds.

Why frozen: the image features become a fixed function of the pixels, which removes the
BatchNorm train/eval asymmetry entirely (DINOv2 is LayerNorm-only, no running statistics),
and it follows the Newt recipe (Hansen et al., 2025) of concatenating a *pooled* DINOv2
embedding with proprioception into an MLP state encoder. DINO-WM instead keeps the full
patch-token grid, because it has no reward/value head to carry the task signal; simdist
does, so the pooled vector is the consistent choice here.

Pooling. A ViT emits a token sequence, not a spatial map, so there is no single "the"
feature vector:
  cls       -- the global summary token (hidden_size)
  mean      -- mean over patch tokens (hidden_size)
  cls_mean  -- both, concatenated (2 * hidden_size); DINOv2's own linear-probe protocol
               and the default here.
Pooling is permutation-invariant over patches, so absolute spatial layout is discarded.
That is the deliberate trade of this branch.

    !! transformers==4.48.* is REQUIRED !!
Flax support was removed in transformers v5 (modeling_flax_dinov2.py is 404 on main), so
this is the last line that ships FlaxDinov2Model.

    !! the staged position embedding is not an optimisation, it is a bug workaround !!
FlaxDinov2's interpolate_pos_encoding reshapes the position embedding with
`hidden_states.shape[0]` -- the *input batch size* -- instead of 1:

    patch_pos_embed = jnp.transpose(...).reshape((hidden_states.shape[0], -1, dim))
    return jnp.concatenate((class_pos_embed[jnp.newaxis, :], patch_pos_embed), axis=1)

so with batch B > 1 the 256 positions are split into (B, 256/B, dim) and the concatenate
against the (1, 1, dim) class token raises. The stock model therefore only works at batch
size 1 whenever the input resolution differs from the checkpoint's (518px). We avoid the
branch entirely: scripts/stage_dinov2_weights.py bakes the interpolated embedding into the
params using TORCH's interpolate_pos_encoding, and declares config.image_size = our input
size, so `num_patches == num_positions` and interpolate_pos_encoding early-returns. All 12
transformer blocks then run as stock, unmodified transformers code at any batch size.
Verified against the torch reference at max relative difference 5.6e-6.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from simdist.modeling.resnet import imagenet_normalize  # same ImageNet mean/std

POOLINGS = ("cls", "mean", "cls_mean")
KNOWN_VARIANTS = ("dinov2-small", "dinov2-base", "dinov2-large")


class FrozenParam(nnx.Variable):
    """Backbone weights that are never optimized.

    Deliberately NOT nnx.Param: trainer.py builds its optimizer with `wrt=nnx.Param` and
    its grads with `nnx.DiffState(0, nnx.Param)`, so FrozenParam is excluded from both
    with no trainer-side change. It is still ordinary (non-Intermediate) state, so it
    round-trips through the orbax checkpoint and a resumed run does not need the staged
    file -- at the cost of ~88 MB per checkpoint for ViT-S.
    """


class DINOv2Backbone(nnx.Module):
    """Frozen DINOv2 ViT.

    Input:  (..., H, W, 3) NHWC, either raw [0, 255] with normalize=True or already
            ImageNet-normalized. Leading dims are flattened internally, so this serves
            both the current-observation encode and the (..., n_cam, ...) target encode.
    Output: (..., out_features) pooled embedding.
    """

    def __init__(self, staged: dict[str, Any], *, img_size: int, pooling: str = "cls_mean"):
        # transformers is imported lazily so that models without a DINOv2 encoder (Go2)
        # never pay for it, mirroring how resnet.py keeps torch out of model construction.
        from transformers import Dinov2Config, FlaxDinov2Model

        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

        cfg = Dinov2Config(**staged["config"])
        if cfg.image_size != img_size:
            raise ValueError(
                f"staged weights were baked for {cfg.image_size}px inputs but the encoder "
                f"wants {img_size}px. Re-run scripts/stage_dinov2_weights.py --img-size "
                f"{img_size} (the position embedding is resolution-specific)."
            )
        if img_size % cfg.patch_size:
            raise ValueError(
                f"img_size {img_size} is not divisible by patch_size {cfg.patch_size}"
            )

        self.img_size = img_size
        self.pooling = pooling
        self.hidden_size = cfg.hidden_size
        self.out_features = 2 * cfg.hidden_size if pooling == "cls_mean" else cfg.hidden_size
        self.n_tokens = (img_size // cfg.patch_size) ** 2 + 1

        self.model = FlaxDinov2Model(
            cfg, _do_init=False, input_shape=(1, img_size, img_size, cfg.num_channels)
        )
        self.params = FrozenParam(staged["params"])

    def __call__(self, x, train: bool = False, normalize: bool = False):
        """`train` is accepted for interface parity with ResNet18Backbone and ignored: the
        backbone is frozen and has no dropout, drop-path or running statistics, so train
        and eval are the same function. That is the point of this branch."""
        del train

        lead = x.shape[:-3]
        H, W, C = x.shape[-3:]
        if (H, W) != (self.img_size, self.img_size):
            raise ValueError(
                f"DINOv2Backbone was staged for {self.img_size}x{self.img_size} inputs "
                f"(the position embedding is baked at that resolution) but got {H}x{W}."
            )
        x = x.reshape((-1, H, W, C))
        if normalize:
            x = imagenet_normalize(x)

        # FlaxDinov2Model's public __call__ takes NCHW and transposes to NHWC internally
        # (modeling_flax_dinov2.py:604). Our data is NHWC, so we transpose in; XLA folds
        # the round trip away.
        out = self.model(
            jnp.transpose(x, (0, 3, 1, 2)), params=self.params.value, train=False
        )
        tokens = out.last_hidden_state  # (B, 1 + n_patches, hidden)

        if self.pooling == "cls":
            pooled = tokens[:, 0]
        elif self.pooling == "mean":
            pooled = tokens[:, 1:].mean(axis=1)
        else:  # cls_mean -- DINOv2's own linear-probe protocol
            pooled = jnp.concatenate([tokens[:, 0], tokens[:, 1:].mean(axis=1)], axis=-1)

        # Belt and braces: FrozenParam already keeps these out of DiffState, but make the
        # intent explicit at the boundary so a future nnx.Param slip cannot leak grads.
        pooled = jax.lax.stop_gradient(pooled)
        return pooled.reshape(lead + (self.out_features,))


def load_staged_dinov2(npz_path: str) -> dict[str, Any]:
    """Read the artefact written by scripts/stage_dinov2_weights.py.

    Stored flat (numpy .npz cannot nest): params live under "param/<slash/separated/path>"
    and the Dinov2Config under a single JSON string, so the file stays inspectable with
    plain numpy and needs neither msgpack nor pickle.
    """
    import json

    blob = np.load(npz_path, allow_pickle=False)
    cfg = json.loads(str(blob["config_json"]))
    flat = {
        tuple(k[len("param/"):].split("/")): jnp.asarray(blob[k])
        for k in blob.files
        if k.startswith("param/")
    }
    if not flat:
        raise ValueError(f"{npz_path} contains no 'param/...' entries")
    return {"config": cfg, "params": unflatten_dict(flat)}


def save_staged_dinov2(npz_path: str, config: dict[str, Any], params) -> int:
    """Inverse of load_staged_dinov2. Returns the parameter count."""
    import json

    flat = flatten_dict(params)
    arrays = {"param/" + "/".join(k): np.asarray(v) for k, v in flat.items()}
    np.savez(npz_path, config_json=json.dumps(config), **arrays)
    return sum(v.size for v in arrays.values())


def get_dinov2_backbone(cfg: dict[str, Any], img_size: int, npz_path: str) -> DINOv2Backbone:
    """Build a backbone from a `model.encoder.extero_obs.dinov2` config block."""
    variant = cfg.get("variant", "dinov2-small")
    if variant not in KNOWN_VARIANTS:
        raise ValueError(f"unknown dinov2 variant {variant!r}; known: {KNOWN_VARIANTS}")
    return DINOv2Backbone(
        load_staged_dinov2(npz_path),
        img_size=img_size,
        pooling=cfg.get("pooling", "cls_mean"),
    )
