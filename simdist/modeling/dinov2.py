"""DINOv2 image backbone: a thin nnx wrapper around transformers' FlaxDinov2Model.

Drop-in replacement for ResNet18Backbone in ManipulationEncoder -- same call signature and
the same "(..., H, W, 3) -> (..., out_features)" contract, so encode_latent changes only in
which backbone it holds.

Frozen by default (`trainable_blocks: 0`): the image features become a fixed function of
the pixels, and it follows the Newt recipe (Hansen et al., 2025) of concatenating a single
global DINOv2 embedding with proprioception into an MLP state encoder. DINO-WM instead
keeps the full patch-token grid, because it has no reward/value head to carry the task
signal; simdist does, so one vector per camera is the consistent choice here.

    !! fine-tuning needs NO port of DINOv2 to nnx !!
FlaxDinov2Model is a linen FlaxPreTrainedModel, but we never use its variable management:
its forward is a pure function of an externally supplied params dict
(`self.module.apply({"params": params or self.params}, ...)`, modeling_flax_dinov2.py:613),
and we already hold and pass those params ourselves. JAX differentiates straight through
it. Partial fine-tuning is therefore only a question of which nnx Variable *class* holds
which slice of the tree -- see split_staged_params and BackboneParam.

`trainable_blocks: K` adapts the last K transformer blocks plus the final layernorm; set
`lr_mult` in the same config block, because a ViT adapted at the head LR is destroyed
within a few hundred steps. Either way train == eval bit-exactly (all dropout rates are
0.0), so this backbone never has the BatchNorm-style train/eval gap the ResNet arm has.

Pooling. A ViT emits a token sequence, not a spatial map, so there is no single "the"
feature vector:
  cls       -- the global summary token (hidden_size). What Newt and LeWM (Maes et al.,
               2026) both actually use -- neither concatenates the patch mean.
  mean      -- mean over patch tokens (hidden_size)
  cls_mean  -- both, concatenated (2 * hidden_size); DINOv2's own linear-probe protocol
               and the default here.
The `mean` half is permutation-invariant over patches, so it discards absolute spatial
layout by construction; CLS is not, since it attends over position-embedded patches. If a
task needs image-space geometry (plug vs. port offset), that asymmetry matters and `cls`
is the better-attested choice.

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

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from simdist.modeling.resnet import imagenet_normalize  # same ImageNet mean/std

POOLINGS = ("cls", "mean", "cls_mean")
KNOWN_VARIANTS = ("dinov2-small", "dinov2-base", "dinov2-large")

# Joins a staged param path into a single dict key for `trainable_params`. "/" is safe:
# it never occurs inside a FlaxDinov2Model param name (see the staged .npz, which already
# uses "/" as its own flat separator).
PATH_SEP = "/"


class FrozenParam(nnx.Variable):
    """Backbone weights that are never optimized.

    Deliberately NOT nnx.Param: trainer.py builds its optimizer with `wrt=nnx.Param` and
    its grads with `nnx.DiffState(0, nnx.Param)`, so FrozenParam is excluded from both
    with no trainer-side change. It is still ordinary (non-Intermediate) state, so it
    round-trips through the orbax checkpoint and a resumed run does not need the staged
    file -- at the cost of ~88 MB per checkpoint for ViT-S.
    """


class BackboneParam(nnx.Param):
    """Backbone weights that ARE optimized (partial fine-tuning, `trainable_blocks > 0`).

    An nnx.Param subclass, so `wrt=nnx.Param` / `DiffState(0, nnx.Param)` pick it up with
    no trainer change. It is a distinct *class* only so trainer.py can give the backbone
    its own (much lower) learning rate -- see the `dinov2` label in `_backbone_label`.
    """


def _as_plain_dict(tree: Any) -> Any:
    """Normalize a Variable's nested-dict value back to plain dicts.

    A Variable whose value is a nested dict does NOT keep that dict type through nnx's
    transforms: once the Variable is part of a differentiated state under `nnx.jit`, nnx
    hands the value back as a `flax.nnx.statelib.State`, and flax's `flatten_dict`
    asserts on (frozen)dict. Outside a transform (and for FrozenParam, which is never
    differentiated) it stays a plain dict, so this has to accept both. State is a Mapping,
    so one recursive walk covers every case; leaves are the same arrays/tracers either
    way, and gradients are unaffected by the container swap.
    """
    if isinstance(tree, Mapping):
        return {k: _as_plain_dict(v) for k, v in tree.items()}
    return tree


def split_staged_params(
    params: dict[str, Any], *, num_layers: int, trainable_blocks: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition the staged param tree into (frozen, trainable) by transformer block.

    Fine-tunes the LAST `trainable_blocks` blocks plus the final layernorm; `embeddings`
    (patch projection, CLS token, the baked position embedding) and the earlier blocks
    stay frozen. This is the standard ViT partial-adaptation recipe: early blocks encode
    generic low-level structure, late blocks carry the semantics that need to move.

    Freezing the early blocks is also what makes fine-tuning fit in memory, and it does
    so *automatically*: their output depends only on the input pixels and on FrozenParam
    values, neither of which is in the DiffState, so JAX's partial evaluation classifies
    the whole prefix as primal-only and stages no residuals for it. No explicit
    stop_gradient at the block boundary is needed (nor possible -- `module.apply` runs
    all 12 blocks in one opaque call).
    """
    if not 0 <= trainable_blocks <= num_layers:
        raise ValueError(
            f"trainable_blocks must be in [0, {num_layers}], got {trainable_blocks}"
        )
    first_trainable = num_layers - trainable_blocks

    def _is_trainable(path: tuple[str, ...]) -> bool:
        if trainable_blocks == 0:
            return False
        if path[0] == "layernorm":
            return True
        if len(path) >= 3 and path[0] == "encoder" and path[1] == "layer":
            return int(path[2]) >= first_trainable
        return False

    flat = flatten_dict(params)
    frozen = {k: v for k, v in flat.items() if not _is_trainable(k)}
    trainable = {k: v for k, v in flat.items() if _is_trainable(k)}
    return unflatten_dict(frozen), unflatten_dict(trainable)


class DINOv2Backbone(nnx.Module):
    """Frozen DINOv2 ViT.

    Input:  (..., H, W, 3) NHWC, either raw [0, 255] with normalize=True or already
            ImageNet-normalized. Leading dims are flattened internally, so this serves
            both the current-observation encode and the (..., n_cam, ...) target encode.
    Output: (..., out_features) pooled embedding.
    """

    def __init__(
        self,
        staged: dict[str, Any],
        *,
        img_size: int,
        pooling: str = "cls_mean",
        trainable_blocks: int = 0,
    ):
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

        self.trainable_blocks = trainable_blocks
        frozen, trainable = split_staged_params(
            staged["params"],
            num_layers=cfg.num_hidden_layers,
            trainable_blocks=trainable_blocks,
        )
        # `params` keeps its name and holds the whole tree when fully frozen, so
        # trainable_blocks=0 is byte-identical to the pre-fine-tuning branch and existing
        # orbax checkpoints still restore. trainable_blocks>0 is a different state layout
        # (an extra `trainable_params` node) and deliberately cannot resume from those.
        #
        # One Variable per TENSOR, not one Variable holding the whole subtree. A
        # tree-valued Variable does not survive optax: `optax.multi_transform` masks the
        # leaves it is not responsible for with `MaskedNode()` and rebuilds the container
        # as a plain dict, which then fails to match the `State` that nnx turned the
        # Variable's value into -- "Custom node type mismatch: expected State, value
        # {...MaskedNode()...}". Flat leaves are what every other param in the model looks
        # like, so masking, checkpointing and param counting all work natively.
        # Keys are sorted so the state layout is stable across runs.
        self.params = FrozenParam(frozen)
        # nnx.data(...): a bare dict attribute is treated as a STATIC attribute and nnx
        # refuses to hold arrays in one ("Found data on value of type dict assigned to
        # static attribute"). nnx.data marks it as pytree data instead.
        self.trainable_params = (
            nnx.data(
                {
                    PATH_SEP.join(path): BackboneParam(arr)
                    for path, arr in sorted(flatten_dict(trainable).items())
                }
            )
            if trainable_blocks
            else None
        )

    def __call__(self, x, train: bool = False, normalize: bool = False):
        """`train` is accepted for interface parity with ResNet18Backbone and ignored.

        DINOv2 is LayerNorm-only, and `hidden_dropout_prob`, `attention_probs_dropout_prob`
        and `drop_path_rate` are all 0.0 in the checkpoint config (FlaxDinov2DropPath
        early-returns at rate 0.0). So train and eval are the same function whether or not
        the backbone is being fine-tuned -- unlike the ResNet arm, this branch has no
        train/eval gap to reason about. We therefore always call with train=False and
        never need a dropout rng.
        """
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
        # module.apply is a pure function of the params dict handed to it, so gradients
        # flow into whichever half of the tree is a BackboneParam. The two halves are
        # recombined here rather than held as one Variable so that only the trainable
        # half enters the DiffState (see split_staged_params).
        out = self.model(
            jnp.transpose(x, (0, 3, 1, 2)), params=self._merged_params(), train=False
        )
        tokens = out.last_hidden_state  # (B, 1 + n_patches, hidden)

        if self.pooling == "cls":
            pooled = tokens[:, 0]
        elif self.pooling == "mean":
            pooled = tokens[:, 1:].mean(axis=1)
        else:  # cls_mean -- DINOv2's own linear-probe protocol
            pooled = jnp.concatenate([tokens[:, 0], tokens[:, 1:].mean(axis=1)], axis=-1)

        if not self.trainable_blocks:
            # Belt and braces: FrozenParam already keeps these out of DiffState, but make
            # the intent explicit at the boundary so a future nnx.Param slip cannot leak
            # grads. Omitted when fine-tuning, which needs exactly that gradient.
            pooled = jax.lax.stop_gradient(pooled)
        return pooled.reshape(lead + (self.out_features,))

    def _merged_params(self) -> dict[str, Any]:
        """Recombine the frozen and trainable halves into the tree module.apply wants.

        Pure Python tree surgery on traced arrays, so it costs nothing at runtime -- it
        happens once per jit trace, not once per step.
        """
        frozen = _as_plain_dict(self.params.value)
        if self.trainable_params is None:
            return frozen
        merged = flatten_dict(frozen)
        for key, var in self.trainable_params.items():
            merged[tuple(key.split(PATH_SEP))] = var.value
        return unflatten_dict(merged)


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
        trainable_blocks=int(cfg.get("trainable_blocks", 0) or 0),
    )
