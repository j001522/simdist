from typing import Any, Tuple, TypeVar, cast
import os

from omegaconf import OmegaConf, DictConfig
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import numpy as np
import orbax.checkpoint as ocp

from simdist.data.dataset import DatasetBatch
from simdist.utils import paths
from simdist.modeling import models, pixel_decoder


T = TypeVar("T")


def load_model_from_ckpt(
    ckpt_dir: str,
    step: int | None = None,  # if none, load the latest
) -> Tuple[nnx.Module, DictConfig, int]:
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist. ")

    model_cfg = OmegaConf.load(
        os.path.join(ckpt_dir, paths.get_model_config_filename())
    )
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    dummy_scaler_params = make_dummy_scaler_params(model_cfg)
    model = models.get_model(model_cfg, dummy_scaler_params, nnx.Rngs(0))
    # Match the SAVE-side filter (trainer.py: `nnx.state(model, nnx.Not(nnx.Intermediate))`).
    # The 8 `debug_*` Intermediate scalars in encoders.py/models.py are deliberately not
    # written to disk, so including them here makes the strict structure check fail and
    # drops us into the lossy fallback below for every checkpoint written since they were
    # added (2026-07-21, 027004f).
    graphdef, model_state, intermediate_state = nnx.split(
        model, nnx.Not(nnx.Intermediate), ...
    )

    model_pure_dict = model_state.to_pure_dict()
    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as mngr:
        if step is None:
            step = mngr.latest_step()
        try:
            restored_pure_dict = mngr.restore(
                step,
                args=ocp.args.StandardRestore(item=model_pure_dict),
            )
        except ValueError as err:
            # The checkpoint was saved under a flax version whose nnx state tree
            # differs from the current one (e.g. attention rng streams nested as
            # `rngs.default.{count,key}` in flax 0.10.x vs `rngs.{count,key}` in
            # 0.12.x), which makes the strict StandardRestore structure check
            # fail. Restore the raw on-disk tree and copy only the leaves whose
            # path exists in the current model graph (all learned parameters and
            # the scaler params); leave freshly-initialised rng streams untouched
            # — they are re-seeded per run and do not affect deterministic
            # inference.
            print(
                f"WARNING: strict restore of {ckpt_dir} @ {step} failed "
                f"({type(err).__name__}: {err}); falling back to path-matched copy."
            )
            raw = mngr.restore(step, args=ocp.args.StandardRestore())
            n_copied, skipped = _copy_matching_leaves(model_pure_dict, raw)
            print(f"  fallback copied {n_copied} leaves from disk.")
            if skipped:
                # A skipped leaf keeps its `nnx.Rngs(0)` random init, which is silent and
                # ruins every downstream number. Never let that pass unnoticed.
                raise RuntimeError(
                    f"Checkpoint restore from {ckpt_dir} @ {step} is INCOMPLETE: "
                    f"{sum(n for _, n in skipped)} on-disk leaves had no matching path in "
                    f"the model state and were dropped, under: "
                    f"{', '.join('/'.join(p) for p, _ in skipped[:12])}. "
                    "Those parameters would have stayed randomly initialised."
                ) from err
            restored_pure_dict = model_pure_dict

    _replace_by_pure_dict(model_state, restored_pure_dict)
    model = nnx.merge(graphdef, model_state, intermediate_state)

    return model, model_cfg, step


def read_decoder_config(ckpt_dir: str) -> dict:
    """The decoder run's own config record, written by ``decoder_trainer.train``.

    Carries the world model it was trained against (``world_model.run_name`` and the
    ``resolved_step`` that was actually loaded), so a decoder checkpoint is enough to
    reconstruct the exact pair.
    """
    cfg_path = os.path.join(ckpt_dir, paths.get_decoder_config_filename())
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No decoder config at {cfg_path}.")
    return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)


def load_decoder_from_ckpt(
    ckpt_dir: str,
    latent_dim: int,
    n_cam: int,
    image_size: int,
    step: int | None = None,  # if none, load the latest
) -> Tuple[nnx.Module, dict, int]:
    """Restore a debug pixel decoder. The shape args come from the world model it was
    trained on -- ``decoder_config.yaml`` records ``latent_dim`` but not the camera
    count or resolution, which are properties of the system config."""
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Decoder checkpoint directory {ckpt_dir} not found.")

    dec_cfg = read_decoder_config(ckpt_dir)
    decoder = pixel_decoder.get_decoder(
        dec_cfg["decoder"],
        latent_dim=latent_dim,
        n_cam=n_cam,
        image_size=image_size,
        rngs=nnx.Rngs(0),
    )
    # Same filter the trainer saved under, so the trees match leaf for leaf.
    graphdef, decoder_state = nnx.split(decoder, nnx.Not(nnx.Intermediate))
    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as mngr:
        if step is None:
            step = mngr.latest_step()
        restored_pure_dict = mngr.restore(
            step,
            args=ocp.args.StandardRestore(item=decoder_state.to_pure_dict()),
        )

    _replace_by_pure_dict(decoder_state, restored_pure_dict)
    return nnx.merge(graphdef, decoder_state), dec_cfg, step


def _replace_by_pure_dict(model_state: nnx.State, pure_dict: dict) -> None:
    """``State.replace_by_pure_dict`` that tolerates TREE-VALUED Variables.

    nnx's own version walks the pure dict and requires every path in it to be a leaf
    path of the state. That breaks on a Variable whose *value* is a nested dict: the
    DINOv2 backbone keeps its frozen weights as one ``FrozenParam(frozen_subtree)``
    (dinov2.py -- deliberately, so `trainable_blocks=0` stays byte-identical to the
    pre-fine-tuning layout), so the state has a single leaf at
    ``encoder.dinov2.params`` while ``to_pure_dict()`` expands it into 223 paths.
    Saving is symmetric (orbax flattens the dict on the way out), restoring is not,
    and every DINOv2 checkpoint therefore failed to load with
    "key in pure_dict not available in state: ('encoder', 'dinov2', 'params',
    'embeddings', 'cls_token')".

    Driving the walk from the STATE's leaves instead of the pure dict's fixes it: a
    leaf takes whatever the pure dict holds at its path, array or subtree. Paths
    missing from the pure dict keep their freshly-initialised value, which is the same
    tolerance ``_copy_matching_leaves`` relies on for rng streams.
    """
    flat = nnx.to_flat_state(model_state)
    for path, variable in flat:
        value = pure_dict
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            continue
        variable.set_value(jax.tree.map(jnp.asarray, value))


def _copy_matching_leaves(
    dst: dict, src: dict, _prefix: Tuple[str, ...] = ()
) -> Tuple[int, list]:
    """Recursively copy leaves from ``src`` into ``dst`` where the same nested path
    exists in both. Leaves present only in ``dst`` keep their value. Returns
    ``(n_copied, skipped)`` where ``skipped`` lists ``(path, n_leaves)`` for every
    on-disk subtree that found no home -- the caller MUST treat that as an error, since
    a skipped leaf silently keeps its random init.

    Sequence indices do not survive the round trip with a consistent key type: orbax
    hands back the children of a numbered container keyed by the STRINGS "0", "1", ...
    while ``to_pure_dict()`` keys them by the INTS 0, 1, .... A plain ``k in dst`` test
    therefore missed every numbered layer, which silently dropped the whole of
    ``dynamics``/``policy``/``reward``. Match on ``str(k)``.
    """
    lookup = {str(k): k for k in dst}
    n_copied, skipped = 0, []
    for k, v in src.items():
        path = _prefix + (str(k),)
        dk = lookup.get(str(k))
        if dk is None:
            skipped.append((path, _count_leaves(v)))
            continue
        if isinstance(v, dict) and isinstance(dst[dk], dict):
            c, s = _copy_matching_leaves(dst[dk], v, path)
            n_copied += c
            skipped.extend(s)
        elif not isinstance(v, dict) and not isinstance(dst[dk], dict):
            dst[dk] = jnp.asarray(v)
            n_copied += 1
        else:
            skipped.append((path, _count_leaves(v)))
    return n_copied, skipped


def _count_leaves(v) -> int:
    if not isinstance(v, dict):
        return 1
    return sum(_count_leaves(x) for x in v.values())


def make_dummy_scaler_params(cfg: dict):
    struct = cfg["scaler_params_struct"]
    dummy_scalar_params = {}
    for k, dim in struct.items():
        dummy_scalar_params[k] = {
            "mean": jnp.zeros(dim, dtype=jnp.float32),
            "std": jnp.ones(dim, dtype=jnp.float32),
        }
    return dummy_scalar_params


def dataset_batch_to_jax(batch: DatasetBatch) -> DatasetBatch:
    return _numpy_dict_to_jax(batch)


def maybe_load_pretrained_encoder(model: nnx.Module, cfg: dict) -> None:
    """Load ImageNet-pretrained ResNet-18 weights into the encoder backbone when the
    model config asks for them (``model.encoder.extero_obs.resnet.pretrained ==
    "imagenet"``). No-op for models without a ResNet encoder (e.g. Go2). Kept out of
    ``__init__`` so model construction stays torch-free; only called for a freshly
    created model, never when resuming (the checkpoint already holds the weights).

    The DINOv2 encoder does NOT go through here: it is frozen, so its weights are read
    from the staged .npz inside ``DINOv2Backbone.__init__`` and there is no random-init
    state to overwrite. This function early-returns for it (no ``resnet`` config key)."""
    enc_cfg = cfg["model"].get("encoder", {}) or {}
    resnet_cfg = (enc_cfg.get("extero_obs", {}) or {}).get("resnet", {}) or {}
    if resnet_cfg.get("pretrained") != "imagenet":
        return
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "resnet", None)
    if backbone is None:
        return
    from simdist.modeling import resnet as resnet_mod

    print("Loading ImageNet-pretrained ResNet-18 weights into the encoder backbone...")
    resnet_mod.load_torchvision_resnet18(backbone)


def repeat_along_batch_dim(x: T, B: int) -> T:
    """Repeat the input array or dict of arrays along the batch dimension B."""
    return cast(
        T,
        jax.tree.map(lambda y: jnp.tile(y[None, ...], (B,) + (1,) * y.ndim), x),
    )


def create_param_filter(param_names: list[str]):
    filters = []
    for path in param_names:
        names = path.split("/")
        f = []
        for name in names:
            if name.isdigit():
                name = int(name)
            f.append(nnx.PathContains(name))
        f = tuple(f)
        f = nnx.All(nnx.Param, nnx.All(*f))
        filters.append(f)
    return nnx.Any(*tuple(filters))


def count_params(model: nnx.Module, filter=None):
    full_state = nnx.state(model)
    if filter is None:
        sel_state = full_state
    else:
        sel_state = full_state.filter(filter)

    def _leaf_numel(x: Any) -> int:
        # Handle JAX/NumPy arrays and Python scalars uniformly.
        try:
            return int(np.size(x))
        except Exception:
            return 0

    leaves = jax.tree_util.tree_leaves(sel_state)
    return int(sum(_leaf_numel(x) for x in leaves))


def _numpy_dict_to_jax(numpy_dict: dict[Any, np.ndarray]) -> dict[Any, jnp.ndarray]:
    """Convert a dict of numpy arrays to a dict of JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x), numpy_dict)
