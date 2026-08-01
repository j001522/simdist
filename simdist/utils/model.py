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
from simdist.modeling import models


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
    graphdef, model_state = nnx.split(model)

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
        except ValueError:
            # The checkpoint was saved under a flax version whose nnx state tree
            # differs from the current one (e.g. attention rng streams nested as
            # `rngs.default.{count,key}` in flax 0.10.x vs `rngs.{count,key}` in
            # 0.12.x), which makes the strict StandardRestore structure check
            # fail. Restore the raw on-disk tree and copy only the leaves whose
            # path exists in the current model graph (all learned parameters and
            # the scaler params); leave freshly-initialised rng streams untouched
            # — they are re-seeded per run and do not affect deterministic
            # inference.
            raw = mngr.restore(step, args=ocp.args.StandardRestore())
            _copy_matching_leaves(model_pure_dict, raw)
            restored_pure_dict = model_pure_dict

    # FrozenParam (the DINOv2 backbone) holds an entire param TREE as the value of
    # ONE variable. `to_pure_dict` expands that into nested dicts, but
    # `replace_by_pure_dict` walks the State, where it is a single leaf -- so the
    # expanded paths are rejected ("key in pure_dict not available in state") under
    # flax 0.12.7. Detach the subtree here and assign it straight onto the variable
    # after the merge, which is what the expanded form meant in the first place.
    frozen_trees = _detach_frozen_trees(model, restored_pure_dict)

    model_state.replace_by_pure_dict(restored_pure_dict)
    model = nnx.merge(graphdef, model_state)

    for get_var, subtree in frozen_trees:
        get_var(model).value = jax.tree.map(jnp.asarray, subtree)

    return model, model_cfg, step


def _detach_frozen_trees(model, pure: dict):
    """Pop tree-valued Variable subtrees out of ``pure`` before a strict replace.

    Returns [(accessor, subtree)] where ``accessor(model)`` yields the Variable to
    assign after ``nnx.merge``. Only the DINOv2 backbone uses this shape today;
    ResNet checkpoints have no tree-valued variables and are untouched.
    """
    out = []
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "dinov2", None) if encoder is not None else None
    if backbone is None or not isinstance(getattr(backbone, "params", None), nnx.Variable):
        return out
    node = pure.get("encoder", {}).get("dinov2")
    if not isinstance(node, dict) or not isinstance(node.get("params"), dict):
        return out
    out.append((lambda m: m.encoder.dinov2.params, node.pop("params")))
    if not node:
        pure["encoder"].pop("dinov2")
    return out


def _copy_matching_leaves(dst: dict, src: dict) -> None:
    """Recursively copy leaves from ``src`` into ``dst`` where the same nested
    path exists in both. Leaves present only in ``dst`` keep their value; leaves
    present only in ``src`` are ignored."""
    for k, v in src.items():
        if k not in dst:
            continue
        if isinstance(v, dict) and isinstance(dst[k], dict):
            _copy_matching_leaves(dst[k], v)
        elif not isinstance(v, dict) and not isinstance(dst[k], dict):
            dst[k] = jnp.asarray(v)


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
