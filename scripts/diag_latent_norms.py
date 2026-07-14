"""Diagnostic: is the latent-dynamics target unstable, and is BatchNorm the cause?

Trains the world model exactly as trainer.train() does (same model, loss, optimizer,
data), but each step also records:

  * latent_dynamics loss                       -- the term that explodes in real runs
  * ||z|| of the TARGET latents, encoded in eval mode (BN running averages), which is
    how losses.py computes them (deterministic=True)
  * ||z|| of the same observations encoded in train mode (BN batch statistics), which
    is how the prediction path sees them
  * min/max of every ResNet BatchNorm's running variance

Hypothesis under test: BN running stats drift away from the batch statistics; if a
running variance collapses toward zero, the eval-mode pass divides by ~sqrt(eps) and
the TARGET latent explodes, so the MSE explodes on that batch while the model itself
is fine. If true: ||z||_eval diverges from ||z||_train, and min(running_var) -> 0,
both correlating with the latent_dynamics spikes.

If instead both norms grow together and the running variances stay healthy, the cause
is an unbounded latent chasing a moving target, and BN is exonerated.

Output: CSV lines prefixed with "DIAG," (grep-friendly) to stdout.
"""

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, get_worker_info, random_split

from simdist.utils.jax import configure_jax_compilation_cache

configure_jax_compilation_cache()

from simdist.data.dataset import get_dataset
from simdist.modeling import losses, models
from simdist.utils import io, model as model_utils
from simdist.utils.paths import get_train_model_hydra_config


def _bn_running_stats(model):
    """min/max running variance and max |running mean| over all BatchNorm layers."""
    vars_, means_ = [], []
    for _, node in nnx.iter_graph(model):
        if isinstance(node, nnx.BatchNorm):
            vars_.append(np.asarray(node.var.value))
            means_.append(np.asarray(node.mean.value))
    if not vars_:
        return None
    return {
        "bn_var_min": float(min(v.min() for v in vars_)),
        "bn_var_max": float(max(v.max() for v in vars_)),
        "bn_mean_absmax": float(max(np.abs(m).max() for m in means_)),
        "bn_layers": len(vars_),
    }


def _token_norm(t):
    """mean/max per-token L2 norm, flattening every leading dim."""
    n = jnp.linalg.norm(t.reshape(-1, t.shape[-1]), axis=-1)
    return float(jnp.mean(n)), float(jnp.max(n))


def _latent_norms(model, x, y_scaled):
    """||z|| for the future obs, encoded in eval mode (the loss's target) and train mode.

    Train-mode encoding is done on a detached copy: BatchNorm mutates its running stats
    in place when use_running_average=False, and the diagnostic must not perturb the
    very training run it is measuring.

    Also reports the HISTORY token norms. The LayerNorm only bounds the latent token;
    the history path (proprio_obs_proj / act_proj) is still unnormalized, and both feed
    the same dynamics cross-attention. If the history tokens drift while ||z|| is pinned,
    the instability has moved rather than been fixed.
    """
    proprio, extero = y_scaled["proprio_obs"], y_scaled["extero_obs"]

    z_eval = model.encode_latent(proprio, extero, deterministic=True)

    model_copy = nnx.merge(*nnx.split(model))
    z_train = model_copy.encode_latent(proprio, extero, deterministic=False)

    # history tokens + the latent token as the dynamics context actually sees them
    enc = nnx.merge(*nnx.split(model)).encoder(x, deterministic=True)
    hist_mean, hist_max = _token_norm(enc["history"])
    ctx_z_mean, _ = _token_norm(enc["latent"])

    z_eval_mean, z_eval_max = _token_norm(z_eval)
    z_train_mean, z_train_max = _token_norm(z_train)
    return {
        "z_eval_mean": z_eval_mean,
        "z_eval_max": z_eval_max,
        "z_train_mean": z_train_mean,
        "z_train_max": z_train_max,
        "z_absmax": float(jnp.max(jnp.abs(z_eval))),
        "hist_tok_mean": hist_mean,
        "hist_tok_max": hist_max,
        "ctx_z_mean": ctx_z_mean,
    }


@hydra.main(**get_train_model_hydra_config())
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    steps = int(cfg["training"].get("max_steps") or 600)

    dataset = get_dataset(cfg)
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_dataset, _ = random_split(
        dataset,
        [
            cfg["training"]["training_data_ratio"],
            1 - cfg["training"]["training_data_ratio"],
        ],
        generator=generator,
    )
    # mirrors trainer.train(): workers put their dataset copy in train mode so the same
    # image augmentations are applied as in a real run
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_info.dataset.dataset.train()

    train_set = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        shuffle=True,
        num_workers=cfg["data"]["num_train_workers"],
        worker_init_fn=worker_init_fn,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )

    scaler_params = io.load_scaler_params(dataset.data_dir)
    model = models.get_model(
        cfg, scaler_params, rngs=nnx.Rngs(cfg["training"]["seed"])
    )
    model_utils.maybe_load_pretrained_encoder(model, cfg)

    loss_fn = losses.get_loss(cfg)
    heads = cfg["heads"]
    filt = nnx.Param if len(heads) == 0 else model_utils.create_param_filter(heads)
    diff_state = nnx.DiffState(0, filt)

    train_cfg = cfg["training"]
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg["learning_rate"],
        warmup_steps=max(1, int(0.05 * steps)),
        decay_steps=steps,
        end_value=train_cfg["end_learning_rate"],
    )
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    tx = (
        optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr_sched))
        if grad_clip_norm is not None
        else optax.adam(lr_sched)
    )
    optimizer = nnx.Optimizer(model, tx, wrt=filt)

    @nnx.jit
    def train_step(model, optimizer, x, y):
        grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
        (loss, losses_), grads = grad_fn(model, x, y, False)
        gnorm = optax.global_norm(grads)
        optimizer.update(model, grads)
        return loss, losses_, gnorm

    cols = [
        "step", "loss", "latent_dynamics", "grad_norm",
        "z_eval_mean", "z_eval_max", "z_train_mean", "z_train_max", "z_absmax",
        "hist_tok_mean", "hist_tok_max", "ctx_z_mean",
        "bn_var_min", "bn_var_max", "bn_mean_absmax",
    ]
    print("DIAG," + ",".join(cols), flush=True)

    step = 0
    for batch in train_set:
        if batch is None:
            continue
        batch = model_utils.dataset_batch_to_jax(batch)
        x, y = batch["model_in"], batch["labels"]

        loss, losses_, gnorm = train_step(model, optimizer, x, y)

        # measure the target latents the loss actually regresses onto
        y_scaled = model.get_scaler().scale(y)
        row = {"step": step, "loss": float(loss),
               "latent_dynamics": float(losses_["latent_dynamics"]),
               "grad_norm": float(gnorm)}
        row.update(_latent_norms(model, x, y_scaled))
        bn = _bn_running_stats(model)
        row.update(bn if bn else {"bn_var_min": np.nan, "bn_var_max": np.nan,
                                  "bn_mean_absmax": np.nan})

        print("DIAG," + ",".join(f"{row[c]:.6g}" if c != "step" else str(row[c])
                                 for c in cols), flush=True)

        step += 1
        if step >= steps:
            break

    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
