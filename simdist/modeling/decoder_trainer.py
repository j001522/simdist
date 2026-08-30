"""Post-hoc training of a debug pixel decoder on a FROZEN world-model checkpoint.

The world model is loaded, never optimized, and never differentiated through: the
latents are produced by a separate jitted function (`encode_batch`) and handed to the
loss as plain arrays, so the decoder's gradient graph does not contain the encoder,
the dynamics or DINOv2 at all. That is stronger than a stop_gradient and it is what
makes this cheap -- no backbone backward, so the decoder trains at a batch size the
world-model run could not afford.

ONE target, and deliberately only one: decode ``z = E(o_t)`` into the frame at t, one
image per camera. The decoder is a renderer for a single latent -- it has no horizon,
no time axis and no knowledge of what step it is drawing. Feed it ``z_hat_{t+k}`` and
you get that latent drawn with exactly the same function that drew ``E(o_t)``.

That is what makes it a measurement. A decoder also fitted on predicted latents learns
to map a DRIFTED latent back onto the true future frame, absorbing part of the dynamics
error into itself, and then rollouts look better than the dynamics are. Cost of the
version that did this, test reconstruction at depth 64: resnet 0.00190 -> 0.00173, dino
0.00282 -> 0.00267, dinoft 0.00283 -> 0.00264 once the rollout term was removed.

Rollouts are an EVAL-time operation on top of this: roll the dynamics, decode each
latent independently, look at the strip (scripts/render_wm_rollout.py).

For the record, since this file used to cite DINO-WM for the rollout term: DINO-WM does
have both terms in ``models/visual_world_model.py``, but its ``z_pred =
self.predict(z_src)`` is a single-step, teacher-forced prediction, and its decoder is
trained jointly with the world model rather than post-hoc as a probe. Its multi-step
``rollout()`` is inference-only and never enters a loss.

The world model's own checkpoint config supplies the system, the dataset and H/T, so
the decoder can never be trained against a dataset the encoder did not see.
"""

import copy
import datetime
import os
import time

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, get_worker_info, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from simdist.data.dataset import DatasetBase, get_dataset
from simdist.modeling import pixel_decoder
from simdist.utils import config as config_utils
from simdist.utils import model as model_utils
from simdist.utils import paths


def train(cfg: dict):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg["run_name"] or f"decoder_{date_time}"

    print("Running decoder training with config: ", cfg)

    # ---- frozen world model -------------------------------------------------------
    wm_run = cfg["world_model"]["run_name"]
    if not wm_run:
        raise ValueError("world_model.run_name must be given (the checkpoint to probe)")
    wm_ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), wm_run)
    model, wm_cfg, wm_step = model_utils.load_model_from_ckpt(
        wm_ckpt_dir, cfg["world_model"]["step"]
    )
    print(f"Loaded world model '{wm_run}' at step {wm_step} from {wm_ckpt_dir}")

    latent_dim = wm_cfg["model"]["latent_dim"]
    image_shapes = config_utils.extero_obs_image_shapes_from_sys_config(
        wm_cfg["system"]
    )
    n_cam = len(image_shapes)
    img_h, img_w, channels = image_shapes[0]
    if img_h != img_w:
        raise ValueError(f"decoder assumes square images, got {img_h}x{img_w}")
    print(
        f"  latent_dim={latent_dim} n_cam={n_cam} image={img_h}x{img_w}x{channels} "
        f"encoder={'dinov2' if 'dinov2' in wm_cfg['model']['encoder']['extero_obs'] else 'resnet'}"
    )

    # ---- dataset: taken from the checkpoint's own config ---------------------------
    ds_cfg = dataset_cfg_from_world_model(cfg, wm_cfg)
    dataset = get_dataset(ds_cfg)
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_dataset, test_dataset = random_split(
        dataset,
        [
            cfg["training"]["training_data_ratio"],
            1 - cfg["training"]["training_data_ratio"],
        ],
        generator=generator,
    )

    training = True

    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        if worker_info is not None:
            ds: DatasetBase = worker_info.dataset.dataset
            ds.train() if training else ds.eval()

    def set_dataset_mode(train_mode: bool) -> None:
        nonlocal training
        training = train_mode
        dataset.train() if train_mode else dataset.eval()

    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )
    train_set = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=cfg["data"]["num_train_workers"],
        **loader_kwargs,
    )
    test_set = DataLoader(
        test_dataset, num_workers=cfg["data"]["num_test_workers"], **loader_kwargs
    )

    # ---- decoder ------------------------------------------------------------------
    decoder = pixel_decoder.get_decoder(
        cfg["decoder"],
        latent_dim=latent_dim,
        n_cam=n_cam,
        image_size=img_h,
        rngs=nnx.Rngs(cfg["training"]["seed"]),
    )
    print(f"Decoder parameters: {model_utils.count_params(decoder, nnx.Param)}")
    print(
        f"  transposed-conv stack reaches {decoder.pre_resize_size}px, "
        f"resized to {img_h}px"
    )

    # ---- checkpointing ------------------------------------------------------------
    ckpt_dir = os.path.join(paths.get_decoder_checkpoints_dir(), run_name)
    sample_dir = os.path.join(ckpt_dir, "samples")
    if cfg["checkpoint"]["enabled"]:
        os.makedirs(ckpt_dir, exist_ok=True)
        record = copy.deepcopy(cfg)
        record["world_model"]["resolved_step"] = int(wm_step)
        record["world_model"]["latent_dim"] = int(latent_dim)
        with open(os.path.join(ckpt_dir, paths.get_decoder_config_filename()), "w") as f:
            yaml.dump(record, f, default_flow_style=False)
        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg["checkpoint"]["max_to_keep"],
            keep_period=cfg["checkpoint"].get("keep_period"),
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=False,
        )
    os.makedirs(sample_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(paths.get_decoder_checkpoints_dir(), run_name, "tensorboard")
    )

    # ---- optimizer (decoder only) --------------------------------------------------
    train_cfg = cfg["training"]
    total_steps = train_cfg["max_steps"] or (train_cfg["num_epochs"] * len(train_set))
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg["learning_rate"],
        warmup_steps=int(train_cfg["warmup_ratio"] * total_steps),
        decay_steps=total_steps,
        end_value=train_cfg["end_learning_rate"],
    )
    tx = optax.adam(lr_sched)
    if train_cfg.get("grad_clip_norm"):
        tx = optax.chain(optax.clip_by_global_norm(train_cfg["grad_clip_norm"]), tx)
    optimizer = nnx.Optimizer(decoder, tx, wrt=nnx.Param)

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    metrics.reset()

    # ---- steps ---------------------------------------------------------------------
    @nnx.jit
    def encode_batch(model, x):
        """Frozen forward pass: o_t -> z. Returned as a constant for the decoder loss."""
        return jax.lax.stop_gradient(model.encode_context(x)["latent"])

    def _loss(decoder, z, img):
        """One latent -> one frame per camera. No horizon, no time axis."""
        return jnp.mean((decoder(z) - img) ** 2)

    @nnx.jit
    def train_step(decoder, optimizer, metrics, z, img):
        loss, grads = nnx.value_and_grad(_loss)(decoder, z, img)
        metrics.update(loss=loss)
        optimizer.update(decoder, grads)
        return loss

    @nnx.jit
    def eval_step(decoder, metrics, z, img):
        loss = _loss(decoder, z, img)
        metrics.update(loss=loss)
        return loss

    def prepare(batch):
        """Batch -> (z, img), images as float in [0, 1].

        ``labels`` (the future frames) is never touched: it is (B, T, n_cam, H, W, 3),
        ~1.4 GB as float32 at batch 64, and this loss has no use for it.
        """
        batch = model_utils.dataset_batch_to_jax(batch)
        x = batch["model_in"]
        return encode_batch(model, x), _to_unit(x["extero_obs"])  # (B, n_cam, H, W, C)

    print("Starting decoder training")
    train_steps = 0
    metrics_dict = {}
    interval_start_time = time.time()
    stop = False
    for epoch in range(train_cfg["num_epochs"]):
        print(f"Starting epoch {epoch + 1}/{train_cfg['num_epochs']}")
        epoch_start_time = time.time()
        set_dataset_mode(True)
        pbar = tqdm(train_set, desc="Training", unit="batch")
        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            loss = train_step(decoder, optimizer, metrics, *prepare(batch))
            pbar.set_description(f"Training (Loss: {float(loss):.5f})")
            train_steps += 1

            is_last = i == len(train_set) - 1 and epoch == train_cfg["num_epochs"] - 1
            stop = train_cfg["max_steps"] is not None and (
                train_steps >= train_cfg["max_steps"]
            )
            due = train_steps % train_cfg["eval_interval"] == 0
            if not (due or is_last or stop):
                continue

            metrics_dict["steps_per_second"] = train_cfg["eval_interval"] / (
                time.time() - interval_start_time
            )
            for k, v in metrics.compute().items():
                metrics_dict[f"train/{k}"] = float(v)
            metrics.reset()

            # eval. Capped by max_eval_batches: the test split is 1% of ~2.2M windows,
            # so a full pass is ~345 batches at batch 64 and would cost a third of the
            # run for a reconstruction MSE that is already stable after ~100.
            set_dataset_mode(False)
            last_eval_batch = None
            max_eval = train_cfg.get("max_eval_batches")
            pbar_test = tqdm(test_set, desc="Testing", unit="batch", total=max_eval)
            for n_eval, test_batch in enumerate(pbar_test):
                if max_eval is not None and n_eval >= max_eval:
                    break
                if test_batch is None:
                    continue
                last_eval_batch = prepare(test_batch)
                loss = eval_step(decoder, metrics, *last_eval_batch)
                pbar_test.set_description(f"Testing (Loss: {float(loss):.5f})")
            pbar_test.close()
            set_dataset_mode(True)

            for k, v in metrics.compute().items():
                metrics_dict[f"test/{k}"] = float(v)
            metrics.reset()
            metrics_dict["epoch"] = epoch

            # The whole point of a debug decoder is looking at it.
            if last_eval_batch is not None:
                _save_samples(
                    decoder,
                    last_eval_batch,
                    os.path.join(sample_dir, f"step_{train_steps:07d}.png"),
                    n_rows=cfg["samples"]["num_rows"],
                )
                last_eval_batch = None

            if cfg["checkpoint"]["enabled"]:
                state = nnx.state(decoder, nnx.Not(nnx.Intermediate))
                with ocp.CheckpointManager(ckpt_dir, options=ckpt_options) as mngr:
                    mngr.save(
                        train_steps,
                        args=ocp.args.StandardSave(state.to_pure_dict()),
                        metrics=metrics_dict,
                    )

            for k, v in metrics_dict.items():
                tb_writer.add_scalar(k, v, train_steps)
            print(f"Steps: {train_steps}, Metrics: {metrics_dict}")
            metrics_dict = {}
            interval_start_time = time.time()

            if stop:
                print(f"Stopping decoder training after {train_steps} steps.")
                break

        if stop:
            break
        print(
            f"Epoch {epoch + 1}/{train_cfg['num_epochs']} completed in "
            f"{time.time() - epoch_start_time:.2f} seconds."
        )

    tb_writer.close()
    print(f"Samples written to {sample_dir}")


def dataset_cfg_from_world_model(cfg: dict, wm_cfg: dict) -> dict:
    """Dataset config for the decoder run, derived from the world model's own config.

    Everything that decides WHAT the encoder sees (system, model.dataset, H/T) comes
    from the checkpoint; only the loader knobs come from the decoder config. Image
    augmentation is off by default: we are probing what z retains about the actual
    scene, and jittered/blurred/cropped targets only add noise to that question.
    """
    ds_cfg = copy.deepcopy(wm_cfg)
    if cfg["data"]["dataset_name"]:
        ds_cfg["data"]["dataset_name"] = cfg["data"]["dataset_name"]
    ds_cfg["data"]["num_train_workers"] = cfg["data"]["num_train_workers"]
    ds_cfg["data"]["num_test_workers"] = cfg["data"]["num_test_workers"]
    ds_cfg["data"]["prefetch_factor"] = cfg["data"]["prefetch_factor"]
    ds_cfg["training"]["batch_size"] = cfg["training"]["batch_size"]
    ds_cfg["training"]["seed"] = cfg["training"]["seed"]
    if not cfg["data"]["augment"]:
        augs = ds_cfg["model"]["dataset"]["augmentations"]
        augs["add_noise"] = False
        augs["image"] = {}
    return ds_cfg


def _to_unit(imgs: jnp.ndarray) -> jnp.ndarray:
    """uint8 frames -> float32 in [0, 1]."""
    return imgs.astype(jnp.float32) / 255.0


def _save_samples(decoder, prepared, out_path: str, n_rows: int) -> None:
    """Write a target/reconstruction contact sheet.

    Layout: one image row per sample, cameras left to right, and for each sample the
    ground-truth strip above its reconstruction. Rollouts are not part of training and
    so are not part of this sheet -- render them with scripts/render_wm_rollout.py.
    """
    z, img = prepared
    n = min(n_rows, int(img.shape[0]))

    strips = [np.asarray(img[:n]), np.asarray(decoder(z[:n]))]

    rows = []
    for i in range(n):
        for strip in strips:
            frames = np.clip(strip[i], 0.0, 1.0)  # (n_cam, H, W, C)
            rows.append(np.concatenate(list(frames), axis=1))
    sheet = np.concatenate(rows, axis=0)
    sheet = (sheet * 255.0).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
