import datetime
import os
import yaml
import time

import wandb
import torch
from torch.utils.data import random_split, get_worker_info, DataLoader
from torch.utils.tensorboard import SummaryWriter
import flax.nnx as nnx
import orbax.checkpoint as ocp
import optax
from tqdm import tqdm

from simdist.data.dataset import get_dataset, DatasetBase
from simdist.utils import io, model as model_utils, paths
from simdist.modeling import models, losses, types


def train(cfg: dict):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{date_time}" if cfg["run_name"] is None else cfg["run_name"]
    finetuning = "finetune" in cfg and cfg["finetune"]
    train_steps = 0

    max_steps = None
    if "max_steps" in cfg["training"] and cfg["training"]["max_steps"] is not None:
        if cfg["training"]["max_steps"] > 0:
            max_steps = cfg["training"]["max_steps"]
            print(f"Training will stop after {max_steps} steps.")

    # start wandb
    if cfg["wandb"]["log"]:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=run_name,
            config=cfg,
        )
    print("Running training with config: ", cfg)

    # TensorBoard runs independently of wandb (useful when wandb.log=false, e.g. on Snellius).
    tb_dir = os.path.join(paths.get_model_checkpoints_dir(), run_name, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)

    # Create datasets
    dataset = get_dataset(cfg)
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_dataset, test_dataset = random_split(
        dataset,
        [
            cfg["training"]["training_data_ratio"],
            1 - cfg["training"]["training_data_ratio"],
        ],
        generator=generator,
    )

    # holds the current training mode
    training = True

    # used to update each worker's dataset each time all of the data is seen
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset: DatasetBase = worker_info.dataset.dataset
            if training:
                dataset.train()
            else:
                dataset.eval()

    def set_dataset_mode(train_mode: bool) -> None:
        """Switch augmentation on (train) / off (eval) for subsequently spawned workers.

        worker_init_fn reads `training` when a DataLoader spawns its workers, which with
        persistent_workers=False happens on every fresh iteration of that loader. So the
        flag must be True whenever the train loader starts an epoch and False for the eval
        pass -- hence the restore after the test loop below. Workers already running are
        untouched (separate processes; they captured the flag at spawn), so flipping this
        mid-epoch cannot disturb the in-flight train loader.

        The direct train()/eval() call covers num_workers=0, where worker_init_fn never
        runs at all and the loader reads this very object.
        """
        nonlocal training
        training = train_mode
        if train_mode:
            dataset.train()
        else:
            dataset.eval()

    # create dataloaders
    train_set = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        shuffle=True,
        num_workers=cfg["data"]["num_train_workers"],
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_to_numpy_safe,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )
    test_set = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        num_workers=cfg["data"]["num_test_workers"],
        worker_init_fn=worker_init_fn,
        # collate_fn=collate_to_numpy_safe,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )

    # get the model
    ckpt_dir = None
    if (
        "resume_checkpoint" in cfg["checkpoint"]
        and cfg["checkpoint"]["resume_checkpoint"] is not None
    ):
        # load model from checkpoint if specified
        print("Resuming from checkpoint...")
        resume_checkpoint = cfg["checkpoint"]["resume_checkpoint"]
        ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), resume_checkpoint)
        model, model_cfg, train_steps = model_utils.load_model_from_ckpt(ckpt_dir)
        cfg["model"] = model_cfg["model"]  # use the model cfg from the checkpoint
        if "scaler_params_struct" in model_cfg:
            # If the model cfg has a scaler_params_struct, use it
            cfg["scaler_params_struct"] = model_cfg["scaler_params_struct"]
        if max_steps is not None:
            max_steps += train_steps
        if finetuning:
            print("Finetuning the model...")
    elif finetuning:
        raise ValueError("checkpoint.resume_checkpoint must be given when finetuning.")
    else:
        # otherwise, create a new model
        print("Creating a new model...")
        scaler_params = io.load_scaler_params(dataset.data_dir)
        model = models.get_model(
            cfg, scaler_params, rngs=nnx.Rngs(cfg["training"]["seed"])
        )
        # Vision encoders (manip) start from ImageNet weights; no-op otherwise.
        model_utils.maybe_load_pretrained_encoder(model, cfg)

        # get scaler params structure to store it later
        struct = {}
        for k, v in scaler_params.items():
            struct[k] = len(v["mean"])
        cfg["scaler_params_struct"] = struct

    # Setup checkpointing with Orbax
    if cfg["checkpoint"]["enabled"]:
        # keep_period is optional and absent from configs written before it existed
        # (resuming reads the checkpoint's own cfg), so read it defensively.
        keep_period = cfg["checkpoint"].get("keep_period")
        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg["checkpoint"]["max_to_keep"],
            keep_period=keep_period,
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=False,
        )
        if ckpt_dir is None or finetuning:
            ckpt_dir = os.path.join(paths.get_model_checkpoints_dir(), run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(
                os.path.join(ckpt_dir, paths.get_model_config_filename()), "w"
            ) as f:
                yaml.dump(cfg, f, default_flow_style=False)

    # Get the loss function
    loss_fn = losses.get_loss(cfg)

    # Determine trainable parameters
    heads = cfg["heads"]
    if len(heads) == 0:
        filt = nnx.Param
    else:
        filt = model_utils.create_param_filter(heads)
    diff_state = nnx.DiffState(0, filt)
    print(f"Parameters to be optimized: {model_utils.count_params(model, filt)}")
    print(f"Total trainable parameters: {model_utils.count_params(model, nnx.Param)}")
    print(f"Total parameters: {model_utils.count_params(model)}")

    # set up optimizer and metrics
    train_cfg = cfg["training"]
    # Absolute warmup_steps/decay_steps are sized for the full (massive) dataset. On a
    # smaller dataset the whole run can finish inside `warmup_steps`, so the LR never
    # ramps or decays. `warmup_ratio`/`decay_ratio` (if set) express the schedule as a
    # fraction of the ACTUAL total steps so it adapts to the dataset/epoch count.
    total_steps = max_steps or (train_cfg["num_epochs"] * len(train_set))
    warmup_ratio = train_cfg.get("warmup_ratio")
    decay_ratio = train_cfg.get("decay_ratio")
    warmup_steps = (
        int(warmup_ratio * total_steps)
        if warmup_ratio is not None
        else train_cfg["warmup_steps"]
    )
    decay_steps = (
        int(decay_ratio * total_steps)
        if decay_ratio is not None
        else train_cfg["decay_steps"]
    )
    print(
        f"LR schedule: total_steps={total_steps} warmup_steps={warmup_steps} "
        f"decay_steps={decay_steps} peak={train_cfg['learning_rate']}"
    )
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg["learning_rate"],
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=train_cfg["end_learning_rate"],
    )
    # Optional gradient clipping (global-norm). A moving, unnormalized latent-dynamics
    # target can spike on a bad batch and permanently inflate the latent scale; clipping
    # bounds that. Off by default (grad_clip_norm=None) -> Go2 behaviour unchanged.
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    if grad_clip_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr_sched))
        print(f"Gradient clipping enabled: global-norm {grad_clip_norm}")
    else:
        tx = optax.adam(lr_sched)
    optimizer = nnx.Optimizer(model, tx, wrt=filt)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        **{k: nnx.metrics.Average(k) for k in loss_fn.loss_terms},
    )
    metrics.reset()

    @nnx.jit
    def train_step(
        model,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        x: types.ModelInputs,
        y: types.TrainingLabels,
    ):
        grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
        (loss, losses), grads = grad_fn(model, x, y, False)
        metrics.update(loss=loss, **losses)
        optimizer.update(model, grads)  # flax >=0.11 requires (model, grads)
        return loss

    @nnx.jit
    def eval_step(
        model,
        metrics: nnx.MultiMetric,
        x: types.ModelInputs,
        y: types.ModelOutputs,
    ):
        loss, losses = loss_fn(model, x, y, True)
        metrics.update(loss=loss, **losses)
        return loss

    def debug_stats() -> dict[str, float]:
        # Snapshot of the debug scalars left behind by the forward pass that just ran.
        #
        # Mean/std (not norm) throughout: norm grows with sqrt(dim), which makes proprio
        # (6-dim) and image feats (512/1536-dim) look artificially far apart even when
        # both are properly scaled. Mean/std are per-feature and directly comparable
        # regardless of dimension. latent_norm is the one exception -- it watches a
        # single fixed-dimensional (latent_dim) quantity over training time (the ||z||
        # inflation concern), so dimension-count confounds don't apply there.
        #
        # proprio_scaled_* comes from models.py (captured right after model.__call__'s
        # own scaler.scale(x)) -- the SCALED values the encoder actually sees, not the
        # raw batch.
        stats = {
            "debug/proprio_scaled_mean": float(model.debug_proprio_scaled_mean.value),
            "debug/proprio_scaled_std": float(model.debug_proprio_scaled_std.value),
        }
        # latent/image-feature stats only exist on ManipulationEncoder (see encoders.py
        # debug_*_input captures); guard so this stays a no-op for the Go2 encoder.
        if hasattr(model.encoder, "debug_latent_norm_input"):
            stats["debug/latent_norm"] = float(
                model.encoder.debug_latent_norm_input.value
            )
        if hasattr(model.encoder, "debug_image_feat_mean_input"):
            stats["debug/image_feat_mean"] = float(
                model.encoder.debug_image_feat_mean_input.value
            )
            stats["debug/image_feat_std"] = float(
                model.encoder.debug_image_feat_std_input.value
            )
        # The proprio block as latent_mlp receives it (post-projection, pre-LayerNorm).
        # Read against image_feat_* this is the balance metric: proprio_scaled_* above
        # describes the 6 raw inputs, which says nothing about how much of the concat
        # they occupy once the projection is on.
        if hasattr(model.encoder, "debug_proprio_feat_mean_input"):
            stats["debug/proprio_feat_mean"] = float(
                model.encoder.debug_proprio_feat_mean_input.value
            )
            stats["debug/proprio_feat_std"] = float(
                model.encoder.debug_proprio_feat_std_input.value
            )
        return stats

    def log_debug_stats(prefix: str, samples: list[dict[str, float]]) -> None:
        # These live in nnx.Intermediate scratch state, overwritten every forward pass, so
        # reading them once reports only the LAST batch of the window. For image feats a
        # single batch is tail-dominated and swung 2-4x between evals while the underlying
        # trend was flat. Average over the window instead, which also matches the loss
        # metrics (train/* = eval_interval average, test/* = full eval-pass average).
        if not samples:
            return
        for key in samples[0]:
            metrics_dict[f"{prefix}/{key}"] = sum(s[key] for s in samples) / len(samples)

    print("Starting training")
    num_epochs = cfg["training"]["num_epochs"]
    metrics_dict = {}
    # Accumulated per-step debug snapshots for the current logging window. Lives outside
    # the epoch loop because a window spans an epoch boundary whenever eval_interval does
    # not divide len(train_set).
    debug_samples_train = []
    interval_start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_start_time = time.time()

        # train
        set_dataset_mode(True)
        pbar_train = tqdm(train_set, desc="Training", unit="batch")
        for i, batch in enumerate(pbar_train):
            if batch is None:
                continue
            batch = model_utils.dataset_batch_to_jax(batch)
            loss = train_step(
                model, optimizer, metrics, batch["model_in"], batch["labels"]
            )
            pbar_train.set_description(f"Training (Loss: {float(loss):.4f})")
            debug_samples_train.append(debug_stats())
            train_steps += 1

            is_last_step = i == len(train_set) - 1 and epoch == num_epochs - 1
            steps_done = max_steps is not None and train_steps >= max_steps
            if (
                train_steps % cfg["training"]["eval_interval"] == 0
                or is_last_step
                or steps_done
            ):
                interval_time = time.time() - interval_start_time
                metrics_dict["steps_per_second"] = (
                    cfg["training"]["eval_interval"] / interval_time
                )

                for metric, value in metrics.compute().items():
                    metrics_dict[f"train/{metric}"] = float(value)
                metrics.reset()
                log_debug_stats("train", debug_samples_train)
                debug_samples_train.clear()

                # test -- eval mode, so the test loader's workers spawn with augmentation
                # off. (Was `training = True`, which put them in train() mode: every test
                # loss up to now was measured on colour-jittered/blurred/cropped images
                # with proprio noise added.)
                set_dataset_mode(False)
                pbar_test = tqdm(test_set, desc="Testing", unit="batch")
                debug_samples_test = []
                for batch in pbar_test:
                    if batch is None:
                        continue
                    batch = model_utils.dataset_batch_to_jax(batch)
                    loss = eval_step(model, metrics, batch["model_in"], batch["labels"])
                    debug_samples_test.append(debug_stats())
                    pbar_test.set_description(f"Testing (Loss: {float(loss):.4f})")
                # Restore before the next epoch's train loader spawns its workers,
                # otherwise every epoch after the first trains without augmentation.
                set_dataset_mode(True)

                for metric, value in metrics.compute().items():
                    metrics_dict[f"test/{metric}"] = float(value)
                metrics.reset()
                log_debug_stats("test", debug_samples_test)

                if cfg["checkpoint"]["enabled"]:
                    # Exclude debug-only Intermediate captures (encoders.py debug_*) --
                    # pure scratch recomputed every forward pass, no reason to persist it
                    # or risk a structure mismatch when resuming across a debug-var change.
                    state = nnx.state(model, nnx.Not(nnx.Intermediate))
                    pure_dict_state = state.to_pure_dict()
                    with ocp.CheckpointManager(ckpt_dir, options=ckpt_options) as mngr:
                        mngr.save(
                            train_steps,
                            args=ocp.args.StandardSave(pure_dict_state),
                            metrics=metrics_dict,
                        )

                metrics_dict["epoch"] = epoch

                # Wandb logging
                if cfg["wandb"]["log"]:
                    wandb.log(metrics_dict, step=train_steps)

                # TensorBoard logging (mirrors whatever wandb would have logged)
                for key, value in metrics_dict.items():
                    tb_writer.add_scalar(key, value, train_steps)

                print(f"Steps: {train_steps}, Metrics: {metrics_dict}")
                metrics_dict = {}
                interval_start_time = time.time()

            if steps_done:
                print(f"Stopping training after {train_steps} steps.")
                break

        if steps_done:
            break

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds.")

    tb_writer.close()
