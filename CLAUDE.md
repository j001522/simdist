# CLAUDE.md — `simdist`

Reference implementation of **Simulation Distillation**: pretrain a latent world model in simulation (IsaacLab + RSL-RL expert), then finetune on real data and deploy via sampling-based MPC. Original target is the Unitree Go2 quadruped. **JAX/Flax** for the world model, **PyTorch (RSL-RL)** for the expert RL policy.

## Pipeline (4 stages)

1. **Train expert RL policy** in IsaacLab (PPO via RSL-RL) → checkpoints in `checkpoints/rl/<run>/`
2. **Generate data** by rolling out expert + non-expert iterations with action corruption → `datasets/sim/<run>/`
3. **Pretrain world model** (JAX) on the processed dataset → `checkpoints/models/<run>/`
4. **Deploy** via MPPI sampling-based planner using the world model (sim or real Go2)

Optional: **finetune** the dynamics on real-world data (`scripts/aggregate_realworld_data.py`, `docs/adaptation.md`).

## Repo layout

```
simdist/                         # Python package (installed editable: pip install -e .)
   __init__.py
   modeling/                     # JAX/Flax world model
       trainer.py                # train(cfg) — Hydra entrypoint logic
       models.py                 # world model architectures
       encoders.py               # observation encoders
       resnet.py                 # ResNet-18 backbone (nnx port of torchvision weights)
       dinov2.py                 # frozen DINOv2 backbone (wraps FlaxDinov2Model)
       modules.py losses.py types.py scaler.py
   data/
       data_recorder.py          # buffers rollouts to HDF5
       data_processor.py         # post-processing into processed_data_*
       episode_aggregator.py     # real-world data aggregation
       dataset.py                # PyTorch Dataset → JAX
   rl/
       go2.py                    # IsaacLab Manager-based env config (Go2)
       go2_mdp.py                # observations, rewards, terminations, events
       rsl_rl_ppo_cfg.py         # PPO hyperparameters
       cli_args.py               # rsl_rl + AppLauncher CLI plumbing
   control/
       mppi.py                   # sampling-based MPC (planner)
       controller_base.py
   utils/                        # paths, jax, torch, io, buffer, registry, model, config, extero

scripts/
   train_rl.py                   # PPO expert training (IsaacLab + RSL-RL)
   train_model.py                # World model training (Hydra + JAX)
   generate_data.py              # Rollout expert + mixed iterations into HDF5
   process_data.py               # Build processed dataset + scaler params
   simulate_go2.py               # MPPI rollout in IsaacLab using world model
   play_rl.py                    # Replay a trained PPO checkpoint
   export_policies.py            # Export ONNX policy/critic for data-gen
   aggregate_realworld_data.py   # Concatenate real-world rosbag-derived data

config/                          # Hydra configs
   train_model.yaml              # entrypoint config (defaults: system, model, loss, heads)
   generate_data.yaml            # rollout (envs, steps, expert mix, action corruption)
   process_data.yaml
   simulate_go2.yaml
   finetune_model.yaml
   aggregate_realworld_data.yaml
   system/go2.yaml               # proprio/extero/action specs + augmentation noise
   model/quadruped_world_model.yaml
   loss/world_model{,_dynamics_only}.yaml
   heads/{all,world_model_dynamics_only}.yaml
   control/mppi.yaml             # planner hyperparams (450 samples, 8 iters, 64 elites)

go2_ros2_ws/                     # Real-Go2 deployment (ROS 2 + Docker + tmuxp)
   src/{control,measurement,bringup,visualization,simdist_controller,...}
   docker/{Dockerfile,build.sh,entrypoint.sh}
   setup/{docker_install.sh,nvidia-container-toolkit.sh}
   scripts/run.sh                # launches container + tmuxp

checkpoints/
   rl/                           # PPO runs (per-iteration .pt)
   models/                       # World-model orbax checkpoints + model_config.yaml

datasets/{sim,real}/             # generated/aggregated episodes (HDF5)

docs/
   installation.md               # ⭐ canonical install steps
   pretraining_go2.md            # ⭐ stages 1–4 walkthrough
   adaptation.md                 # real-data finetuning
   deployment_go2.md             # real Go2 hardware + Docker
tests/                           # pytest: test_models, test_control
pyproject.toml                   # JAX, Flax, optax, orbax, rsl-rl-lib, hydra, h5py, wandb
```

## Running on Snellius HPC (Apptainer)

Use the top-level wrapper — **never raw `apptainer exec`**:

```bash
# From /gpfs/work4/0/prjs0951/Giacomo/AIC-Thesis/
./isaaclab.sh -p simdist/<script.py> [args...]
```

Verify install: `./isaaclab.sh -p installation/scripts/verify_install.py`

### IsaacLab/UWLab version shims (applied to simdist/IsaacLab)

UWLab targets a newer IsaacLab than the simdist submodule. These shims were added:

| File | Added |
|---|---|
| `isaaclab/utils/math.py` | `quat_apply_inverse`, `yaw_quat` |
| `isaaclab/sensors/ray_caster/ray_caster_cfg.py` | `ray_alignment` (shim field) |
| `isaaclab/actuators/actuator_cfg.py` | `effort_limit_sim`, `velocity_limit_sim` |
| `isaaclab/assets/articulation/articulation_data.py` | `joint_armature`, `joint_friction_coeff` fields; `joint_vel_limits` property (alias for `joint_velocity_limits` — PhysX hard limits, inf for unconstrained joints; must NOT use `soft_joint_vel_limits` which is zero for unactuated mimic joints) |
| `isaaclab/scene/interactive_scene.py` | `_surface_grippers = {}` in `__init__` |
| `isaaclab/assets/articulation/articulation.py` | `joint_friction_coeff = joint_friction` alias at init |
| `isaaclab_rl/rsl_rl/rl_cfg.py` | `normalize_advantage_per_mini_batch` |
| `isaaclab_rl/rsl_rl/__init__.py` | `RslRlBaseRunnerCfg = RslRlOnPolicyRunnerCfg` (alias); `handle_deprecated_rsl_rl_cfg` no-op shim |
| `isaaclab_rl/rsl_rl/rl_cfg.py` | `class_name = "OnPolicyRunner"`, `clip_actions = None` in `RslRlOnPolicyRunnerCfg` |
| `isaaclab_rl/rsl_rl/vecenv_wrapper.py` | `clip_actions` kwarg added to `__init__`; `get_observations()` and `step()` return `TensorDict` instead of plain tensor/tuple (uw_rsl_rl 3.1.2 VecEnv interface) |
| `isaaclab_rl/utils/pretrained_checkpoint.py` | `get_published_pretrained_checkpoint` stub (returns None; no Nucleus on HPC) |

UWLab shims (in `UWLab/source/uwlab_rl/uwlab_rl/rsl_rl/rl_cfg.py`):
- `actor_obs_normalization`, `critic_obs_normalization` in `RslRlFancyActorCriticCfg`

pytorch3d in `uwlab_tasks/omnireset/mdp/utils.py`: imports made lazy (inside function) because the installed wheel was built for torch 2.7.0 but container has 2.5.1.

---

## Critical install facts (`docs/installation.md`)

```bash
conda create -n simdist python=3.10
conda activate simdist
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
sudo apt install cmake build-essential
git clone --recurse-submodules https://github.com/CLeARoboticsLab/simdist.git
cd simdist
./IsaacLab/isaaclab.sh -i none      # IsaacLab is a git submodule
pip install -e .
```

> Submodule note: this checkout was cloned without `--recurse-submodules`, so `IsaacLab/` is **not present**. It needs to be initialized (`git submodule update --init --recursive`) or installed separately and pointed at via `ISAACLAB_PATH`.

## Key dependencies (pyproject.toml)

`flax==0.10.4`, `jax[cuda]==0.6.0`, `optax==0.2.4`, `orbax-checkpoint==0.11.25`, `numpy==1.26.4`, `h5py==3.15.1`, `hydra-core==1.3.2`, `rsl-rl-lib==2.2.1`, `wandb`, `tqdm`. Python ≥3.10. Isaac Sim 4.5 with Torch 2.5.1+cu121.

## Reference invocations (`docs/pretraining_go2.md`)

```bash
# 1. expert
WANDB_USERNAME=<u> python scripts/train_rl.py --task Go2 --headless
python scripts/play_rl.py --task Go2Play --num_envs 32 -r <run> --real-time
python scripts/export_policies.py -r <run>

# 2. data
python scripts/generate_data.py rl_run=<run>
python scripts/process_data.py dataset_name=<dataset>

# 3. world model
python scripts/train_model.py data.dataset_name=<dataset> run_name=<name>

# 4. sim deploy (MPPI)
python scripts/simulate_go2.py model.checkpoint=<checkpoint>
```

`generate_data.yaml` mixes the expert (iter 4999) at 50% with non-expert iterations 0..2000 and applies bursty action corruption — important for world-model coverage.

## Frozen DINOv2 encoder (branch `dinov2-frozen-encoder`)

Alternative image backbone for `ManipulationEncoder`, selected by config: put a `dinov2`
block under `model.encoder.extero_obs` instead of `resnet`
(`config/model/manipulation_world_model_dino.yaml`). Follows **Newt** (Hansen et al.,
2025): a *pooled* frozen DINOv2 embedding concatenated with proprioception into the MLP
state encoder. DINO-WM keeps the full patch-token grid instead, but it has no
reward/value head to carry the task signal; simdist has one, so pooling is consistent.

ViT-S/14 @224 → `cls_mean` pooling (CLS ‖ mean-of-patches) = 768/camera → 3×768 + 6
proprio = 2310 into `latent_mlp`. Trainable params drop 12.0M → 0.93M (the ResNet was
11M of the old total); 21.6M frozen backbone params sit in `FrozenParam`, which is *not*
`nnx.Param`, so `trainer.py`'s `wrt=nnx.Param` optimizer and `DiffState(0, nnx.Param)`
grads skip them with no trainer change. Verified: 0 grad entries under `encoder.dinov2`.

Three non-obvious things:

1. **`transformers==4.48.*` is required.** Flax support was removed in v5 —
   `modeling_flax_dinov2.py` is 404 on `main`, present on tag `v4.48.3`. Installed with
   `--no-deps` (+ `huggingface-hub==0.27.1`, which must stay <1.0), verified not to move
   jax/flax/numpy/torch.
2. **`FlaxDinov2Model` is broken for batch > 1** whenever input resolution ≠ the
   checkpoint's 518px: `interpolate_pos_encoding` reshapes the position embedding with
   `hidden_states.shape[0]` (the input batch size) instead of 1. Workaround:
   `scripts/stage_dinov2_weights.py` bakes the interpolated embedding into the params
   using *torch's* implementation and sets `config.image_size=224`, so that branch
   early-returns and all 12 blocks run as stock transformers code. Matches the torch
   reference to rel 1.4e-6. **Do not "simplify" the baked embedding away.**
3. **`feature_norm: blockwise`, not `layernorm`.** Measured on 384 real frames, CLS std
   ≈2.32 vs mean-patch std ≈1.44 (ratio ~1.6, since averaging 256 tokens shrinks the
   vector). One LayerNorm over the concatenation keeps that ratio at 1.53; normalizing
   each block separately gives 1.00. Per-image scale varies only ~1.5% of the
   within-vector std, so affine-free LayerNorm discards nothing worth keeping and no
   dataset-statistics artifact is needed.

Staging (login node — compute nodes have no outbound network):

```bash
python scripts/stage_dinov2_weights.py            # -> assets/dinov2-small-224.npz (86.6 MB)
```

The script self-verifies against the torch reference and exits non-zero on divergence.
`assets/*.npz` is gitignored; regenerate rather than commit. Being LayerNorm-only with all
dropout rates at 0.0, this backbone makes train and eval bit-identical (`max|train-eval|
= 0`) whether or not it is being fine-tuned, which removes the BatchNorm train/eval gap the
ResNet had.

### Partial fine-tuning (`trainable_blocks` / `lr_mult`)

**Fine-tuning needed no port of DINOv2 to nnx.** `FlaxDinov2Model` is a linen
`FlaxPreTrainedModel`, but we never use its variable management: its forward is a pure
function of an externally supplied params dict (`self.module.apply({"params": params or
self.params}, ...)`, `modeling_flax_dinov2.py:613`), and we already hold and pass those
params. JAX differentiates straight through it, so this is only a question of which nnx
Variable *class* holds which slice of the tree.

`model.encoder.extero_obs.dinov2.trainable_blocks: K` adapts the last K of 12 blocks plus
the final layernorm (K=4 → 7.10M of 21.6M backbone params). `lr_mult` gives the backbone a
fraction of the head LR via `optax.multi_transform` in `trainer.py`. `trainable_blocks: 0`
is byte-identical to the frozen branch and still restores frozen-run checkpoints; K>0 is a
different state layout and deliberately cannot.

Four things that are not obvious:

1. **One Variable per TENSOR, not one Variable holding the subtree.** A tree-valued
   Variable does not survive the transforms: under `nnx.jit` its value comes back as a
   `State` (breaking `flatten_dict`), and under `optax.multi_transform` the masked group
   comes back as a plain dict of `MaskedNode()` (breaking the `State` structure match).
   Hence the flat `nnx.data({"a/b/c": BackboneParam(...)})` layout — a *bare* dict
   attribute is treated as static and nnx refuses to put arrays in it.
2. **Discriminative LR has to be in the optimizer.** Scaling the gradient inside the module
   is a no-op: Adam normalizes by gradient RMS. At `lr_mult: 1.0` the ViT is destroyed
   within a few hundred steps.
3. **No `stop_gradient` at the frozen/trainable boundary** — impossible (`module.apply`
   runs all 12 blocks in one call) and unnecessary. JAX's partial evaluation classifies the
   frozen prefix as primal-only and stages no backward residuals for it.
4. **Memory scales ~linearly in K, and the batch is the multiplier.** `encode_latent` runs
   on `batch * n_cam` images. Measured backward temps: at batch 256 (768 images) K=4 needs
   ~40 GiB, K=2 ~20 GiB; halved at batch 128. Snellius H100 is 95,830 MiB. The forward-only
   target encode in `WorldModelLoss` is under `stop_gradient` and does not add to this.

Launcher: `train_wm_manip_dino_ft.sbatch` (batch 128, 36 h, `eval_interval` and
`keep_period` doubled to keep the per-epoch eval cadence of the frozen sweep). Keep
`train_wm_manip_dino.sbatch` intact as the frozen baseline. Note the fine-tuned runs are
**not batch-matched** to that baseline.

## Proprioception projection (branch `proprio-projection`)

`ManipulationEncoder.encode_latent` originally concatenated the **raw** 6 joint obs onto
the flattened image features. That leaves proprio at 0.26–0.52 % of `latent_mlp`'s input
dims (2310 for DINOv2 `cls_mean`, 1542 for ResNet-18, 1158 for DINOv2 `cls`).

**It is a dimension problem, not a scale one.** Proprio is already scaler-standardised to
~unit std (`models.py:190`) and each image block is affine-free LayerNormed to exactly
unit std; the first `Linear` is lecun_normal (var = 1/fan_in), so every input dim
contributes equal variance and proprio's share of the pre-activation variance is just its
share of the dims. This is exactly why the existing `debug/proprio_scaled_std` vs
`debug/image_feat_std` pair looks healthy and shows nothing.

`model.encoder.proprio_embed_dim` fixes the count:

| value | effect |
|---|---|
| absent / `null` / `none` | raw concat — the pre-branch behaviour, byte-identical |
| `image` | project to `per_image_embed_dim`; proprio becomes one more camera-sized block → **25 % of input dims for every backbone** (512 ResNet, 384 DINOv2 `cls`, 768 `cls_mean`) |
| an int | that width directly |

Costs +148k params (ResNet) / +214k (DINOv2 `cls_mean`). Both manip model YAMLs now default
to `image`.

Four things worth knowing:

1. **`proprio_norm: layernorm` is not cosmetic.** Measured at init the projected block
   arrives at std 0.16 against the image blocks' 1.0 — without it, ~6× of the
   variance-per-feature balance the projection just bought is handed straight back.
   Affine-free like the image norms, so a learnable gamma can't re-inflate the branch.
   `proprio_norm: none` is the escape hatch: LayerNorm removes the mean-over-features and
   the norm, which with only 6 informative inputs is a real fraction of the signal (the
   2-layer gelu makes those nonlinear functions of the joints, which dilutes but doesn't
   eliminate the cost). **Never** normalize the raw 6-vector.
2. **Not weight-shared with `proprio_obs_proj`.** That one is pinned to `latent_dim` by the
   history tokens and its representation plays a different role (a sequence token, with
   temporal/type encodings added). `QuadrupedEncoder` *does* share, which is why the Go2
   path was already balanced 64/64 — the manip encoder was the outlier, following the
   paper's literal "concat the 6 joint obs".
3. **Not a bare `Linear`.** A linear 6→512 is rank-6: it rebalances init variance but adds
   no capacity. The projection reuses `proprio_obs_layers`/`h_size` for a 2-layer gelu MLP.
4. **This moves the prediction TARGET, not just the input.** `losses.py` encodes the future
   obs through the same `encode_latent`, so under the raw concat the latent-dynamics target
   is ~99.7 % image and the dynamics head is barely graded on joint motion. Standing
   suspect for `latent_dynamics` flooring at ~0.07 across all six encoders in the
   2026-07-31 sweep and for rollout error being encoder-independent.

Old checkpoints are unaffected: `load_model_from_ckpt` rebuilds from each run's own
`model_config.yaml` snapshot, which predates the key. New runs are a different state layout
and deliberately cannot resume from them.

New TensorBoard metrics `debug/proprio_feat_{mean,std}` report the proprio block **as
`latent_mlp` receives it** (post-projection, pre-LayerNorm); read against
`debug/image_feat_*` that pair is the actual balance metric.

Launcher: `train_wm_proprio_proj.sbatch` (4 arms: ResNet/DINOv2-`cls`-ft × projected/raw;
default `--array=0-1` runs the two projected arms against the existing
`wm_resnet_l*_26096399` / `wm_dinoft_cls_l*_26096400` baselines — the raw arm on this
branch is numerically identical to those, since the only diff on that path is two extra
`nnx.Intermediate` debug scalars that never enter the loss or the checkpoint).

**Not addressed here, and probably the bigger lever:** `system/ur5e.yaml` defines proprio
as `arm_joint_pos` alone — no joint velocity, no EE pose, no gripper state, no wrench. For
insertion, EE pose and contact wrench are the channels carrying the signal. Adding them
costs a dataset regeneration (processor + scaler stats).

## Adapting to AIC (UR5e cable insertion)

The IsaacLab side here is **Go2-specific** (`simdist/rl/go2.py`, `go2_mdp.py`, `quadruped_world_model.yaml`, `system/go2.yaml`). To target the AIC challenge we need analogous files for a UR5e cable-insertion env. That env is the role of the sibling `aic_utils` extension and `UWLab` (which already provides the UR5e + OmniReset manipulation tasks).

Conceptual mapping (Go2 → AIC):
| simdist concept | AIC analog |
|---|---|
| `simdist/rl/go2.py` (env cfg) | new `simdist/rl/ur5e_aic.py` registering an IsaacLab env that mirrors the AIC scene (UR5e + task board + cable) |
| `simdist/rl/go2_mdp.py` | observations: joint_states, wrist_wrench, 3 cameras; actions: cartesian impedance target deltas |
| `system/go2.yaml` (proprio/extero/action) | replace with UR5e proprio + RGB extero + cartesian action spec |
| `quadruped_world_model.yaml` | likely keep latent dynamics core; swap encoder for RGB (and/or include wrench) |
| Real Go2 ROS workspace (`go2_ros2_ws/`) | the AIC eval container + `my_policy_node` policy that calls into the deployed world model |

Things to verify when porting:
- World model in `simdist/modeling/encoders.py` — check whether it already supports image inputs (Go2 uses extero/proprio, not RGB).
- Action space dimension: Go2 uses 12 joint targets; AIC cartesian impedance is 6-DoF + stiffness, very different.
- Frame rate: Go2 sim runs at ~50–100 Hz; AIC cameras are 20 FPS.

## Updating this file

Edit when: new pipeline stage added, env/system YAML schema changes, deployment workflow shifts, or AIC-specific code lands under `simdist/rl/`.
