# SimDist → Manipulation (UR5e peg insertion) port

Branch: `manipulation`. Goal: adapt SimDist's data-collection + world-model pipeline
from Go2 locomotion to OmniReset UR5e peg insertion, using our trained OmniReset
state expert + intermediate checkpoints as the data source.

## Key asymmetry (drives the whole design)

- **Expert policy `π_i` and value `V^e` are STATE-based** (OmniReset `-State-` expert).
  They consume the State env's `policy` / `critic` obs groups.
- **The world model is VISION-based**: 3× 224×224 RGB (front/overhead, side, wrist) +
  6 joint obs → ResNet-18 per image → MLP → 64-d latent `z` (paper Table II).
- So during data-gen: drive the env with `π_i(s_t)` (state), query `V^e(s_t)` (state),
  but **record `o_t` = vision** (int8 images + proprio). This is Algorithm 2 line 16:
  `Add (o_t, a_t, b^e_t, r_t, v_t)` — note **no command term** (unlike Go2).

## Base env to wrap

`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0`
(cfg `data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg`).
It already provides: scene (inherits `RlStateSceneCfg`) + 3 TiledCameras
(`front_camera`, `side_camera`, `wrist_camera`, 224×224), RelCartesianOSC action
(6-d rel EE pose + binary gripper = 7), camera-pose/focal randomization events, and a
`data_collection` obs group of **unprocessed int8** images + proprio.
It does NOT have state `policy`/`critic` obs groups (it's built for DAgger). We add them.

## Obs groups in the recorder env (`__post_init__` overrides)

- `policy`  ← State `ObservationsCfg.PolicyCfg` (rl_state_cfg.py:369). 6 terms,
  `concatenate_terms=True`, **`history_length=5`**. Consumed by `π_i`.
- `critic`  ← State `ObservationsCfg.CriticCfg` (rl_state_cfg.py:418). Adds privileged:
  time_left, joint_vel, ee_vel, 4× material_properties, 4× mass, joint
  friction/armature/stiffness/damping; `history_length=1`. Consumed by `V^e`.
- `recorded_obs` ← the RGB `data_collection` group (int8 `front/side/wrist_rgb` +
  `arm_joint_pos` (6) + last actions), `concatenate_terms=False`. Saved as `o_t`.

All these terms are computable because the RGB scene inherits `RlStateSceneCfg`
(robot/insertive_object/receptive_object/table all present).

## What to port in SimDist

### 1. `simdist/rl/<ur5e_omnireset>.py` (mirror of `simdist/rl/go2.py`)
- `ManipRecordEnvCfg(<RGB DataCollection cfg>)` setting the 3 obs groups above +
  `recorders = ManipRecorderManagerCfg()`, disabling policy-obs corruption.
- `ManagerBasedRLEnvRecord(cfg, critic)` — reuse as-is from go2.py (generic).
- Recorder terms (mirror go2 lines 550-610) **with two changes**:
  - `ValueRecorder`: `self._env.critic(self._env.obs_buf["critic"])`  ← **not `["policy"]`** (asymmetric!).
  - **drop** the commands recorder (no task command recorded; paper has none).
  - obs recorder returns `obs_buf["recorded_obs"]` (a dict of image+proprio terms).

### 2. `simdist/data/data_recorder.py`
- Swap `Go2RecordEnvCfg` / import for the manip cfg (lines 8, 50, 56).
- `get_action` already uses `obs["policy"]` ✓. Action corruption Σ over `dim(A)=7`.

### 3. `scripts/export_policies.py`
- Mostly generic. Point `--task` at the OmniReset State task + its
  `rsl_rl_cfg_entry_point`; it already exports actor (`policy_i.pt`, with obs
  normalizer) and critic (`critic_i.pt`, with critic_obs normalizer). Asymmetric
  obs handled by the two normalizers.

### 4. configs
- `config/generate_data.yaml`: `expert` = converged ckpt (default model_2000),
  `non_experts` = [0,100,…,1000] (paper saves every 100 up to 1000),
  `expert_prob = 0.5`, action_corruption intervals: noise U[1,5], no-noise U[5,10]
  (paper). `system.actions` = 7 entries with per-dim min/max noise σ.
- `config/system/*`: manip action/obs spec (action_dim=7).

### 5. Downstream (later, not data-gen): world-model encoder
- Per paper Table II: ResNet-18 (imagenet) per image → 3×512 stacked, concat 6 joint
  obs → MLP → z(64). This replaces Go2's flat-vector encoder in the modeling stage.

## Open decisions / prereqs

- **[PREREQ] UWLab import path**: `uwlab_tasks` (OmniReset) must be importable from the
  python that runs `generate_data`/`export_policies`. The OmniReset training ran on the
  Spark Isaac python (`/shared/giacomo/isaac/.../python.sh`); confirm SimDist data-gen
  uses the same interpreter so the tasks register. Verify before any run.
- **Checkpoint range**: paper uses non_experts up to 1000; we have 0–2000 (21 ckpts).
  Default: expert=model_2000, non_experts=[0..1000]. Revisit if quality differs.
- **Recorded proprio**: RESOLVED — data has `arm_joint_pos [6]` as proprio; privileged
  poses ARE in `obs/` but excluded from the WM input by config name-selection (see 3.0.0).
- **Dataset format**: RESOLVED — confirmed robomimic `/data/demo_<int>` layout with
  JPEG-vlen images + separate `reward`/`value`/`expert_policy_flag` (schema in 3.0.0).
  `process_data.py`/`dataset.py` DO need image-path changes (3.3) + length filter.

---

# Stage 3 — World-model training port (train on the collected dataset)

Data-gen (stages 1–2, above) is built — it produced the Spark dataset. The
**training** stage is still Go2-shaped: the encoder, dataset/processor, model
config, and the base-policy goal query all need manipulation versions. This
section is the plan for that.

## 3.0.0 Confirmed dataset schema (raw HDF5)

Inspected `datasets/sim/2026-06-30_11-09-59/raw_data.hdf5` (17 GB, robomimic-style,
IsaacLab RecorderManager output). Structure per demo, and its consequences:

- Layout: `/data` (attrs `total`, `env_args`), children `demo_<int>` (NOT
  zero-padded / contiguous — **index by key, not by counting**). 10374 demos,
  `total=1_280_000` steps.
- Per `demo_<int>` (attrs `num_samples`=T, `success`): datasets
  `actions [T,7] f32` (ee_dx,dy,dz,drx,dry,drz,gripper), `expert_policy_flag [T] bool`
  (→ the BC indicator `1_e(a)` in eq:simloss), `reward [T] f32`, `value [T] f32`
  (expert critic V^e), and group `obs/`.
- `obs/` (RICHER than the WM uses): `arm_joint_pos [T,6]` (**the proprio**),
  `front_rgb / side_rgb / wrist_rgb [T]` vlen JPEG bytes with attrs
  `image_shape=[224,224,3], jpeg=true` (**the extero**, decode via `jpeg_io`), PLUS
  privileged fields **to be EXCLUDED from the WM input**: `end_effector_pose[6]`,
  `insertive_asset_pose[6]`, `receptive_asset_pose[6]`,
  `insertive_asset_in_receptive_asset_frame[6]`, `binary_contact[1]`,
  `last_arm_action[6]`, `last_gripper_action[1]`.
  `system/ur5e.yaml` already names exactly `arm_joint_pos` + 3 rgb → **select by
  name; ignore the rest** (do not choke / do not scale privileged fields).

Key facts & required handling:
- **Control rate 10 Hz** (`sim dt 0.008333 × decimation 12 = 0.1s`); max episode
  160 steps = 16 s → informs `H`/`T` (25/25 ≈ 2.5 s; tunable, not sacred).
- **Length filter REQUIRED**: `num_samples` min 1, **427 demos (4.1%) are T=1**,
  left-skewed (mean 123 / median 160). A window needs `H+T+1` steps → **drop demos
  shorter than history_length+prediction_length+1** in process_data/dataset.
- **`success=0` for ALL 10374 demos.** Almost certainly a flag-wiring artifact of
  the simplified `GraspedVertHigh` env (prior OmniReset expert hit ~75% peg success),
  NOT expert failure — but **VERIFY before a full train**: on long demos, `value`
  should rise over the episode and `reward` spike near `binary_contact==1`. If flat →
  the reward/value targets are garbage and training is pointless. Success flag itself
  is unused by the WM (loss uses reward/value/expert_flag), so if targets are good the
  cosmetic flag doesn't matter.
- **Image-target cost (inherent to the method):** the latent-dynamics loss encodes
  targets `o_{t+1:t+T}`, so each sample needs images at the current step AND all T
  future steps → `(1+T)×3 ≈ 78` JPEG-decode+ResNet per sample (Go2 hid this — tiny
  vector extero). History uses only the *latest* image ("minimal history"); the cost
  is in the TARGETS. Drives dataloader (decode workers) + throughput design.
- Gen config (`record.yaml`): 256 envs × 5000 steps, `expert_prob 0.5`,
  `rl_run omnireset_2026-06-21_17-10-39`, `simplify.reset_types=[GraspedVertHigh]`.

## 3.0 Environment (JAX, no Isaac needed for training)

`train_model.py → trainer → modeling → dataset` has **no Isaac dependency**
(pure JAX/Flax/optax/orbax/h5py/hydra + torch only for the `Dataset`/`DataLoader`
wrapper). The Go2 `data/data_processor.py` imports Isaac (loads episodes via
`HDF5DatasetFileHandler`), but the **manip processor**
(`data/manip_data_processor.py`) reads the raw HDF5 **directly with h5py** — so the
**entire manipulation stage-3 loop is Isaac-free**.

Consequence — the whole loop runs in ONE conda env `simdist-jax` (no apptainer):
- **`process_data`** (manip) → `ManipulationDataProcessor`, h5py-direct, JPEG
  passthrough, Isaac-free. `scripts/process_data.py` auto-selects it when extero is
  images (imports the Isaac-coupled Go2 `DataProcessor` lazily, only in the else-branch).
- **`train_model`** → same env. Deps: `jax[cuda12]`, `flax`, `optax`,
  `orbax-checkpoint`, `numpy`, `h5py`, `hydra-core`, `opencv` (JPEG decode),
  `torch` (dataloader only). **torch MUST be the CPU build** (`torch==2.5.1+cpu`
  from the pytorch cpu index): the CUDA build hard-pins `nvidia-cudnn-cu12==9.1.0.70`
  which clashes with `jax-cuda12` (needs cudnn>=9.8). CPU torch has no nvidia deps →
  JAX owns the GPU/cuDNN. (Same 9.1-vs-9.8 cuDNN issue the apptainer solved via an
  LD_LIBRARY_PATH shim; CPU torch sidesteps it entirely.)

**GPU:** the old "H100-only / A100 deadlocks" note was an Isaac-Sim CUDA-context
issue and **has been fixed** — training runs on both A100 and H100. In practice
the **H100 partition is more available**, so prefer it.

**Versions (confirmed):** the nnx port was validated under **flax 0.12.7 + jax
0.10.1 (py3.12)** on the Spark; `environment_jax.yml` replicates that exact pair
(CUDA12 wheels for Snellius). This supersedes simdist's stale pyproject pins
(`jax 0.6.0` / `flax 0.10.4` / py3.10) — **do not downgrade to match pyproject**.
optax/orbax/numpy are floated (old simdist pins conflict with flax 0.12.7).
Version choice is API-compat only — it does not affect training results; the Spark
constraint does not bind this Snellius-only env.

### Isaac-free JPEG decode (the "vendoring" question, resolved)

The manip dataset must call `decode_jpeg_dataset` (read side) at train time, but
that helper currently lives in the Isaac-coupled `jpeg_hdf5_handler.py` (top-level
`import isaaclab`). Importing it from `dataset.py` would drag Isaac into the conda
train env. **Fix = decouple the import** (not full vendoring):
- Make the `isaaclab` import **lazy** (inside the write class `__init__`/method), or
- **Split** the read-side helpers (`decode_jpeg_dataset`, `_is_image`,
  `is_jpeg_dataset`) into an Isaac-free module (`data/jpeg_io.py`), leaving only
  the write handler in `jpeg_hdf5_handler.py`.

Prefer the split — cleanest. Then `dataset.py` imports decode from `jpeg_io` with
no Isaac in the train env.

## 3.1 Encoder — ResNet-18 now, swappable (DINOv2/VFM later)

`encoders.py` already exposes the swap point: `WorldModelEncoderBase`
(`__call__` + `encode_latent`). `QuadrupedEncoder` subclasses it.

- Add `ManipulationEncoder(WorldModelEncoderBase)`: ResNet-18 (ImageNet) applied
  to each of the 3 cameras → `3×512` → stack, concat 6 joint obs → MLP → `z(64)`
  (paper Table II / `app:manip`).
- **Register encoders by name** and select via `config/model/*.yaml` `encoder.type`
  so a later `DINOv2Encoder(WorldModelEncoderBase)` is a **config-only** swap
  (same `(proprio, images) → z(64)` contract; freeze the VFM, train the projection
  MLP). Design the registry from day one.
- **ResNet-18 ImageNet weights in JAX** is the main risk (no torchvision in
  Flax/nnx). Options: port torchvision R18 weights → nnx pytree; use a JAX zoo
  (`flaxmodels`); or train from scratch (worse on a small dataset — paper uses
  pretrained). Pick before implementing 3.1.

## 3.2 Goal conditioning — `cmd_dim=0`, temporal embeddings for the policy head

How Go2 threads the goal (`fut_cmds` = velocity command), in `modeling/models.py`:
- reward head (`:199`): `concat(z, fut_acts, fut_cmds)`
- value head (`:209`): `concat(latents, fut_cmds)`
- base-policy head (`:217`): `self.policy(fut_cmds_emb, context)` — the command
  embedding **is the decoder query sequence** that gives the policy its per-step
  output slots for the action chunk.

Manipulation has a **fixed** insertion goal → `system/ur5e.yaml` already sets
`cmd.dim=0`. Therefore:
- **reward/value:** the `fut_cmds` concats become **zero-width appends** (harmless).
  The goal is **implicit in the pretrained reward/value heads** (the OmniReset
  expert/`V^e` were trained for insertion; `V^e` encodes distance-to-inserted).
  MPPI does the planning over these heads — no goal input. Guard the concats for
  the 0-dim case so they don't crash.
- **base-policy head:** losing `fut_cmds` removes its query → it can't emit a
  chunk. **Fix (confirmed design): replace the command query with learned temporal
  (positional) embeddings** — a `(chunk_len, emb)` parameter table broadcast over
  the batch, used as the decoder query; it cross-attends `context` at each slot.
- **dynamics head is unaffected** — its query is `fut_acts_emb` (`:190`), not
  commands.

Net: `ManipulationWorldModel` overrides exactly one thing (policy-head query →
temporal embeddings) plus the `cmd_dim=0` concat guards.

## 3.3 Dataset / processor image path

Currently `data_processor.py` cats extero into a flat vector + z-scores it via
`_RunningStats(extero_obs_dim)` + flat `_H5Appender`; `dataset.py`
(`QuadrupedWorldModelDataset`) gaussian-noises extero as a vector. For images:
- **process_data:** decode JPEG (or pass JPEG through), do **not** z-score pixels
  (only proprio + reward/value/action get scaler stats). Store images per-camera,
  not concatenated.
- **`ManipulationWorldModelDataset`:** JPEG-decode in `__getitem__`, apply image
  augmentations (color jitter, gaussian blur, random crop — paper `app:manip`),
  proprio gaussian noise only.
- Register `ManipulationWorldModelDataset` + select via model cfg `dataset.type`.

## 3.4 Config-util image-shape helper

Correction: `utils/config.py` `extero_obs_dim_from_sys_config` **already** handles
list dims (`np.prod` over `[224,224,3]`) — it does not crash. But it *flattens* all
cameras to one scalar (~451k), which is useless for the ResNet path.
- ✅ Added `extero_obs_image_shapes_from_sys_config` → `[(224,224,3), ...]` per
  camera (config order), the shape-aware helper the vision encoder needs.
- **Remaining (couple to 3.1/3.3, do after data inspection):** rewire the manip
  encoder + `ManipulationWorldModelDataset` + `get_dummy_item` to consume per-camera
  shapes instead of the flat `extero_obs_dim`. `dataset.py:164/174/182` currently
  build flat `(extero_obs_dim,)` arrays — wrong for images.

## 3.5 New model config

✅ Added `config/model/manipulation_world_model.yaml` (paper Table II): `latent_dim=64`,
all-transformer MLP hidden 256 (factor 4), dynamics 3L/4H, reward 1L/1H, value 1L/1H,
policy 4L/8H; `encoder.type: manipulation` with a `resnet` block (resnet18/imagenet,
not frozen, 512/image); `dataset.type: manipulation_world_model` + image augs.
Validated to parse; key-diff vs quadruped is only `height_cnn`→`resnet` + image-aug keys.
**Open [TBD] in the file:** `history_length`/`prediction_length` (H/T — tune to control
rate), dropout (kept from quadruped), and the `encoder`/`resnet` sub-schema (finalized
when `ManipulationEncoder` lands in 3.1). Values there are the *contract* the encoder
code must read.

## Stage-3 task order

1. **Env** — ✅ JPEG read helpers split into Isaac-free `data/jpeg_io.py`
   (`jpeg_hdf5_handler.py` now holds only the Isaac-coupled write class and
   re-exports the helpers for back-compat). Env spec at `environment_jax.yml`
   (`conda env create -f environment_jax.yml`). **TODO: confirm jax/flax pins vs
   Spark before creating the env.**
2. **Inspect** collected HDF5 — ✅ done, schema in 3.0.0. Confirms JPEG vlen images,
   `arm_joint_pos` proprio, `reward`/`value`/`expert_policy_flag`, privileged extras
   to exclude. ✅ value/reward sanity verified on Snellius (2026-07-06): reward mean
   0.0134 std 0.0077 (min −0.151, not flat), value −12.2…−8.1 std 0.61 — informative.
   Lengths: 1.28M steps, median 160, 427 demos T=1, 2241 demos T<50 (length filter in
   step 4 handles these).
3. **config util** — ✅ added `extero_obs_image_shapes_from_sys_config` (3.4);
   downstream rewire deferred to 5/6 (needs data-inspection shapes).
4. **process_data** — ✅ `data/manip_data_processor.py` (`ManipulationDataProcessor`):
   h5py-direct raw read (Isaac-free), proprio+3-cam-by-name select, **JPEG passthrough**
   into `images.hdf5`, length filter, start_idxs, scaler (proprio/act/reward/value real,
   extero placeholder, cmd zero-width). `process_data.py` auto-selects it for image
   extero; `paths.get_images_path` added. ✅ Slice-verified on Snellius (2026-07-06):
   30-demo slice (`datasets/sim/sliceverify30`, mixed lengths 1–114) processed with
   H=T=25, padding 5/5 → min_len 60 filter kept exactly the 20 demos ≥60; outputs
   `start_idxs` (954 windows), `proprio_obs` (2154×6), `acts` (2154×7), 3-cam
   `images.hdf5` (2154 frames each, `image_shape`/`jpeg` attrs set); JPEG passthrough
   **byte-identical** to raw; frames decode via cv2 to (224,224,3) uint8. Note: front/side
   cams are dimly lit (mean pixel ~8–12/255) but content is visible; wrist normal.
5. **Encoder** — (5a ✅) `modeling/resnet.py`: nnx `ResNet18Backbone` (→512 GAP) +
   `load_torchvision_resnet18` weight port (DECISION: hand-nnx + torchvision weights) +
   `imagenet_normalize`; `scripts/verify_resnet_port.py` checks it vs torch (run once,
   needs torch+jax). ✅ Verified on Snellius (2026-07-06, CPU, conda `simdist-jax` +
   torchvision 0.20.1/pillow added `--no-deps`): max abs err 2.38e-06, rel L2 5.59e-07 —
   PORT OK. ✅ RGB confirmed by construction: recorder `cv2.imencode`s Isaac RGB frames
   (BGR-assumed swap into the file) and `decode_jpeg_dataset` `cv2.imdecode`s (swap back)
   — the two swaps cancel, so decoded arrays are in the original Isaac RGB order.
   (5b ✅) `ManipulationEncoder(WorldModelEncoderBase)` in `modeling/encoders.py`:
   SHARED ResNet over 3 cams → 3×512, concat raw 6 joints → `latent_mlp` → z; history
   path mirrors Quadruped (proprio+act tokens, temporal/type enc), no images in history;
   `encode_latent` shape-agnostic (also serves target encoding `E(o_{t+1:t+T})`);
   `freeze` honored via stop_gradient. Syntax-clean. Weight-load is an explicit step
   (trainer must call `resnet.load_torchvision_resnet18(model.encoder.resnet)` when
   `pretrained==imagenet`).

### Locked image Inputs-schema (contract for step 6)

`ManipulationEncoder` + "encode inside the model on raw pixels" fix the manip schema:
- `extero_obs` now holds the **latest** camera images `(B, n_cam, H, W, 3)` uint8
  (replaces Go2's flat height vector). The encoder does its own `imagenet_normalize`.
- History carries **no** images (only proprio+action tokens) — dataloader must NOT pack
  image history, only the single latest frame-set.
- **Targets** for the latent-dynamics loss need future images `(B, T, n_cam, H, W, 3)`
  (to compute `sg(E(o_{t+1:t+T}))`); `encode_latent` already handles the extra T axis.
- Scaler: keep the manip `scaler_params_mapping` free of `extero_obs` (images unscaled);
  the placeholder identity Stats written by the processor is a belt-and-suspenders no-op.
- Dataloader delivers decoded uint8 (JPEG-decode via `jpeg_io`); image augs (crop/jitter/
  blur) applied on uint8 in the dataset before the model. cmd_dim=0 → `fut_cmds` width-0.
6. **`ManipulationWorldModel`** + **`ManipulationWorldModelDataset`** — ✅ DONE
   (2026-07-08, verified on Snellius CPU in `simdist-jax`).
   - `models.ManipulationWorldModel` (`@register_model("manipulation_world_model")`):
     overrides encoder → `ManipulationEncoder`; scaler mapping drops **both**
     `extero_obs` (images stay uint8, encoder normalizes) **and** `fut_cmds`; learned
     `policy_temporal_query` `(T, latent)` replaces the fut_cmds policy-decoder query.
   - `WorldModelBase` gained a `_policy_query()` hook (Go2 returns `fut_cmds_emb`;
     manip returns the temporal table) + **cmd_dim==0 guards**: `fut_cmds_embed` is not
     built (0-input Linear div-by-zero) and the reward/value heads skip the fut_cmds
     concat when `cmd_dim==0`. Go2 (cmd_dim>0) behaviour is byte-unchanged.
   - `dataset.ManipulationWorldModelDataset` (`@register_dataset("manipulation_world_model")`):
     loads `images.hdf5` (per-camera vlen-JPEG, not `extero_obs.hdf5`), overrides
     `_load_files`/`_load_data`/`get_item_by_t_H`; input `extero_obs` = latest frame-set
     `(n_cam,H,W,3)` uint8, label = future `(T,n_cam,H,W,3)`; decode via
     `jpeg_io.decode_jpeg_frames` (new slice-decode helper). Image augs (crop/jitter/blur)
     on the INPUT frame-set only (targets kept clean); proprio gaussian noise as Go2.
   - Trainer: `model_utils.maybe_load_pretrained_encoder(model, cfg)` loads ImageNet
     ResNet weights for a fresh manip model (no-op for Go2 / on resume).
   - **NOTE (quirk):** the processor writes `cmds.hdf5` 1-D `(N,)` zeros, so real
     `fut_cmds` windows are `(T+1,)` not `(T+1,0)`; harmless because the manip model
     never reads fut_cmds (guards + temporal query).
   - **NOTE (pre-existing fix):** `trainer.py` used `optimizer.update(grads)` (flax ≤0.10);
     the `simdist-jax` env is flax 0.12.7 which requires `optimizer.update(model, grads)`.
     Fixed — this blocked ANY WM training (Go2 too) under the current env, not just manip.
   - Verified end-to-end on `sliceverify30` (954 windows): dataset item shapes, JPEG
     decode, default_collate→jax (uint8 preserved), model forward, `WorldModelLoss`
     (label scaling + future-image target encode), and one jitted `value_and_grad` +
     `optimizer.update` gradient step (params move, losses finite).
7. **Smoke / full train** — launcher ready: `train_wm_manip.sbatch` (repo root; GPU +
   `simdist-jax`, `SMOKE=1` = 2-step entrypoint check, else 2-epoch pretrain). CPU full
   loop is impractical (224² images); run on `gpu_h100`. ResNet-18 ImageNet weights
   download on first use (compute nodes have internet; `$TORCH_HOME` caches them across
   jobs), or submit `PRETRAINED=null` for random init. **Not yet run on GPU / real data.**

See `ARCHITECTURE.md` for the full manipulation world-model schematic (all dims/flows).
