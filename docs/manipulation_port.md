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
- **Recorded proprio**: paper says "6 joint observations" → record `arm_joint_pos`
  (+ optionally last actions). Keep extra privileged poses out of `recorded_obs`.
- **Dataset format**: `recorded_obs` is a dict (images can't concat with vectors);
  confirm SimDist's RecorderManager HDF5 export + `data/dataset.py` handle nested/image
  obs (Go2 was flat vectors). Likely needs `process_data.py`/`dataset.py` changes.
