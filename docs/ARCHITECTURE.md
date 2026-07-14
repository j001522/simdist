# Manipulation World Model — Architecture

Detailed schematic of the SimDist **manipulation** world model (UR5e peg insertion,
paper `app:manip`). It is the latent world model that MPPI plans against at deploy time:
given a short history of proprioception + the latest camera images, and a candidate
action sequence, it predicts the resulting **future latents, rewards, values**, and a
**base-policy action chunk** that warm-starts the planner.

Code:
- `simdist/modeling/models.py` — `WorldModelBase`, `ManipulationWorldModel`
- `simdist/modeling/encoders.py` — `ManipulationEncoder`
- `simdist/modeling/resnet.py` — `ResNet18Backbone`
- `simdist/modeling/modules.py` — `MLP`, `Embedding`, `TransformerEncoder/Decoder`
- `simdist/modeling/losses.py` — `WorldModelLoss`
- config: `config/model/manipulation_world_model.yaml`, `config/system/ur5e.yaml`

This model shares the entire latent-dynamics core with the Go2 quadruped model; the only
manipulation-specific pieces are the **vision encoder**, **unscaled images**, and a
**learned temporal query** for the policy head (because there is no goal command). Those
differences are flagged with 🔧 below.

---

## 0. Constants (from the two configs)

| Symbol | Meaning | Value |
|---|---|---|
| `D` | latent dim (`latent_dim`) | **64** |
| `H` | history length | **25** (tunable) |
| `T` | prediction / planning horizon | **25** (tunable) |
| `P` | proprio dim (`arm_joint_pos`) | **6** |
| `A` | action dim (6 EE-pose deltas + gripper) | **7** |
| `C` | command dim (`cmd.dim`) | **0** 🔧 |
| `n_cam` | cameras (front/side/wrist RGB) | **3** |
| image | per-camera frame (H×W×C) | **224×224×3** uint8 |
| `E_img` | ResNet embedding per image | **512** |
| `h_enc` | encoder MLP hidden = `D·2` | 128 |
| `h_emb` | head-embedding MLP hidden = `D·2` | 128 |
| `h_attn` | transformer MLP hidden = `D·4` | 256 |

All MLPs are `Linear → GELU` per hidden layer (dropout 0.2 between hidden layers), then a
final `Linear` with no activation. Attention dropout 0.05.

---

## 1. Inputs & labels (per sample, before batching)

`WorldModelSchema.Inputs` (`x`) — what the model consumes:

| key | shape | notes |
|---|---|---|
| `proprio_obs_hist` | `(H+1, P)` = `(26, 6)` | joints from `t-H … t` |
| `extero_obs` | `(n_cam, 224, 224, 3)` uint8 | 🔧 **latest** frame-set only (no image history) |
| `acts_hist` | `(H, A)` = `(25, 7)` | actions `t-H … t-1` |
| `fut_acts` | `(T, A)` = `(25, 7)` | candidate actions `t … t+T-1` |
| `fut_cmds` | `(T+1, 0)` 🔧 | width-0 (dummy) / `(T+1,)` zeros in real data — **unused** |

`WorldModelSchema.Labels` (`y`), used only by the loss:

| key | shape | notes |
|---|---|---|
| `proprio_obs` | `(T, P)` | future joints `t+1 … t+T` |
| `extero_obs` | `(T, n_cam, 224, 224, 3)` uint8 | 🔧 future frame-sets (latent-dynamics target) |
| `rewards` | `(T,)` | `t … t+T-1` |
| `values` | `(T,)` | `V^e` from the expert critic, `t+1 … t+T` |
| `actions` | `(T, A)` | expert actions `t … t+T-1` |
| `exp_pol_flags` | `(T,)` bool | true while the expert policy was in control |

After the PyTorch `default_collate` + `dataset_batch_to_jax`, every field gains a leading
batch axis `B`. Images stay **uint8** through collation.

### Scaling 🔧
`WorldModelBase.__call__` first runs the `Scaler` (per-field `(x-μ)/σ`). The manip model
**drops `extero_obs` and `fut_cmds` from the scaler mapping**, so camera frames pass
through as uint8 (the encoder does its own ImageNet normalization) and the zero commands
are left alone. `proprio`, `acts`, `rewards`, `values` are scaled as in Go2. The loss
scales the labels the same way (`sg(scaler.scale(y))`) so image targets are also raw uint8.

---

## 2. Encoder — `ManipulationEncoder`

Turns the history + latest observation into a **history token sequence** and a single
**current latent** `z_t`.

### 2a. Current latent `z_t` — `encode_latent(proprio_t, images_t)`
```
images_t (n_cam, 224,224,3) uint8
   │  shared ResNet-18 (ImageNet), per image, GAP → 512
   ▼
img_feats (n_cam, 512) ── reshape ──► (n_cam·512) = (1536)
                                          │
proprio_t (6) ────────────── concat ─────┤ → (1536 + 6) = (1542)
                                          ▼
                             latent_mlp  MLP(1542 → [128,128] → 64)
                                          ▼
                                   z_t  (D=64)
```
- The **same** ResNet weights process all 3 cameras (weight sharing).
- `imagenet_normalize`: `x/255`, then `(x-μ)/σ` with ImageNet RGB stats, inside the ResNet.
- `encode_latent` is **shape-agnostic in its leading dims**, so the loss reuses it to
  encode the whole future window at once: `images (B,T,n_cam,224,224,3) → z (B,T,64)`.
- ResNet is **not frozen** (`freeze:false`); gradients flow into it.

### 2b. History tokens
```
proprio_hist = proprio_obs_hist[:, :-1]   (H, 6)  ─ proprio_obs_proj MLP(6→[128,128]→64) ─► (H, 64)
acts_hist                                  (H, 7)  ─ act_proj        MLP(7→[]→64)        ─► (H, 64)
z_t (current latent, from 2a)                                                             (64)
```
`act_proj` has `action_layers:0` → a single `Linear(7→64)`.

### 2c. Temporal + type encodings (learned additive biases)
Two parameter tables: `temporal_enc (H+1, D)` and `type_enc (3, D)`.
```
proprio_hist += temporal_enc[:-1]   # positions 0..H-1
acts_hist    += temporal_enc[:-1]   # same positions (proprio & act at a step share time)
z_t          += temporal_enc[-1]    # position H (= current time t)

proprio_hist += type_enc[0]         # "this is a proprio token"
acts_hist    += type_enc[1]         # "this is an action token"
z_t          += type_enc[2]         # "this is the latent token"
```

### 2d. Interleave → history sequence
Proprio and action tokens are woven together in time order:
```
hist_enc (2H, D) = [ p_0, a_0, p_1, a_1, …, p_{H-1}, a_{H-1} ]   # (50, 64)
   even indices ← proprio tokens
   odd  indices ← action tokens
```
Encoder returns `{ history: (2H, D)=(50,64),  latent: z_t (D)=(64) }`.

---

## 3. Context assembly

```
context = concat( history (B,50,64), z_t[:,None] (B,1,64) )  →  (B, 51, 64)
```
This 51-token sequence (25 proprio + 25 action + 1 current-latent) is the memory that
every decoder head cross-attends to.

---

## 4. Latent dynamics — predict `z_{t+1 … t+T}`

The action chunk is embedded and used as the **decoder query**; it cross-attends `context`.
```
fut_acts (B,T,7) ─ fut_acts_embed: Embedding(Linear 7→64 + posemb(T,64)) ─► fut_acts_emb (B,T,64)

latents = TransformerDecoder_dynamics( query = fut_acts_emb, memory = context )
          3 layers · 4 heads · MLP hidden 256 · CAUSAL self-attention
        → latents (B, T, D) = (B,25,64)      # predicted future latents ẑ_{t+1..t+T}
```
Causal masking means step `i` only sees actions `≤ i`, so the same forward pass yields a
consistent autoregressive rollout for all horizons.

---

## 5. Prediction heads

All three heads read `context`/`latents`; each is `Embedding → Transformer → MLP decoder`.
The 🔧 `cmd_dim==0` guards mean the `fut_cmds` concatenations below are **omitted** for
manipulation (they are present for Go2, where `C=3`).

### 5a. Reward head → `rewards (B,T)`
```
z_shift  = concat( z_t[:,None] , latents[:, :-1] )               (B,T,64)   # current + ẑ up to T-1
rew_in   = concat( z_shift , fut_acts )        [🔧 + fut_cmds]    (B,T, 64+7 = 71)
rew_emb  = Embedding(Linear 71→64 + posemb)                      (B,T,64)
rew_enc  = TransformerEncoder(1 layer·1 head·MLP256, no mask)    (B,T,64)
rewards  = MLP(64→1).squeeze                                     (B,T)
```

### 5b. Value head → `values (B,T)`
```
val_in   = latents                              [🔧 + fut_cmds]  (B,T,64)
val_emb  = Embedding(Linear 64→64 + posemb)                      (B,T,64)
val_enc  = TransformerEncoder(1 layer·1 head·MLP256, no mask)    (B,T,64)
values   = MLP(64→1).squeeze                                     (B,T)
```
Values are the expert critic `V^e` — this is where the insertion **goal is implicit**:
`V^e` encodes distance-to-inserted, so MPPI needs no explicit goal input.

### 5c. Base-policy head → `actions (B,T,A)`  🔧
Go2 uses the future-command embedding as the decoder query. Manipulation has no command,
so the query is a **learned temporal-embedding table** `policy_temporal_query (T, D)`,
broadcast over the batch:
```
query   = broadcast( policy_temporal_query (T,64) ) → (B,T,64)
policy  = TransformerDecoder(4 layers·8 heads·MLP256·CAUSAL, memory = context)  (B,T,64)
actions = MLP(64→7)                                                             (B,T,7)
```
Each of the `T` query slots is a learned "position", cross-attending `context` to emit one
action of the chunk. This chunk warm-starts MPPI.

### Outputs
```
{ latents (B,T,64), rewards (B,T), values (B,T), actions (B,T,7) }
```

---

## 6. Training loss — `WorldModelLoss`

```
y_pred = model(x)
y      = sg( scaler.scale(labels) )               # images/cmds pass through (see §1)
z_tgt  = sg( encode_latent(y.proprio, y.images) ) # (B,T,64) future latents, ResNet reused

latent_dynamics = mean( (y_pred.latents − z_tgt)² )        # weight 1.0
reward          = mean( (y_pred.rewards  − y.rewards)² )    # weight 1.0
value           = mean( (y_pred.values   − y.values)²  )    # weight 1.0
action          = mean( ((y_pred.actions − y.actions)·M)² ) # weight 4.0
```
`M = cumprod(exp_pol_flags)` masks the action loss from the first step the expert stopped
acting (once a non-expert action appears, the rest of that window is ignored). The latent
target uses a **stop-gradient** (`sg`) so the dynamics head regresses toward the encoder,
not vice-versa.

---

## 7. End-to-end diagram

```
                       proprio_hist (B,26,6)      images_t (B,3,224,224,3)     acts_hist (B,25,7)
                              │                          │                           │
                    ┌─────────┴──────────┐        shared ResNet-18            act_proj Linear
                    │  [:-1]      [-1]    │         per cam → 512                   │
              proprio_proj   proprio_proj│               │                         │
                 (B,25,64)      (B,6)→ concat ◄── (B,3·512=1536)                    │
                    │              │  latent_mlp → z_t (B,64)                        │
                    │              └──────────┬─────────────────┐                   │
              +temporal/type            +temporal/type          │             +temporal/type
                    │                        │                  │                   │
                    └──────── interleave ────┼──────────────────┼─────── acts ──────┘
                                   history (B,50,64)            │
                                             └── concat ────────┘
                                                   context (B,51,64)
                                                        │
         fut_acts (B,25,7) ─embed─► (B,25,64) ─query──► DYNAMICS decoder ──► latents ẑ (B,25,64)
                                                        │(memory)                 │
                     ┌──────────────────────────────────┼─────────────────┐      │
                     ▼                                   ▼                 ▼      ▼
              policy query (learned      value: latents→emb→enc→MLP   reward: [z_shift,fut_acts]
              temporal table, B,25,64)          →values (B,25)         →emb→enc→MLP→rewards(B,25)
                     ▼
              POLICY decoder (memory=context) → MLP → actions (B,25,7)
```

---

## 8. What differs from the Go2 quadruped model (🔧 summary)

| Aspect | Go2 | Manipulation |
|---|---|---|
| Extero input | flat height-scan vector, scaled | 3× RGB `(224,224,3)` uint8, **unscaled** |
| Extero encoder | `HeightMapCNN` | **shared ResNet-18 (ImageNet)** over 3 cams |
| Command `C` | velocity command (`C>0`) | **0** (no goal command) |
| Reward/value concat | append `fut_cmds` | **skipped** (cmd guard) |
| Policy-head query | `fut_cmds` embedding | **learned `policy_temporal_query (T,D)`** |
| `fut_cmds_embed` | built | **not built** (0-input Linear would div-by-zero) |
| Scaler mapping | includes `extero_obs`, `fut_cmds` | **excludes both** |

Everything else — latent dim, transformer stacks, embeddings, temporal/type encodings,
context assembly, the four loss terms — is identical to `QuadrupedWorldModel`.
