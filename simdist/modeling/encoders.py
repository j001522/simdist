import flax.nnx as nnx
import jax.numpy as jnp
import jax

from simdist.modeling import types, modules, resnet
from simdist.modeling import dinov2 as dinov2_mod
from simdist.utils import config, extero, paths


class WorldModelEncoderBase(nnx.Module):
    def __init__(self, cfg: dict, rngs: nnx.Rngs, **kwargs):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.sys_cfg = cfg["system"]
        self.enc_cfg = self.model_cfg["encoder"]
        self.proprio_obs_dim = config.proprio_obs_dim_from_sys_config(self.sys_cfg)
        self.extero_obs_dim = config.extero_obs_dim_from_sys_config(self.sys_cfg)
        self.act_dim = config.action_dim_from_sys_config(self.sys_cfg)
        self.latent_dim = self.model_cfg["latent_dim"]

    def __call__(
        self,
        x: types.WorldModelSchema.Inputs,
        deterministic: bool | None = None,
    ) -> types.WorldModelSchema.Encoding:
        raise NotImplementedError("Must implement __call__ method")

    def encode_latent(
        self,
        proprio_obs: jnp.ndarray,
        extero_obs: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        raise NotImplementedError("Must implement encode_latent method")


class QuadrupedEncoder(WorldModelEncoderBase):
    def __init__(self, cfg: dict, rngs: nnx.Rngs):
        super().__init__(cfg, rngs)

        h_size = self.latent_dim * self.enc_cfg["mlp_hidden_size_factor"]
        H = self.model_cfg["dataset"]["history_length"]
        self.hist_enc_len = 2 * H

        self.proprio_obs_proj = modules.MLP(
            input_dim=self.proprio_obs_dim,
            hidden_dims=[h_size] * self.enc_cfg["proprio_obs_layers"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )
        self.act_proj = modules.MLP(
            input_dim=self.act_dim,
            hidden_dims=[h_size] * self.enc_cfg["action_layers"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )
        self.hm_enc = HeightMapEncoder(cfg, rngs)
        self.latent_mlp = modules.MLP(
            input_dim=2 * self.latent_dim,
            hidden_dims=[h_size] * self.enc_cfg["latent_layers"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )

        self.temporal_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (H + 1, self.latent_dim)) * 0.02
        )

        num_types = 3  # proprio_obs, act, latent
        self.type_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (num_types, self.latent_dim)) * 0.02
        )

    def __call__(
        self,
        x: types.WorldModelSchema.Inputs,
        deterministic: bool | None = None,
    ) -> types.WorldModelSchema.Encoding:
        B = x["proprio_obs_hist"].shape[0]

        # encode history
        proprio_obs_hist = x["proprio_obs_hist"][:, :-1]
        proprio_obs_hist = self.proprio_obs_proj(
            proprio_obs_hist, deterministic=deterministic
        )
        act_hist = self.act_proj(x["acts_hist"], deterministic=deterministic)

        # encode latent
        last_proprio_obs = x["proprio_obs_hist"][:, -1]
        latent = self.encode_latent(
            last_proprio_obs, x["extero_obs"], deterministic=deterministic
        )

        # temporal encoding
        proprio_obs_hist += self.temporal_enc[:-1]
        act_hist += self.temporal_enc[:-1]
        latent += self.temporal_enc[-1]  # last time step

        # type encoding
        proprio_obs_hist += self.type_enc[0]
        act_hist += self.type_enc[1]
        latent += self.type_enc[2]

        # interleave history
        hist_enc = jnp.zeros((B, self.hist_enc_len, self.latent_dim))
        hist_enc = hist_enc.at[:, 0::2].set(proprio_obs_hist)
        hist_enc = hist_enc.at[:, 1::2].set(act_hist)

        return {"history": hist_enc, "latent": latent}

    def encode_latent(
        self,
        proprio_obs: jnp.ndarray,
        extero_obs: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        proprio_obs = self.proprio_obs_proj(proprio_obs, deterministic=deterministic)
        hm_enc = self.hm_enc(extero_obs, deterministic=deterministic)
        concatenated = jnp.concatenate([proprio_obs, hm_enc], axis=-1)
        latent = self.latent_mlp(concatenated, deterministic=deterministic)
        return latent


class ManipulationEncoder(WorldModelEncoderBase):
    """Vision encoder (paper app:manip): each of the 3 cameras -> a SHARED ResNet-18
    -> 3x512, concat the raw 6 joint obs -> MLP -> z(latent_dim).

    History path mirrors QuadrupedEncoder (proprio + action tokens with temporal/type
    encodings). Per "minimal history", only the LATEST image enters encode_latent; the
    history carries no images. extero_obs in the Inputs schema holds the camera images
    (..., n_cam, H, W, 3), replacing Go2's flat height-scan vector.

    The ResNet is random-initialized here; ImageNet weights are loaded as an explicit
    pipeline step (trainer calls resnet.load_torchvision_resnet18(encoder.resnet) when
    encoder.extero_obs.resnet.pretrained == "imagenet"). Keeps __init__ torch-free.
    """

    def __init__(self, cfg: dict, rngs: nnx.Rngs):
        super().__init__(cfg, rngs)

        h_size = self.latent_dim * self.enc_cfg["mlp_hidden_size_factor"]
        H = self.model_cfg["dataset"]["history_length"]
        self.hist_enc_len = 2 * H

        extero_cfg = self.enc_cfg["extero_obs"]
        self.n_cam = len(config.extero_obs_names_from_sys_config(self.sys_cfg))

        # Backbone: frozen DINOv2 (encoder.extero_obs.dinov2) or the fine-tuned
        # ResNet-18 (encoder.extero_obs.resnet). Held as two separate attributes rather
        # than one renamed `backbone` so that ResNet runs trained before this branch keep
        # their orbax state layout and still resume.
        self.use_dinov2 = "dinov2" in extero_cfg
        if self.use_dinov2:
            dino_cfg = extero_cfg["dinov2"]
            shapes = config.extero_obs_image_shapes_from_sys_config(self.sys_cfg)
            sizes = {(s[0], s[1]) for s in shapes}
            if len(sizes) != 1:
                raise ValueError(
                    f"DINOv2 encoder needs one image size for all cameras, got {sizes}"
                )
            h, w = sizes.pop()
            if h != w:
                raise ValueError(f"DINOv2 encoder expects square images, got {h}x{w}")
            npz = paths.resolve_asset_path(dino_cfg["weights"])
            self.dinov2 = dinov2_mod.get_dinov2_backbone(dino_cfg, h, npz)
            self.resnet = None
            # Frozen unless `dinov2.trainable_blocks > 0`. When frozen the weights are
            # FrozenParam and the optimizer never sees them (see dinov2.py), so this flag
            # only drives the extra stop_gradient below; when fine-tuning it must be off
            # or that stop_gradient would sever the very gradient we are adding.
            self.freeze_resnet = self.dinov2.trainable_blocks == 0
            self.per_image_embed_dim = self.dinov2.out_features
        else:
            resnet_cfg = extero_cfg["resnet"]
            self.per_image_embed_dim = resnet_cfg["per_image_embed_dim"]
            self.freeze_resnet = resnet_cfg.get("freeze", False)
            self.resnet = resnet.ResNet18Backbone(rngs=rngs)
            self.dinov2 = None
            assert self.resnet.out_features == self.per_image_embed_dim

        self.proprio_obs_proj = modules.MLP(
            input_dim=self.proprio_obs_dim,
            hidden_dims=[h_size] * self.enc_cfg["proprio_obs_layers"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )
        self.act_proj = modules.MLP(
            input_dim=self.act_dim,
            hidden_dims=[h_size] * self.enc_cfg["action_layers"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )
        # Proprio branch into the LATENT (distinct from the history path above, which
        # keeps its own projection to latent_dim for the transformer tokens).
        #
        # `proprio_embed_dim` absent/null/0 reproduces the paper's raw concat: the 6 joint
        # obs enter latent_mlp unprojected, i.e. 6 of 1542 (ResNet) or 6 of 2310 (DINOv2
        # cls_mean) input dims -- 0.26-0.39%. Scale is NOT the problem (proprio is already
        # scaler-standardised to ~unit std and the image blocks are affine-free
        # LayerNormed to exactly unit std); the first Linear is lecun_normal, so every
        # input dim contributes equal variance and proprio's share of the pre-activation
        # variance is just its share of the dims. It is a dimension-count problem, which
        # is why the existing debug/proprio_scaled_std vs debug/image_feat_std pair cannot
        # see it.
        #
        # Setting it projects proprio to its own block first, so the concat is
        # [n_cam blocks of images | 1 block of proprio]. "image" sizes that block to
        # per_image_embed_dim, i.e. proprio counts as one more camera: 512 (ResNet),
        # 384 (DINOv2 `cls`), 768 (DINOv2 `cls_mean`) -> a 20-25% share.
        #
        # This also moves the PREDICTION TARGET: losses.py encodes the future obs through
        # this same encode_latent, so under the raw concat the latent-dynamics target is
        # ~99.7% image and the dynamics head is barely graded on joint motion at all.
        #
        # Old checkpoints are unaffected -- load_model_from_ckpt rebuilds from each run's
        # own model_config.yaml snapshot, which predates this key. New runs are a
        # different state layout and deliberately cannot resume from them.
        self.proprio_embed_dim = self._resolve_proprio_embed_dim(
            self.enc_cfg.get("proprio_embed_dim")
        )
        if self.proprio_embed_dim:
            # Depth/width reuse `proprio_obs_layers` and h_size for parity with
            # proprio_obs_proj. Deliberately NOT weight-shared with it: that one is
            # pinned to latent_dim by the history tokens, and its representation serves a
            # different role (a sequence token, with temporal/type encodings added).
            # Deliberately not a bare Linear either -- a linear 6->512 is rank-6, so it
            # rebalances the init variance but adds no capacity; the gelu MLP does.
            self.proprio_latent_proj = modules.MLP(
                input_dim=self.proprio_obs_dim,
                hidden_dims=[h_size] * self.enc_cfg["proprio_obs_layers"],
                output_dim=self.proprio_embed_dim,
                rngs=rngs,
            )
        else:
            self.proprio_latent_proj = None
        proprio_block_dim = self.proprio_embed_dim or self.proprio_obs_dim

        # concat = [n_cam x per_image_embed_dim image feats, proprio block] -> z(latent_dim)
        self.latent_mlp = modules.MLP(
            input_dim=self.n_cam * self.per_image_embed_dim + proprio_block_dim,
            hidden_dims=[h_size] * self.enc_cfg["latent_layers"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )

        self.temporal_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (H + 1, self.latent_dim)) * 0.02
        )
        num_types = 3  # proprio_obs, act, latent
        self.type_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (num_types, self.latent_dim)) * 0.02
        )

        # Per-camera feature norm: each camera's pooled feature is normalized
        # independently (applied before the n_cam*embed flatten in encode_latent), so one
        # camera's raw scale can't skew the shared statistic the others get normalized
        # against, and so the image branch isn't dominating the proprio branch purely on
        # magnitude/dimension count going into latent_mlp.
        # use_scale/use_bias off: a learnable gamma would let the encoder re-inflate this
        # branch, which is the pressure we're removing. Affine-free LayerNorm therefore
        # has no parameters at all, which is why "blockwise" below can afford a second one.
        #
        # feature_norm modes:
        #   "layernorm" -- one LayerNorm over the whole per-image vector (ResNet default,
        #                  unchanged behaviour).
        #   "blockwise" -- DINOv2 "cls_mean" pooling concatenates two blocks with
        #                  different natural scales (the CLS token vs the mean of 256
        #                  patch tokens, which is an average and so shrinks toward its
        #                  own mean). A single LayerNorm over the concatenation rescales
        #                  both by one shared statistic and preserves that imbalance;
        #                  normalizing each block separately removes it.
        #   "none"      -- feed raw backbone output; the first Linear of latent_mlp has
        #                  to absorb the scale.
        self.feature_norm = extero_cfg.get(
            "feature_norm", "blockwise" if self.use_dinov2 else "layernorm"
        )
        if self.feature_norm not in ("layernorm", "blockwise", "none"):
            raise ValueError(f"unknown feature_norm {self.feature_norm!r}")
        self.norm_block_dim = None
        if self.feature_norm == "blockwise":
            if not self.use_dinov2:
                raise ValueError("feature_norm='blockwise' is only defined for DINOv2")
            # cls_mean splits into two equal halves; cls/mean pooling is a single block,
            # in which case blockwise degenerates to plain layernorm.
            self.norm_block_dim = (
                self.dinov2.hidden_size
                if self.dinov2.pooling == "cls_mean"
                else self.per_image_embed_dim
            )
        self.layer_norm_1 = nnx.LayerNorm(
            self.norm_block_dim or self.per_image_embed_dim,
            use_scale=False,
            use_bias=False,
            rngs=rngs,
        )
        self.layer_norm_2 = nnx.LayerNorm(
            self.latent_dim, use_scale=False, use_bias=False, rngs=rngs
        )

        # Normalization of the PROJECTED proprio block, so the whole point of the
        # projection -- a balanced concat -- isn't undone by the projection's own output
        # scale (a 2-layer gelu MLP lands near std 0.5, not 1, so without this the block
        # arrives at ~1/4 the per-feature variance of each LayerNormed image block).
        # Affine-free like the image norms, for the same reason: a learnable gamma would
        # let the encoder re-inflate or re-deflate the branch.
        #
        # Caveat worth knowing before reading an ablation: LayerNorm removes the
        # mean-over-features and the norm, which with only 6 informative inputs is a
        # meaningful fraction of the signal (the 2-layer gelu makes those nonlinear
        # functions of the joints rather than one clean linear direction, which dilutes
        # but does not eliminate the cost). Hence `proprio_norm: none` as an escape hatch:
        # it keeps every dof at the price of an unguaranteed scale match.
        # NEVER normalize the RAW 6-vector -- there it would delete real joint signal, and
        # the scaler has already standardised it per-dim anyway.
        self.proprio_norm = self.enc_cfg.get(
            "proprio_norm", "layernorm" if self.proprio_embed_dim else "none"
        )
        if self.proprio_norm not in ("layernorm", "none"):
            raise ValueError(f"unknown proprio_norm {self.proprio_norm!r}")
        if self.proprio_norm == "layernorm" and not self.proprio_embed_dim:
            raise ValueError(
                "proprio_norm='layernorm' requires proprio_embed_dim to be set; "
                "normalizing the raw proprio vector would discard joint signal."
            )
        # Affine-free LayerNorm has no parameters, so creating this conditionally does not
        # change the checkpointed state layout either way.
        self.layer_norm_proprio = (
            nnx.LayerNorm(
                self.proprio_embed_dim, use_scale=False, use_bias=False, rngs=rngs
            )
            if self.proprio_norm == "layernorm"
            else None
        )

        # Debug-only captures for TensorBoard (see trainer.py's "debug/" metrics). Plain
        # nnx.Intermediate state -- same mechanism nnx.BatchNorm uses for its running
        # stats, so it's inert to grads/optimizer.update and doesn't touch any existing
        # return signature.
        # encode_latent runs twice per step (current obs here in __call__, future target
        # in WorldModelLoss via model.encode_latent) and would clobber a single slot with
        # whichever ran last. "_input" is the one __call__ snapshots right after its own
        # (current-obs) call, so it always reflects THIS step's input, not the target.
        #
        # Image feats: mean/std (not norm) of the RAW pre-LayerNorm ResNet output. Norm
        # is the wrong metric here -- after layer_norm_1 it's mathematically pinned at
        # sqrt(512) forever (mean=0/var=1 per feature is exactly what affine-free
        # LayerNorm guarantees), so it can never show anything changing. Pre-LayerNorm
        # mean/std is the part that actually moves as the backbone fine-tunes, and mean/
        # std (unlike norm) aren't dimension-dependent, so they're the fair comparison
        # against proprio's mean/std (models.py debug_proprio_scaled_*) despite the huge
        # dimension mismatch (6 vs 1536).
        self.debug_image_feat_mean = nnx.Intermediate(jnp.zeros(()))
        self.debug_image_feat_std = nnx.Intermediate(jnp.zeros(()))
        self.debug_latent_norm = nnx.Intermediate(jnp.zeros(()))
        self.debug_image_feat_mean_input = nnx.Intermediate(jnp.zeros(()))
        self.debug_image_feat_std_input = nnx.Intermediate(jnp.zeros(()))
        self.debug_latent_norm_input = nnx.Intermediate(jnp.zeros(()))
        # Proprio block as it ARRIVES AT THE CONCAT (post-projection, pre-LayerNorm), so
        # it is directly comparable against debug_image_feat_* above -- that is the pair
        # that says whether the two branches actually reach latent_mlp balanced. Under
        # the raw-concat default this just restates debug/proprio_scaled_*.
        self.debug_proprio_feat_mean = nnx.Intermediate(jnp.zeros(()))
        self.debug_proprio_feat_std = nnx.Intermediate(jnp.zeros(()))
        self.debug_proprio_feat_mean_input = nnx.Intermediate(jnp.zeros(()))
        self.debug_proprio_feat_std_input = nnx.Intermediate(jnp.zeros(()))
        # Writing an Intermediate works under trainer.py's nnx.jit (the module is an
        # nnx argument, so nnx threads the state out) but NOT under the MPPI planner's
        # jit, which closes over the model -> TraceContextError "cannot mutate
        # Intermediate from a different trace level". Inference turns these off; see
        # mppi_server.build_controller.
        self.collect_debug_stats = True

    def _resolve_proprio_embed_dim(self, value) -> int:
        """`encoder.proprio_embed_dim` -> width of the proprio block, 0 = raw concat.

        Accepts None/0/"none" (disabled, the pre-existing behaviour), the string "image"
        (match per_image_embed_dim, i.e. proprio counts as one more camera), or an int.
        """
        if value is None or value == 0 or value == "none":
            return 0
        if value == "image":
            return int(self.per_image_embed_dim)
        try:
            dim = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"proprio_embed_dim must be null, 'none', 'image' or an int, "
                f"got {value!r}"
            ) from None
        if dim < 0:
            raise ValueError(f"proprio_embed_dim must be >= 0, got {dim}")
        return dim

    def __call__(
        self,
        x: types.WorldModelSchema.Inputs,
        deterministic: bool | None = None,
    ) -> types.WorldModelSchema.Encoding:
        B = x["proprio_obs_hist"].shape[0]

        # encode history (proprio + action tokens); no images in history
        proprio_obs_hist = x["proprio_obs_hist"][:, :-1]
        proprio_obs_hist = self.proprio_obs_proj(
            proprio_obs_hist, deterministic=deterministic
        )
        act_hist = self.act_proj(x["acts_hist"], deterministic=deterministic)

        # encode latent from the most recent proprio + the latest camera images
        last_proprio_obs = x["proprio_obs_hist"][:, -1]
        latent = self.encode_latent(
            last_proprio_obs, x["extero_obs"], deterministic=deterministic
        )
        # Snapshot debug stats now, before WorldModelLoss's separate encode_latent call
        # (on the future target) overwrites the scratch slots above.
        if self.collect_debug_stats:
            self.debug_image_feat_mean_input.value = self.debug_image_feat_mean.value
            self.debug_image_feat_std_input.value = self.debug_image_feat_std.value
            self.debug_latent_norm_input.value = self.debug_latent_norm.value
            self.debug_proprio_feat_mean_input.value = self.debug_proprio_feat_mean.value
            self.debug_proprio_feat_std_input.value = self.debug_proprio_feat_std.value

        proprio_obs_hist += self.temporal_enc[:-1]
        act_hist += self.temporal_enc[:-1]
        latent += self.temporal_enc[-1]

        proprio_obs_hist += self.type_enc[0]
        act_hist += self.type_enc[1]
        latent += self.type_enc[2]

        hist_enc = jnp.zeros((B, self.hist_enc_len, self.latent_dim))
        hist_enc = hist_enc.at[:, 0::2].set(proprio_obs_hist)
        hist_enc = hist_enc.at[:, 1::2].set(act_hist)

        return {"history": hist_enc, "latent": latent}

    def encode_latent(
        self,
        proprio_obs: jnp.ndarray,
        extero_obs: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        # extero_obs: (..., n_cam, H, W, 3) camera images. ResNet flattens leading dims;
        # shape-agnostic so this also serves target encoding E(o_{t+1:t+T}) in the loss.
        backbone = self.dinov2 if self.use_dinov2 else self.resnet
        feats = backbone(
            extero_obs, train=not bool(deterministic), normalize=True
        )  # (..., n_cam, per_image_embed_dim)
        if self.freeze_resnet:
            feats = jax.lax.stop_gradient(feats)

        # Capture BEFORE layer_norm_1: post-norm this is pinned at mean=0/std=1 by
        # construction, so it's the raw feature that actually reflects backbone training.
        # With a frozen backbone these are constants of the dataset rather than a training
        # signal -- still worth logging, since a drift means the INPUT distribution moved.
        if self.collect_debug_stats:
            self.debug_image_feat_mean.value = feats.mean()
            self.debug_image_feat_std.value = feats.std()

        if self.feature_norm == "blockwise":
            # Normalize each pooling block independently, then re-concatenate.
            blocks = jnp.split(feats, feats.shape[-1] // self.norm_block_dim, axis=-1)
            feats = jnp.concatenate([self.layer_norm_1(b) for b in blocks], axis=-1)
        elif self.feature_norm == "layernorm":
            feats = self.layer_norm_1(feats)

        feats = feats.reshape(
            feats.shape[:-2] + (self.n_cam * self.per_image_embed_dim,)
        )

        # Proprio block: raw (paper default) or projected to its own camera-sized block.
        if self.proprio_latent_proj is not None:
            proprio_feat = self.proprio_latent_proj(
                proprio_obs, deterministic=deterministic
            )
        else:
            proprio_feat = proprio_obs
        # Captured pre-norm, mirroring the image feats above: post-LayerNorm this is
        # pinned at mean=0/std=1 by construction and could never show anything moving.
        if self.collect_debug_stats:
            self.debug_proprio_feat_mean.value = proprio_feat.mean()
            self.debug_proprio_feat_std.value = proprio_feat.std()
        if self.layer_norm_proprio is not None:
            proprio_feat = self.layer_norm_proprio(proprio_feat)

        concatenated = jnp.concatenate([feats, proprio_feat], axis=-1)
        enc_latent = self.latent_mlp(concatenated, deterministic=deterministic)
        if self.collect_debug_stats:
            self.debug_latent_norm.value = jnp.linalg.norm(enc_latent, axis=-1).mean()
        enc_latent = self.layer_norm_2(enc_latent)
        return enc_latent


class HeightMapEncoder(nnx.Module):
    def __init__(
        self,
        cfg: dict,
        rngs: nnx.Rngs,
    ):
        model_cfg = cfg["model"]
        sys_cfg = cfg["system"]
        cnn_cfg = model_cfg["encoder"]["extero_obs"]["height_cnn"]
        self.latent_dim = model_cfg["latent_dim"]
        hx, hy = config.height_map_dims_from_sys_cfg(sys_cfg)

        self.hm_cnn = HeightMapCNN(
            h_l=hx,
            h_w=hy,
            latent_dim=cnn_cfg["features"][-1],
            features=cnn_cfg["features"][:-1],
            strides=cnn_cfg["strides"],
            kernel_size=cnn_cfg["kernel_size"],
            rngs=rngs,
        )
        self.h_l_conv, self.h_w_conv, self.h_lat_dim = self.hm_cnn.output_shape
        self.hm_proj = modules.MLP(
            input_dim=self.h_l_conv * self.h_w_conv * self.h_lat_dim,
            hidden_dims=cnn_cfg["projection_hidden_dims"],
            output_dim=self.latent_dim,
            rngs=rngs,
        )

        self.h_l_conv_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (self.h_l_conv, self.h_lat_dim)) * 0.02
        )
        self.h_w_conv_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (self.h_w_conv, self.h_lat_dim)) * 0.02
        )

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool | None = None,
    ):
        # cnn
        hm_enc = self.hm_cnn(x)
        # spatial encoding, flatten, and projection
        hm_enc = self._spatial_encoding(hm_enc)
        hm_enc = hm_enc.reshape(
            hm_enc.shape[:-3] + (self.h_l_conv * self.h_w_conv * self.h_lat_dim,)
        )
        hm_enc = self.hm_proj(hm_enc, deterministic=deterministic)
        return hm_enc

    @property
    def height_map_encoded_shape(self):
        return self.h_l_conv, self.h_w_conv, self.h_lat_dim

    def _spatial_encoding(self, x: jnp.ndarray) -> jnp.ndarray:
        h_l_conv_enc = self.h_l_conv_enc[:, None, :]  # (H, 1, D)
        h_w_conv_enc = self.h_w_conv_enc[None, :, :]  # (1, W, D)
        enc = h_l_conv_enc + h_w_conv_enc  # (H, W, D)
        return x + enc


class HeightMapCNN(nnx.Module):
    def __init__(
        self,
        h_l: int,  # Input height
        h_w: int,  # Input width
        latent_dim: int,
        features: list[int],
        strides: list[int],
        kernel_size: int,
        rngs: nnx.Rngs,
    ):
        self.h_l = h_l
        self.h_w = h_w

        self.cnn = modules.CNN(
            in_channels=1,
            features=features,
            strides=strides,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            ht_in=self.h_l,
            wd_in=self.h_w,
            rngs=rngs,
        )

    @property
    def output_shape(self):
        return self.cnn.output_shape

    def __call__(self, x: jnp.ndarray):
        # Reshape from flattened input to (h_l, h_w, 1)
        x = extero.restore_height_map(x, self.h_l, self.h_w)
        x = jnp.expand_dims(x, axis=-1)
        x = self.cnn(x)
        return x  # Shape: (..., h_l_conv, h_w_conv, latent_dim)
