import flax.nnx as nnx
import jax.numpy as jnp
import jax

from simdist.modeling import types, modules, resnet
from simdist.utils import config, extero


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

        resnet_cfg = self.enc_cfg["extero_obs"]["resnet"]
        self.n_cam = len(config.extero_obs_names_from_sys_config(self.sys_cfg))
        self.per_image_embed_dim = resnet_cfg["per_image_embed_dim"]
        self.freeze_resnet = resnet_cfg.get("freeze", False)
        self.resnet = resnet.ResNet18Backbone(rngs=rngs)
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
        # concat = [3x512 image feats, raw 6 joint obs] -> z(latent_dim)
        self.latent_mlp = modules.MLP(
            input_dim=self.n_cam * self.per_image_embed_dim + self.proprio_obs_dim,
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

        # Bounds ||z||: the latent-dynamics target is the encoder's own output under
        # stop_grad, so nothing else anchors its scale (it inflated 4 -> 46 in 600 steps
        # and the dynamics MSE followed). Applied inside encode_latent so the target and
        # encoding["latent"] are normalized identically.
        # use_scale/use_bias off: a learnable gamma would let the encoder re-inflate the
        # latent, which is the pressure we are removing. ||z|| is then pinned at
        # sqrt(latent_dim), same property SimNorm gives TD-MPC2.
        self.layer_norm_1 = nnx.LayerNorm(
            self.latent_dim, use_scale=False, use_bias=False, rngs=rngs
        )

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
        feats = self.resnet(
            extero_obs, train=not bool(deterministic), normalize=True
        )  # (..., n_cam, 512)
        if self.freeze_resnet:
            feats = jax.lax.stop_gradient(feats)
        feats = feats.reshape(
            feats.shape[:-2] + (self.n_cam * self.per_image_embed_dim,)
        )
        concatenated = jnp.concatenate([feats, proprio_obs], axis=-1)
        enc_latent = self.latent_mlp(concatenated, deterministic=deterministic)
        enc_latent = self.layer_norm_1(enc_latent)
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
