import flax.nnx as nnx
import jax
import jax.numpy as jnp

from simdist.modeling import types, scaler, encoders, modules
from simdist.utils import config, registry


_MODEL_REGISTRY: registry.Registry["ModelBase"] = registry.Registry("Model")


def register_model(name: str):
    return _MODEL_REGISTRY.register(name)


def get_model(
    cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs
) -> "ModelBase":
    model_name = cfg["model"]["type"]
    return _MODEL_REGISTRY.create(model_name, cfg, scaler_params, rngs)


class ModelBase(nnx.Module):
    def __init__(
        self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs, **kwargs
    ):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.sys_cfg = cfg["system"]

    def __call__(
        self,
        x: types.ModelInputs,
        deterministic: bool | None = None,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement __call__ method")

    def inference(
        self,
        x: types.ModelInputs,
        **kwargs,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement inference method")


class WorldModelBase(ModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.proprio_obs_dim = config.proprio_obs_dim_from_sys_config(self.sys_cfg)
        self.extero_obs_dim = config.extero_obs_dim_from_sys_config(self.sys_cfg)
        self.action_dim = config.action_dim_from_sys_config(self.sys_cfg)
        self.cmd_dim = config.cmd_dim_from_sys_config(self.sys_cfg)
        self.latent_dim = self.model_cfg["latent_dim"]
        self.mlp_dropout_rate = self.model_cfg["dropout"]["mlp"]
        self.attention_dropout_rate = self.model_cfg["dropout"]["attention"]
        T = self.model_cfg["dataset"]["prediction_length"]
        emb_cfg = self.model_cfg["embedding"]
        emb_mlp_hsf = emb_cfg["mlp_hidden_size_factor"]
        emb_hidden_size = emb_mlp_hsf * self.latent_dim

        # input processing
        self.scaler = scaler.Scaler(
            scaler_params,
            types.WorldModelSchema.scaler_params_mapping,
        )
        self.encoder = encoders.WorldModelEncoderBase(cfg, rngs)
        self.fut_acts_embed = modules.Embedding(
            seq_len=T,
            input_dim=self.action_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["future_acts_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        # cmd_dim == 0 (e.g. manipulation: no goal command) would give a 0-input Linear
        # whose fan-in init divides by zero; skip the embed entirely — the policy-head
        # query is then supplied by a subclass override (see _policy_query).
        if self.cmd_dim > 0:
            self.fut_cmds_embed = modules.Embedding(
                seq_len=T,
                input_dim=self.cmd_dim,
                hidden_dims=[emb_hidden_size] * emb_cfg["future_cmds_layers"],
                embed_dim=self.latent_dim,
                rngs=rngs,
            )
        else:
            self.fut_cmds_embed = None

        # dynamics
        attn_cfg = self.model_cfg["dynamics"]["attention"]
        self.dynamics = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )

        # reward head
        attn_cfg = self.model_cfg["reward"]["attention"]
        dec_cfg = self.model_cfg["reward"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.reward_emb = modules.Embedding(
            seq_len=T,
            input_dim=self.latent_dim + self.action_dim + self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["reward_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.reward = modules.TransformerEncoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.reward_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=1,
            rngs=rngs,
        )

        # value head
        attn_cfg = self.model_cfg["value"]["attention"]
        dec_cfg = self.model_cfg["value"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.value_emb = modules.Embedding(
            seq_len=T,
            input_dim=self.latent_dim + self.cmd_dim,
            hidden_dims=[emb_hidden_size] * emb_cfg["value_layers"],
            embed_dim=self.latent_dim,
            rngs=rngs,
        )
        self.value = modules.TransformerEncoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.value_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=1,
            rngs=rngs,
        )

        # policy head
        attn_cfg = self.model_cfg["policy"]["attention"]
        dec_cfg = self.model_cfg["policy"]["decoder"]
        dec_h_size = self.latent_dim * dec_cfg["mlp_hidden_size_factor"]
        self.policy = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.latent_dim,
            mlp_hidden_dim=self.latent_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.policy_dec = modules.MLP(
            input_dim=self.latent_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=self.action_dim,
            rngs=rngs,
        )

        # Debug-only capture (see trainer.py "debug/" metrics): mean/std of the SCALED
        # proprio history, i.e. what the encoder actually receives, not the raw batch.
        # mean/std (not norm) so it's directly comparable across signals of different
        # dimensionality -- norm grows with sqrt(dim), mean/std doesn't.
        self.debug_proprio_scaled_mean = nnx.Intermediate(jnp.zeros(()))
        self.debug_proprio_scaled_std = nnx.Intermediate(jnp.zeros(()))

    def __call__(
        self,
        x: types.WorldModelSchema.Inputs,
        deterministic: bool | None = None,
    ) -> types.WorldModelSchema.Outputs:

        # pre-processing, encoding, and embedding
        x = self.scaler.scale(x)
        self.debug_proprio_scaled_mean.value = x["proprio_obs_hist"].mean()
        self.debug_proprio_scaled_std.value = x["proprio_obs_hist"].std()
        encoding = self.encoder(x, deterministic=deterministic)
        fut_acts_emb = self.fut_acts_embed(x["fut_acts"], deterministic=deterministic)
        if self.fut_cmds_embed is not None:
            fut_cmds_emb = self.fut_cmds_embed(
                x["fut_cmds"][:, :-1], deterministic=deterministic
            )
        else:
            fut_cmds_emb = None

        # concatenate latent to the end of the history encoding
        latent_enc = jnp.expand_dims(encoding["latent"], axis=1)
        context = jnp.concatenate([encoding["history"], latent_enc], axis=1)

        # dynamics
        latents = self.dynamics(fut_acts_emb, context, deterministic=deterministic)

        # reward head
        # concatenate last latent with latent prediction
        z_t_tm1 = jnp.concatenate(
            (encoding["latent"][:, None, :], latents[:, :-1]), axis=1
        )
        # concatenate future actions (and commands, when present) to latents
        rew_parts = [z_t_tm1, x["fut_acts"]]
        if self.cmd_dim > 0:
            rew_parts.append(x["fut_cmds"][:, :-1])
        rew_in = jnp.concatenate(rew_parts, axis=-1)
        # embedding
        rew_in_emb = self.reward_emb(rew_in, deterministic=deterministic)
        # prediction
        rew_pred = self.reward(rew_in_emb, deterministic=deterministic)
        rewards = self.reward_dec(rew_pred, deterministic=deterministic).squeeze()

        # value head
        # concatenate latent prediction with future commands (when present)
        value_parts = [latents]
        if self.cmd_dim > 0:
            value_parts.append(x["fut_cmds"][:, 1:])
        value_in = jnp.concatenate(value_parts, axis=-1)
        # embedding
        value_in_emb = self.value_emb(value_in, deterministic=deterministic)
        # prediction
        value_pred = self.value(value_in_emb, deterministic=deterministic)
        values = self.value_dec(value_pred, deterministic=deterministic).squeeze()

        # policy head
        policy_query = self._policy_query(x, fut_cmds_emb, deterministic=deterministic)
        latent_acts_pred = self.policy(
            policy_query, context, deterministic=deterministic
        )
        actions = self.policy_dec(latent_acts_pred, deterministic=deterministic)

        return {
            "latents": latents,
            "rewards": rewards,
            "values": values,
            "actions": actions,
        }

    def _policy_query(
        self,
        x: types.WorldModelSchema.Inputs,
        fut_cmds_emb: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """Decoder query for the base-policy head. Default (Go2): the future-command
        embedding, so the policy is conditioned on the goal command. Manipulation
        overrides this with learned temporal embeddings because there is no goal
        command (cmd_dim=0); see manipulation_port.md 3.2."""
        return fut_cmds_emb

    def inference(
        self,
        x: types.WorldModelSchema.Inputs,
    ) -> types.WorldModelSchema.Outputs:
        y = self(x, deterministic=True)
        y = self.scaler.unscale(y)
        return y

    def encode_latent(
        self,
        proprio_obs: jnp.ndarray,
        extero_obs: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """For latent dynamics consistency loss"""
        return self.encoder.encode_latent(
            proprio_obs, extero_obs, deterministic=deterministic
        )

    def get_scaler(self) -> scaler.Scaler:
        """
        Return the scaler.
        """
        return self.scaler


@register_model("quadruped_world_model")
class QuadrupedWorldModel(WorldModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.encoder = encoders.QuadrupedEncoder(cfg, rngs)


@register_model("manipulation_world_model")
class ManipulationWorldModel(WorldModelBase):
    """UR5e peg-insertion world model (paper app:manip).

    Same latent-dynamics core as the quadruped model, differing in three places:
      1. a vision encoder (``ManipulationEncoder``) over the 3 RGB cameras;
      2. images are left UNSCALED — the encoder does ImageNet normalization — by
         dropping ``extero_obs`` from the scaler mapping (both the input scaling here
         and the label scaling in ``WorldModelLoss`` pass the uint8 frames through);
      3. the base-policy decoder query is a learned temporal (positional) embedding
         table instead of the future-command embedding, because peg insertion has no
         goal command (cmd_dim=0). See manipulation_port.md 3.2 / 3.3.
    """

    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.encoder = encoders.ManipulationEncoder(cfg, rngs)

        # Images must not be scaled: omit extero_obs from the mapping so the uint8
        # camera frames pass through untouched (the processor writes an identity
        # placeholder for extero_obs stats, so this is belt-and-suspenders).
        # Drop extero_obs (images unscaled) and fut_cmds (cmd_dim=0 -> nothing to scale;
        # commands are an all-zero placeholder the model never reads, see _policy_query
        # and the cmd_dim guards in the reward/value heads).
        manip_scaler_mapping = {
            k: v
            for k, v in types.WorldModelSchema.scaler_params_mapping.items()
            if k not in ("extero_obs", "fut_cmds")
        }
        self.scaler = scaler.Scaler(scaler_params, manip_scaler_mapping)

        # Learned temporal query for the policy head: one slot per predicted step,
        # broadcast over the batch (replaces Go2's fut_cmds query at cmd_dim=0).
        self.pred_len = self.model_cfg["dataset"]["prediction_length"]
        self.policy_temporal_query = nnx.Param(
            jax.random.normal(rngs["params"](), (self.pred_len, self.latent_dim)) * 0.02
        )

    def _policy_query(
        self,
        x: types.WorldModelSchema.Inputs,
        fut_cmds_emb: jnp.ndarray,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        B = x["proprio_obs_hist"].shape[0]
        return jnp.broadcast_to(
            self.policy_temporal_query.value, (B, self.pred_len, self.latent_dim)
        )
