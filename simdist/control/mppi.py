import functools

import flax
from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np

from simdist.control.controller_base import (
    ControllerBase,
    ControllerInput,
    ControllerOutput,
)
from simdist.modeling import models, types
from simdist.utils.model import repeat_along_batch_dim


@flax.struct.dataclass
class MppiState:
    mean: jnp.ndarray
    std: jnp.ndarray
    key_data: jnp.ndarray


class MppiController(ControllerBase):
    def __init__(
        self,
        model: models.WorldModelBase,
        model_cfg: dict,
        controller_cfg: dict,
        *args,
        **kwargs,
    ):
        super().__init__(model, model_cfg, controller_cfg, *args, **kwargs)

        self.num_base_trajs = int(
            self.ctrl_cfg["num_samples"] * self.ctrl_cfg["mixture_coef"]
        )
        assert self.num_base_trajs > 0
        self.shift = self.ctrl_cfg["delay"]
        assert self.shift > 0

        self.mppi_state = None
        self.discounts = self.ctrl_cfg["discount"] ** jnp.arange(self.T)
        self.final_discount = self.ctrl_cfg["discount"] ** (self.T + 1)
        self.rngs = nnx.Rngs(self.ctrl_cfg["seed"])
        self.dummy_actions = jnp.zeros((self.T, self.act_dim))

    def reset(self, x: ControllerInput, cmd: np.ndarray):
        super().reset(x, cmd)
        self.mppi_state = MppiState(
            mean=self._to_dev_f32(jnp.zeros((self.T, self.act_dim))),
            std=self._to_dev_f32(
                jnp.ones((self.T, self.act_dim)) * self.ctrl_cfg["init_std"]
            ),
            key_data=self._to_key_data(self.rngs()),
        )

    def run_control(self) -> ControllerOutput:
        model_inputs = self._make_model_inputs(self.dummy_actions)
        # The encoder output does not depend on the candidate actions, so it is computed
        # once here and reused by every sampled trajectory in every solver iteration.
        encoding = self._encode(model_inputs)
        fut_cmds = jnp.asarray(model_inputs["fut_cmds"])
        base_policy_actions = self._get_base_policy_actions(encoding, fut_cmds)
        self.mppi_state = self._mppi_step(
            self.mppi_state, base_policy_actions, encoding, fut_cmds
        )
        actions = self.mppi_state.mean
        output: ControllerOutput = {"actions": np.array(actions)}
        return output

    def _make_model_inputs(
        self, fut_acts: jnp.ndarray
    ) -> types.WorldModelSchema.Inputs:
        """
        Create model inputs for the world model. fut_acts should be shape (T, act_dim)
        """
        with self.buf_lock:
            hist: ControllerInput = self.buf.get()
            model_inputs: types.WorldModelSchema.Inputs = {
                "proprio_obs_hist": hist["proprio_obs"][-(self.H + 1) :],
                "extero_obs": hist["extero_obs"][-1],
                "acts_hist": hist["prev_action"][-self.H :],
                "fut_acts": fut_acts,
                "fut_cmds": self.fut_cmds,
            }
            model_inputs = jax.tree.map(jnp.asarray, model_inputs)
        return model_inputs

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _encode(
        self, model_inputs: types.WorldModelSchema.Inputs
    ) -> types.WorldModelSchema.Encoding:
        """Encode the current observation once, at batch size 1."""
        return self.model.encode_context(repeat_along_batch_dim(model_inputs, 1))

    def _broadcast_encoding(
        self, encoding: types.WorldModelSchema.Encoding, batch_size: int
    ) -> types.WorldModelSchema.Encoding:
        """Expand a batch-1 encoding to the sample batch. This is a broadcast of a
        (2H+1, latent_dim) tensor, not of the raw observation -- which is the whole
        point: for manipulation the images never reach the sample batch dimension."""
        return jax.tree.map(
            lambda z: jnp.broadcast_to(z, (batch_size,) + z.shape[1:]), encoding
        )

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _get_base_policy_actions(
        self, encoding: types.WorldModelSchema.Encoding, fut_cmds: jnp.ndarray
    ) -> jnp.ndarray:
        x = {"fut_acts": self.dummy_actions[None], "fut_cmds": fut_cmds[None]}
        model_outputs = self.model.inference_from_encoding(x, encoding)
        return model_outputs["actions"][0]

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _mppi_step(
        self,
        mppi_state: MppiState,
        base_policy_actions: jnp.ndarray,
        encoding: types.WorldModelSchema.Encoding,
        fut_cmds: jnp.ndarray,
        **kwargs,
    ) -> MppiState:
        key = self._to_key(mppi_state.key_data)

        # the sample batch is fixed across solver iterations, so broadcast the encoding
        # and commands once, outside the scan
        batch_size = self.ctrl_cfg["num_samples"] + self.num_base_trajs
        batch_encoding = self._broadcast_encoding(encoding, batch_size)
        batch_fut_cmds = jnp.broadcast_to(
            fut_cmds[None], (batch_size,) + fut_cmds.shape
        )

        # add noise to base actions
        key, key_base_act = jax.random.split(key)
        base_act_noise = (
            jax.random.normal(key_base_act, (self.num_base_trajs, self.T, self.act_dim))
            * self.ctrl_cfg["base_act_std"]
        )
        noised_base_policy_actions = base_policy_actions + base_act_noise

        # initialize
        prev_mean = mppi_state.mean
        mean = jnp.roll(prev_mean, shift=self.shift, axis=0)
        mean = mean.at[self.shift :].set(0.0)
        std = jnp.ones((self.T, self.act_dim)) * self.ctrl_cfg["init_std"]

        # each iterations do the following
        def f(carry, xs_unused):
            mean, std, key = carry

            # get noised actions
            key, key_act = jax.random.split(key)
            act_noise = (
                jax.random.normal(
                    key_act, (self.ctrl_cfg["num_samples"], self.T, self.act_dim)
                )
                * std
            )
            noised_acts = mean + act_noise
            # run model with both base policy and noised actions
            acts = jnp.concatenate([noised_acts, noised_base_policy_actions], axis=0)
            x = {"fut_acts": acts, "fut_cmds": batch_fut_cmds}
            y = self.model.inference_from_encoding(x, batch_encoding)
            returns = self._calc_returns(y)

            # select elites
            _, elite_idxs = jax.lax.top_k(returns, self.ctrl_cfg["num_elites"])
            elite_returns = jnp.take(returns, elite_idxs, axis=0)
            elite_acts = jnp.take(acts, elite_idxs, axis=0)

            # update
            max_rew = jnp.max(elite_returns)
            score = jnp.exp(self.ctrl_cfg["temperature"] * (elite_returns - max_rew))
            _mean = jnp.sum(score[:, None, None] * elite_acts, axis=0) / jnp.sum(score)
            _std = jnp.sqrt(
                jnp.sum(score[:, None, None] * (elite_acts - _mean) ** 2, axis=0)
                / jnp.sum(score)
            )

            mean = (
                self.ctrl_cfg["momentum"] * mean
                + (1 - self.ctrl_cfg["momentum"]) * _mean
            )
            std = jnp.clip(_std, min=self.ctrl_cfg["min_std"])

            return (mean, std, key), None

        # run iterations
        (mean, std, key), _ = jax.lax.scan(
            f, (mean, std, key), jnp.arange(self.ctrl_cfg["iterations"])
        )

        return MppiState(
            self._to_dev_f32(mean),
            self._to_dev_f32(std),
            self._to_key_data(key),
        )

    def _calc_returns(self, y: types.WorldModelSchema.Outputs) -> jnp.ndarray:
        rewards = jnp.sum(y["rewards"] * self.discounts, axis=-1)
        value = y["values"][:, -1] * self.final_discount
        returns = rewards + value
        return returns
