"""Manipulation (UR5e peg insertion) recorder env for SimDist data generation.

Manipulation analog of ``simdist/rl/go2.py``. See ``docs/manipulation_port.md``.

Key asymmetry vs Go2: the OmniReset expert policy and value function are
STATE-based, while the world model is VISION-based. So during data generation we

  * drive the env and query V^e from the STATE obs groups (``policy`` / ``critic``),
  * but RECORD ``o_t`` from the VISION group (``recorded_obs``: int8 images + proprio).

We wrap OmniReset's RGB data-collection env (which already has the 3 cameras and the
RelCartesianOSC 6-DoF + gripper action) and add back the state ``policy``/``critic``
obs groups so the state expert/critic can be evaluated while the cameras render.
"""

from typing import Any

import torch

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)

from simdist.data.jpeg_hdf5_handler import JpegHDF5DatasetFileHandler

# OmniReset configs (uwlab_tasks is installed in the Spark Isaac python).
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.data_collection_rgb_cfg import (  # noqa: E501
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import (  # noqa: E501
    ObservationsCfg as StateObservationsCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (  # noqa: E501
    Ur5eRobotiq2f85RelativeOSCAction,
)


class ManagerBasedRLEnvRecord(ManagerBasedRLEnv):
    """Env that holds the state critic and an expert-action flag for recording.

    Identical contract to the Go2 version: ``DataRecorder`` constructs it with the
    state value function and sets ``expert_policy_flag_buf`` each step.
    """

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        critic: Any,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.critic = critic
        self.expert_policy_flag_buf = None


# --------------------------------------------------------------------------- #
# Recorder terms (mirror go2.py, with the two manipulation changes)           #
# --------------------------------------------------------------------------- #
class ManipObservationsRecorder(RecorderTerm):
    """Records the vision observation o_t (int8 images + proprio) for the world model.

    Reads the env's existing ``data_collection`` obs group (unprocessed int8 RGB +
    proprio). We deliberately do NOT alias it into a second group: sharing the same
    SceneEntityCfg objects across two groups makes IsaacLab resolve them twice and
    raises a joint_names/joint_ids inconsistency.
    """

    def record_pre_step(self):
        return "obs", self._env.obs_buf["data_collection"]


class RewardRecorder(RecorderTerm):
    def record_post_step(self):
        return "reward", self._env.reward_buf


class ValueRecorder(RecorderTerm):
    def record_post_step(self):
        with torch.no_grad():
            # CHANGE vs go2: OmniReset critic is asymmetric -> feed obs["critic"],
            # not obs["policy"]. The exported critic_i.pt expects the privileged
            # critic obs group.
            return "value", self._env.critic(self._env.obs_buf["critic"]).squeeze()


class ExpertPolicyFlagRecorder(RecorderTerm):
    def record_pre_step(self):
        with torch.no_grad():
            return "expert_policy_flag", self._env.expert_policy_flag_buf.squeeze()


@configclass
class ManipObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ManipObservationsRecorder


@configclass
class ExpertPolicyFlagRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ExpertPolicyFlagRecorder


@configclass
class ValueRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ValueRecorder


@configclass
class RewardRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = RewardRecorder


@configclass
class ManipRecorderManagerCfg(RecorderManagerBaseCfg):
    # JPEG-encode image obs on write (~6x smaller than gzip, visually lossless).
    dataset_file_handler_class_type: type = JpegHDF5DatasetFileHandler

    # NOTE vs go2: no commands recorder (peg insertion task is fixed; Algorithm 2
    # records only (o_t, a_t, b^e_t, r_t, v_t)).
    record_pre_step_obs = ManipObservationsRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_expert_policy_flag = ExpertPolicyFlagRecorderCfg()
    record_post_step_reward = RewardRecorderCfg()
    record_post_step_value = ValueRecorderCfg()


# --------------------------------------------------------------------------- #
# Recorder env cfg                                                            #
# --------------------------------------------------------------------------- #
@configclass
class Ur5eRecordEnvCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """RGB data-collection env + state policy/critic obs groups + recorders."""

    # Early-terminate a collection episode once the peg is seated (5-step dwell,
    # via the base RGB `success` DoneTerm). Gated to >= success_min_episode_length
    # so every kept episode is still long enough for the WM window (H+T+beg+end;
    # H=T=5 -> 20). Set stop_on_success=False for the old time_out-only behavior.
    # DataRecorder overrides these from the generate_data `stop_on_success` block.
    stop_on_success: bool = True
    success_min_episode_length: int = 20

    def __post_init__(self):
        super().__post_init__()

        # State obs groups consumed by the expert (policy) and value fn (critic).
        # PolicyCfg uses history_length=5, CriticCfg adds privileged terms (incl.
        # material properties) at history_length=1 -- matching what the exported
        # policy_i.pt / critic_i.pt were trained on.
        self.observations.policy = StateObservationsCfg.PolicyCfg()
        self.observations.critic = StateObservationsCfg.CriticCfg()

        # The world-model observation o_t is the env's existing unprocessed int8
        # vision group `data_collection` (front/side/wrist int8 RGB + arm_joint_pos
        # + last actions). The recorder reads obs_buf["data_collection"] directly --
        # we do NOT add a second aliased group (that double-resolves SceneEntityCfgs).

        # Attach the recorder manager.
        self.recorders = ManipRecorderManagerCfg()

        # No action-noise corruption on the policy obs fed to the expert.
        self.observations.policy.enable_corruption = False

        # Horizon shortened 16 s -> 10 s (80 -> 50 steps at 5 Hz): at 16 s ~62% of
        # recorded steps were the seated V~21 plateau, collapsing the value-target
        # distribution (Snellius scaler: 59% of normalized targets within +-0.1 of
        # median). The critic's `time_left` obs is a fraction of the episode, so
        # its range is unchanged; the full-distribution probe showed V is
        # clock-insensitive (seated V holds ~+21 across the clock), re-verified
        # with probe_critic.py at 10 s before collection.
        self.episode_length_s = 10.0

        # Terminations. Always keep time_out + abnormal_robot. `early_success`
        # (which ends fast-seaters as FAILURES before min_episode_length) stays
        # OFF -- for collection we want those episodes kept, not discarded. The
        # `success` DoneTerm (>=5 consecutive seated steps) ends seated episodes
        # early: this stops recording the long redundant seated V~21 plateau
        # (further easing the value-target collapse above) and frees sim time for
        # more distinct rollouts. It is gated to success_min_episode_length so
        # every kept episode still spans the WM window. Enabling it also makes the
        # recorded per-demo `success` attr meaningful (RecorderManager reads the
        # active `success` termination). Note the expert critic was trained WITHOUT
        # success termination, so its recorded V at a seated step still reads ~21
        # (a valid distillation target); we simply record fewer such steps.
        self.terminations.early_success = None
        if self.stop_on_success:
            self.terminations.success.params["min_episode_length"] = self.success_min_episode_length
        else:
            self.terminations.success = None

        # Match the Stage-1 (base) dynamics the expert was trained + evaluated on.
        # The RGB data-collection cfg inherits the Stage-2/sim2real stack (eval OSC
        # action scale with z=0.002 + high Kp, plus randomize_arm_sysid /
        # randomize_osc_gains that write real-robot friction onto the arm). The
        # expert (run omnireset_2026-06-21_17-10-39, ~0.90 on State-v0) never
        # trained through the finetune curriculum, so under that stack it saturates
        # (|action|~6) and the arm barely moves -> it never seats the peg. Restore
        # the base OSC action (z-scale 0.02, soft Kp 200/3) and the ideal actuator
        # (no sysid / OSC-gain DR), exactly as the working State-Play env. Confirmed
        # to insert (probe_recorder_episode.py). Users stay in sim (no sim2real), so
        # the finetune dynamics are not needed. See memory: recorder-zero-insertion.
        self.actions = Ur5eRobotiq2f85RelativeOSCAction()
        self.events.randomize_arm_sysid = None
        self.events.randomize_osc_gains = None

        # Match the CONTROL RATE the expert was trained at. The base RlStateCfg runs
        # decimation=12 @ sim.dt=1/120 -> 10 Hz. The current expert (run
        # omnireset_2026-07-01_16-31-09) was trained at 5 Hz (decimation doubled to
        # 24, sim.dt unchanged). The OSC action is a per-control-step delta, so a
        # decimation mismatch changes the per-step dynamics and breaks the policy
        # (same failure class as the action-scale/DR mismatch above). Keep this equal
        # to the training decimation of whatever expert `generate_data.yaml` points at.
        self.decimation = 24
        self.sim.render_interval = self.decimation  # one rendered frame per control step
