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

        # Match the State env the expert/critic were trained on: fixed 16 s horizon,
        # terminate only on time_out + abnormal_robot. We drop the RGB env's
        # success/early_success terminations because (1) they cut episodes short on
        # insertion, and process_data discards episodes shorter than
        # H+T+beg+end (=60 steps), wasting the most valuable expert rollouts; and
        # (2) the critic V^e is conditioned on `time_left`, calibrated to the 16 s
        # training horizon -- a different horizon makes value targets OOD. Matches
        # the locomotion reference too (reset on failure + time_out, no success).
        self.episode_length_s = 16.0
        self.terminations.success = None
        self.terminations.early_success = None
