"""Collapse the OmniReset manipulation env down to a single nominal task.

Single source of truth for the *simplified* peg-insertion task used to validate
the SimDist pipeline (data collection -> train on Snellius -> MPPI on Spark)
cheaply. The world model is data hungry, so we shrink the manifold it has to fit:

  * every domain randomization OFF (appearance/texture, camera pose+intrinsics,
    HDRI lighting, material friction, mass, gripper gains),
  * arm/controller dynamics PINNED to the deterministic sysid nominal (so the
    exported expert -- finetuned with sysid+OSC DR -- stays at the center of its
    training distribution rather than dropping to USD defaults), and
  * one grasped reset bank (prob 1.0) instead of the 4-way start mix.

Apply the SAME function to the data-collection (RGB) env and to any eval / MPPI
env so they are guaranteed to be in identical conditions ("record with the
grasped peg and test in the same conditions"). Disabling a term by setting it to
``None`` is the established idiom in this codebase (cf. ``Ur5eRecordEnvCfg``
setting ``terminations.success = None``); the IsaacLab managers skip ``None``
terms.

Note on start-state diversity: a narrow start bank does NOT starve the world
model. The recorder (Algorithm 2) injects per-env Gaussian action noise and
mixes sub-optimal checkpoints, so rollouts fan out into a tube of transitions
around the nominal insertion corridor -- which is exactly the region MPPI plans
in. If diagnostics show the WM underfits, widen ``reset_types`` (e.g. add
``GraspedVertLow``) -- a one-line change, no code edits here.
"""

from __future__ import annotations

# Grasped reset banks built by experiments/ood_insertion/build_reset_banks.py.
# The MultiResetManager loader reads <dataset_dir>/Resets/<pair>/resets_<type>.pt.
DEFAULT_RESET_DIR = "/shared/giacomo/experiments/ood_insertion/resets"
DEFAULT_RESET_TYPES = ("GraspedVertHigh",)  # held, upright, above the hole
DEFAULT_RESET_PROBS = (1.0,)

# DR event terms nulled when present. Hasattr-guarded so one function serves both
# the RGB events (FinetuneEval base + camera/appearance/lighting DR) and the
# leaner State events (TrainEval base: material/mass/gripper only).
_NULL_DR_TERMS = (
    # physics DR (startup)
    "robot_material",
    "insertive_object_material",
    "receptive_object_material",
    "table_material",
    "randomize_robot_mass",
    "randomize_insertive_object_mass",
    "randomize_receptive_object_mass",
    "randomize_table_mass",
    # gripper actuator gains (reset)
    "randomize_gripper_actuator_parameters",
    # camera pose / intrinsics (reset)
    "randomize_front_camera",
    "randomize_front_camera_focal_length",
    "randomize_side_camera",
    "randomize_side_camera_focal_length",
    "randomize_wrist_camera",
    "randomize_wrist_camera_focal_length",
    # appearance / texture (interval)
    "randomize_wrist_mount_appearance",
    "randomize_inner_finger_appearance",
    "randomize_insertive_object_appearance",
    "randomize_receptive_object_appearance",
    "randomize_table_appearance",
    "randomize_curtain_left_appearance",
    "randomize_curtain_back_appearance",
    "randomize_curtain_right_appearance",
    # lighting (interval)
    "randomize_sky_light",
)

# Per-reset arm/controller randomizers. With pin_arm_nominal=True we set their
# scale to the deterministic identity (1.0, 1.0) instead of nulling, so the arm
# runs at the sysid nominal the expert was finetuned around. With False we drop
# them entirely (USD-default arm dynamics).
_ARM_DR_TERMS = ("randomize_arm_sysid", "randomize_osc_gains")


def simplify_events(
    events,
    *,
    dataset_dir: str = DEFAULT_RESET_DIR,
    reset_types=DEFAULT_RESET_TYPES,
    probs=DEFAULT_RESET_PROBS,
    pin_arm_nominal: bool = True,
):
    """Mutate an events configclass in place into the simplified task.

    Returns the same ``events`` for convenience. Safe to call on any OmniReset
    events cfg (RGB or State): unknown terms are skipped.
    """
    reset_types = list(reset_types)
    probs = list(probs)
    if len(reset_types) != len(probs):
        raise ValueError(f"reset_types ({len(reset_types)}) and probs ({len(probs)}) length mismatch")

    for name in _NULL_DR_TERMS:
        if getattr(events, name, None) is not None:
            setattr(events, name, None)

    for name in _ARM_DR_TERMS:
        term = getattr(events, name, None)
        if term is None:
            continue
        if pin_arm_nominal:
            term.params["scale_range"] = (1.0, 1.0)
            if "delay_range" in term.params:
                term.params["delay_range"] = (0, 0)
        else:
            setattr(events, name, None)

    reset = getattr(events, "reset_from_reset_states", None)
    if reset is None:
        raise AttributeError("events cfg has no reset_from_reset_states term to pin")
    reset.params["dataset_dir"] = dataset_dir
    reset.params["reset_types"] = reset_types
    reset.params["probs"] = probs

    return events
