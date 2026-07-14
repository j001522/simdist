"""Collapse the OmniReset manipulation env down to a single nominal task.

Single source of truth for the *simplified* peg-insertion task used to validate
the SimDist pipeline (data collection -> train on Snellius -> MPPI on Spark)
cheaply. The world model is data hungry, so we shrink the manifold it has to fit
to a single deterministic condition -- but *deterministic* is not the same as
*disabled*. Some "randomization" events are the only thing that sets a value at
all, so nulling them silently breaks the scene. We therefore split terms into:

  * NULL (variability that has a sane default when removed): physics material /
    mass / gripper-gain DR, and robot/peg/hole *appearance* DR (their native USD
    materials are correct -- the peg and hole are already green, matching the
    OmniReset/SimDist eval look);
  * PIN to a deterministic value (events that *place* or *set* something):
      - arm/controller dynamics -> sysid nominal (expert was finetuned around it);
      - the three cameras -> their base pose + a fixed focal length.
        ``randomize_tiled_cameras`` is what actually AIMS the cameras each reset;
        the static ``TiledCameraCfg`` offset uses a different convention and
        mis-aims the wrist cam into empty space (uniform frame -> tripped the
        ``corrupted_camera`` termination). So we pin, never null, the camera terms;
      - table + curtains *appearance* -> a uniform white, so the cell reads as the
        clean white OmniReset/SimDist scene instead of the dark native materials.
  * Lighting: pinned in the scene cfg (``rl_state_cfg`` ``sky_light``), NOT here --
    ``randomize_hdri`` hardcodes a random HDRI choice and rotation, so it cannot be
    pinned via params.
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
    # physics DR (startup) -- not visual; sane USD defaults when removed
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
    # robot-part appearance (interval) -- keep native robot materials
    "randomize_wrist_mount_appearance",
    "randomize_inner_finger_appearance",
    # peg / hole appearance (interval) -- keep NATIVE materials (green), which is
    # the OmniReset/SimDist eval look
    "randomize_insertive_object_appearance",
    "randomize_receptive_object_appearance",
    # lighting (interval) -- pinned in the scene cfg instead (see module docstring)
    "randomize_sky_light",
)

# Camera-placement events. NOT cosmetic DR: randomize_tiled_cameras is what aims the
# cameras each reset. We PIN them (zero delta / single focal), never null them.
_CAMERA_POSE_TERMS = ("randomize_front_camera", "randomize_side_camera", "randomize_wrist_camera")
_CAMERA_FOCAL_TERMS = (
    "randomize_front_camera_focal_length",
    "randomize_side_camera_focal_length",
    "randomize_wrist_camera_focal_length",
)

# Enclosure appearance pinned to a uniform white -> clean white cell (green peg+hole
# against white) matching the OmniReset/SimDist eval scene, not the dark native
# curtain/table materials.
_WHITE_APPEARANCE_TERMS = (
    "randomize_table_appearance",
    "randomize_curtain_left_appearance",
    "randomize_curtain_back_appearance",
    "randomize_curtain_right_appearance",
)
_WHITE_RGB = 0.5

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

    # Pin the cameras to their deterministic base pose + fixed focal length. These
    # events AIM the cameras; nulling them mis-aims the wrist cam (see docstring).
    for name in _CAMERA_POSE_TERMS:
        term = getattr(events, name, None)
        if term is None:
            continue
        term.params["position_deltas"] = {k: (0.0, 0.0) for k in term.params["position_deltas"]}
        term.params["euler_deltas"] = {k: (0.0, 0.0) for k in term.params["euler_deltas"]}
    for name in _CAMERA_FOCAL_TERMS:
        term = getattr(events, name, None)
        if term is None:
            continue
        lo, hi = term.params["focal_length_range"]
        term.params["focal_length_range"] = (0.5 * (lo + hi),) * 2

    # Pin table + curtains to a uniform white solid color (deterministic): the
    # appearance term samples every value through rng.uniform(range), so single-
    # valued ranges + texture_prob=0 give a fixed matte-white surface every reset.
    for name in _WHITE_APPEARANCE_TERMS:
        term = getattr(events, name, None)
        if term is None:
            continue
        p = term.params
        p["texture_prob"] = 0.0          # always solid color, never a random texture
        p["texture_config_path"] = None  # don't load textures we won't use
        p["colors"] = {c: (_WHITE_RGB, _WHITE_RGB) for c in ("r", "g", "b")}
        for rk, val in (("roughness_range", 0.7), ("metallic_range", 0.0), ("specular_range", 0.5)):
            if rk in p:
                p[rk] = (val, val)
        if "texture_scale_range" in p:
            p["texture_scale_range"] = (1.0, 1.0)
        if "diffuse_tint_range" in p:
            p["diffuse_tint_range"] = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    reset = getattr(events, "reset_from_reset_states", None)
    if reset is None:
        raise AttributeError("events cfg has no reset_from_reset_states term to pin")
    reset.params["dataset_dir"] = dataset_dir
    reset.params["reset_types"] = reset_types
    reset.params["probs"] = probs

    return events
