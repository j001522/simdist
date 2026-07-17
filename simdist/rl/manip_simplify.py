"""Collapse the OmniReset manipulation env down to a single nominal task.

Single source of truth for the *simplified* peg-insertion task used to validate
the SimDist pipeline (data collection -> train on Snellius -> MPPI on Spark)
cheaply. The world model is data hungry, so we shrink the manifold it has to fit
to a single deterministic condition.

POLICY (post value-target incident, 2026-07-16): **nothing is nulled — every
randomization event stays present and is PINNED to a deterministic value inside
its training range** (midpoint; geometric mean for log_uniform). Nulling an
event is never safe here, for two reasons this codebase has now been bitten by:

  1. some events don't just randomize — they *create/initialize* the property
     they touch (first hit: nulling ``randomize_tiled_cameras`` mis-aimed the
     wrist cam, tripping ``corrupted_camera``);
  2. the values sampled by the physics events (materials, masses, gripper
     gains) are **privileged critic observations** (``CriticCfg``). With the
     events nulled, the USD defaults those obs fall back to are OUT OF
     DISTRIBUTION for the expert critic, whose recorded V^e then collapses to
     inverted garbage — silently, because the *policy* obs contain no DR dims,
     so the expert keeps inserting while the value labels rot. This poisoned
     the WM value head and made MPPI drive away from the goal. Evidence:
     experiments/value_target_analysis/ (probe_critic.py: critic healthy on the
     full distribution, +14..+21 monotone toward seated; inverted ~[-0.5, +2]
     under the nulled events). Memory: value-target-inverted.

Pinning to the *center of the training range* keeps the critic in-distribution
AND the dynamics deterministic. After changing anything here, re-verify with
  isaac-python experiments/value_target_analysis/probe_critic.py --mode simplified
(V must be ~+14..+21 and increase toward the seated peg).

The one exception is ``randomize_sky_light``: ``randomize_hdri`` picks the HDRI
with ``random.choice`` and applies a hardcoded fully-random ``R.random()``
rotation — its params cannot pin it. It is nulled, and lighting is pinned in
the scene cfg instead (``rl_state_cfg`` ``sky_light``), validated by the
lighting check. It touches only a stage light — no obs, no physics.

Other knobs (unchanged rationale):
  * cameras -> pinned to base pose + fixed focal length (they AIM the cameras;
    see incident 1 above);
  * table + curtains -> pinned uniform white (clean white cell, green peg/hole
    look of the OmniReset/SimDist eval scene);
  * robot-part / peg / hole appearance -> pinned solid colors (matte green for
    peg+hole to approximate the previously validated native look, gray for the
    two robot-part meshes). Re-run the lighting check after color changes.
  * arm/controller dynamics -> sysid nominal (``pin_arm_nominal``);
  * one grasped reset bank (prob 1.0) instead of the 4-way start mix.

Apply the SAME function to the data-collection (RGB) env and to any eval / MPPI
env so they are guaranteed to be in identical conditions ("record with the
grasped peg and test in the same conditions").

Note on start-state diversity: a narrow start bank does NOT starve the world
model. The recorder (Algorithm 2) injects per-env Gaussian action noise and
mixes sub-optimal checkpoints, so rollouts fan out into a tube of transitions
around the nominal insertion corridor -- which is exactly the region MPPI plans
in. If diagnostics show the WM underfits, widen ``reset_types`` (e.g. add
``GraspedVertLow``) -- a one-line change, no code edits here.
"""

from __future__ import annotations

import math

# Grasped reset banks built by experiments/ood_insertion/build_reset_banks.py.
# The MultiResetManager loader reads <dataset_dir>/Resets/<pair>/resets_<type>.pt.
DEFAULT_RESET_DIR = "/shared/giacomo/experiments/ood_insertion/resets"
DEFAULT_RESET_TYPES = ("GraspedVertHigh",)  # held, upright, above the hole
DEFAULT_RESET_PROBS = (1.0,)

# ---- physics DR: PIN inside the training range (critic obs stay in-dist) ----
# Values are chosen inside each training range but BIASED toward the measured
# USD defaults / low contact friction, not blindly at the midpoint: midpoint
# pins (peg friction 1.5, peg mass 0.11) produced a deterministic jam-on-entry
# stall at ~3.8 cm and dropped the expert to 0.783 (512 eps). Measured defaults
# (probe_defaults.py): peg friction 0.5/0.5 (BELOW the 1.0..2.0 training range
# -- the old nulled env was slippery easy-mode the critic never saw), peg mass
# 0.054 kg, hole/table 0.5/0.5. Pins: peg friction at the range minimum, hole
# at its range minimum, peg mass at the true default. None means midpoint.
_MATERIAL_TERMS = {
    "robot_material": None,  # midpoints 0.75/0.6 -- training-typical; the event
    # always overwrote the USD 100-friction gripper pads in training too
    "insertive_object_material": {"static_friction_range": 1.0, "dynamic_friction_range": 0.9},
    "receptive_object_material": {"static_friction_range": 0.2, "dynamic_friction_range": 0.15},
    "table_material": {"static_friction_range": 0.5, "dynamic_friction_range": 0.5},
}
_MATERIAL_RANGE_KEYS = ("static_friction_range", "dynamic_friction_range", "restitution_range")

# randomize_rigid_body_mass terms: scale terms pin to 1.0 (= USD default); the
# insertive "abs" term pins to the measured default 0.054 kg (in range 0.02..0.2).
_MASS_TERMS = {
    "randomize_robot_mass": None,
    "randomize_insertive_object_mass": 0.054,
    "randomize_receptive_object_mass": None,
    "randomize_table_mass": None,
}

# randomize_actuator_gains (log_uniform): pin to the geometric mean of the
# range — for the training (0.5, 2.0) that is exactly 1.0 = nominal gains.
_GRIPPER_GAIN_TERM = "randomize_gripper_actuator_parameters"
_GRIPPER_GAIN_KEYS = ("stiffness_distribution_params", "damping_distribution_params")

# Lighting: the ONE nulled event — randomize_hdri cannot be pinned via params
# (random.choice HDRI + hardcoded R.random() rotation). Lighting is pinned in
# the scene cfg instead. Touches only a stage light: no obs, no physics.
_NULL_TERMS = ("randomize_sky_light",)

# Camera-placement events. NOT cosmetic DR: randomize_tiled_cameras is what aims the
# cameras each reset. We PIN them (zero delta / single focal), never null them.
_CAMERA_POSE_TERMS = ("randomize_front_camera", "randomize_side_camera", "randomize_wrist_camera")
_CAMERA_FOCAL_TERMS = (
    "randomize_front_camera_focal_length",
    "randomize_side_camera_focal_length",
    "randomize_wrist_camera_focal_length",
)

# Appearance events -> pinned solid matte colors (deterministic every reset).
# Enclosure white; peg + hole green (approximates the validated native look);
# the two randomized robot-part meshes a neutral gray.
_WHITE = (0.5, 0.5, 0.5)
_GREEN = (0.10, 0.60, 0.15)
_GRAY = (0.35, 0.35, 0.35)
_APPEARANCE_TERMS = {
    "randomize_table_appearance": _WHITE,
    "randomize_curtain_left_appearance": _WHITE,
    "randomize_curtain_back_appearance": _WHITE,
    "randomize_curtain_right_appearance": _WHITE,
    "randomize_insertive_object_appearance": _GREEN,
    "randomize_receptive_object_appearance": _GREEN,
    "randomize_wrist_mount_appearance": _GRAY,
    "randomize_inner_finger_appearance": _GRAY,
}

# Per-reset arm/controller randomizers. With pin_arm_nominal=True we set their
# scale to the deterministic identity (1.0, 1.0) instead of nulling, so the arm
# runs at the sysid nominal the expert was finetuned around. With False we drop
# them entirely (USD-default arm dynamics).
_ARM_DR_TERMS = ("randomize_arm_sysid", "randomize_osc_gains")


def _mid(rng):
    m = 0.5 * (rng[0] + rng[1])
    return (m, m)


def _geomean(rng):
    m = math.sqrt(rng[0] * rng[1])
    return (m, m)


def _pin_solid_appearance(term, rgb):
    """Pin a randomize_visual_appearance_multiple_meshes term to one matte color.

    Every value is sampled through rng.uniform(range), so single-valued ranges +
    texture_prob=0 give the same solid surface every reset.
    """
    p = term.params
    p["texture_prob"] = 0.0          # always solid color, never a random texture
    p["texture_config_path"] = None  # don't load textures we won't use
    p["colors"] = {c: (v, v) for c, v in zip("rgb", rgb)}
    for rk, val in (("roughness_range", 0.6), ("metallic_range", 0.0), ("specular_range", 0.5)):
        if rk in p:
            p[rk] = (val, val)
    if "texture_scale_range" in p:
        p["texture_scale_range"] = (1.0, 1.0)
    if "diffuse_tint_range" in p:
        p["diffuse_tint_range"] = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


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
    events cfg (RGB or State): unknown terms are skipped (hasattr-guarded, so
    one function serves both the RGB events and the leaner State events).
    """
    reset_types = list(reset_types)
    probs = list(probs)
    if len(reset_types) != len(probs):
        raise ValueError(f"reset_types ({len(reset_types)}) and probs ({len(probs)}) length mismatch")

    # Physics materials: pin every friction/restitution range to a single value
    # (per-term override, else midpoint -- see _MATERIAL_TERMS comment). One
    # bucket -- every body gets the identical (deterministic) material.
    for name, overrides in _MATERIAL_TERMS.items():
        term = getattr(events, name, None)
        if term is None:
            continue
        for key in _MATERIAL_RANGE_KEYS:
            if key in term.params:
                if overrides and key in overrides:
                    term.params[key] = (overrides[key], overrides[key])
                else:
                    term.params[key] = _mid(term.params[key])
        if "num_buckets" in term.params:
            term.params["num_buckets"] = 1

    # Masses: pin to the override value if given, else midpoint (= 1.0 nominal
    # for the scale terms).
    for name, override in _MASS_TERMS.items():
        term = getattr(events, name, None)
        if term is None:
            continue
        if override is not None:
            term.params["mass_distribution_params"] = (override, override)
        else:
            term.params["mass_distribution_params"] = _mid(term.params["mass_distribution_params"])

    # Gripper actuator gains (log_uniform): pin to the geometric mean (= 1.0).
    term = getattr(events, _GRIPPER_GAIN_TERM, None)
    if term is not None:
        for key in _GRIPPER_GAIN_KEYS:
            if key in term.params:
                term.params[key] = _geomean(term.params[key])

    # The lone null (see module docstring).
    for name in _NULL_TERMS:
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

    # Appearance: pin each surface to one solid matte color (deterministic).
    for name, rgb in _APPEARANCE_TERMS.items():
        term = getattr(events, name, None)
        if term is None:
            continue
        _pin_solid_appearance(term, rgb)

    reset = getattr(events, "reset_from_reset_states", None)
    if reset is None:
        raise AttributeError("events cfg has no reset_from_reset_states term to pin")
    reset.params["dataset_dir"] = dataset_dir
    reset.params["reset_types"] = reset_types
    reset.params["probs"] = probs

    return events
