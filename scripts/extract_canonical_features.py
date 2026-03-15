"""
extract_canonical_features.py
==============================
Shared canonical anthropometric feature extractor used by:
  - robust_child_shape_opt_upperbody_200.py  (objective, evaluation)
  - step1_extract_child_gt_features.py       (CSV export)

The function signature  extract_features_from_joints(verts, joints, joint_idx)
is the single source of truth for feature computation. All downstream scripts
import from here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  JOINT INDICES assumed from the pipeline's JOINT_IDX dict
  (matches robust_child_shape_opt_upperbody_200.py exactly):

    pelvis        : 0
    left_hip      : 1    right_hip       : 2
    left_knee     : 4    right_knee      : 5
    left_ankle    : 7    right_ankle     : 8
    spine3        : 9    (upper thorax — NOT in JOINT_IDX name map but always
                          present in SMPL-X joint output; used for neck_length)
    neck          : 12
    head          : 15   (skull-level joint — used for height)
    left_shoulder : 16   right_shoulder  : 17
    left_elbow    : 18   right_elbow     : 19
    left_wrist    : 20   right_wrist     : 21

  Joints 9 and 15 are accessed by raw index, not by name, because they are
  not included in the pipeline's JOINT_IDX name map.  They are always present
  in the SMPL-X forward-pass output (out.joints has ≥ 22 body joints).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HEIGHT COMPUTATION (used to normalise ALL ratios):

    height_m = joints[15, Y] − min(joints[7, Y], joints[8, Y])
             = head joint Y  − lowest ankle joint Y

  This is purely joint-based and does NOT change across versions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXISTING FEATURE KEYS  (FEATURE_KEYS in the pipeline — unchanged):

    height_canonical         metres  (same as height_m above)
    shoulder_width_ratio     ||L_shoulder − R_shoulder|| / height_m
    pelvis_width_ratio       ||L_hip      − R_hip||      / height_m
    torso_height_ratio       ||pelvis     − neck||        / height_m
    arm_length_ratio         mean(||L_sh−L_wr||, ||R_sh−R_wr||) / height_m
    thigh_ratio              mean(||L_hip−L_knee||, ...) / height_m
    shank_ratio              mean(||L_knee−L_ankle||,...) / height_m
    leg_length_ratio         mean(||L_hip−L_ankle||,...)  / height_m

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NEW HEAD / NECK FEATURE KEYS  (HEAD_NECK_FEATURE_KEYS — optional):

    head_height_ratio        VERTEX-assisted (see details below)
    head_width_ratio         VERTEX-assisted
    neck_length_ratio        joint-based
    head_width_to_shoulder_ratio  derived
    head_height_to_torso_ratio    derived

  These are returned unconditionally when verts is not None (which is always
  the case when called from step1 or the optimization pipeline).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HEAD FEATURE DEFINITIONS IN DETAIL:

  head_height_ratio
    head_top_y    = max Y of all vertices with Y > joints[12, Y]  (above neck)
    head_base_y   = joints[12, Y]                                  (neck joint)
    head_height_m = head_top_y − head_base_y
    head_height_ratio = head_height_m / height_m

    Rationale: vertices above the neck joint in canonical T-pose form the
    head/skull region.  The neck joint (12) is a stable anatomical anchor.
    Using the actual vertex crown is more accurate than joint 15, which sits
    inside the skull geometry rather than at the very top.
    Limitation: shoulder/arm vertices could theoretically be above the neck
    joint in unusual poses.  In canonical T-pose this does not occur.

  head_width_ratio
    head_verts    = vertices with Y > joints[12, Y]
    head_width_m  = max(head_verts[:, X]) − min(head_verts[:, X])
    head_width_ratio = head_width_m / height_m

    Rationale: same vertex set as above; lateral (X-axis) bounding width.
    In canonical T-pose the arms are extended, so arm vertices will be above
    the neck joint in X range BUT significantly lower in Y than the head top.
    To avoid arm contamination we further restrict to vertices within a
    vertical band close to the head:
      Y > (neck_y + 0.5 * head_height_m)  (upper half of the head region only)
    This confines the width measurement to the skull cap, not the jaw/chin.

  neck_length_ratio
    neck_length_m = ||joints[9] − joints[12]||   (spine3 → neck, Euclidean)
    neck_length_ratio = neck_length_m / height_m

    Rationale: joint 9 (spine3) is the uppermost thoracic spine joint;
    joint 12 (neck) is the mid-cervical level in the SMPL-X kinematic chain.
    The distance between them is the most stable joint-only proxy for
    anatomical neck length.  Normalised by body height so the ratio
    represents the neck-length-to-stature proportion, which differs between
    children and adults.
    Limitation: the SMPL-X kinematic "neck" joint does not correspond exactly
    to the anatomical cervical-thoracic junction, so treat this as a proxy.

  head_width_to_shoulder_ratio
    = head_width_m / shoulder_width_m
    Children have proportionally larger head-to-shoulder ratios than adults.

  head_height_to_torso_ratio
    = head_height_m / torso_height_m
    Children have proportionally larger head-to-torso ratios.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Raw fixed joint indices (used by raw-index access, not JOINT_IDX names)
# ---------------------------------------------------------------------------
_IDX_SPINE3 = 9    # upper thorax / base of neck
_IDX_NECK   = 12   # mid-cervical
_IDX_HEAD   = 15   # skull-level (top of head in SMPL-X regressed joint)

# Y axis index in SMPL-X canonical pose (Y = up)
_Y = 1
_X = 0

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Keys that are already part of the pipeline's FEATURE_KEYS — never changed.
CORE_FEATURE_KEYS = [
    "height_canonical",
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "arm_length_ratio",
    "thigh_ratio",
    "shank_ratio",
    "leg_length_ratio",
]

# New optional head/neck feature keys added in this version.
# step2 detects these automatically via get_analysis_features().
HEAD_NECK_FEATURE_KEYS = [
    "head_height_ratio",
    "head_width_ratio",
    "neck_length_ratio",
    "head_width_to_shoulder_ratio",
    "head_height_to_torso_ratio",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dist(joints: np.ndarray, a: int, b: int) -> float:
    """Euclidean distance between joint a and joint b."""
    return float(np.linalg.norm(joints[a] - joints[b]))


def _bilateral(joints: np.ndarray, left: int, left2: int,
               right: int, right2: int) -> float:
    """Mean of bilateral Euclidean distances."""
    return (_dist(joints, left, left2) + _dist(joints, right, right2)) / 2.0


# ---------------------------------------------------------------------------
# Core feature extraction  (existing behaviour — DO NOT change return values)
# ---------------------------------------------------------------------------

def _compute_core_features(
    joints: np.ndarray,
    joint_idx: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute all existing CORE_FEATURE_KEYS.

    joints    : (J, 3)  all SMPL-X joints in canonical pose, pelvis at origin
    joint_idx : name -> raw index mapping (the pipeline's JOINT_IDX)
    """
    ji = joint_idx   # shorthand

    # ── Height ──────────────────────────────────────────────────────────
    # head joint Y (skull level) minus lowest ankle Y.
    # Purely joint-based; uses raw index 15 because 'head' is not in JOINT_IDX.
    head_y  = float(joints[_IDX_HEAD, _Y])
    foot_y  = float(min(joints[ji["left_ankle"], _Y],
                        joints[ji["right_ankle"], _Y]))
    height_m = head_y - foot_y
    if height_m < 1e-4:
        raise ValueError(
            f"Computed height_m={height_m:.6f} is near zero. "
            "Check joint conventions or model output."
        )

    # ── Width / breadth features ─────────────────────────────────────────
    shoulder_width = _dist(joints, ji["left_shoulder"], ji["right_shoulder"])
    pelvis_width   = _dist(joints, ji["left_hip"],      ji["right_hip"])

    # ── Length features ──────────────────────────────────────────────────
    torso_height = _dist(joints, ji["pelvis"], ji["neck"])

    arm_length  = _bilateral(joints,
                              ji["left_shoulder"],  ji["left_wrist"],
                              ji["right_shoulder"], ji["right_wrist"])
    leg_length  = _bilateral(joints,
                              ji["left_hip"],  ji["left_ankle"],
                              ji["right_hip"], ji["right_ankle"])
    thigh       = _bilateral(joints,
                              ji["left_hip"],  ji["left_knee"],
                              ji["right_hip"], ji["right_knee"])
    shank       = _bilateral(joints,
                              ji["left_knee"],  ji["left_ankle"],
                              ji["right_knee"], ji["right_ankle"])

    return {
        "height_canonical":      height_m,
        "shoulder_width_ratio":  shoulder_width / height_m,
        "pelvis_width_ratio":    pelvis_width   / height_m,
        "torso_height_ratio":    torso_height   / height_m,
        "arm_length_ratio":      arm_length     / height_m,
        "thigh_ratio":           thigh          / height_m,
        "shank_ratio":           shank          / height_m,
        "leg_length_ratio":      leg_length     / height_m,
        # Store raw values for use by head/neck feature computation
        "_shoulder_width_m":     shoulder_width,
        "_torso_height_m":       torso_height,
        "_height_m":             height_m,
    }


# ---------------------------------------------------------------------------
# Head / neck feature extraction  (new — optional, uses verts)
# ---------------------------------------------------------------------------

def _compute_head_neck_features(
    verts: np.ndarray,
    joints: np.ndarray,
    core: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute HEAD_NECK_FEATURE_KEYS using vertex+joint approach.

    verts  : (V, 3)  SMPL-X mesh vertices in canonical pose, pelvis at origin
    joints : (J, 3)  canonical joints
    core   : output of _compute_core_features (for derived ratios)

    Returns a dict of new features.  Returns empty dict if verts is None or
    the joint array is too short.
    """
    if verts is None or joints.shape[0] <= _IDX_HEAD:
        return {}

    height_m      = core["_height_m"]
    shoulder_w_m  = core["_shoulder_width_m"]
    torso_h_m     = core["_torso_height_m"]

    neck_y = float(joints[_IDX_NECK, _Y])

    # ── Identify "above-neck" vertices ───────────────────────────────────
    above_neck_mask = verts[:, _Y] > neck_y
    n_above = int(above_neck_mask.sum())

    if n_above < 10:
        # Fallback: use the head joint as a last resort
        head_joint_y = float(joints[_IDX_HEAD, _Y])
        head_height_m = max(head_joint_y - neck_y, 1e-4)
        head_width_m  = head_height_m * 0.65   # rough proportion placeholder
        _fallback = True
    else:
        head_verts_y = verts[above_neck_mask, _Y]
        head_top_y   = float(head_verts_y.max())
        head_height_m = max(head_top_y - neck_y, 1e-4)
        _fallback = False

        # Head width: restrict to upper half of head to exclude arm vertices
        # Arms in T-pose extend horizontally and may have Y > neck_y, but they
        # are in the lower part of the above-neck region.  Taking only vertices
        # in the upper 50% of the head height effectively selects the skull cap.
        mid_head_y = neck_y + 0.5 * head_height_m
        skull_cap_mask = verts[:, _Y] > mid_head_y
        n_skull_cap = int(skull_cap_mask.sum())

        if n_skull_cap >= 3:
            skull_x = verts[skull_cap_mask, _X]
            head_width_m = float(skull_x.max() - skull_x.min())
        else:
            # Very few verts in upper half — use full above-neck X span
            head_x = verts[above_neck_mask, _X]
            head_width_m = float(head_x.max() - head_x.min())

    # ── neck_length_ratio: joint-based ──────────────────────────────────
    # spine3 (9) → neck (12); always available if joint array has ≥ 13 joints
    if joints.shape[0] > max(_IDX_SPINE3, _IDX_NECK):
        neck_length_m = _dist(joints, _IDX_SPINE3, _IDX_NECK)
    else:
        neck_length_m = float("nan")

    # ── Assemble features ────────────────────────────────────────────────
    result: Dict[str, float] = {}

    result["head_height_ratio"] = head_height_m / height_m
    result["head_width_ratio"]  = head_width_m  / height_m

    if not np.isnan(neck_length_m):
        result["neck_length_ratio"] = neck_length_m / height_m

    if shoulder_w_m > 1e-6:
        result["head_width_to_shoulder_ratio"] = head_width_m / shoulder_w_m

    if torso_h_m > 1e-6:
        result["head_height_to_torso_ratio"] = head_height_m / torso_h_m

    # Diagnostic flag: always written so step1 can include it as a CSV column.
    # Value: 1 if fallback logic was used (verts above neck too sparse),
    #        0 if normal vertex-based computation succeeded.
    result["head_features_used_fallback"] = 1.0 if _fallback else 0.0

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_features_from_joints(
    verts: Optional[np.ndarray],
    joints: np.ndarray,
    joint_idx: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute all anthropometric features from SMPL-X canonical-pose output.

    Parameters
    ----------
    verts      : (V, 3) vertex array, pelvis at origin.
                 May be None; head/neck features will be skipped if so.
    joints     : (J, 3) joint array, pelvis at origin.
    joint_idx  : dict mapping joint name → raw index (pipeline's JOINT_IDX).

    Returns
    -------
    dict with keys from CORE_FEATURE_KEYS (always) + HEAD_NECK_FEATURE_KEYS
    (when verts is available and the joint array has sufficient joints).

    The returned dict may also contain internal keys prefixed with '_' that
    are used only for derived-feature computation and should not be written
    to CSV.  step1 filters these via FEATURE_KEYS / HEAD_NECK_FEATURE_KEYS.
    """
    core = _compute_core_features(joints, joint_idx)
    head = _compute_head_neck_features(verts, joints, core)

    # Merge; internal '_*' keys are present but harmless (step1 ignores them)
    return {**core, **head}
