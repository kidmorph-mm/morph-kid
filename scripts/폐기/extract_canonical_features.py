# scripts/extract_canonical_features.py
from __future__ import annotations

from typing import Dict
import numpy as np


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def get_height_from_vertices(verts: np.ndarray, up_axis: int = 1) -> float:
    vmin = float(np.min(verts[:, up_axis]))
    vmax = float(np.max(verts[:, up_axis]))
    return vmax - vmin


def extract_features_from_joints(
    verts: np.ndarray,
    joints: np.ndarray,
    joint_idx: Dict[str, int],
) -> Dict[str, float]:
    """
    joint_idx 예시:
    {
        "pelvis": 0,
        "left_hip": ...,
        "right_hip": ...,
        "spine3": ...,
        "neck": ...,
        "left_shoulder": ...,
        "right_shoulder": ...,
        "left_elbow": ...,
        "right_elbow": ...,
        "left_wrist": ...,
        "right_wrist": ...,
        "left_knee": ...,
        "right_knee": ...,
        "left_ankle": ...,
        "right_ankle": ...,
    }
    """
    height_canonical = get_height_from_vertices(verts, up_axis=1)

    pelvis = joints[joint_idx["pelvis"]]
    neck = joints[joint_idx["neck"]]

    l_sh = joints[joint_idx["left_shoulder"]]
    r_sh = joints[joint_idx["right_shoulder"]]
    l_el = joints[joint_idx["left_elbow"]]
    r_el = joints[joint_idx["right_elbow"]]
    l_wr = joints[joint_idx["left_wrist"]]
    r_wr = joints[joint_idx["right_wrist"]]

    l_hip = joints[joint_idx["left_hip"]]
    r_hip = joints[joint_idx["right_hip"]]
    l_kn = joints[joint_idx["left_knee"]]
    r_kn = joints[joint_idx["right_knee"]]
    l_an = joints[joint_idx["left_ankle"]]
    r_an = joints[joint_idx["right_ankle"]]

    shoulder_width = l2(l_sh, r_sh)
    pelvis_width = l2(l_hip, r_hip)
    torso_height = l2(pelvis, neck)

    upper_arm_l = l2(l_sh, l_el)
    upper_arm_r = l2(r_sh, r_el)
    forearm_l = l2(l_el, l_wr)
    forearm_r = l2(r_el, r_wr)

    upper_arm = (upper_arm_l + upper_arm_r) / 2.0
    forearm = (forearm_l + forearm_r) / 2.0
    arm_length = upper_arm + forearm

    thigh_l = l2(l_hip, l_kn)
    thigh_r = l2(r_hip, r_kn)
    shank_l = l2(l_kn, l_an)
    shank_r = l2(r_kn, r_an)

    thigh = (thigh_l + thigh_r) / 2.0
    shank = (shank_l + shank_r) / 2.0
    leg_length = thigh + shank

    feats = {
        "height_canonical": height_canonical,
        "shoulder_width": shoulder_width,
        "pelvis_width": pelvis_width,
        "torso_height": torso_height,
        "upper_arm": upper_arm,
        "forearm": forearm,
        "arm_length": arm_length,
        "thigh": thigh,
        "shank": shank,
        "leg_length": leg_length,
    }

    h = max(height_canonical, 1e-8)
    for k, v in list(feats.items()):
        if k != "height_canonical":
            feats[f"{k}_ratio"] = v / h

    return feats