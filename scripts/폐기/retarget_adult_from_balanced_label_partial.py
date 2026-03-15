from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import trimesh

from utils_smplx import load_smplx_model, center_by_pelvis
from extract_canonical_features import extract_features_from_joints


MODEL_ROOT = "/home/jaeson1012/agora_dataset/models"
BALANCED_LABEL_CSV = Path("/home/jaeson1012/agora_dataset/runs/pseudo_labels_balanced/adult_pseudo_labels_balanced.csv")

JOINT_IDX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

FEATURE_KEYS = [
    "height_canonical",
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "arm_length_ratio",
    "thigh_ratio",
    "shank_ratio",
    "leg_length_ratio",
]

DEFAULT_ALPHA_CONFIG = {
    "global": 0.75,
    "pelvis": 0.70,
    "torso": 0.40,
    "shoulder": 0.50,
    "arm": 0.45,
    "leg": 0.55,
    "posed_delta": 0.70,
}


def load_pkl(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_gender(gender_value: Any) -> str:
    if gender_value is None:
        return "neutral"
    if isinstance(gender_value, bytes):
        gender_value = gender_value.decode("utf-8", errors="ignore")
    if isinstance(gender_value, np.ndarray):
        if gender_value.size == 1:
            gender_value = gender_value.reshape(-1)[0]
        else:
            gender_value = gender_value.tolist()
    gender = str(gender_value).strip().lower()
    if gender in {"male", "m"}:
        return "male"
    if gender in {"female", "f"}:
        return "female"
    return "neutral"


def ensure_batch(x: Any, dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if dim is not None and arr.shape[-1] != dim:
        raise ValueError(f"Expected dim {dim}, got {arr.shape}")
    return arr


def build_smplx_output(model, fit_data: Dict[str, Any], canonical: bool = False):
    import torch

    device = next(model.parameters()).device
    dtype = torch.float32

    betas = torch.tensor(ensure_batch(fit_data["betas"]), dtype=dtype, device=device)
    batch_size = betas.shape[0]

    def get_or_zero(key: str, dim: int):
        if canonical:
            return torch.zeros((batch_size, dim), dtype=dtype, device=device)
        if key not in fit_data:
            return torch.zeros((batch_size, dim), dtype=dtype, device=device)
        return torch.tensor(ensure_batch(fit_data[key], dim), dtype=dtype, device=device)

    num_expression_coeffs = getattr(model, "num_expression_coeffs", 10)
    num_body_joints = getattr(model, "NUM_BODY_JOINTS", 21)
    num_hand_joints = getattr(model, "NUM_HAND_JOINTS", 15)

    out = model(
        betas=betas,
        transl=get_or_zero("transl", 3),
        global_orient=get_or_zero("global_orient", 3),
        body_pose=get_or_zero("body_pose", num_body_joints * 3),
        left_hand_pose=get_or_zero("left_hand_pose", num_hand_joints * 3),
        right_hand_pose=get_or_zero("right_hand_pose", num_hand_joints * 3),
        jaw_pose=get_or_zero("jaw_pose", 3),
        leye_pose=get_or_zero("leye_pose", 3),
        reye_pose=get_or_zero("reye_pose", 3),
        expression=get_or_zero("expression", num_expression_coeffs),
        return_verts=True,
    )

    return (
        out.vertices[0].detach().cpu().numpy(),
        out.joints[0].detach().cpu().numpy(),
        model.faces.astype(np.int32),
    )


def load_balanced_target(adult_pkl: str | Path, label_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(label_csv)
    adult_pkl = str(Path(adult_pkl))
    row = df[df["adult_sample_path"] == adult_pkl]
    if len(row) == 0:
        raise KeyError(f"adult sample not found in balanced label csv: {adult_pkl}")
    row = row.iloc[0]

    return {
        "group": str(row["assigned_group"]),
        "target_height_cm": float(row["assigned_target_height_cm"]),
        "height_canonical": float(row["assigned_target_height_cm"]) / 100.0,
        "shoulder_width_ratio": float(row["assigned_target_shoulder_width_ratio"]),
        "pelvis_width_ratio": float(row["assigned_target_pelvis_width_ratio"]),
        "torso_height_ratio": float(row["assigned_target_torso_HEIGHT_ratio".lower()] if "assigned_target_torso_height_ratio" not in row else row["assigned_target_torso_height_ratio"]),
        "arm_length_ratio": float(row["assigned_target_arm_length_ratio"]),
        "thigh_ratio": float(row["assigned_target_thigh_ratio"]),
        "shank_ratio": float(row["assigned_target_shank_ratio"]),
        "leg_length_ratio": float(row["assigned_target_leg_length_ratio"]),
        "source_assignment_score": float(row["assignment_score"]),
        "source_height_bin": str(row["assigned_height_bin"]),
    }


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def point_to_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = np.sum(ab * ab)
    if ab2 < 1e-10:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1)


def point_to_line_t(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = np.sum(ab * ab)
    if ab2 < 1e-10:
        return np.zeros((points.shape[0],), dtype=np.float32)
    t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / ab2
    return t.astype(np.float32)


def gaussian_segment_mask(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    d = point_to_segment_distance(points, a, b)
    w = np.exp(- (d / max(radius, 1e-6)) ** 2)
    return np.clip(w, 0.0, 1.0)


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def save_feature_compare(path: Path, before: Dict[str, float], after: Dict[str, float], target: Dict[str, float]):
    rows = []
    for k in FEATURE_KEYS:
        rows.append({
            "feature": k,
            "before": before.get(k, np.nan),
            "after": after.get(k, np.nan),
            "target": target.get(k, np.nan),
            "abs_err_before": abs(before.get(k, np.nan) - target.get(k, np.nan)),
            "abs_err_after": abs(after.get(k, np.nan) - target.get(k, np.nan)),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def build_target_lengths(current_height: float, target: Dict[str, float]) -> Dict[str, float]:
    target_height = target["height_canonical"]
    return {
        "target_height": target_height,
        "shoulder_width": target["shoulder_width_ratio"] * target_height,
        "pelvis_width": target["pelvis_width_ratio"] * target_height,
        "torso_height": target["torso_height_ratio"] * target_height,
        "arm_length": target["arm_length_ratio"] * target_height,
        "thigh": target["thigh_ratio"] * target_height,
        "shank": target["shank_ratio"] * target_height,
        "leg_length": target["leg_length_ratio"] * target_height,
    }


def lerp_point(old: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return old + alpha * (target - old)


def retarget_canonical_joints_partial(
    joints: np.ndarray,
    verts: np.ndarray,
    target: Dict[str, float],
    alpha_cfg: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    j = joints.copy()
    pelvis = j[JOINT_IDX["pelvis"]].copy()

    current_height = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
    tlen = build_target_lengths(current_height, target)

    # 1) partial global scale
    g_full = tlen["target_height"] / max(current_height, 1e-8)
    g = 1.0 + alpha_cfg["global"] * (g_full - 1.0)
    j = pelvis[None, :] + g * (j - pelvis[None, :])

    # refresh
    pelvis = j[JOINT_IDX["pelvis"]]
    neck = j[JOINT_IDX["neck"]]
    l_sh = j[JOINT_IDX["left_shoulder"]]
    r_sh = j[JOINT_IDX["right_shoulder"]]
    l_el = j[JOINT_IDX["left_elbow"]]
    r_el = j[JOINT_IDX["right_elbow"]]
    l_wr = j[JOINT_IDX["left_wrist"]]
    r_wr = j[JOINT_IDX["right_wrist"]]
    l_hip = j[JOINT_IDX["left_hip"]]
    r_hip = j[JOINT_IDX["right_hip"]]
    l_kn = j[JOINT_IDX["left_knee"]]
    r_kn = j[JOINT_IDX["right_knee"]]
    l_an = j[JOINT_IDX["left_ankle"]]
    r_an = j[JOINT_IDX["right_ankle"]]

    # 2) pelvis width
    hip_center = 0.5 * (l_hip + r_hip)
    hip_axis = normalize(r_hip - l_hip)
    half_pw_old = 0.5 * l2(l_hip, r_hip)
    half_pw_new = 0.5 * tlen["pelvis_width"]
    l_hip_target = hip_center - half_pw_new * hip_axis
    r_hip_target = hip_center + half_pw_new * hip_axis
    j[JOINT_IDX["left_hip"]] = lerp_point(l_hip, l_hip_target, alpha_cfg["pelvis"])
    j[JOINT_IDX["right_hip"]] = lerp_point(r_hip, r_hip_target, alpha_cfg["pelvis"])

    # 3) shoulder width
    shoulder_center = 0.5 * (l_sh + r_sh)
    shoulder_axis = normalize(r_sh - l_sh)
    half_sw_new = 0.5 * tlen["shoulder_width"]
    l_sh_target = shoulder_center - half_sw_new * shoulder_axis
    r_sh_target = shoulder_center + half_sw_new * shoulder_axis
    j[JOINT_IDX["left_shoulder"]] = lerp_point(l_sh, l_sh_target, alpha_cfg["shoulder"])
    j[JOINT_IDX["right_shoulder"]] = lerp_point(r_sh, r_sh_target, alpha_cfg["shoulder"])

    # 4) torso height
    pelvis_now = j[JOINT_IDX["pelvis"]]
    neck_now = j[JOINT_IDX["neck"]]
    torso_axis = normalize(neck_now - pelvis_now)
    neck_target = pelvis_now + tlen["torso_height"] * torso_axis
    neck_new = lerp_point(neck_now, neck_target, alpha_cfg["torso"])
    neck_delta = neck_new - neck_now
    j[JOINT_IDX["neck"]] = neck_new

    # move upper-body chain with neck shift partially
    for name in [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]:
        j[JOINT_IDX[name]] += neck_delta

    # 5) arm chain
    for side in ["left", "right"]:
        sh_id = JOINT_IDX[f"{side}_shoulder"]
        el_id = JOINT_IDX[f"{side}_elbow"]
        wr_id = JOINT_IDX[f"{side}_wrist"]

        shoulder = j[sh_id]
        elbow = j[el_id]
        wrist = j[wr_id]

        arm_axis = normalize(wrist - shoulder)
        current_upper = l2(shoulder, elbow)
        current_fore = l2(elbow, wrist)
        current_total = max(current_upper + current_fore, 1e-8)

        target_upper = tlen["arm_length"] * (current_upper / current_total)
        target_fore = tlen["arm_length"] * (current_fore / current_total)

        elbow_target = shoulder + target_upper * arm_axis
        wrist_target = elbow_target + target_fore * arm_axis

        j[el_id] = lerp_point(elbow, elbow_target, alpha_cfg["arm"])
        j[wr_id] = lerp_point(wrist, wrist_target, alpha_cfg["arm"])

    # 6) leg chain
    for side in ["left", "right"]:
        hip_id = JOINT_IDX[f"{side}_hip"]
        knee_id = JOINT_IDX[f"{side}_knee"]
        ankle_id = JOINT_IDX[f"{side}_ankle"]

        hip = j[hip_id]
        knee = j[knee_id]
        ankle = j[ankle_id]

        leg_axis = normalize(ankle - hip)
        knee_target = hip + tlen["thigh"] * leg_axis
        ankle_target = knee_target + tlen["shank"] * leg_axis

        j[knee_id] = lerp_point(knee, knee_target, alpha_cfg["leg"])
        j[ankle_id] = lerp_point(ankle, ankle_target, alpha_cfg["leg"])

    return j, {"global_scale_applied": float(g), "global_scale_full": float(g_full)}


def bone_chain_update(
    points: np.ndarray,
    old_a: np.ndarray,
    old_b: np.ndarray,
    new_a: np.ndarray,
    new_b: np.ndarray,
    radius: float,
) -> np.ndarray:
    w = gaussian_segment_mask(points, old_a, old_b, radius)
    t = point_to_line_t(points, old_a, old_b)
    t = np.clip(t, 0.0, 1.0)

    old_ab = old_b - old_a
    new_ab = new_b - new_a
    old_u = normalize(old_ab)

    rel = points - old_a[None, :]
    parallel = np.sum(rel * old_u[None, :], axis=1)[:, None] * old_u[None, :]
    perp = rel - parallel

    mapped = new_a[None, :] + t[:, None] * new_ab[None, :] + perp
    return points * (1.0 - w[:, None]) + mapped * w[:, None]


def smooth_joint_displacement(points: np.ndarray, old_joints: np.ndarray, new_joints: np.ndarray, radius: float) -> np.ndarray:
    moved = new_joints - old_joints
    accum = np.zeros_like(points)
    denom = np.zeros((points.shape[0], 1), dtype=np.float32)

    use_joint_ids = [
        JOINT_IDX["pelvis"], JOINT_IDX["neck"],
        JOINT_IDX["left_shoulder"], JOINT_IDX["right_shoulder"],
        JOINT_IDX["left_elbow"], JOINT_IDX["right_elbow"],
        JOINT_IDX["left_wrist"], JOINT_IDX["right_wrist"],
        JOINT_IDX["left_hip"], JOINT_IDX["right_hip"],
        JOINT_IDX["left_knee"], JOINT_IDX["right_knee"],
        JOINT_IDX["left_ankle"], JOINT_IDX["right_ankle"],
    ]

    for jid in use_joint_ids:
        center = old_joints[jid]
        d = np.linalg.norm(points - center[None, :], axis=1)
        w = np.exp(- (d / max(radius, 1e-6)) ** 2).astype(np.float32)[:, None]
        accum += w * moved[jid][None, :]
        denom += w

    return points + accum / np.maximum(denom, 1e-6)


def deform_mesh_from_joint_retarget_partial(
    verts_in: np.ndarray,
    old_joints: np.ndarray,
    new_joints: np.ndarray,
    alpha_cfg: Dict[str, float],
) -> np.ndarray:
    verts = verts_in.copy()
    h = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
    r_torso = 0.11 * h
    r_pelvis = 0.11 * h
    r_shoulder = 0.10 * h
    r_limb = 0.07 * h

    # torso
    torso_target = bone_chain_update(
        verts,
        old_joints[JOINT_IDX["pelvis"]],
        old_joints[JOINT_IDX["neck"]],
        new_joints[JOINT_IDX["pelvis"]],
        new_joints[JOINT_IDX["neck"]],
        r_torso,
    )
    verts = verts + alpha_cfg["torso"] * (torso_target - verts)

    # pelvis
    pelvis_target = bone_chain_update(
        verts,
        old_joints[JOINT_IDX["left_hip"]],
        old_joints[JOINT_IDX["right_hip"]],
        new_joints[JOINT_IDX["left_hip"]],
        new_joints[JOINT_IDX["right_hip"]],
        r_pelvis,
    )
    verts = verts + alpha_cfg["pelvis"] * (pelvis_target - verts)

    # shoulders
    shoulder_target = bone_chain_update(
        verts,
        old_joints[JOINT_IDX["left_shoulder"]],
        old_joints[JOINT_IDX["right_shoulder"]],
        new_joints[JOINT_IDX["left_shoulder"]],
        new_joints[JOINT_IDX["right_shoulder"]],
        r_shoulder,
    )
    verts = verts + alpha_cfg["shoulder"] * (shoulder_target - verts)

    # arms
    for a_name, b_name in [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]:
        target = bone_chain_update(
            verts,
            old_joints[JOINT_IDX[a_name]],
            old_joints[JOINT_IDX[b_name]],
            new_joints[JOINT_IDX[a_name]],
            new_joints[JOINT_IDX[b_name]],
            r_limb,
        )
        verts = verts + alpha_cfg["arm"] * (target - verts)

    # legs
    for a_name, b_name in [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]:
        target = bone_chain_update(
            verts,
            old_joints[JOINT_IDX[a_name]],
            old_joints[JOINT_IDX[b_name]],
            new_joints[JOINT_IDX[a_name]],
            new_joints[JOINT_IDX[b_name]],
            r_limb,
        )
        verts = verts + alpha_cfg["leg"] * (target - verts)

    # smooth overall displacement
    smooth_target = smooth_joint_displacement(verts, old_joints, new_joints, radius=0.09 * h)
    verts = verts + 0.35 * (smooth_target - verts)

    return verts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-pkl", type=str, required=True)
    parser.add_argument("--balanced-label-csv", type=str, default=str(BALANCED_LABEL_CSV))
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fit_data = load_pkl(args.adult_pkl)
    gender = normalize_gender(fit_data.get("gender", "neutral"))
    target = load_balanced_target(args.adult_pkl, Path(args.balanced_label_csv))
    alpha_cfg = DEFAULT_ALPHA_CONFIG.copy()

    model = load_smplx_model(
        model_root=MODEL_ROOT,
        gender=gender,
        device="cuda",
        num_betas=int(np.asarray(fit_data["betas"]).shape[-1]),
        kid_template_path=None,
    )

    can_verts, can_joints, faces = build_smplx_output(model, fit_data, canonical=True)
    posed_verts, posed_joints, _ = build_smplx_output(model, fit_data, canonical=False)

    can_verts, can_joints = center_by_pelvis(can_verts, can_joints, pelvis_idx=JOINT_IDX["pelvis"])
    posed_verts, posed_joints = center_by_pelvis(posed_verts, posed_joints, pelvis_idx=JOINT_IDX["pelvis"])

    before_feats = extract_features_from_joints(can_verts, can_joints, JOINT_IDX)

    new_can_joints, retarget_info = retarget_canonical_joints_partial(can_joints, can_verts, target, alpha_cfg)
    new_can_verts = deform_mesh_from_joint_retarget_partial(can_verts, can_joints, new_can_joints, alpha_cfg)

    joint_delta = new_can_joints - can_joints
    new_posed_joints = posed_joints + alpha_cfg["posed_delta"] * joint_delta
    new_posed_verts = deform_mesh_from_joint_retarget_partial(
        posed_verts,
        posed_joints,
        new_posed_joints,
        alpha_cfg,
    )

    after_feats = extract_features_from_joints(new_can_verts, new_can_joints, JOINT_IDX)

    stem = Path(args.adult_pkl).stem
    save_obj(outdir / f"{stem}_adult_canonical.obj", can_verts, faces)
    save_obj(outdir / f"{stem}_child_canonical.obj", new_can_verts, faces)
    save_obj(outdir / f"{stem}_adult_posed.obj", posed_verts, faces)
    save_obj(outdir / f"{stem}_child_posed.obj", new_posed_verts, faces)

    save_feature_compare(outdir / f"{stem}_feature_compare.csv", before_feats, after_feats, target)

    report = {
        "adult_pkl": args.adult_pkl,
        "gender": gender,
        "balanced_target": target,
        "alpha_config": alpha_cfg,
        "retarget_info": retarget_info,
        "before_feats": {k: float(v) for k, v in before_feats.items()},
        "after_feats": {k: float(v) for k, v in after_feats.items()},
    }

    with open(outdir / f"{stem}_retarget_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print("[INFO] adult:", args.adult_pkl)
    print("[INFO] assigned group:", target["group"])
    print("[INFO] assigned target height cm:", target["target_height_cm"])
    print("[INFO] assigned height bin:", target["source_height_bin"])
    print("[INFO] before height:", before_feats["height_canonical"])
    print("[INFO] after height:", after_feats["height_canonical"])
    print("[INFO] target height:", target["height_canonical"])


if __name__ == "__main__":
    main()