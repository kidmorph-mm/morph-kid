from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import trimesh
from scipy.optimize import minimize

from utils_smplx import load_smplx_model, center_by_pelvis
from extract_canonical_features import extract_features_from_joints


MODEL_ROOT = "/home/jaeson1012/agora_dataset/models"
FINAL_GT_DIR = Path("/home/jaeson1012/agora_dataset/data/final_child_gt")

JOINT_IDX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "neck": 12,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
}


# -----------------------------
# I/O
# -----------------------------
def load_pkl(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
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
        raise ValueError(f"Expected last dim {dim}, got shape={arr.shape}")
    return arr


# -----------------------------
# SMPL-X forward
# -----------------------------
def build_smplx_output(
    model,
    fit_data: Dict[str, Any],
    canonical: bool = False,
):
    import torch

    device = next(model.parameters()).device
    dtype = torch.float32

    betas = ensure_batch(fit_data["betas"])
    betas_t = torch.tensor(betas, dtype=dtype, device=device)

    batch_size = betas_t.shape[0]

    def get_or_zero(key: str, dim: int) -> torch.Tensor:
        if canonical:
            return torch.zeros((batch_size, dim), dtype=dtype, device=device)
        if key not in fit_data:
            return torch.zeros((batch_size, dim), dtype=dtype, device=device)
        return torch.tensor(ensure_batch(fit_data[key], dim), dtype=dtype, device=device)

    num_expression_coeffs = getattr(model, "num_expression_coeffs", 10)
    num_body_joints = getattr(model, "NUM_BODY_JOINTS", 21)
    num_hand_joints = getattr(model, "NUM_HAND_JOINTS", 15)

    output = model(
        betas=betas_t,
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

    verts = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)
    return verts, joints, faces


# -----------------------------
# Child target selection
# -----------------------------
def load_target_prototype(group: str, target_height_cm: float) -> Dict[str, float]:
    group = group.lower()
    if group not in {"core", "sub"}:
        raise ValueError("group must be 'core' or 'sub'")

    bin_csv = FINAL_GT_DIR / f"gt_child_{group}_bin_prototypes_5cm.csv"
    if not bin_csv.exists():
        raise FileNotFoundError(f"Missing prototype csv: {bin_csv}")

    df = pd.read_csv(bin_csv)
    if df.empty:
        raise RuntimeError(f"Empty prototype csv: {bin_csv}")

    idx = (df["height_cm_median"] - target_height_cm).abs().idxmin()
    row = df.loc[idx].to_dict()

    target = {
        "height_canonical": float(row["height_cm_median"]) / 100.0,
        "shoulder_width_ratio": float(row["shoulder_width_ratio_median"]),
        "pelvis_width_ratio": float(row["pelvis_width_ratio_median"]),
        "torso_height_ratio": float(row["torso_height_ratio_median"]),
        "arm_length_ratio": float(row["arm_length_ratio_median"]),
        "thigh_ratio": float(row["thigh_ratio_median"]),
        "shank_ratio": float(row["shank_ratio_median"]),
        "leg_length_ratio": float(row["leg_length_ratio_median"]),
    }

    # 시각적 아이 느낌 보정용 head prior
    target["head_scale_prior"] = 1.10 if group == "core" else 1.06
    target["shoulder_scale_prior"] = 0.90 if group == "core" else 0.94
    return target


# -----------------------------
# Geometry helpers
# -----------------------------
def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(v) + eps)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def point_to_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = np.sum(ab * ab)
    if ab2 < 1e-10:
        return np.linalg.norm(points - a[None, :], axis=1)

    t = np.sum((points - a[None, :]) * ab[None, :], axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1)


def soft_segment_mask(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    d = point_to_segment_distance(points, a, b)
    w = np.exp(- (d / max(radius, 1e-6)) ** 2)
    return np.clip(w, 0.0, 1.0)


def blend_points(orig: np.ndarray, new: np.ndarray, w: np.ndarray) -> np.ndarray:
    return orig * (1.0 - w[:, None]) + new * w[:, None]


def scale_along_bone(points: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    u = normalize(b - a)
    rel = points - a[None, :]
    parallel = np.sum(rel * u[None, :], axis=1)[:, None] * u[None, :]
    perp = rel - parallel
    return a[None, :] + scale * parallel + perp


def scale_along_axis(points: np.ndarray, center: np.ndarray, axis: np.ndarray, scale: float) -> np.ndarray:
    u = normalize(axis)
    rel = points - center[None, :]
    parallel = np.sum(rel * u[None, :], axis=1)[:, None] * u[None, :]
    perp = rel - parallel
    return center[None, :] + scale * parallel + perp


def scale_isotropic(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return center[None, :] + scale * (points - center[None, :])


# -----------------------------
# Visual-friendly articulated warp
# -----------------------------
def apply_child_warp(
    verts_in: np.ndarray,
    joints_in: np.ndarray,
    params: Dict[str, float],
    joint_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    verts = verts_in.copy()
    joints = joints_in.copy()

    pelvis = joints[joint_idx["pelvis"]].copy()
    h = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
    if h < 1e-6:
        return verts, joints

    # 1) global scale
    g = params["global_scale"]
    verts = pelvis[None, :] + g * (verts - pelvis[None, :])
    joints = pelvis[None, :] + g * (joints - pelvis[None, :])

    # refresh anchors
    pelvis = joints[joint_idx["pelvis"]]
    l_hip = joints[joint_idx["left_hip"]]
    r_hip = joints[joint_idx["right_hip"]]
    neck = joints[joint_idx["neck"]]
    l_sh = joints[joint_idx["left_shoulder"]]
    r_sh = joints[joint_idx["right_shoulder"]]
    l_el = joints[joint_idx["left_elbow"]]
    r_el = joints[joint_idx["right_elbow"]]
    l_wr = joints[joint_idx["left_wrist"]]
    r_wr = joints[joint_idx["right_wrist"]]
    l_kn = joints[joint_idx["left_knee"]]
    r_kn = joints[joint_idx["right_knee"]]
    l_an = joints[joint_idx["left_ankle"]]
    r_an = joints[joint_idx["right_ankle"]]

    # radius
    r_torso = 0.11 * h
    r_limb = 0.07 * h
    r_pelvis = 0.10 * h

    # 2) pelvis width
    pelvis_axis = r_hip - l_hip
    pelvis_center = 0.5 * (l_hip + r_hip)
    v_new = scale_along_axis(verts, pelvis_center, pelvis_axis, params["pelvis_scale"])
    j_new = scale_along_axis(joints, pelvis_center, pelvis_axis, params["pelvis_scale"])
    pelvis_mask_v = soft_segment_mask(verts, l_hip, r_hip, r_pelvis)
    pelvis_mask_j = soft_segment_mask(joints, l_hip, r_hip, r_pelvis)
    verts = blend_points(verts, v_new, pelvis_mask_v)
    joints = blend_points(joints, j_new, pelvis_mask_j)

    # refresh
    pelvis = joints[joint_idx["pelvis"]]
    l_hip = joints[joint_idx["left_hip"]]
    r_hip = joints[joint_idx["right_hip"]]
    neck = joints[joint_idx["neck"]]
    l_sh = joints[joint_idx["left_shoulder"]]
    r_sh = joints[joint_idx["right_shoulder"]]

    # 3) torso length
    v_new = scale_along_bone(verts, pelvis, neck, params["torso_scale"])
    j_new = scale_along_bone(joints, pelvis, neck, params["torso_scale"])
    torso_mask_v = soft_segment_mask(verts, pelvis, neck, r_torso)
    torso_mask_j = soft_segment_mask(joints, pelvis, neck, r_torso)
    # upper body points는 torso scale 영향 조금 더 받게
    upper_v = (verts[:, 1] > pelvis[1]).astype(np.float32)
    upper_j = (joints[:, 1] > pelvis[1]).astype(np.float32)
    torso_mask_v = np.clip(np.maximum(torso_mask_v, 0.35 * upper_v), 0.0, 1.0)
    torso_mask_j = np.clip(np.maximum(torso_mask_j, 0.35 * upper_j), 0.0, 1.0)
    verts = blend_points(verts, v_new, torso_mask_v)
    joints = blend_points(joints, j_new, torso_mask_j)

    # refresh
    pelvis = joints[joint_idx["pelvis"]]
    neck = joints[joint_idx["neck"]]
    l_sh = joints[joint_idx["left_shoulder"]]
    r_sh = joints[joint_idx["right_shoulder"]]
    l_el = joints[joint_idx["left_elbow"]]
    r_el = joints[joint_idx["right_elbow"]]
    l_wr = joints[joint_idx["left_wrist"]]
    r_wr = joints[joint_idx["right_wrist"]]

    # 4) shoulder width
    shoulder_axis = r_sh - l_sh
    shoulder_center = 0.5 * (l_sh + r_sh)
    v_new = scale_along_axis(verts, shoulder_center, shoulder_axis, params["shoulder_scale"])
    j_new = scale_along_axis(joints, shoulder_center, shoulder_axis, params["shoulder_scale"])
    shoulder_mask_v = np.maximum(
        soft_segment_mask(verts, l_sh, r_sh, 0.09 * h),
        (verts[:, 1] > neck[1] - 0.05 * h).astype(np.float32) * 0.25,
    )
    shoulder_mask_j = np.maximum(
        soft_segment_mask(joints, l_sh, r_sh, 0.09 * h),
        (joints[:, 1] > neck[1] - 0.05 * h).astype(np.float32) * 0.25,
    )
    verts = blend_points(verts, v_new, np.clip(shoulder_mask_v, 0.0, 1.0))
    joints = blend_points(joints, j_new, np.clip(shoulder_mask_j, 0.0, 1.0))

    # refresh arm joints
    l_sh = joints[joint_idx["left_shoulder"]]
    r_sh = joints[joint_idx["right_shoulder"]]
    l_el = joints[joint_idx["left_elbow"]]
    r_el = joints[joint_idx["right_elbow"]]
    l_wr = joints[joint_idx["left_wrist"]]
    r_wr = joints[joint_idx["right_wrist"]]

    # 5) arm length
    for a, b in [(l_sh, l_el), (l_el, l_wr), (r_sh, r_el), (r_el, r_wr)]:
        v_new = scale_along_bone(verts, a, b, params["arm_scale"])
        j_new = scale_along_bone(joints, a, b, params["arm_scale"])
        mask_v = soft_segment_mask(verts, a, b, r_limb)
        mask_j = soft_segment_mask(joints, a, b, r_limb)
        verts = blend_points(verts, v_new, mask_v)
        joints = blend_points(joints, j_new, mask_j)

    # refresh leg joints
    l_hip = joints[joint_idx["left_hip"]]
    r_hip = joints[joint_idx["right_hip"]]
    l_kn = joints[joint_idx["left_knee"]]
    r_kn = joints[joint_idx["right_knee"]]
    l_an = joints[joint_idx["left_ankle"]]
    r_an = joints[joint_idx["right_ankle"]]

    # 6) thigh
    for a, b in [(l_hip, l_kn), (r_hip, r_kn)]:
        v_new = scale_along_bone(verts, a, b, params["thigh_scale"])
        j_new = scale_along_bone(joints, a, b, params["thigh_scale"])
        mask_v = soft_segment_mask(verts, a, b, r_limb)
        mask_j = soft_segment_mask(joints, a, b, r_limb)
        verts = blend_points(verts, v_new, mask_v)
        joints = blend_points(joints, j_new, mask_j)

    # refresh shank joints
    l_kn = joints[joint_idx["left_knee"]]
    r_kn = joints[joint_idx["right_knee"]]
    l_an = joints[joint_idx["left_ankle"]]
    r_an = joints[joint_idx["right_ankle"]]

    # 7) shank
    for a, b in [(l_kn, l_an), (r_kn, r_an)]:
        v_new = scale_along_bone(verts, a, b, params["shank_scale"])
        j_new = scale_along_bone(joints, a, b, params["shank_scale"])
        mask_v = soft_segment_mask(verts, a, b, r_limb)
        mask_j = soft_segment_mask(joints, a, b, r_limb)
        verts = blend_points(verts, v_new, mask_v)
        joints = blend_points(joints, j_new, mask_j)

    # 8) head scale (visual)
    neck = joints[joint_idx["neck"]]
    current_h = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
    y_th = neck[1] + 0.02 * current_h
    head_mask_v = (verts[:, 1] > y_th).astype(np.float32)
    head_mask_j = (joints[:, 1] > y_th).astype(np.float32)
    if head_mask_v.sum() > 0:
        head_center = verts[head_mask_v > 0.5].mean(axis=0)
    else:
        head_center = neck.copy()
    v_new = scale_isotropic(verts, head_center, params["head_scale"])
    j_new = scale_isotropic(joints, head_center, params["head_scale"])
    verts = blend_points(verts, v_new, head_mask_v * 0.85)
    joints = blend_points(joints, j_new, head_mask_j * 0.85)

    return verts, joints


# -----------------------------
# Optimization
# -----------------------------
PARAM_NAMES = [
    "global_scale",
    "shoulder_scale",
    "torso_scale",
    "arm_scale",
    "thigh_scale",
    "shank_scale",
    "pelvis_scale",
    "head_scale",
]


def vec_to_params(x: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(PARAM_NAMES, x.tolist())}


def get_initial_params(group: str, target_h: float, source_h: float, target: Dict[str, float]) -> np.ndarray:
    g = target_h / max(source_h, 1e-8)
    if group == "core":
        return np.array([
            g,
            target["shoulder_scale_prior"],
            0.96,
            0.92,
            0.88,
            0.88,
            0.92,
            target["head_scale_prior"],
        ], dtype=np.float64)
    return np.array([
        g,
        target["shoulder_scale_prior"],
        0.98,
        0.95,
        0.93,
        0.93,
        0.95,
        target["head_scale_prior"],
    ], dtype=np.float64)


def get_bounds(group: str):
    if group == "core":
        return [
            (0.55, 0.85),  # global
            (0.75, 1.00),  # shoulder
            (0.80, 1.05),  # torso
            (0.78, 1.00),  # arm
            (0.72, 0.98),  # thigh
            (0.72, 0.98),  # shank
            (0.80, 1.00),  # pelvis
            (1.00, 1.22),  # head
        ]
    return [
        (0.65, 0.92),
        (0.82, 1.02),
        (0.86, 1.08),
        (0.84, 1.02),
        (0.80, 1.00),
        (0.80, 1.00),
        (0.86, 1.04),
        (1.00, 1.16),
    ]


def objective_factory(
    can_verts: np.ndarray,
    can_joints: np.ndarray,
    target: Dict[str, float],
    group: str,
):
    def obj(x: np.ndarray) -> float:
        p = vec_to_params(x)
        tv, tj = apply_child_warp(can_verts, can_joints, p, JOINT_IDX)
        feats = extract_features_from_joints(tv, tj, JOINT_IDX)

        loss = 0.0

        # height
        loss += 8.0 * ((feats["height_canonical"] - target["height_canonical"]) / target["height_canonical"]) ** 2

        # key ratios
        weights = {
            "shoulder_width_ratio": 2.5,
            "pelvis_width_ratio": 1.5,
            "torso_height_ratio": 3.0,
            "arm_length_ratio": 2.0,
            "thigh_ratio": 3.0,
            "shank_ratio": 3.0,
            "leg_length_ratio": 4.0,
        }
        for k, w in weights.items():
            loss += w * (feats[k] - target[k]) ** 2

        # child-like visual priors
        loss += 1.2 * (p["shoulder_scale"] - target["shoulder_scale_prior"]) ** 2
        loss += 1.0 * (p["head_scale"] - target["head_scale_prior"]) ** 2

        # regularization
        reg = {
            "global_scale": 0.5,
            "shoulder_scale": 0.6,
            "torso_scale": 0.6,
            "arm_scale": 0.6,
            "thigh_scale": 0.8,
            "shank_scale": 0.8,
            "pelvis_scale": 0.5,
            "head_scale": 0.4,
        }
        for k, w in reg.items():
            loss += w * (p[k] - 1.0) ** 2

        # keep thigh/shank balanced
        loss += 0.8 * (p["thigh_scale"] - p["shank_scale"]) ** 2

        return float(loss)

    return obj


# -----------------------------
# Save / report
# -----------------------------
def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(path)


def save_feature_compare(path: Path, before: Dict[str, float], after: Dict[str, float], target: Dict[str, float]) -> None:
    keys = [
        "height_canonical",
        "shoulder_width_ratio",
        "pelvis_width_ratio",
        "torso_height_ratio",
        "arm_length_ratio",
        "thigh_ratio",
        "shank_ratio",
        "leg_length_ratio",
    ]
    rows = []
    for k in keys:
        rows.append({
            "feature": k,
            "before": before.get(k, np.nan),
            "after": after.get(k, np.nan),
            "target": target.get(k, np.nan),
            "abs_err_before": abs(before.get(k, np.nan) - target.get(k, np.nan)),
            "abs_err_after": abs(after.get(k, np.nan) - target.get(k, np.nan)),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-pkl", type=str, required=True)
    parser.add_argument("--group", type=str, default="core", choices=["core", "sub"])
    parser.add_argument("--target-height-cm", type=float, default=120.0)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fit_data = load_pkl(args.adult_pkl)
    gender = normalize_gender(fit_data.get("gender", "neutral"))

    # adult는 일반 SMPL-X
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
    target = load_target_prototype(args.group, args.target_height_cm)

    x0 = get_initial_params(args.group, target["height_canonical"], before_feats["height_canonical"], target)
    bounds = get_bounds(args.group)

    objective = objective_factory(can_verts, can_joints, target, args.group)
    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-9},
    )

    best_params = vec_to_params(result.x)

    can_child_verts, can_child_joints = apply_child_warp(can_verts, can_joints, best_params, JOINT_IDX)
    posed_child_verts, posed_child_joints = apply_child_warp(posed_verts, posed_joints, best_params, JOINT_IDX)

    after_feats = extract_features_from_joints(can_child_verts, can_child_joints, JOINT_IDX)

    # save objs
    stem = Path(args.adult_pkl).stem
    save_obj(outdir / f"{stem}_adult_canonical.obj", can_verts, faces)
    save_obj(outdir / f"{stem}_child_canonical.obj", can_child_verts, faces)
    save_obj(outdir / f"{stem}_adult_posed.obj", posed_verts, faces)
    save_obj(outdir / f"{stem}_child_posed.obj", posed_child_verts, faces)

    # save params
    report = {
        "adult_pkl": str(args.adult_pkl),
        "group": args.group,
        "target_height_cm": args.target_height_cm,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "gender": gender,
        "best_params": best_params,
        "target": target,
        "before_feats": {k: float(v) for k, v in before_feats.items()},
        "after_feats": {k: float(v) for k, v in after_feats.items()},
    }

    with open(outdir / f"{stem}_optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    save_feature_compare(outdir / f"{stem}_feature_compare.csv", before_feats, after_feats, target)

    print("[INFO] done")
    print("[INFO] adult pkl:", args.adult_pkl)
    print("[INFO] outdir:", outdir)
    print("[INFO] optimizer success:", result.success)
    print("[INFO] best params:", best_params)
    print("[INFO] before height:", before_feats["height_canonical"])
    print("[INFO] after height:", after_feats["height_canonical"])
    print("[INFO] target height:", target["height_canonical"])


if __name__ == "__main__":
    main()