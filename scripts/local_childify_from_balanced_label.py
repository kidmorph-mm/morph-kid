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

# 사람 형태 보존용 안전 제약
SAFE_CONFIG = {
    "global_height_alpha": 0.35,   # target height로 35%만 접근
    "posed_delta_alpha": 0.55,     # posed에는 canonical 변화의 55%만 적용

    "pelvis_ratio_cap_down": 0.18, # 현재 대비 최대 18% 축소
    "pelvis_ratio_cap_up": 0.05,   # 최대 5% 확대
    "shoulder_ratio_cap_down": 0.10,
    "shoulder_ratio_cap_up": 0.08,
    "arm_ratio_cap_down": 0.12,
    "arm_ratio_cap_up": 0.05,
    "leg_ratio_cap_down": 0.14,
    "leg_ratio_cap_up": 0.05,

    "pelvis_mesh_alpha": 0.75,
    "shoulder_mesh_alpha": 0.65,
    "arm_mesh_alpha": 0.60,
    "leg_mesh_alpha": 0.65,
    "smooth_mesh_alpha": 0.20,
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
        "torso_height_ratio": float(row["assigned_target_torso_height_ratio"]),
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


def clamp_ratio_change(current: float, target: float, cap_down: float, cap_up: float) -> float:
    if current <= 1e-8:
        return target
    ratio = target / current
    min_ratio = 1.0 - cap_down
    max_ratio = 1.0 + cap_up
    ratio = max(min_ratio, min(max_ratio, ratio))
    return current * ratio


def build_safe_target_metrics(before_feats: Dict[str, float], target: Dict[str, float], cfg: Dict[str, float]) -> Dict[str, float]:
    safe = {}

    # 전체 키도 부분 접근만
    safe["height_canonical"] = (
        before_feats["height_canonical"]
        + cfg["global_height_alpha"] * (target["height_canonical"] - before_feats["height_canonical"])
    )

    safe["pelvis_width_ratio"] = clamp_ratio_change(
        before_feats["pelvis_width_ratio"],
        target["pelvis_width_ratio"],
        cfg["pelvis_ratio_cap_down"],
        cfg["pelvis_ratio_cap_up"],
    )

    safe["shoulder_width_ratio"] = clamp_ratio_change(
        before_feats["shoulder_width_ratio"],
        target["shoulder_width_ratio"],
        cfg["shoulder_ratio_cap_down"],
        cfg["shoulder_ratio_cap_up"],
    )

    safe["arm_length_ratio"] = clamp_ratio_change(
        before_feats["arm_length_ratio"],
        target["arm_length_ratio"],
        cfg["arm_ratio_cap_down"],
        cfg["arm_ratio_cap_up"],
    )

    safe["leg_length_ratio"] = clamp_ratio_change(
        before_feats["leg_length_ratio"],
        target["leg_length_ratio"],
        cfg["leg_ratio_cap_down"],
        cfg["leg_ratio_cap_up"],
    )

    # torso, thigh, shank는 이번 버전에서는 직접 목표로 안 몰고
    # 기존 비율을 크게 유지
    safe["torso_height_ratio"] = before_feats["torso_height_ratio"]
    safe["thigh_ratio"] = before_feats["thigh_ratio"]
    safe["shank_ratio"] = before_feats["shank_ratio"]

    return safe


def retarget_joints_local_safe(
    joints: np.ndarray,
    verts: np.ndarray,
    safe_target: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    j = joints.copy()
    pelvis = j[JOINT_IDX["pelvis"]].copy()

    current_height = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))
    target_height = safe_target["height_canonical"]

    # 1) partial global scale only
    g = target_height / max(current_height, 1e-8)
    j = pelvis[None, :] + g * (j - pelvis[None, :])

    # refresh
    pelvis = j[JOINT_IDX["pelvis"]]
    l_hip = j[JOINT_IDX["left_hip"]]
    r_hip = j[JOINT_IDX["right_hip"]]
    l_sh = j[JOINT_IDX["left_shoulder"]]
    r_sh = j[JOINT_IDX["right_shoulder"]]
    l_el = j[JOINT_IDX["left_elbow"]]
    r_el = j[JOINT_IDX["right_elbow"]]
    l_wr = j[JOINT_IDX["left_wrist"]]
    r_wr = j[JOINT_IDX["right_wrist"]]
    l_kn = j[JOINT_IDX["left_knee"]]
    r_kn = j[JOINT_IDX["right_knee"]]
    l_an = j[JOINT_IDX["left_ankle"]]
    r_an = j[JOINT_IDX["right_ankle"]]

    h_now = target_height

    # 2) pelvis width
    current_pw = l2(l_hip, r_hip)
    target_pw = safe_target["pelvis_width_ratio"] * h_now
    hip_center = 0.5 * (l_hip + r_hip)
    hip_axis = normalize(r_hip - l_hip)
    half_pw = 0.5 * target_pw
    j[JOINT_IDX["left_hip"]] = hip_center - half_pw * hip_axis
    j[JOINT_IDX["right_hip"]] = hip_center + half_pw * hip_axis

    # 3) shoulder width
    current_sw = l2(l_sh, r_sh)
    target_sw = safe_target["shoulder_width_ratio"] * h_now
    shoulder_center = 0.5 * (l_sh + r_sh)
    shoulder_axis = normalize(r_sh - l_sh)
    half_sw = 0.5 * target_sw
    j[JOINT_IDX["left_shoulder"]] = shoulder_center - half_sw * shoulder_axis
    j[JOINT_IDX["right_shoulder"]] = shoulder_center + half_sw * shoulder_axis

    # 4) arm length (chain만 약하게)
    for side in ["left", "right"]:
        sh_id = JOINT_IDX[f"{side}_shoulder"]
        el_id = JOINT_IDX[f"{side}_elbow"]
        wr_id = JOINT_IDX[f"{side}_wrist"]

        sh = j[sh_id]
        el = j[el_id]
        wr = j[wr_id]

        current_upper = l2(sh, el)
        current_fore = l2(el, wr)
        current_total = max(current_upper + current_fore, 1e-8)
        target_total = safe_target["arm_length_ratio"] * h_now

        axis = normalize(wr - sh)
        target_upper = target_total * (current_upper / current_total)
        target_fore = target_total * (current_fore / current_total)

        j[el_id] = sh + target_upper * axis
        j[wr_id] = j[el_id] + target_fore * axis

    # 5) leg length (knee/ankle chain만 약하게)
    for side in ["left", "right"]:
        hip_id = JOINT_IDX[f"{side}_hip"]
        knee_id = JOINT_IDX[f"{side}_knee"]
        ankle_id = JOINT_IDX[f"{side}_ankle"]

        hip = j[hip_id]
        knee = j[knee_id]
        ankle = j[ankle_id]

        current_thigh = l2(hip, knee)
        current_shank = l2(knee, ankle)
        current_total = max(current_thigh + current_shank, 1e-8)
        target_total = safe_target["leg_length_ratio"] * h_now

        axis = normalize(ankle - hip)
        target_thigh = target_total * (current_thigh / current_total)
        target_shank = target_total * (current_shank / current_total)

        j[knee_id] = hip + target_thigh * axis
        j[ankle_id] = j[knee_id] + target_shank * axis

    info = {
        "global_scale_applied": float(g),
        "safe_target_height": float(target_height),
        "safe_target_pelvis_ratio": float(safe_target["pelvis_width_ratio"]),
        "safe_target_shoulder_ratio": float(safe_target["shoulder_width_ratio"]),
        "safe_target_arm_ratio": float(safe_target["arm_length_ratio"]),
        "safe_target_leg_ratio": float(safe_target["leg_length_ratio"]),
    }
    return j, info


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
        JOINT_IDX["pelvis"],
        JOINT_IDX["left_hip"], JOINT_IDX["right_hip"],
        JOINT_IDX["left_shoulder"], JOINT_IDX["right_shoulder"],
        JOINT_IDX["left_elbow"], JOINT_IDX["right_elbow"],
        JOINT_IDX["left_wrist"], JOINT_IDX["right_wrist"],
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


def deform_mesh_local_safe(
    verts_in: np.ndarray,
    old_joints: np.ndarray,
    new_joints: np.ndarray,
    cfg: Dict[str, float],
) -> np.ndarray:
    verts = verts_in.copy()
    h = float(np.max(verts[:, 1]) - np.min(verts[:, 1]))

    r_pelvis = 0.10 * h
    r_shoulder = 0.09 * h
    r_limb = 0.07 * h

    # pelvis band
    pelvis_target = bone_chain_update(
        verts,
        old_joints[JOINT_IDX["left_hip"]],
        old_joints[JOINT_IDX["right_hip"]],
        new_joints[JOINT_IDX["left_hip"]],
        new_joints[JOINT_IDX["right_hip"]],
        r_pelvis,
    )
    verts = verts + cfg["pelvis_mesh_alpha"] * (pelvis_target - verts)

    # shoulder band
    shoulder_target = bone_chain_update(
        verts,
        old_joints[JOINT_IDX["left_shoulder"]],
        old_joints[JOINT_IDX["right_shoulder"]],
        new_joints[JOINT_IDX["left_shoulder"]],
        new_joints[JOINT_IDX["right_shoulder"]],
        r_shoulder,
    )
    verts = verts + cfg["shoulder_mesh_alpha"] * (shoulder_target - verts)

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
        verts = verts + cfg["arm_mesh_alpha"] * (target - verts)

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
        verts = verts + cfg["leg_mesh_alpha"] * (target - verts)

    # 매우 약한 smoothing
    smooth_target = smooth_joint_displacement(verts, old_joints, new_joints, radius=0.08 * h)
    verts = verts + cfg["smooth_mesh_alpha"] * (smooth_target - verts)

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
    cfg = SAFE_CONFIG.copy()

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
    safe_target = build_safe_target_metrics(before_feats, target, cfg)

    new_can_joints, info = retarget_joints_local_safe(can_joints, can_verts, safe_target)
    new_can_verts = deform_mesh_local_safe(can_verts, can_joints, new_can_joints, cfg)

    joint_delta = new_can_joints - can_joints
    new_posed_joints = posed_joints + cfg["posed_delta_alpha"] * joint_delta
    new_posed_verts = deform_mesh_local_safe(posed_verts, posed_joints, new_posed_joints, cfg)

    after_feats = extract_features_from_joints(new_can_verts, new_can_joints, JOINT_IDX)

    stem = Path(args.adult_pkl).stem
    save_obj(outdir / f"{stem}_adult_canonical.obj", can_verts, faces)
    save_obj(outdir / f"{stem}_child_canonical.obj", new_can_verts, faces)
    save_obj(outdir / f"{stem}_adult_posed.obj", posed_verts, faces)
    save_obj(outdir / f"{stem}_child_posed.obj", new_posed_verts, faces)

    save_feature_compare(outdir / f"{stem}_feature_compare.csv", before_feats, after_feats, safe_target)

    report = {
        "adult_pkl": args.adult_pkl,
        "gender": gender,
        "balanced_target": target,
        "safe_target": safe_target,
        "safe_config": cfg,
        "retarget_info": info,
        "before_feats": {k: float(v) for k, v in before_feats.items()},
        "after_feats": {k: float(v) for k, v in after_feats.items()},
    }

    with open(outdir / f"{stem}_local_childify_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print("[INFO] adult:", args.adult_pkl)
    print("[INFO] assigned group:", target["group"])
    print("[INFO] assigned target height cm:", target["target_height_cm"])
    print("[INFO] assigned height bin:", target["source_height_bin"])
    print("[INFO] before height:", before_feats["height_canonical"])
    print("[INFO] safe target height:", safe_target["height_canonical"])
    print("[INFO] after height:", after_feats["height_canonical"])


if __name__ == "__main__":
    main()