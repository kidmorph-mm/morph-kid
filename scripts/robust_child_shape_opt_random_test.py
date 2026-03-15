from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import smplx
import trimesh
from scipy.optimize import minimize

from extract_canonical_features import extract_features_from_joints


MODEL_ROOT = Path("/home/jaeson1012/agora_dataset/models")
BALANCED_LABEL_CSV = Path("/home/jaeson1012/agora_dataset/runs/pseudo_labels_balanced/adult_pseudo_labels_balanced.csv")
FINAL_GT_DIR = Path("/home/jaeson1012/agora_dataset/data/final_child_gt")
DEFAULT_KID_TEMPLATE = Path("/home/jaeson1012/agora_dataset/models/smplx_kid_template.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

CANDIDATE_SCORE_WEIGHTS = {
    "height": 1.8,
    "shoulder_width_ratio": 1.5,
    "pelvis_width_ratio": 1.3,
    "torso_height_ratio": 1.3,
    "arm_length_ratio": 1.1,
    "thigh_ratio": 1.8,
    "shank_ratio": 1.6,
    "leg_length_ratio": 2.2,
}

STAGE1_WEIGHTS = {
    "height_canonical": 8.0,
    "shoulder_width_ratio": 1.0,
    "pelvis_width_ratio": 6.0,
    "torso_height_ratio": 1.0,
    "arm_length_ratio": 1.0,
    "thigh_ratio": 5.0,
    "shank_ratio": 4.5,
    "leg_length_ratio": 6.5,
}

STAGE2_WEIGHTS = {
    "height_canonical": 7.0,
    "shoulder_width_ratio": 4.5,
    "pelvis_width_ratio": 6.0,
    "torso_height_ratio": 2.8,
    "arm_length_ratio": 4.8,
    "thigh_ratio": 4.0,
    "shank_ratio": 3.8,
    "leg_length_ratio": 5.0,
}

MOVEMENT_THRESH = {
    "height_canonical": 0.03,
    "shoulder_width_ratio": 0.005,
    "pelvis_width_ratio": 0.004,
    "arm_length_ratio": 0.006,
    "leg_length_ratio": 0.006,
}

MOVEMENT_PENALTY = {
    "height_canonical": 2.5,
    "shoulder_width_ratio": 2.0,
    "pelvis_width_ratio": 2.0,
    "arm_length_ratio": 2.2,
    "leg_length_ratio": 2.5,
}

REG = {
    "adult_prior": 0.18,
    "beta_l2": 0.05,
    "kid_axis": 0.05,
    "scale": 0.32,
}

# Recommended stronger-child defaults for stagnant sub cases.
# These can be overridden from CLI for ablation tests.
DEFAULT_LOG_SCALE_MIN = -0.40
DEFAULT_LOG_SCALE_MAX = 0.06
DEFAULT_INIT_SCALE0_MIN = 0.72
DEFAULT_INIT_SCALE1_MIN = 0.66
DEFAULT_INIT_SCALE2_MIN = 0.60
DEFAULT_STRONGER_LAM_HIGH = 0.50
DEFAULT_STRONGER_LAM_MID = 0.35

TOPK_CANDIDATES = 3
TOPK_FINALISTS = 3


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
        raise ValueError(f"Expected dim {dim}, got shape={arr.shape}")
    return arr


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
            "improvement": abs(before.get(k, np.nan) - target.get(k, np.nan)) - abs(after.get(k, np.nan) - target.get(k, np.nan)),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def find_kid_template_path(explicit_path: Optional[str]) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"kid template not found: {p}")
        return p

    if DEFAULT_KID_TEMPLATE.exists():
        return DEFAULT_KID_TEMPLATE

    patterns = [
        "*smplx_kid_template*.npy",
        "*kid_template*.npy",
        "*kid*.npy",
        "*SMIL*.npy",
        "*smil*.npy",
        "*template*.npy",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(MODEL_ROOT.rglob(pattern))
    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if not candidates:
        raise FileNotFoundError("Could not find kid template under MODEL_ROOT")
    return candidates[0]


def create_adult_model(gender: str, num_betas: int):
    return smplx.create(
        str(MODEL_ROOT),
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        use_pca=False,
        flat_hand_mean=True,
        ext="npz",
    ).to(DEVICE)


def create_kid_model(gender: str, create_num_betas: int, kid_template_path: Path):
    return smplx.create(
        str(MODEL_ROOT),
        model_type="smplx",
        gender=gender,
        num_betas=create_num_betas,
        use_pca=False,
        flat_hand_mean=True,
        ext="npz",
        age="kid",
        kid_template_path=str(kid_template_path),
    ).to(DEVICE)


def prepare_kid_beta_init(raw_betas: np.ndarray) -> Tuple[np.ndarray, int]:
    raw = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
    raw_dim = raw.shape[0]
    if raw_dim == 10:
        beta_init = np.concatenate([raw, np.zeros((1,), dtype=np.float32)], axis=0)
        return beta_init, 10
    if raw_dim >= 11:
        return raw.copy(), raw_dim - 1
    raise ValueError(f"Unexpected beta dim: {raw_dim}")


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def zeros(shape):
    return torch.zeros(shape, dtype=torch.float32, device=DEVICE)


def build_output_from_model(
    model,
    betas_np: np.ndarray,
    fit_data: Dict[str, Any],
    canonical: bool,
    global_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    betas = to_torch(ensure_batch(betas_np))
    batch_size = betas.shape[0]

    num_expression_coeffs = getattr(model, "num_expression_coeffs", 10)
    num_body_joints = getattr(model, "NUM_BODY_JOINTS", 21)
    num_hand_joints = getattr(model, "NUM_HAND_JOINTS", 15)

    def get_or_zero(key: str, dim: int) -> torch.Tensor:
        if canonical:
            return zeros((batch_size, dim))
        if key not in fit_data:
            return zeros((batch_size, dim))
        return to_torch(ensure_batch(fit_data[key], dim))

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

    verts = out.vertices[0].detach().cpu().numpy()
    joints = out.joints[0].detach().cpu().numpy()

    pelvis = joints[JOINT_IDX["pelvis"]].copy()
    verts = pelvis[None, :] + global_scale * (verts - pelvis[None, :])
    joints = pelvis[None, :] + global_scale * (joints - pelvis[None, :])

    pelvis = joints[JOINT_IDX["pelvis"]].copy()
    verts = verts - pelvis[None, :]
    joints = joints - pelvis[None, :]

    faces = model.faces.astype(np.int32)
    return verts, joints, faces


def load_balanced_target_map(label_csv: Path) -> pd.DataFrame:
    return pd.read_csv(label_csv)


def load_child_prototypes() -> pd.DataFrame:
    core_csv = FINAL_GT_DIR / "gt_child_core_bin_prototypes_5cm.csv"
    sub_csv = FINAL_GT_DIR / "gt_child_sub_bin_prototypes_5cm.csv"
    if not core_csv.exists():
        raise FileNotFoundError(core_csv)
    if not sub_csv.exists():
        raise FileNotFoundError(sub_csv)

    core = pd.read_csv(core_csv).copy()
    core["group"] = "core"
    sub = pd.read_csv(sub_csv).copy()
    sub["group"] = "sub"

    df = pd.concat([core, sub], ignore_index=True)
    df["height_cm_median"] = df["height_cm_median"].astype(float)
    return df


def row_to_target(row: pd.Series) -> Dict[str, Any]:
    return {
        "group": str(row["group"]),
        "height_bin": str(row["height_bin"]),
        "target_height_cm": float(row["height_cm_median"]),
        "height_canonical": float(row["height_cm_median"]) / 100.0,
        "shoulder_width_ratio": float(row["shoulder_width_ratio_median"]),
        "pelvis_width_ratio": float(row["pelvis_width_ratio_median"]),
        "torso_height_ratio": float(row["torso_height_ratio_median"]),
        "arm_length_ratio": float(row["arm_length_ratio_median"]),
        "thigh_ratio": float(row["thigh_ratio_median"]),
        "shank_ratio": float(row["shank_ratio_median"]),
        "leg_length_ratio": float(row["leg_length_ratio_median"]),
    }


def target_to_serializable(target: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in target.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def score_candidate(adult_feats: Dict[str, float], target: Dict[str, Any]) -> float:
    adult_h = adult_feats["height_canonical"]
    target_h = target["height_canonical"]
    scale_ratio = target_h / max(adult_h, 1e-8)

    score = 0.0
    score += CANDIDATE_SCORE_WEIGHTS["height"] * abs(scale_ratio - 0.72)

    if scale_ratio < 0.57:
        score += 4.0 * (0.57 - scale_ratio)
    if scale_ratio > 0.82:
        score += 1.0 * (scale_ratio - 0.82)

    for k in [
        "shoulder_width_ratio",
        "pelvis_width_ratio",
        "torso_height_ratio",
        "arm_length_ratio",
        "thigh_ratio",
        "shank_ratio",
        "leg_length_ratio",
    ]:
        score += CANDIDATE_SCORE_WEIGHTS[k] * abs(adult_feats[k] - target[k])

    # childness bonus
    child_bonus = 0.0
    child_bonus += max(0.0, 0.46 - target["leg_length_ratio"]) * 3.0
    child_bonus += max(0.0, 0.09 - target["pelvis_width_ratio"]) * 2.5
    child_bonus += max(0.0, target["torso_height_ratio"] - 0.25) * 1.5
    child_bonus += max(0.0, 1.35 - target["target_height_cm"] / 100.0) * 0.4

    score -= 0.22 * child_bonus

    # penalize too-close sub targets
    if target["group"] == "sub" and scale_ratio > 0.77:
        score += 0.7 * (scale_ratio - 0.77) / 0.05

    # very large sub bins get mild penalty because they often stagnate
    if target["group"] == "sub" and target["target_height_cm"] >= 130.0:
        score += 0.18

    return float(score)


def find_neighbor_smaller_target(df: pd.DataFrame, target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cur_h = target["target_height_cm"]

    same_group = df[df["group"] == target["group"]].copy()
    smaller_same = same_group[same_group["height_cm_median"] < cur_h].sort_values("height_cm_median", ascending=False)
    if len(smaller_same) > 0:
        return row_to_target(smaller_same.iloc[0])

    other = df[df["height_cm_median"] < cur_h].sort_values("height_cm_median", ascending=False)
    if len(other) > 0:
        return row_to_target(other.iloc[0])

    return None


def blend_targets(base: Dict[str, Any], childier: Optional[Dict[str, Any]], lam: float) -> Dict[str, Any]:
    if childier is None or lam <= 0.0:
        return dict(base)

    out = dict(base)
    for k in FEATURE_KEYS:
        if k == "height_canonical":
            out[k] = (1.0 - lam) * base[k] + lam * childier[k]
        else:
            out[k] = (1.0 - lam) * base[k] + lam * childier[k]
    out["target_height_cm"] = 100.0 * out["height_canonical"]
    out["blended_from"] = {
        "base_group": base["group"],
        "base_bin": base["height_bin"],
        "childier_group": childier["group"],
        "childier_bin": childier["height_bin"],
        "lambda": lam,
    }
    return out


def build_candidate_targets(
    adult_feats: Dict[str, float],
    all_protos: pd.DataFrame,
    label_row: Optional[pd.Series],
    topk: int = TOPK_CANDIDATES,
    stronger_lam_high: float = DEFAULT_STRONGER_LAM_HIGH,
    stronger_lam_mid: float = DEFAULT_STRONGER_LAM_MID,
) -> List[Dict[str, Any]]:
    scored = []
    for _, row in all_protos.iterrows():
        t = row_to_target(row)
        s = score_candidate(adult_feats, t)
        scored.append((s, t))

    scored.sort(key=lambda x: x[0])
    candidates = [dict(t, candidate_score=float(s), selection="global_topk") for s, t in scored[:topk]]

    # also include balanced assigned target if available
    if label_row is not None:
        assigned = {
            "group": str(label_row["assigned_group"]),
            "height_bin": str(label_row["assigned_height_bin"]),
            "target_height_cm": float(label_row["assigned_target_height_cm"]),
            "height_canonical": float(label_row["assigned_target_height_cm"]) / 100.0,
            "shoulder_width_ratio": float(label_row["assigned_target_shoulder_width_ratio"]),
            "pelvis_width_ratio": float(label_row["assigned_target_pelvis_width_ratio"]),
            "torso_height_ratio": float(label_row["assigned_target_torso_height_ratio"]),
            "arm_length_ratio": float(label_row["assigned_target_arm_length_ratio"]),
            "thigh_ratio": float(label_row["assigned_target_thigh_ratio"]),
            "shank_ratio": float(label_row["assigned_target_shank_ratio"]),
            "leg_length_ratio": float(label_row["assigned_target_leg_length_ratio"]),
            "candidate_score": float(score_candidate(adult_feats, {
                "group": str(label_row["assigned_group"]),
                "height_bin": str(label_row["assigned_height_bin"]),
                "target_height_cm": float(label_row["assigned_target_height_cm"]),
                "height_canonical": float(label_row["assigned_target_height_cm"]) / 100.0,
                "shoulder_width_ratio": float(label_row["assigned_target_shoulder_width_ratio"]),
                "pelvis_width_ratio": float(label_row["assigned_target_pelvis_width_ratio"]),
                "torso_height_ratio": float(label_row["assigned_target_torso_height_ratio"]),
                "arm_length_ratio": float(label_row["assigned_target_arm_length_ratio"]),
                "thigh_ratio": float(label_row["assigned_target_thigh_ratio"]),
                "shank_ratio": float(label_row["assigned_target_shank_ratio"]),
                "leg_length_ratio": float(label_row["assigned_target_leg_length_ratio"]),
            })),
            "selection": "balanced_assigned",
        }
        candidates.append(assigned)

    # de-duplicate by group/bin
    uniq = {}
    for c in candidates:
        key = (c["group"], c["height_bin"])
        if key not in uniq or c["candidate_score"] < uniq[key]["candidate_score"]:
            uniq[key] = c
    candidates = list(uniq.values())

    # build stronger-child variants for close or high sub targets
    expanded = []
    for c in candidates:
        expanded.append(c)
        adult_h = adult_feats["height_canonical"]
        scale_ratio = c["height_canonical"] / max(adult_h, 1e-8)

        need_stronger = (c["group"] == "sub" and (scale_ratio > 0.75 or c["target_height_cm"] >= 130.0))
        if need_stronger:
            childier = find_neighbor_smaller_target(all_protos, c)
            lam = stronger_lam_high if scale_ratio > 0.78 else stronger_lam_mid
            stronger = blend_targets(c, childier, lam)
            stronger["candidate_score"] = c["candidate_score"] - 0.08
            stronger["selection"] = "stronger_child_blend"
            expanded.append(stronger)

    # de-duplicate again
    final = {}
    for c in expanded:
        key = (
            c["group"],
            c["height_bin"],
            round(float(c["height_canonical"]), 4),
            c.get("selection", ""),
        )
        final[key] = c

    # final rerank
    ranked = sorted(final.values(), key=lambda x: x["candidate_score"])
    return ranked[:max(topk, TOPK_FINALISTS)]


def build_inits(
    beta_init: np.ndarray,
    target: Dict[str, Any],
    before_feats: Dict[str, float],
    init_scale0_min: float = DEFAULT_INIT_SCALE0_MIN,
    init_scale1_min: float = DEFAULT_INIT_SCALE1_MIN,
    init_scale2_min: float = DEFAULT_INIT_SCALE2_MIN,
) -> List[np.ndarray]:
    beta_dim = beta_init.shape[0]
    adult_h = before_feats["height_canonical"]
    target_h = target["height_canonical"]
    scale_ratio = target_h / max(adult_h, 1e-8)

    # log scale is optimized; use several stronger starts
    scale0 = np.clip(scale_ratio, init_scale0_min, 1.02)
    scale1 = np.clip(scale_ratio * 0.97, init_scale1_min, 1.00)
    scale2 = np.clip(scale_ratio * 0.93, init_scale2_min, 0.98)

    starts = []

    x0 = np.concatenate([beta_init.astype(np.float64), np.array([math.log(scale0)], dtype=np.float64)], axis=0)
    starts.append(x0)

    x1_beta = beta_init.copy()
    if beta_dim >= 11:
        x1_beta[-1] += 0.8
    x1 = np.concatenate([x1_beta.astype(np.float64), np.array([math.log(scale1)], dtype=np.float64)], axis=0)
    starts.append(x1)

    x2_beta = beta_init.copy()
    if beta_dim >= 11:
        x2_beta[-1] += 1.3
    # mild shape push
    if beta_dim >= 3:
        x2_beta[0] *= 0.75
        x2_beta[1] *= 0.75
        x2_beta[2] *= 0.75
    x2 = np.concatenate([x2_beta.astype(np.float64), np.array([math.log(scale2)], dtype=np.float64)], axis=0)
    starts.append(x2)

    return starts


def objective_factory(
    kid_model,
    fit_data: Dict[str, Any],
    adult_beta_prior: np.ndarray,
    target: Dict[str, Any],
    before_feats: Dict[str, float],
    stage: int,
    beta_dim: int,
):
    weights = STAGE1_WEIGHTS if stage == 1 else STAGE2_WEIGHTS
    beta_prior_dim = min(10, adult_beta_prior.shape[0], beta_dim)

    def obj(x: np.ndarray) -> float:
        beta_vec = x[:-1].astype(np.float32)
        log_scale = float(x[-1])
        global_scale = float(np.exp(log_scale))

        verts, joints, _ = build_output_from_model(
            kid_model,
            beta_vec,
            fit_data,
            canonical=True,
            global_scale=global_scale,
        )
        feats = extract_features_from_joints(verts, joints, JOINT_IDX)

        loss = 0.0
        for k, w in weights.items():
            if k == "height_canonical":
                denom = max(target[k], 1e-8)
                loss += w * ((feats[k] - target[k]) / denom) ** 2
            else:
                loss += w * (feats[k] - target[k]) ** 2

        # movement-aware: if barely changes from adult, penalize
        for k, th in MOVEMENT_THRESH.items():
            move = abs(feats[k] - before_feats[k])
            if move < th:
                loss += MOVEMENT_PENALTY[k] * (th - move) ** 2

        # stronger push for sub / close targets
        adult_h = before_feats["height_canonical"]
        scale_ratio = target["height_canonical"] / max(adult_h, 1e-8)
        if target["group"] == "sub" and scale_ratio > 0.75:
            loss += 1.0 * max(0.0, 0.78 - abs(feats["height_canonical"] / max(adult_h, 1e-8))) ** 2

        # regularization
        loss += REG["adult_prior"] * float(np.sum((beta_vec[:beta_prior_dim] - adult_beta_prior[:beta_prior_dim]) ** 2))
        loss += REG["beta_l2"] * float(np.sum(beta_vec ** 2))

        if beta_dim >= 11:
            # encourage some kid-axis motion, but not extreme
            loss += REG["kid_axis"] * float((beta_vec[-1] - 1.0) ** 2)

        loss += REG["scale"] * (global_scale - 1.0) ** 2

        # stage-2 specific focus on hard features
        if stage == 2:
            for k in ["arm_length_ratio", "shoulder_width_ratio", "torso_height_ratio"]:
                loss += 1.2 * (feats[k] - target[k]) ** 2

        return float(loss)

    return obj


def evaluate_solution(
    kid_model,
    fit_data: Dict[str, Any],
    x: np.ndarray,
    target: Dict[str, Any],
    before_feats: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    beta_vec = x[:-1].astype(np.float32)
    global_scale = float(np.exp(float(x[-1])))
    verts, joints, _ = build_output_from_model(kid_model, beta_vec, fit_data, canonical=True, global_scale=global_scale)
    feats = extract_features_from_joints(verts, joints, JOINT_IDX)

    total = 0.0
    for k in FEATURE_KEYS:
        total += abs(feats[k] - target[k])

    # reward actual movement
    movement = 0.0
    for k in ["height_canonical", "shoulder_width_ratio", "pelvis_width_ratio", "arm_length_ratio", "leg_length_ratio"]:
        movement += abs(feats[k] - before_feats[k])

    score = total - 0.5 * movement
    return feats, float(score)


def optimize_for_candidate(
    adult_model,
    kid_model,
    fit_data: Dict[str, Any],
    raw_betas: np.ndarray,
    kid_beta_init: np.ndarray,
    target: Dict[str, Any],
    no_obj: bool,
    outdir: Path,
    stem: str,
    log_scale_min: float = DEFAULT_LOG_SCALE_MIN,
    log_scale_max: float = DEFAULT_LOG_SCALE_MAX,
    init_scale0_min: float = DEFAULT_INIT_SCALE0_MIN,
    init_scale1_min: float = DEFAULT_INIT_SCALE1_MIN,
    init_scale2_min: float = DEFAULT_INIT_SCALE2_MIN,
) -> Dict[str, Any]:
    adult_can_verts, adult_can_joints, adult_faces = build_output_from_model(
        adult_model, raw_betas, fit_data, canonical=True, global_scale=1.0
    )
    adult_posed_verts, adult_posed_joints, _ = build_output_from_model(
        adult_model, raw_betas, fit_data, canonical=False, global_scale=1.0
    )
    before_feats = extract_features_from_joints(adult_can_verts, adult_can_joints, JOINT_IDX)

    beta_dim = kid_beta_init.shape[0]
    starts = build_inits(
        kid_beta_init,
        target,
        before_feats,
        init_scale0_min=init_scale0_min,
        init_scale1_min=init_scale1_min,
        init_scale2_min=init_scale2_min,
    )
    bounds = [(-4.0, 4.0)] * beta_dim + [(log_scale_min, log_scale_max)]

    best = None
    runs = []

    for si, x0 in enumerate(starts, start=1):
        stage1_obj = objective_factory(
            kid_model=kid_model,
            fit_data=fit_data,
            adult_beta_prior=raw_betas.astype(np.float32),
            target=target,
            before_feats=before_feats,
            stage=1,
            beta_dim=beta_dim,
        )
        res1 = minimize(
            stage1_obj,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 140, "ftol": 1e-10},
        )

        stage2_obj = objective_factory(
            kid_model=kid_model,
            fit_data=fit_data,
            adult_beta_prior=raw_betas.astype(np.float32),
            target=target,
            before_feats=before_feats,
            stage=2,
            beta_dim=beta_dim,
        )
        res2 = minimize(
            stage2_obj,
            x0=res1.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 180, "ftol": 1e-10},
        )

        feats2, final_score = evaluate_solution(
            kid_model=kid_model,
            fit_data=fit_data,
            x=res2.x,
            target=target,
            before_feats=before_feats,
        )

        runs.append({
            "start_idx": si,
            "stage1_success": bool(res1.success),
            "stage2_success": bool(res2.success),
            "stage1_fun": float(res1.fun),
            "stage2_fun": float(res2.fun),
            "final_eval_score": float(final_score),
            "after_feats": {k: float(v) for k, v in feats2.items()},
        })

        cand = {
            "x": res2.x.copy(),
            "optimizer_success": bool(res2.success),
            "optimizer_message": str(res2.message),
            "optimizer_fun": float(res2.fun),
            "final_eval_score": float(final_score),
            "after_feats": feats2,
        }

        if best is None or cand["final_eval_score"] < best["final_eval_score"]:
            best = cand

    assert best is not None

    best_beta = best["x"][:-1].astype(np.float32)
    best_scale = float(np.exp(float(best["x"][-1])))

    child_can_verts, child_can_joints, child_faces = build_output_from_model(
        kid_model, best_beta, fit_data, canonical=True, global_scale=best_scale
    )
    child_posed_verts, child_posed_joints, _ = build_output_from_model(
        kid_model, best_beta, fit_data, canonical=False, global_scale=best_scale
    )
    after_feats = extract_features_from_joints(child_can_verts, child_can_joints, JOINT_IDX)

    if not no_obj:
        save_obj(outdir / f"{stem}_adult_canonical.obj", adult_can_verts, adult_faces)
        save_obj(outdir / f"{stem}_adult_posed.obj", adult_posed_verts, adult_faces)
        save_obj(outdir / f"{stem}_child_canonical.obj", child_can_verts, child_faces)
        save_obj(outdir / f"{stem}_child_posed.obj", child_posed_verts, child_faces)

    save_feature_compare(outdir / f"{stem}_feature_compare.csv", before_feats, after_feats, target)

    report = {
        "balanced_target": target_to_serializable(target),
        "optimizer_success": bool(best["optimizer_success"]),
        "optimizer_message": best["optimizer_message"],
        "optimizer_fun": float(best["optimizer_fun"]),
        "best_global_scale": best_scale,
        "best_beta": best_beta.tolist(),
        "before_feats": {k: float(v) for k, v in before_feats.items()},
        "after_feats": {k: float(v) for k, v in after_feats.items()},
        "multistart_runs": runs,
        "tuning": {
            "log_scale_min": float(log_scale_min),
            "log_scale_max": float(log_scale_max),
            "init_scale0_min": float(init_scale0_min),
            "init_scale1_min": float(init_scale1_min),
            "init_scale2_min": float(init_scale2_min),
            "reg_scale": float(REG["scale"]),
        },
    }

    return report


def process_one(
    adult_pkl: str,
    label_df: pd.DataFrame,
    all_protos: pd.DataFrame,
    kid_template_path: Path,
    outdir: Path,
    no_obj: bool,
    log_scale_min: float = DEFAULT_LOG_SCALE_MIN,
    log_scale_max: float = DEFAULT_LOG_SCALE_MAX,
    init_scale0_min: float = DEFAULT_INIT_SCALE0_MIN,
    init_scale1_min: float = DEFAULT_INIT_SCALE1_MIN,
    init_scale2_min: float = DEFAULT_INIT_SCALE2_MIN,
    stronger_lam_high: float = DEFAULT_STRONGER_LAM_HIGH,
    stronger_lam_mid: float = DEFAULT_STRONGER_LAM_MID,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(adult_pkl).stem

    fit_data = load_pkl(adult_pkl)
    gender = normalize_gender(fit_data.get("gender", "neutral"))

    raw_betas = ensure_batch(fit_data["betas"])[0]
    adult_model_num_betas = int(raw_betas.shape[0])
    kid_beta_init, kid_create_num_betas = prepare_kid_beta_init(raw_betas)

    adult_model = create_adult_model(gender, adult_model_num_betas)
    kid_model = create_kid_model(gender, kid_create_num_betas, kid_template_path)

    adult_can_verts, adult_can_joints, _ = build_output_from_model(
        adult_model, raw_betas, fit_data, canonical=True, global_scale=1.0
    )
    before_feats = extract_features_from_joints(adult_can_verts, adult_can_joints, JOINT_IDX)

    label_row = None
    row_match = label_df[label_df["adult_sample_path"] == str(Path(adult_pkl))]
    if len(row_match) > 0:
        label_row = row_match.iloc[0]

    candidates = build_candidate_targets(
        before_feats,
        all_protos,
        label_row=label_row,
        topk=TOPK_CANDIDATES,
        stronger_lam_high=stronger_lam_high,
        stronger_lam_mid=stronger_lam_mid,
    )

    candidate_reports = []
    finalist_pool = []

    for ci, target in enumerate(candidates, start=1):
        candidate_outdir = outdir / f"candidate_{ci}_{target['group']}_{str(target['height_bin']).replace('-', '_')}"
        candidate_outdir.mkdir(parents=True, exist_ok=True)

        report = optimize_for_candidate(
            adult_model=adult_model,
            kid_model=kid_model,
            fit_data=fit_data,
            raw_betas=raw_betas,
            kid_beta_init=kid_beta_init,
            target=target,
            no_obj=True if len(candidates) > 1 else no_obj,
            outdir=candidate_outdir,
            stem=stem,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            init_scale0_min=init_scale0_min,
            init_scale1_min=init_scale1_min,
            init_scale2_min=init_scale2_min,
        )

        after_feats = report["after_feats"]
        total_improve = 0.0
        for k in FEATURE_KEYS:
            total_improve += abs(report["before_feats"][k] - target[k]) - abs(after_feats[k] - target[k])

        report["total_improvement"] = float(total_improve)
        report["candidate_meta"] = target_to_serializable(target)
        candidate_reports.append(report)

        finalist_pool.append({
            "report": report,
            "score": float(report["optimizer_fun"]) - 0.35 * float(total_improve),
            "target": target,
            "candidate_outdir": candidate_outdir,
        })

    finalist_pool = sorted(finalist_pool, key=lambda x: x["score"])[:TOPK_FINALISTS]
    best_item = finalist_pool[0]

    # rerun best candidate once with user-requested no_obj setting for final outputs
    final_report = optimize_for_candidate(
        adult_model=adult_model,
        kid_model=kid_model,
        fit_data=fit_data,
        raw_betas=raw_betas,
        kid_beta_init=kid_beta_init,
        target=best_item["target"],
        no_obj=no_obj,
        outdir=outdir,
        stem=stem,
        log_scale_min=log_scale_min,
        log_scale_max=log_scale_max,
        init_scale0_min=init_scale0_min,
        init_scale1_min=init_scale1_min,
        init_scale2_min=init_scale2_min,
    )

    final_report["adult_pkl"] = adult_pkl
    final_report["gender"] = gender
    final_report["kid_template_path"] = str(kid_template_path)
    final_report["candidate_pool"] = [r["candidate_meta"] for r in candidate_reports]
    final_report["candidate_reports"] = candidate_reports
    final_report["chosen_candidate"] = target_to_serializable(best_item["target"])
    final_report["selection_tuning"] = {
        "stronger_lam_high": float(stronger_lam_high),
        "stronger_lam_mid": float(stronger_lam_mid),
    }

    with open(outdir / f"{stem}_optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    return final_report


def run_batch(
    label_csv: Path,
    kid_template_path: Path,
    outroot: Path,
    no_obj: bool,
    limit: Optional[int],
    seed: int,
    log_scale_min: float = DEFAULT_LOG_SCALE_MIN,
    log_scale_max: float = DEFAULT_LOG_SCALE_MAX,
    init_scale0_min: float = DEFAULT_INIT_SCALE0_MIN,
    init_scale1_min: float = DEFAULT_INIT_SCALE1_MIN,
    init_scale2_min: float = DEFAULT_INIT_SCALE2_MIN,
    stronger_lam_high: float = DEFAULT_STRONGER_LAM_HIGH,
    stronger_lam_mid: float = DEFAULT_STRONGER_LAM_MID,
):
    label_df = pd.read_csv(label_csv)
    all_protos = load_child_prototypes()

    if "adult_sample_path" not in label_df.columns:
        raise KeyError("adult_sample_path not found in label csv")

    adult_paths = label_df["adult_sample_path"].dropna().astype(str).tolist()
    adult_paths = [p for p in adult_paths if Path(p).exists()]

    rng = random.Random(seed)
    rng.shuffle(adult_paths)

    if limit is not None:
        adult_paths = adult_paths[:limit]

    results = []
    total = len(adult_paths)

    for i, adult_pkl in enumerate(adult_paths, start=1):
        stem = Path(adult_pkl).stem
        case_outdir = outroot / f"{i:05d}_{stem}"
        case_outdir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] ({i}/{total}) {stem}")
        row = {
            "idx": i,
            "adult_sample_path": adult_pkl,
            "stem": stem,
            "status": "fail",
        }

        try:
            rep = process_one(
                adult_pkl=adult_pkl,
                label_df=label_df,
                all_protos=all_protos,
                kid_template_path=kid_template_path,
                outdir=case_outdir,
                no_obj=no_obj,
                log_scale_min=log_scale_min,
                log_scale_max=log_scale_max,
                init_scale0_min=init_scale0_min,
                init_scale1_min=init_scale1_min,
                init_scale2_min=init_scale2_min,
                stronger_lam_high=stronger_lam_high,
                stronger_lam_mid=stronger_lam_mid,
            )

            row["status"] = "ok"
            row["optimizer_success"] = rep.get("optimizer_success", "")
            row["optimizer_fun"] = rep.get("optimizer_fun", "")
            row["optimizer_message"] = rep.get("optimizer_message", "")
            row["best_global_scale"] = rep.get("best_global_scale", "")
            row["target_group"] = rep.get("balanced_target", {}).get("group", "")
            row["target_height_cm"] = rep.get("balanced_target", {}).get("target_height_cm", "")
            row["target_height_bin"] = rep.get("balanced_target", {}).get("height_bin", "")

            before_feats = rep.get("before_feats", {})
            after_feats = rep.get("after_feats", {})
            target_feats = rep.get("balanced_target", {})

            for k in FEATURE_KEYS:
                b = before_feats.get(k, "")
                a = after_feats.get(k, "")
                t = target_feats.get(k, "")
                row[f"before_{k}"] = b
                row[f"after_{k}"] = a
                row[f"target_{k}"] = t
                if b != "" and a != "" and t != "":
                    b = float(b)
                    a = float(a)
                    t = float(t)
                    row[f"abs_err_before_{k}"] = abs(b - t)
                    row[f"abs_err_after_{k}"] = abs(a - t)
                    row[f"improve_{k}"] = abs(b - t) - abs(a - t)

        except Exception as e:
            row["status"] = "fail"
            row["error"] = repr(e)
            row["traceback_tail"] = traceback.format_exc()[-2000:]

        results.append(row)

        if i % 50 == 0 or i == total:
            pd.DataFrame(results).to_csv(outroot / "batch_results_partial.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(outroot / "batch_results.csv", index=False)

    optimizer_success_series = results_df["optimizer_success"] if "optimizer_success" in results_df.columns else pd.Series([], dtype=object)
    summary = {
        "total": int(len(results_df)),
        "ok": int((results_df["status"] == "ok").sum()),
        "fail": int((results_df["status"] == "fail").sum()),
        "optimizer_success_true": int(sum(x is True or x == True or str(x).lower() == "true" for x in optimizer_success_series)),
        "optimizer_success_false": int(sum(x is False or x == False or str(x).lower() == "false" for x in optimizer_success_series)),
        "seed": int(seed),
        "tuning": {
            "log_scale_min": float(log_scale_min),
            "log_scale_max": float(log_scale_max),
            "init_scale0_min": float(init_scale0_min),
            "init_scale1_min": float(init_scale1_min),
            "init_scale2_min": float(init_scale2_min),
            "stronger_lam_high": float(stronger_lam_high),
            "stronger_lam_mid": float(stronger_lam_mid),
            "reg_scale": float(REG["scale"]),
        },
    }

    with open(outroot / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print("[INFO] batch_results.csv saved")
    print("[INFO] batch_summary.json saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-pkl", type=str, default=None)
    parser.add_argument("--balanced-label-csv", type=str, default=str(BALANCED_LABEL_CSV))
    parser.add_argument("--kid-template", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--no-obj", action="store_true")

    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Random sample size in batch mode after shuffling full list")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-outroot", type=str, default=None)

    parser.add_argument("--log-scale-min", type=float, default=DEFAULT_LOG_SCALE_MIN)
    parser.add_argument("--log-scale-max", type=float, default=DEFAULT_LOG_SCALE_MAX)
    parser.add_argument("--init-scale0-min", type=float, default=DEFAULT_INIT_SCALE0_MIN)
    parser.add_argument("--init-scale1-min", type=float, default=DEFAULT_INIT_SCALE1_MIN)
    parser.add_argument("--init-scale2-min", type=float, default=DEFAULT_INIT_SCALE2_MIN)
    parser.add_argument("--stronger-lam-high", type=float, default=DEFAULT_STRONGER_LAM_HIGH)
    parser.add_argument("--stronger-lam-mid", type=float, default=DEFAULT_STRONGER_LAM_MID)

    args = parser.parse_args()

    label_csv = Path(args.balanced_label_csv)
    kid_template_path = find_kid_template_path(args.kid_template)

    if args.batch:
        if args.batch_outroot is None:
            raise ValueError("--batch requires --batch-outroot")
        outroot = Path(args.batch_outroot)
        outroot.mkdir(parents=True, exist_ok=True)
        run_batch(
            label_csv=label_csv,
            kid_template_path=kid_template_path,
            outroot=outroot,
            no_obj=args.no_obj,
            limit=args.limit,
            seed=args.seed,
            log_scale_min=args.log_scale_min,
            log_scale_max=args.log_scale_max,
            init_scale0_min=args.init_scale0_min,
            init_scale1_min=args.init_scale1_min,
            init_scale2_min=args.init_scale2_min,
            stronger_lam_high=args.stronger_lam_high,
            stronger_lam_mid=args.stronger_lam_mid,
        )
        return

    if args.adult_pkl is None or args.outdir is None:
        raise ValueError("single mode requires --adult-pkl and --outdir")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(label_csv)
    all_protos = load_child_prototypes()

    rep = process_one(
        adult_pkl=args.adult_pkl,
        label_df=label_df,
        all_protos=all_protos,
        kid_template_path=kid_template_path,
        outdir=outdir,
        no_obj=args.no_obj,
        log_scale_min=args.log_scale_min,
        log_scale_max=args.log_scale_max,
        init_scale0_min=args.init_scale0_min,
        init_scale1_min=args.init_scale1_min,
        init_scale2_min=args.init_scale2_min,
        stronger_lam_high=args.stronger_lam_high,
        stronger_lam_mid=args.stronger_lam_mid,
    )

    print("[INFO] done")
    print("[INFO] adult:", args.adult_pkl)
    print("[INFO] chosen group:", rep["balanced_target"]["group"])
    print("[INFO] chosen height bin:", rep["balanced_target"]["height_bin"])
    print("[INFO] chosen target height cm:", rep["balanced_target"]["target_height_cm"])
    print("[INFO] optimizer success:", rep["optimizer_success"])
    print("[INFO] optimizer message:", rep["optimizer_message"])
    print("[INFO] tuning:", rep.get("selection_tuning", {}), rep.get("tuning", {}))


if __name__ == "__main__":
    main()