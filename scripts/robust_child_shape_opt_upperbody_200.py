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

UPPER_BODY_KEYS = [
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "arm_length_ratio",
]

LOWER_BODY_KEYS = [
    "thigh_ratio",
    "shank_ratio",
    "leg_length_ratio",
]

CANDIDATE_SCORE_WEIGHTS = {
    "height": 1.7,
    "shoulder_width_ratio": 2.1,
    "pelvis_width_ratio": 2.1,
    "torso_height_ratio": 2.5,
    "arm_length_ratio": 2.0,
    "thigh_ratio": 1.2,
    "shank_ratio": 1.0,
    "leg_length_ratio": 1.4,
}

# ── Retrieval weights for the analysis-driven child bank lookup ───────────────
# Weighted standardised L2 over proportion features only.
# Height is deliberately excluded (set to 0.0) so retrieval is
# proportion-driven, not height-driven.
# Optional head/neck features are included at a moderate weight when present.
RETRIEVAL_WEIGHTS = {
    "shoulder_width_ratio":        1.3,
    "pelvis_width_ratio":          1.3,
    "torso_height_ratio":          1.4,
    "arm_length_ratio":            1.2,
    "leg_length_ratio":            1.5,
    "thigh_ratio":                 1.0,
    "shank_ratio":                 1.0,
    # optional head/neck features — used when present in both query and bank
    "head_height_ratio":           1.2,
    "head_width_ratio":            1.0,
    "neck_length_ratio":           0.8,
    "head_width_to_shoulder_ratio": 0.9,
    "head_height_to_torso_ratio":  1.0,
    # height included at near-zero weight; helps break ties without dominating
    "height_canonical":            0.0,   # set > 0 to enable weak height signal
}

STAGE1_WEIGHTS = {
    "height_canonical": 8.0,
    "shoulder_width_ratio": 2.4,
    "pelvis_width_ratio": 6.8,
    "torso_height_ratio": 3.0,
    "arm_length_ratio": 2.8,
    "thigh_ratio": 4.2,
    "shank_ratio": 3.8,
    "leg_length_ratio": 5.4,
}

STAGE2_WEIGHTS = {
    "height_canonical": 6.5,
    "shoulder_width_ratio": 8.5,
    "pelvis_width_ratio": 8.8,
    "torso_height_ratio": 9.6,
    "arm_length_ratio": 8.6,
    "thigh_ratio": 2.6,
    "shank_ratio": 2.4,
    "leg_length_ratio": 3.0,
}

STAGE3_WEIGHTS = {
    "height_canonical": 5.5,
    "shoulder_width_ratio": 12.0,
    "pelvis_width_ratio": 12.0,
    "torso_height_ratio": 13.5,
    "arm_length_ratio": 12.0,
    "thigh_ratio": 1.4,
    "shank_ratio": 1.4,
    "leg_length_ratio": 1.8,
}

MOVEMENT_THRESH = {
    "height_canonical": 0.03,
    "shoulder_width_ratio": 0.005,
    "pelvis_width_ratio": 0.004,
    "torso_height_ratio": 0.004,
    "arm_length_ratio": 0.006,
    "leg_length_ratio": 0.006,
}

MOVEMENT_PENALTY = {
    "height_canonical": 2.5,
    "shoulder_width_ratio": 2.6,
    "pelvis_width_ratio": 2.6,
    "torso_height_ratio": 3.0,
    "arm_length_ratio": 2.8,
    "leg_length_ratio": 1.8,
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


# ═══════════════════════════════════════════════════════════════════════════════
# A.  HEIGHT-CONDITIONED BETA10 PRIOR helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_beta10_bin_csv(path: str | Path) -> pd.DataFrame:
    """
    Load beta10_by_height_bin.csv produced by step2_analyze_beta10.py.

    Expected columns (minimum): height_bin, beta10_mean, n
    The 'ALL' summary row is excluded from bin lookup.
    """
    df = pd.read_csv(path)
    # drop the overall-summary row so we only match real height bins
    df = df[df["height_bin"] != "ALL"].copy()
    if "beta10_mean" not in df.columns:
        raise KeyError(f"beta10_by_height_bin CSV missing 'beta10_mean' column: {path}")
    return df


def lookup_beta10_prior(
    target_height_cm: float,
    beta10_bin_df: pd.DataFrame,
    fallback: float = 1.0,
) -> Tuple[float, str]:
    """
    Given a target child height (cm) and the bin statistics table, return
    (beta10_prior_target, matched_bin_label).

    Bin labels follow the step2 convention: '<100', '100-110', ..., '140+'.
    We parse the bin range numerically and find the containing bin.
    If no bin matches (e.g. height out of range), return the closest bin by
    midpoint distance.
    Falls back to `fallback` value if the table is empty or unparseable.
    """
    if beta10_bin_df is None or len(beta10_bin_df) == 0:
        return fallback, "fallback"

    def _parse_bin(label: str) -> Tuple[float, float]:
        """Return (lo, hi) for a bin label like '100-110', '<100', '140+'."""
        label = str(label).strip()
        if label.startswith("<"):
            return 0.0, float(label[1:])
        if label.endswith("+"):
            return float(label[:-1]), float("inf")
        parts = label.split("-")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        return float("nan"), float("nan")

    best_row = None
    best_dist = float("inf")
    for _, row in beta10_bin_df.iterrows():
        lo, hi = _parse_bin(row["height_bin"])
        if lo <= target_height_cm < hi:
            return float(row["beta10_mean"]), str(row["height_bin"])
        # distance to bin midpoint for fallback
        mid = (lo + min(hi, lo + 20.0)) / 2.0
        d = abs(target_height_cm - mid)
        if d < best_dist:
            best_dist = d
            best_row = row

    # no exact match — use nearest bin
    if best_row is not None:
        return float(best_row["beta10_mean"]), str(best_row["height_bin"]) + "_nearest"
    return fallback, "fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# B.  RETRIEVAL-BASED CHILD INITIALIZATION helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Proportion features used in retrieval (height excluded by default via weight=0)
_RETRIEVAL_FEATURES = [k for k, w in RETRIEVAL_WEIGHTS.items() if w > 0.0]


def load_child_bank_csv(path: str | Path) -> pd.DataFrame:
    """
    Load child_gt_beta_features.csv produced by step1_extract_child_gt_features.py.

    Expected columns (minimum): sample_id, beta_0..beta_10, height_canonical,
    and the proportion feature columns in FEATURE_KEYS.
    """
    df = pd.read_csv(path)
    required = ["sample_id", "beta_10", "height_canonical"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"child_bank CSV missing columns {missing}: {path}")
    # add height_cm convenience column if absent
    if "height_cm" not in df.columns:
        df["height_cm"] = df["height_canonical"] * 100.0
    return df


def _compute_retrieval_sigmas(bank_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute per-feature standard deviation from the child bank (sigma_i).
    Used as the normaliser in the weighted standardised L2 retrieval metric.
    A floor of 1e-6 prevents division-by-zero for near-constant features.
    """
    sigmas: Dict[str, float] = {}
    for feat in RETRIEVAL_WEIGHTS:
        if feat in bank_df.columns:
            s = float(bank_df[feat].std())
            sigmas[feat] = max(s, 1e-6)
    return sigmas


def retrieve_child_candidates(
    adult_feats: Dict[str, float],
    bank_df: pd.DataFrame,
    sigmas: Dict[str, float],
    topk: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve top-k child bank samples by weighted standardised L2.

    d(a, c) = sum_i  w_i * ((f_i(a) - f_i(c)) / sigma_i)^2

    Features used: those in RETRIEVAL_WEIGHTS with w_i > 0 that exist in
    both adult_feats and bank_df.

    Returns a dict with:
      retrieved_ids          : list[str]  (top-k sample_id values)
      retrieved_distances    : list[float]
      retrieved_heights_cm   : list[float]
      mean_beta              : np.ndarray  (mean of top-k beta_0..beta_10)
      mean_target_feats      : dict        (mean of top-k proportion features)
      mean_target_height_cm  : float
      n_features_used        : int
      features_used          : list[str]
      mode                   : "retrieval"
    """
    # decide which features to use
    active_feats = [
        f for f in _RETRIEVAL_FEATURES
        if f in adult_feats and f in bank_df.columns and f in sigmas
    ]

    if not active_feats:
        raise ValueError(
            "No overlapping retrieval features between adult query and child bank. "
            "Check that the bank CSV contains proportion columns."
        )

    # build (N, F) matrix from bank
    bank_mat = bank_df[active_feats].values.astype(np.float64)  # (N, F)
    query_vec = np.array([adult_feats[f] for f in active_feats], dtype=np.float64)  # (F,)
    weights_vec = np.array([RETRIEVAL_WEIGHTS[f] for f in active_feats], dtype=np.float64)
    sigma_vec   = np.array([sigmas[f] for f in active_feats], dtype=np.float64)

    # weighted standardised L2
    diff = (bank_mat - query_vec[None, :]) / sigma_vec[None, :]    # (N, F)
    dists = np.sum(weights_vec[None, :] * diff ** 2, axis=1)       # (N,)

    # guard against NaN rows in bank (missing optional features)
    dists = np.where(np.isnan(dists), np.inf, dists)

    topk_actual = min(topk, len(bank_df))
    top_idx = np.argsort(dists)[:topk_actual]
    top_rows = bank_df.iloc[top_idx]
    top_dists = dists[top_idx]

    # mean beta from top-k (use beta_0..beta_10 if available)
    beta_cols = [f"beta_{i}" for i in range(11) if f"beta_{i}" in bank_df.columns]
    mean_beta = top_rows[beta_cols].mean().values.astype(np.float32)

    # mean target feature vector (all FEATURE_KEYS that exist in bank)
    mean_target_feats: Dict[str, float] = {}
    for k in FEATURE_KEYS:
        if k in top_rows.columns:
            mean_target_feats[k] = float(top_rows[k].mean())

    mean_height_cm = float(top_rows["height_cm"].mean()) if "height_cm" in top_rows.columns else float("nan")

    return {
        "retrieved_ids":         top_rows["sample_id"].tolist(),
        "retrieved_distances":   [float(d) for d in top_dists],
        "retrieved_heights_cm":  top_rows["height_cm"].tolist() if "height_cm" in top_rows.columns else [],
        "mean_beta":             mean_beta,
        "mean_target_feats":     mean_target_feats,
        "mean_target_height_cm": mean_height_cm,
        "n_features_used":       len(active_feats),
        "features_used":         active_feats,
        "mode":                  "retrieval",
    }


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
    beta10_prior_target: float = 1.0,   # ← A: height-conditioned prior; default=1.0 (old behaviour)
):
    if stage == 1:
        weights = STAGE1_WEIGHTS
    elif stage == 2:
        weights = STAGE2_WEIGHTS
    else:
        weights = STAGE3_WEIGHTS

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

        # upper-body stagnation penalty: if target differs from adult but current shape barely moved
        for k in UPPER_BODY_KEYS:
            target_gap = abs(before_feats[k] - target[k])
            desired_move = max(0.0025, 0.45 * target_gap) if stage == 1 else max(0.0035, 0.60 * target_gap)
            if stage == 3:
                desired_move = max(0.0045, 0.75 * target_gap)
            move = abs(feats[k] - before_feats[k])
            if move < desired_move:
                coeff = 0.8 if stage == 1 else 1.6
                if stage == 3:
                    coeff = 2.2
                loss += coeff * (desired_move - move) ** 2

        # stronger push for sub / close targets
        adult_h = before_feats["height_canonical"]
        scale_ratio = target["height_canonical"] / max(adult_h, 1e-8)
        if target["group"] == "sub" and scale_ratio > 0.75:
            loss += 1.0 * max(0.0, 0.78 - abs(feats["height_canonical"] / max(adult_h, 1e-8))) ** 2

        # regularization
        loss += REG["adult_prior"] * float(np.sum((beta_vec[:beta_prior_dim] - adult_beta_prior[:beta_prior_dim]) ** 2))
        loss += REG["beta_l2"] * float(np.sum(beta_vec ** 2))

        if beta_dim >= 11:
            # Regularise beta[10] (kid axis) toward the height-conditioned prior.
            # When beta10_prior_target=1.0 (default) this is identical to the
            # original fixed prior.  When a bin-based prior is supplied from
            # --beta10-bin-csv, the target is replaced per-sample.
            loss += REG["kid_axis"] * float((beta_vec[-1] - beta10_prior_target) ** 2)

        loss += REG["scale"] * (global_scale - 1.0) ** 2

        # stronger upper-body focus after stage 1
        if stage >= 2:
            for k in ["arm_length_ratio", "shoulder_width_ratio", "pelvis_width_ratio", "torso_height_ratio"]:
                loss += 2.0 * (feats[k] - target[k]) ** 2

        # final refine: suppress lower-body overfitting while rescuing torso/shoulder/pelvis
        if stage == 3:
            upper_err = sum((feats[k] - target[k]) ** 2 for k in UPPER_BODY_KEYS)
            lower_err = sum((feats[k] - target[k]) ** 2 for k in LOWER_BODY_KEYS)
            loss += 1.25 * upper_err + 0.15 * lower_err

        return float(loss)

    return obj


def evaluate_solution(
    kid_model,
    fit_data: Dict[str, Any],
    x: np.ndarray,
    target: Dict[str, Any],
    before_feats: Dict[str, float],
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    beta_vec = x[:-1].astype(np.float32)
    global_scale = float(np.exp(float(x[-1])))
    verts, joints, _ = build_output_from_model(kid_model, beta_vec, fit_data, canonical=True, global_scale=global_scale)
    feats = extract_features_from_joints(verts, joints, JOINT_IDX)

    height_err = abs(feats["height_canonical"] - target["height_canonical"]) / max(target["height_canonical"], 1e-8)
    upper_total = float(sum(abs(feats[k] - target[k]) for k in UPPER_BODY_KEYS))
    lower_total = float(sum(abs(feats[k] - target[k]) for k in LOWER_BODY_KEYS))
    upper_move = float(sum(abs(feats[k] - before_feats[k]) for k in UPPER_BODY_KEYS))
    lower_move = float(sum(abs(feats[k] - before_feats[k]) for k in LOWER_BODY_KEYS))
    pelvis_move = abs(feats["pelvis_width_ratio"] - before_feats["pelvis_width_ratio"])
    torso_move = abs(feats["torso_height_ratio"] - before_feats["torso_height_ratio"])

    score = (
        1.25 * height_err
        + 3.2 * upper_total
        + 0.90 * lower_total
        - 0.70 * upper_move
        - 0.15 * lower_move
        - 0.35 * pelvis_move
        - 0.45 * torso_move
    )

    metrics = {
        "height_rel_error": float(height_err),
        "upper_body_total_error": float(upper_total),
        "lower_body_total_error": float(lower_total),
        "upper_body_total_movement": float(upper_move),
        "lower_body_total_movement": float(lower_move),
    }
    return feats, float(score), metrics


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
    beta10_prior_target: float = 1.0,   # ← A: threaded from process_one
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
            beta10_prior_target=beta10_prior_target,
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
            beta10_prior_target=beta10_prior_target,
        )
        res2 = minimize(
            stage2_obj,
            x0=res1.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 180, "ftol": 1e-10},
        )

        stage3_obj = objective_factory(
            kid_model=kid_model,
            fit_data=fit_data,
            adult_beta_prior=raw_betas.astype(np.float32),
            target=target,
            before_feats=before_feats,
            stage=3,
            beta_dim=beta_dim,
            beta10_prior_target=beta10_prior_target,
        )
        res3 = minimize(
            stage3_obj,
            x0=res2.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 160, "ftol": 1e-10},
        )

        feats2, final_score2, metrics2 = evaluate_solution(
            kid_model=kid_model,
            fit_data=fit_data,
            x=res2.x,
            target=target,
            before_feats=before_feats,
        )
        feats3, final_score3, metrics3 = evaluate_solution(
            kid_model=kid_model,
            fit_data=fit_data,
            x=res3.x,
            target=target,
            before_feats=before_feats,
        )

        use_stage3 = final_score3 <= final_score2
        chosen_res = res3 if use_stage3 else res2
        chosen_feats = feats3 if use_stage3 else feats2
        chosen_score = final_score3 if use_stage3 else final_score2
        chosen_metrics = metrics3 if use_stage3 else metrics2

        runs.append({
            "start_idx": si,
            "stage1_success": bool(res1.success),
            "stage2_success": bool(res2.success),
            "stage3_success": bool(res3.success),
            "stage1_fun": float(res1.fun),
            "stage2_fun": float(res2.fun),
            "stage3_fun": float(res3.fun),
            "used_stage3": bool(use_stage3),
            "final_eval_score": float(chosen_score),
            "eval_metrics": chosen_metrics,
            "after_feats": {k: float(v) for k, v in chosen_feats.items()},
        })

        cand = {
            "x": chosen_res.x.copy(),
            "optimizer_success": bool(chosen_res.success),
            "optimizer_message": str(chosen_res.message),
            "optimizer_fun": float(chosen_res.fun),
            "final_eval_score": float(chosen_score),
            "eval_metrics": chosen_metrics,
            "after_feats": chosen_feats,
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

    before_upper_err = float(sum(abs(before_feats[k] - target[k]) for k in UPPER_BODY_KEYS))
    after_upper_err = float(sum(abs(after_feats[k] - target[k]) for k in UPPER_BODY_KEYS))
    before_lower_err = float(sum(abs(before_feats[k] - target[k]) for k in LOWER_BODY_KEYS))
    after_lower_err = float(sum(abs(after_feats[k] - target[k]) for k in LOWER_BODY_KEYS))

    report = {
        "balanced_target": target_to_serializable(target),
        "optimizer_success": bool(best["optimizer_success"]),
        "optimizer_message": best["optimizer_message"],
        "optimizer_fun": float(best["optimizer_fun"]),
        "final_eval_score": float(best["final_eval_score"]),
        "best_global_scale": best_scale,
        "best_beta": best_beta.tolist(),
        "before_feats": {k: float(v) for k, v in before_feats.items()},
        "after_feats": {k: float(v) for k, v in after_feats.items()},
        "before_upper_body_total_error": before_upper_err,
        "after_upper_body_total_error": after_upper_err,
        "upper_body_total_improvement": before_upper_err - after_upper_err,
        "before_lower_body_total_error": before_lower_err,
        "after_lower_body_total_error": after_lower_err,
        "lower_body_total_improvement": before_lower_err - after_lower_err,
        "eval_metrics": best.get("eval_metrics", {}),
        "multistart_runs": runs,
        "tuning": {
            "log_scale_min": float(log_scale_min),
            "log_scale_max": float(log_scale_max),
            "init_scale0_min": float(init_scale0_min),
            "init_scale1_min": float(init_scale1_min),
            "init_scale2_min": float(init_scale2_min),
            "reg_scale": float(REG["scale"]),
        },
        # ── analysis-driven mode diagnostics ─────────────────────────────
        "beta10_prior_target": float(beta10_prior_target),
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
    # ── analysis-driven additions (both optional) ──────────────────────
    child_bank_df: Optional[pd.DataFrame] = None,       # B: retrieval bank
    child_bank_sigmas: Optional[Dict[str, float]] = None,  # precomputed sigmas
    retrieval_topk: int = 5,                            # B: top-k to retrieve
    beta10_bin_df: Optional[pd.DataFrame] = None,       # A: bin prior table
) -> Dict[str, Any]:
    """
    Process one adult .pkl through the child transformation pipeline.

    Four ablation modes (controlled by optional DataFrames):

      child_bank_df=None,  beta10_bin_df=None  →  MODE 0 (original heuristic)
      child_bank_df=None,  beta10_bin_df=set   →  MODE 1 (bin prior only)
      child_bank_df=set,   beta10_bin_df=None  →  MODE 2 (retrieval only)
      child_bank_df=set,   beta10_bin_df=set   →  MODE 3 (full analysis-driven)
    """
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

    # ── B: retrieval-based initialization ──────────────────────────────────
    retrieval_info: Optional[Dict[str, Any]] = None
    if child_bank_df is not None:
        try:
            sigmas = child_bank_sigmas if child_bank_sigmas is not None else _compute_retrieval_sigmas(child_bank_df)
            retrieval_info = retrieve_child_candidates(
                adult_feats=before_feats,
                bank_df=child_bank_df,
                sigmas=sigmas,
                topk=retrieval_topk,
            )
            # Replace beta init with retrieval mean beta (same dim as kid_beta_init)
            mean_beta = retrieval_info["mean_beta"]
            if mean_beta.shape[0] == kid_beta_init.shape[0]:
                kid_beta_init = mean_beta.copy()
            elif mean_beta.shape[0] < kid_beta_init.shape[0]:
                # bank has fewer beta dims — pad with zeros
                padded = np.zeros_like(kid_beta_init)
                padded[:mean_beta.shape[0]] = mean_beta
                kid_beta_init = padded
            # else: bank has more dims than kid model — use original init
            print(f"  [retrieval] top-{retrieval_topk}: "
                  f"ids={retrieval_info['retrieved_ids'][:3]}  "
                  f"dists={[f'{d:.3f}' for d in retrieval_info['retrieved_distances'][:3]]}  "
                  f"mean_h={retrieval_info['mean_target_height_cm']:.1f}cm  "
                  f"n_feat={retrieval_info['n_features_used']}")
        except Exception as exc:
            print(f"  [WARN] retrieval failed for {stem}: {exc} — falling back to heuristic init")
            retrieval_info = None

    # ── Candidate target construction ──────────────────────────────────────
    label_row = None
    row_match = label_df[label_df["adult_sample_path"] == str(Path(adult_pkl))]
    if len(row_match) > 0:
        label_row = row_match.iloc[0]

    if retrieval_info is not None and retrieval_info.get("mean_target_feats"):
        # Build a retrieval-based candidate and prepend it to heuristic candidates
        ret_feats = retrieval_info["mean_target_feats"]
        ret_height_cm = retrieval_info["mean_target_height_cm"]
        retrieval_candidate = {
            "group": "retrieval",
            "height_bin": f"{ret_height_cm:.0f}cm_mean",
            "target_height_cm": ret_height_cm,
            "height_canonical": ret_height_cm / 100.0,
            "candidate_score": -1.0,   # force this to the front of the pool
            "selection": "retrieval_mean",
        }
        for k in FEATURE_KEYS:
            if k in ret_feats:
                retrieval_candidate[k] = ret_feats[k]
            else:
                # fill missing with heuristic placeholder (will be overridden by optimizer)
                retrieval_candidate.setdefault(k, before_feats.get(k, 0.0))

        heuristic_candidates = build_candidate_targets(
            before_feats,
            all_protos,
            label_row=label_row,
            topk=TOPK_CANDIDATES,
            stronger_lam_high=stronger_lam_high,
            stronger_lam_mid=stronger_lam_mid,
        )
        # Prepend retrieval candidate; keep top heuristic candidates as fallback pool
        candidates = [retrieval_candidate] + heuristic_candidates[:TOPK_CANDIDATES]
    else:
        candidates = build_candidate_targets(
            before_feats,
            all_protos,
            label_row=label_row,
            topk=TOPK_CANDIDATES,
            stronger_lam_high=stronger_lam_high,
            stronger_lam_mid=stronger_lam_mid,
        )

    # ── A: height-conditioned beta10 prior ─────────────────────────────────
    # Resolved once per sample from the first/best candidate target height.
    # If bin CSV is absent, falls back to 1.0 (identical to original behaviour).
    if beta10_bin_df is not None and candidates:
        primary_target_h_cm = float(candidates[0].get("target_height_cm",
                                    candidates[0].get("height_canonical", 1.2) * 100.0))
        beta10_prior_val, beta10_bin_matched = lookup_beta10_prior(primary_target_h_cm, beta10_bin_df)
        print(f"  [beta10 prior] target_h={primary_target_h_cm:.1f}cm  "
              f"bin={beta10_bin_matched}  prior={beta10_prior_val:.4f}")
    else:
        beta10_prior_val = 1.0
        beta10_bin_matched = "fixed_1.0"

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
            beta10_prior_target=beta10_prior_val,   # ← A
        )

        after_feats = report["after_feats"]
        total_improve = 0.0
        for k in FEATURE_KEYS:
            total_improve += abs(report["before_feats"][k] - target[k]) - abs(after_feats[k] - target[k])

        report["total_improvement"] = float(total_improve)
        report["candidate_meta"] = target_to_serializable(target)
        candidate_reports.append(report)

        upper_improve = float(report.get("upper_body_total_improvement", 0.0))
        finalist_pool.append({
            "report": report,
            "score": (
                float(report.get("final_eval_score", report["optimizer_fun"]))
                - 0.18 * float(total_improve)
                - 0.85 * upper_improve
            ),
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
        beta10_prior_target=beta10_prior_val,   # ← A
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

    # ── analysis-driven mode diagnostics in final report ───────────────────
    final_report["analysis_mode"] = {
        "retrieval_enabled":   retrieval_info is not None,
        "beta10_bin_enabled":  beta10_bin_df is not None,
        "mode_label": (
            "full"       if (retrieval_info is not None and beta10_bin_df is not None) else
            "retrieval"  if retrieval_info is not None else
            "beta10_bin" if beta10_bin_df is not None else
            "heuristic"
        ),
        "beta10_prior_target": float(beta10_prior_val),
        "beta10_bin_matched":  beta10_bin_matched,
        "retrieval_info": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in (retrieval_info or {}).items()
            if k != "mean_beta"   # mean_beta is large; omit from JSON report
        } if retrieval_info is not None else None,
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
    # ── analysis-driven additions ──────────────────────────────────────
    child_bank_df: Optional[pd.DataFrame] = None,
    child_bank_sigmas: Optional[Dict[str, float]] = None,
    retrieval_topk: int = 5,
    beta10_bin_df: Optional[pd.DataFrame] = None,
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
                child_bank_df=child_bank_df,
                child_bank_sigmas=child_bank_sigmas,
                retrieval_topk=retrieval_topk,
                beta10_bin_df=beta10_bin_df,
            )

            row["status"] = "ok"
            row["optimizer_success"] = rep.get("optimizer_success", "")
            row["optimizer_fun"] = rep.get("optimizer_fun", "")
            row["optimizer_message"] = rep.get("optimizer_message", "")
            row["best_global_scale"] = rep.get("best_global_scale", "")
            row["final_eval_score"] = rep.get("final_eval_score", "")
            row["target_group"] = rep.get("balanced_target", {}).get("group", "")
            row["target_height_cm"] = rep.get("balanced_target", {}).get("target_height_cm", "")
            row["target_height_bin"] = rep.get("balanced_target", {}).get("height_bin", "")
            row["before_upper_body_total_error"] = rep.get("before_upper_body_total_error", "")
            row["after_upper_body_total_error"] = rep.get("after_upper_body_total_error", "")
            row["upper_body_total_improvement"] = rep.get("upper_body_total_improvement", "")
            row["before_lower_body_total_error"] = rep.get("before_lower_body_total_error", "")
            row["after_lower_body_total_error"] = rep.get("after_lower_body_total_error", "")
            row["lower_body_total_improvement"] = rep.get("lower_body_total_improvement", "")

            # analysis-mode diagnostics in batch CSV
            amode = rep.get("analysis_mode", {})
            row["analysis_mode_label"]   = amode.get("mode_label", "heuristic")
            row["beta10_prior_target"]   = amode.get("beta10_prior_target", 1.0)
            row["beta10_bin_matched"]    = amode.get("beta10_bin_matched", "")
            row["retrieval_enabled"]     = amode.get("retrieval_enabled", False)
            rinfo = amode.get("retrieval_info") or {}
            row["retrieval_top1_id"]   = rinfo.get("retrieved_ids", [""])[0] if rinfo.get("retrieved_ids") else ""
            row["retrieval_top1_dist"] = rinfo.get("retrieved_distances", [float("nan")])[0] if rinfo.get("retrieved_distances") else float("nan")
            row["retrieval_mean_h_cm"] = rinfo.get("mean_target_height_cm", float("nan"))

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
        "analysis_mode": {
            "retrieval_enabled": child_bank_df is not None,
            "beta10_bin_enabled": beta10_bin_df is not None,
            "retrieval_topk": retrieval_topk,
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

    # ── analysis-driven additions ──────────────────────────────────────────
    parser.add_argument(
        "--beta10-bin-csv", type=str, default=None,
        help="Path to beta10_by_height_bin.csv (step2 output). "
             "Enables height-conditioned beta[10] prior (MODE A). "
             "If omitted, prior defaults to fixed 1.0 (original behaviour).",
    )
    parser.add_argument(
        "--child-bank-csv", type=str, default=None,
        help="Path to child_gt_beta_features.csv (step1 output). "
             "Enables retrieval-based child initialization (MODE B). "
             "If omitted, heuristic candidate selection is used unchanged.",
    )
    parser.add_argument(
        "--retrieval-topk", type=int, default=5,
        help="Number of nearest child GT samples to retrieve (default: 5). "
             "Only used when --child-bank-csv is provided.",
    )

    args = parser.parse_args()

    label_csv = Path(args.balanced_label_csv)
    kid_template_path = find_kid_template_path(args.kid_template)

    # ── load optional analysis CSV files once ─────────────────────────────
    beta10_bin_df: Optional[pd.DataFrame] = None
    if args.beta10_bin_csv is not None:
        beta10_bin_df = load_beta10_bin_csv(args.beta10_bin_csv)
        print(f"[INFO] beta10 bin prior loaded: {args.beta10_bin_csv}  ({len(beta10_bin_df)} bins)")

    child_bank_df: Optional[pd.DataFrame] = None
    child_bank_sigmas: Optional[Dict[str, float]] = None
    if args.child_bank_csv is not None:
        child_bank_df = load_child_bank_csv(args.child_bank_csv)
        child_bank_sigmas = _compute_retrieval_sigmas(child_bank_df)
        print(f"[INFO] child bank loaded: {args.child_bank_csv}  ({len(child_bank_df)} samples)")
        print(f"[INFO] retrieval topk: {args.retrieval_topk}")

    mode_label = (
        "full"       if (child_bank_df is not None and beta10_bin_df is not None) else
        "retrieval"  if child_bank_df is not None else
        "beta10_bin" if beta10_bin_df is not None else
        "heuristic"
    )
    print(f"[INFO] analysis mode: {mode_label}")

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
            child_bank_df=child_bank_df,
            child_bank_sigmas=child_bank_sigmas,
            retrieval_topk=args.retrieval_topk,
            beta10_bin_df=beta10_bin_df,
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
        child_bank_df=child_bank_df,
        child_bank_sigmas=child_bank_sigmas,
        retrieval_topk=args.retrieval_topk,
        beta10_bin_df=beta10_bin_df,
    )

    amode = rep.get("analysis_mode", {})
    print("[INFO] done")
    print("[INFO] adult:", args.adult_pkl)
    print("[INFO] analysis mode:", amode.get("mode_label", "heuristic"))
    print("[INFO] chosen group:", rep["balanced_target"]["group"])
    print("[INFO] chosen height bin:", rep["balanced_target"]["height_bin"])
    print("[INFO] chosen target height cm:", rep["balanced_target"]["target_height_cm"])
    print("[INFO] beta10 prior target:", amode.get("beta10_prior_target", 1.0),
          f"  (bin: {amode.get('beta10_bin_matched', 'n/a')})")
    if amode.get("retrieval_enabled"):
        rinfo = amode.get("retrieval_info") or {}
        print("[INFO] retrieval top-1:", rinfo.get("retrieved_ids", ["?"])[0],
              "  dist:", round(rinfo.get("retrieved_distances", [float("nan")])[0], 4))
    print("[INFO] optimizer success:", rep["optimizer_success"])
    print("[INFO] optimizer message:", rep["optimizer_message"])
    print("[INFO] tuning:", rep.get("selection_tuning", {}), rep.get("tuning", {}))


if __name__ == "__main__":
    main()