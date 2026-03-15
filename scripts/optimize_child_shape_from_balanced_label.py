from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import smplx
import trimesh
from scipy.optimize import minimize

from extract_canonical_features import extract_features_from_joints


MODEL_ROOT = Path("/home/jaeson1012/agora_dataset/models")
BALANCED_LABEL_CSV = Path("/home/jaeson1012/agora_dataset/runs/pseudo_labels_balanced/adult_pseudo_labels_balanced.csv")
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


def find_kid_template_path(explicit_path: Optional[str]) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"kid template not found: {p}")
        return p

    if DEFAULT_KID_TEMPLATE.exists():
        return DEFAULT_KID_TEMPLATE

    candidates = []
    patterns = [
        "*smplx_kid_template*.npy",
        "*kid_template*.npy",
        "*kid*.npy",
        "*SMIL*.npy",
        "*smil*.npy",
        "*template*.npy",
    ]
    for pattern in patterns:
        candidates.extend(MODEL_ROOT.rglob(pattern))

    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if not candidates:
        raise FileNotFoundError("Could not find kid template .npy under MODEL_ROOT")
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
    """
    smplx kid model practical rule in this environment:
    - model create num_betas = base dims
    - forward betas often need +1 dim (kid axis)
    """
    raw = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
    raw_dim = raw.shape[0]

    if raw_dim == 10:
        beta_init = np.concatenate([raw, np.zeros((1,), dtype=np.float32)], axis=0)
        create_num_betas = 10
        return beta_init, create_num_betas

    if raw_dim >= 11:
        create_num_betas = raw_dim - 1
        return raw.copy(), create_num_betas

    raise ValueError(f"Unexpected beta dimension: {raw_dim}")


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

    # global scale around pelvis
    pelvis = joints[JOINT_IDX["pelvis"]].copy()
    verts = pelvis[None, :] + global_scale * (verts - pelvis[None, :])
    joints = pelvis[None, :] + global_scale * (joints - pelvis[None, :])

    # center by pelvis
    pelvis = joints[JOINT_IDX["pelvis"]].copy()
    verts = verts - pelvis[None, :]
    joints = joints - pelvis[None, :]

    faces = model.faces.astype(np.int32)
    return verts, joints, faces


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


def objective_factory(
    kid_model,
    adult_beta_prior: np.ndarray,
    fit_data: Dict[str, Any],
    target: Dict[str, float],
    beta_dim: int,
):
    beta_prior_dim = min(10, adult_beta_prior.shape[0], beta_dim)

    feature_weights = {
        "height_canonical": 5.0,
        "shoulder_width_ratio": 4.5,
        "pelvis_width_ratio": 5.0,
        "torso_height_ratio": 3.0,
        "arm_length_ratio": 3.5,
        "thigh_ratio": 3.5,
        "shank_ratio": 3.5,
        "leg_length_ratio": 4.5,
    }

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
        for k, w in feature_weights.items():
            if k == "height_canonical":
                denom = max(target[k], 1e-8)
                loss += w * ((feats[k] - target[k]) / denom) ** 2
            else:
                loss += w * (feats[k] - target[k]) ** 2

        # preserve some adult identity in low-order shape components
        loss += 0.30 * float(np.sum((beta_vec[:beta_prior_dim] - adult_beta_prior[:beta_prior_dim]) ** 2))

        # keep all betas moderate
        loss += 0.06 * float(np.sum(beta_vec ** 2))

        # keep kid-axis / extra axis moderate if present
        if beta_dim >= 11:
            loss += 0.08 * float(beta_vec[-1] ** 2)

        # mild scale prior
        loss += 0.8 * (global_scale - 1.0) ** 2

        return float(loss)

    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-pkl", type=str, required=True)
    parser.add_argument("--balanced-label-csv", type=str, default=str(BALANCED_LABEL_CSV))
    parser.add_argument("--kid-template", type=str, default=None)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--no-obj", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fit_data = load_pkl(args.adult_pkl)
    gender = normalize_gender(fit_data.get("gender", "neutral"))
    target = load_balanced_target(args.adult_pkl, Path(args.balanced_label_csv))
    kid_template_path = find_kid_template_path(args.kid_template)

    raw_betas = ensure_batch(fit_data["betas"])[0]
    adult_model_num_betas = int(raw_betas.shape[0])
    kid_beta_init, kid_create_num_betas = prepare_kid_beta_init(raw_betas)

    adult_model = create_adult_model(gender, adult_model_num_betas)
    kid_model = create_kid_model(gender, kid_create_num_betas, kid_template_path)

    # reference adult canonical / posed
    adult_can_verts, adult_can_joints, adult_faces = build_output_from_model(
        adult_model, raw_betas, fit_data, canonical=True, global_scale=1.0
    )
    adult_posed_verts, adult_posed_joints, _ = build_output_from_model(
        adult_model, raw_betas, fit_data, canonical=False, global_scale=1.0
    )

    before_feats = extract_features_from_joints(adult_can_verts, adult_can_joints, JOINT_IDX)

    beta_dim = kid_beta_init.shape[0]
    x0 = np.concatenate([kid_beta_init.astype(np.float64), np.array([0.0], dtype=np.float64)], axis=0)

    bounds = [(-3.0, 3.0)] * beta_dim + [(-0.18, 0.10)]

    objective = objective_factory(
        kid_model=kid_model,
        adult_beta_prior=raw_betas.astype(np.float32),
        fit_data=fit_data,
        target=target,
        beta_dim=beta_dim,
    )

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-10},
    )

    best_beta = result.x[:-1].astype(np.float32)
    best_log_scale = float(result.x[-1])
    best_scale = float(np.exp(best_log_scale))

    child_can_verts, child_can_joints, child_faces = build_output_from_model(
        kid_model, best_beta, fit_data, canonical=True, global_scale=best_scale
    )
    child_posed_verts, child_posed_joints, _ = build_output_from_model(
        kid_model, best_beta, fit_data, canonical=False, global_scale=best_scale
    )

    after_feats = extract_features_from_joints(child_can_verts, child_can_joints, JOINT_IDX)

    stem = Path(args.adult_pkl).stem
    if not args.no_obj:    
        save_obj(outdir / f"{stem}_adult_canonical.obj", adult_can_verts, adult_faces)
        save_obj(outdir / f"{stem}_adult_posed.obj", adult_posed_verts, adult_faces)
        save_obj(outdir / f"{stem}_child_canonical.obj", child_can_verts, child_faces)
        save_obj(outdir / f"{stem}_child_posed.obj", child_posed_verts, child_faces)

    save_feature_compare(outdir / f"{stem}_feature_compare.csv", before_feats, after_feats, target)

    report = {
        "adult_pkl": args.adult_pkl,
        "gender": gender,
        "kid_template_path": str(kid_template_path),
        "balanced_target": target,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "best_global_scale": best_scale,
        "adult_beta_dim": int(raw_betas.shape[0]),
        "kid_beta_dim": int(best_beta.shape[0]),
        "kid_create_num_betas": int(kid_create_num_betas),
        "best_beta": best_beta.tolist(),
        "before_feats": {k: float(v) for k, v in before_feats.items()},
        "after_feats": {k: float(v) for k, v in after_feats.items()},
    }

    with open(outdir / f"{stem}_optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print("[INFO] adult:", args.adult_pkl)
    print("[INFO] target group:", target["group"])
    print("[INFO] target height cm:", target["target_height_cm"])
    print("[INFO] target bin:", target["source_height_bin"])
    print("[INFO] kid template:", kid_template_path)
    print("[INFO] optimizer success:", result.success)
    print("[INFO] optimizer message:", result.message)
    print("[INFO] best global scale:", best_scale)
    print("[INFO] before height:", before_feats["height_canonical"])
    print("[INFO] after height:", after_feats["height_canonical"])
    print("[INFO] target height:", target["height_canonical"])


if __name__ == "__main__":
    main()