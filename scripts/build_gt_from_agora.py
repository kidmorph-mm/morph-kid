# scripts/build_gt_from_agora.py
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import trimesh

from utils_smplx import load_smplx_model, build_canonical_output, center_by_pelvis
from extract_canonical_features import extract_features_from_joints


MODEL_ROOT = "/home/jaeson1012/agora_dataset/models"
GT_ROOT = Path("/home/jaeson1012/agora_dataset/smplx_gt/smplx_gt")

MODE = "adults"   # "kids" or "adults"

OUT_CSV = Path(f"/home/jaeson1012/agora_dataset/data/gt_{MODE}_canonical_ratios.csv")
OUT_AUDIT_CSV = Path(f"/home/jaeson1012/agora_dataset/data/gt_{MODE}_audit.csv")
OUT_OBJ_DIR = Path(f"/home/jaeson1012/agora_dataset/data/canonical_obj_samples_{MODE}")

SAVE_OBJ_EVERY = 50
MAX_SAMPLES = None   # 테스트면 20 등으로 변경


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


def load_agora_fit(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    else:
        raise ValueError(f"Unsupported file: {path}")


def normalize_fit_to_params(data: Dict[str, Any]) -> Dict[str, Any]:
    if "betas" not in data:
        raise KeyError(f"betas not found in fit file. keys={list(data.keys())}")

    betas = np.asarray(data["betas"], dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]

    return {"betas": betas}


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


def find_kid_template_path(model_root: str | Path) -> Optional[str]:
    root = Path(model_root)
    if not root.exists():
        return None

    candidates = []
    patterns = [
        "*smplx_kid_template*.npy",
        "*kid_template*.npy",
        "*kid*.npy",
        "*Kid*.npy",
        "*SMIL*.npy",
        "*smil*.npy",
        "*template*.npy",
    ]

    for pattern in patterns:
        candidates.extend(root.rglob(pattern))

    uniq = sorted({p.resolve() for p in candidates if p.is_file()})
    if not uniq:
        return None

    priority_keywords = [
        "smplx_kid_template",
        "kid_template",
        "smil",
        "kid",
        "template",
    ]

    def score(p: Path) -> int:
        name = p.name.lower()
        s = 0
        for i, kw in enumerate(priority_keywords):
            if kw in name:
                s += 100 - i * 10
        return s

    uniq = sorted(uniq, key=lambda p: (-score(p), str(p)))
    return str(uniq[0])


def find_all_target_pkls(root: Path, mode: str) -> List[Path]:
    files = []
    mode = mode.lower()
    for p in root.rglob("*.pkl"):
        pstr = str(p).lower()
        if mode in pstr:
            files.append(p)
    return sorted(files)


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(path)


def main():
    if MODE not in {"kids", "adults"}:
        raise ValueError("MODE must be 'kids' or 'adults'")

    kid_template_path = find_kid_template_path(MODEL_ROOT) if MODE == "kids" else None
    print("[INFO] mode:", MODE)
    print("[INFO] kid_template_path:", kid_template_path)

    sample_paths = find_all_target_pkls(GT_ROOT, MODE)
    if MAX_SAMPLES is not None:
        sample_paths = sample_paths[:MAX_SAMPLES]

    print(f"[INFO] total {MODE} pkl files found: {len(sample_paths)}")
    if not sample_paths:
        raise RuntimeError(f"No {MODE} pkl files found under: {GT_ROOT}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_OBJ_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    audit_rows = []

    n_ok = 0
    n_fail = 0

    for idx, sample_path in enumerate(sample_paths, start=1):
        try:
            fit_data = load_agora_fit(sample_path)
            params = normalize_fit_to_params(fit_data)

            gender = normalize_gender(fit_data.get("gender", "neutral"))
            raw_betas = params["betas"]
            raw_num_betas = int(raw_betas.shape[-1])

            if MODE == "kids":
                if kid_template_path is None:
                    raise RuntimeError("kid template not found but MODE='kids'")
                model_num_betas = raw_num_betas - 1
                betas_for_model = raw_betas
            else:
                model_num_betas = raw_num_betas
                betas_for_model = raw_betas

            model = load_smplx_model(
                model_root=MODEL_ROOT,
                gender=gender,
                device="cuda",
                num_betas=model_num_betas,
                kid_template_path=kid_template_path,
            )

            forward_params = {"betas": betas_for_model}

            verts, joints, faces = build_canonical_output(model, forward_params, device="cuda")
            verts, joints = center_by_pelvis(verts, joints, pelvis_idx=JOINT_IDX["pelvis"])

            feats = extract_features_from_joints(verts, joints, JOINT_IDX)

            raw_v_height = None
            if "v" in fit_data:
                v_raw = np.asarray(fit_data["v"], dtype=np.float32)
                if v_raw.ndim == 3:
                    v_raw = v_raw[0]
                raw_v_height = float(v_raw[:, 1].max() - v_raw[:, 1].min())

            row = {
                "sample_path": str(sample_path),
                "mode": MODE,
                "gender": gender,
                "raw_num_betas": raw_num_betas,
                "model_num_betas": model_num_betas,
                "forward_betas_dim": int(betas_for_model.shape[-1]),
                "kid_template_path": kid_template_path if kid_template_path is not None else "",
                **feats,
            }
            rows.append(row)

            audit_rows.append({
                "sample_path": str(sample_path),
                "mode": MODE,
                "status": "ok",
                "gender": gender,
                "raw_num_betas": raw_num_betas,
                "model_num_betas": model_num_betas,
                "forward_betas_dim": int(betas_for_model.shape[-1]),
                "height_canonical": feats["height_canonical"],
                "raw_v_height": raw_v_height if raw_v_height is not None else "",
                "reason": "",
            })

            if SAVE_OBJ_EVERY and ((idx - 1) % SAVE_OBJ_EVERY == 0):
                obj_name = f"{idx:05d}_" + sample_path.stem + "_canonical.obj"
                save_obj(OUT_OBJ_DIR / obj_name, verts, faces)

            n_ok += 1

            if idx % 20 == 0:
                print(f"[INFO] processed {idx}/{len(sample_paths)} | ok={n_ok} fail={n_fail}")

        except Exception as e:
            n_fail += 1
            audit_rows.append({
                "sample_path": str(sample_path),
                "mode": MODE,
                "status": "fail",
                "gender": "",
                "raw_num_betas": "",
                "model_num_betas": "",
                "forward_betas_dim": "",
                "height_canonical": "",
                "raw_v_height": "",
                "reason": repr(e),
            })
            print(f"[WARN] failed: {sample_path}")
            print(f"       reason: {repr(e)}")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if audit_rows:
        audit_fieldnames = list(audit_rows[0].keys())
        with open(OUT_AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=audit_fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)

    print("[INFO] finished")
    print(f"[INFO] total={len(sample_paths)} ok={n_ok} fail={n_fail}")
    print(f"[INFO] saved main csv:  {OUT_CSV}")
    print(f"[INFO] saved audit csv: {OUT_AUDIT_CSV}")
    print(f"[INFO] saved obj dir:   {OUT_OBJ_DIR}")


if __name__ == "__main__":
    main()