#!/usr/bin/env python3
"""
step1_extract_child_gt_features.py
====================================
Extract betas and anthropometric features for all child GT samples and save
to a CSV file for downstream analysis.

Reuses directly from robust_child_shape_opt_upperbody_200.py:
  - JOINT_IDX          (joint name -> index mapping)
  - FEATURE_KEYS       (ordered list of core feature names)
  - load_pkl           (loads a .pkl sample dict)
  - normalize_gender   (normalises gender string)
  - ensure_batch       (reshapes arrays)
  - create_kid_model   (creates SMPL-X kid model)
  - build_output_from_model  (runs SMPL-X forward pass, canonical pose)
  - prepare_kid_beta_init    (pads betas to 11 dims)

Reuses from extract_canonical_features.py  (the shared pipeline extractor):
  - extract_features_from_joints   single entry point for ALL features
  - HEAD_NECK_FEATURE_KEYS         names of the optional head/neck columns

  NOTE: extract_canonical_features.py is the shared module that the main
  pipeline (robust_child_shape_opt_upperbody_200.py) also imports.  The
  head/neck feature computation was added THERE, not duplicated here.
  Do NOT add separate feature logic to this file.

Core output columns (always present):
  sample_id, gender, raw_beta_dim, beta_0to9_norm,
  beta_0..beta_10,
  height_canonical, shoulder_width_ratio, pelvis_width_ratio,
  torso_height_ratio, arm_length_ratio, thigh_ratio, shank_ratio,
  leg_length_ratio, height_cm

Head/neck output columns (present when verts are available — see extractor):
  head_height_ratio, head_width_ratio, neck_length_ratio,
  head_width_to_shoulder_ratio, head_height_to_torso_ratio

  These columns are detected automatically by step2 via get_analysis_features().

ASSUMPTION: child GT samples are individual .pkl files stored in
  FINAL_GT_DIR (default: /home/jaeson1012/agora_dataset/data/final_child_gt)
  Each .pkl contains at minimum: {"betas": array, "gender": str}
  Override with --gt-dir if your layout differs.

Usage:
  python step1_extract_child_gt_features.py \\
      --gt-dir /path/to/final_child_gt \\
      --kid-template /path/to/smplx_kid_template.npy \\
      --outdir ./analysis_outputs

Outputs:
  <outdir>/child_gt_beta_features.csv
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import from the existing pipeline.
# Both files must be on PYTHONPATH (or in the same directory).
# ---------------------------------------------------------------------------
try:
    from robust_child_shape_opt_upperbody_200 import (
        JOINT_IDX,
        FEATURE_KEYS,
        FINAL_GT_DIR,
        DEFAULT_KID_TEMPLATE,
        load_pkl,
        normalize_gender,
        ensure_batch,
        create_kid_model,
        build_output_from_model,
        prepare_kid_beta_init,
        find_kid_template_path,
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import from robust_child_shape_opt_upperbody_200.py.\n"
        f"Make sure both files are in the same directory or PYTHONPATH is set.\n"
        f"Details: {e}"
    )

try:
    from extract_canonical_features import (
        extract_features_from_joints,
        HEAD_NECK_FEATURE_KEYS,
    )
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import from extract_canonical_features.py.\n"
        f"Details: {e}"
    )


# ---------------------------------------------------------------------------
# Child GT loader
# ---------------------------------------------------------------------------

def load_child_gt_pkl_files(gt_dir: Path) -> List[Path]:
    """
    Return sorted list of .pkl files in gt_dir.

    ASSUMPTION: Each .pkl is one child GT sample with keys:
        betas  : (10,) or (11,) float array
        gender : str  [optional; defaults to 'neutral' if missing]

    If your dataset also has sub-directories, change rglob to glob or add depth.
    """
    pkls = sorted(gt_dir.glob("*.pkl"))
    if not pkls:
        # try one level deep
        pkls = sorted(gt_dir.rglob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No .pkl files found under {gt_dir}.\n"
            "Set --gt-dir to the correct path, or adapt load_child_gt_pkl_files()."
        )
    return pkls


# ---------------------------------------------------------------------------
# Feature extraction for one sample
# ---------------------------------------------------------------------------

def extract_sample(
    pkl_path: Path,
    kid_model_cache: Dict[str, Any],
    kid_template_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load one child GT .pkl, run SMPL-X kid in canonical pose, extract features.

    Returns a dict with:
        sample_id, gender,
        beta_0..beta_10,
        raw_beta_dim, beta_0to9_norm,
        all FEATURE_KEYS   (core anthropometric features),
        height_cm,
        all HEAD_NECK_FEATURE_KEYS  (when computable; omitted on failure)
    Returns None on failure (error is printed).

    kid_model_cache maps model_key -> smplx model, so we don't reload per sample.
    """
    try:
        data = load_pkl(pkl_path)
    except Exception as e:
        print(f"  [WARN] failed to load {pkl_path.name}: {e}")
        return None

    gender = normalize_gender(data.get("gender", "neutral"))

    try:
        raw_betas = ensure_batch(data["betas"])[0]          # (D,)
    except KeyError:
        print(f"  [WARN] {pkl_path.name}: no 'betas' key. Available: {list(data.keys())}")
        return None

    # Pad to 11 dims (matches pipeline convention)
    kid_beta, kid_create_num_betas = prepare_kid_beta_init(raw_betas)
    # kid_beta is always 11-dim after prepare_kid_beta_init

    # Lazy-load model (keyed by gender)
    model_key = f"{gender}_{kid_create_num_betas}"
    if model_key not in kid_model_cache:
        kid_model_cache[model_key] = create_kid_model(
            gender, kid_create_num_betas, kid_template_path
        )
    kid_model = kid_model_cache[model_key]

    try:
        # canonical pose, global_scale=1.0
        # build_output_from_model returns (verts, joints, faces)
        verts, joints, _ = build_output_from_model(
            kid_model,
            kid_beta,
            fit_data=data,          # may contain pose params; canonical=True ignores them
            canonical=True,
            global_scale=1.0,
        )
    except Exception as e:
        print(f"  [WARN] {pkl_path.name}: SMPL-X forward pass failed: {e}")
        traceback.print_exc()
        return None

    try:
        # extract_features_from_joints now returns core + head/neck features
        feats = extract_features_from_joints(verts, joints, JOINT_IDX)
    except Exception as e:
        print(f"  [WARN] {pkl_path.name}: feature extraction failed: {e}")
        return None

    row: Dict[str, Any] = {
        "sample_id": pkl_path.stem,
        "gender": gender,
    }

    # Always store all 11 beta dims
    for i, v in enumerate(kid_beta):
        row[f"beta_{i}"] = float(v)

    # Store raw_betas original dim for reference
    row["raw_beta_dim"] = int(raw_betas.shape[0])

    # Derived: norm of beta[0:10] (the shape PCs, excluding kid axis)
    row["beta_0to9_norm"] = float(np.linalg.norm(kid_beta[:10]))

    # Core features (FEATURE_KEYS from pipeline) — skip internal '_*' keys
    for k in FEATURE_KEYS:
        row[k] = float(feats.get(k, np.nan))

    # Convenience: height_cm
    row["height_cm"] = float(feats.get("height_canonical", np.nan)) * 100.0

    # Head / neck features — written when available; silently skipped otherwise.
    # step2 detects these automatically via get_analysis_features().
    n_head_ok = 0
    for k in HEAD_NECK_FEATURE_KEYS:
        v = feats.get(k, None)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
            n_head_ok += 1

    if n_head_ok == 0:
        # This should not happen unless verts were unavailable (see extractor)
        pass   # head features simply absent from this row; CSV column will be NaN

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract child GT betas and anthropometric features to CSV."
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=str(FINAL_GT_DIR),
        help="Directory of child GT .pkl files "
             f"(default: {FINAL_GT_DIR})",
    )
    parser.add_argument(
        "--kid-template",
        type=str,
        default=None,
        help="Path to smplx_kid_template.npy "
             f"(default: auto-detect from {DEFAULT_KID_TEMPLATE})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./analysis_outputs",
        help="Directory to write outputs (default: ./analysis_outputs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N samples (for quick testing)",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kid_template_path = find_kid_template_path(args.kid_template)
    print(f"[INFO] kid template: {kid_template_path}")
    print(f"[INFO] GT dir:       {gt_dir}")
    print(f"[INFO] output dir:   {outdir}")

    pkl_files = load_child_gt_pkl_files(gt_dir)
    if args.limit is not None:
        pkl_files = pkl_files[: args.limit]
    print(f"[INFO] Found {len(pkl_files)} child GT samples to process")

    kid_model_cache: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    for i, pkl_path in enumerate(pkl_files, start=1):
        if i % 20 == 0 or i == len(pkl_files):
            print(f"  [{i}/{len(pkl_files)}] {pkl_path.name}")

        row = extract_sample(pkl_path, kid_model_cache, kid_template_path)
        if row is not None:
            rows.append(row)
            n_ok += 1
        else:
            n_fail += 1

    if not rows:
        print("[ERROR] No samples were successfully processed. Check paths and formats.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Consistent column order — explicit, so step2 sees columns in the right groups
    beta_cols = [f"beta_{i}" for i in range(11)]
    meta_cols = ["sample_id", "gender", "raw_beta_dim", "beta_0to9_norm"]
    core_feat_cols = list(FEATURE_KEYS) + ["height_cm"]
    # Only include HEAD_NECK_FEATURE_KEYS that actually made it into any row
    present_head_neck = [k for k in HEAD_NECK_FEATURE_KEYS if k in df.columns]
    # Diagnostic fallback flag — placed immediately after head/neck feature columns
    fallback_col = ["head_features_used_fallback"] if "head_features_used_fallback" in df.columns else []
    # Any unexpected extra columns last (future-proof)
    known = set(meta_cols + beta_cols + core_feat_cols + present_head_neck + fallback_col)
    extra = [c for c in df.columns if c not in known]

    ordered_cols = meta_cols + beta_cols + core_feat_cols + present_head_neck + fallback_col + extra
    df = df[ordered_cols]

    out_csv = outdir / "child_gt_beta_features.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n[INFO] Processed: {n_ok} OK, {n_fail} failed")
    print(f"[INFO] Saved: {out_csv}")
    if present_head_neck:
        print(f"[INFO] Head/neck features written: {present_head_neck}")
        # How many rows have at least one non-NaN head/neck value
        n_with_head = int(df[present_head_neck].notna().any(axis=1).sum())
        print(f"[INFO] Samples with head/neck features: {n_with_head}/{len(df)}")
    else:
        print("[WARN] No head/neck features were computed. "
              "Check that extract_canonical_features.py is up to date and "
              "verts are non-None from build_output_from_model.")
    print(f"\n--- Preview ---")
    preview_cols = (["sample_id", "height_cm", "beta_10", "beta_0to9_norm"]
                    + list(FEATURE_KEYS[:3])
                    + present_head_neck[:2])
    print(df[[c for c in preview_cols if c in df.columns]].head(8).to_string(index=False))
    print(f"\n[INFO] beta_10 range: "
          f"min={df['beta_10'].min():.4f}  "
          f"max={df['beta_10'].max():.4f}  "
          f"mean={df['beta_10'].mean():.4f}  "
          f"std={df['beta_10'].std():.4f}")


if __name__ == "__main__":
    main()