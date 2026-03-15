from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from itertools import product
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimize_adult_to_child_v2 import (
    MODEL_ROOT,
    JOINT_IDX,
    load_pkl,
    normalize_gender,
    build_smplx_output,
    load_target_prototype,
    get_initial_params,
    get_bounds,
    objective_factory,
    apply_child_warp,
)
from utils_smplx import load_smplx_model, center_by_pelvis
from extract_canonical_features import extract_features_from_joints


def evaluate_single(adult_pkl: str, group: str, target_height_cm: float, weight_cfg: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    fit_data = load_pkl(adult_pkl)
    gender = normalize_gender(fit_data.get("gender", "neutral"))

    model = load_smplx_model(
        model_root=MODEL_ROOT,
        gender=gender,
        device="cuda",
        num_betas=int(np.asarray(fit_data["betas"]).shape[-1]),
        kid_template_path=None,
    )

    can_verts, can_joints, faces = build_smplx_output(model, fit_data, canonical=True)
    can_verts, can_joints = center_by_pelvis(can_verts, can_joints, pelvis_idx=JOINT_IDX["pelvis"])

    before_feats = extract_features_from_joints(can_verts, can_joints, JOINT_IDX)
    target = load_target_prototype(group, target_height_cm)

    x0 = get_initial_params(group, target["height_canonical"], before_feats["height_canonical"], target)
    bounds = get_bounds(group)
    objective = objective_factory(can_verts, can_joints, target, group, weight_cfg)

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-9},
    )

    params = {k: float(v) for k, v in zip(
        ["global_scale", "shoulder_scale", "torso_scale", "arm_scale", "thigh_scale", "shank_scale", "pelvis_scale", "head_scale"],
        result.x.tolist()
    )}

    v2, j2 = apply_child_warp(can_verts, can_joints, params, JOINT_IDX)
    after_feats = extract_features_from_joints(v2, j2, JOINT_IDX)

    score = 0.0
    for k in [
        "shoulder_width_ratio",
        "pelvis_width_ratio",
        "torso_height_ratio",
        "arm_length_ratio",
        "thigh_ratio",
        "shank_ratio",
        "leg_length_ratio",
    ]:
        score += abs(after_feats[k] - target[k])

    score += 0.5 * abs(after_feats["height_canonical"] - target["height_canonical"]) / target["height_canonical"]

    # global shrink only 방지
    score += 0.5 * abs(params["global_scale"] - 0.75)
    score += 0.7 * abs(params["pelvis_scale"] - 1.0)
    score += 0.7 * abs(params["thigh_scale"] - 1.0)
    score += 0.7 * abs(params["shank_scale"] - 1.0)

    return {
        "adult_pkl": adult_pkl,
        "group": group,
        "target_height_cm": target_height_cm,
        "score": float(score),
        "optimizer_fun": float(result.fun),
        "success": bool(result.success),
        "global_scale": params["global_scale"],
        "shoulder_scale": params["shoulder_scale"],
        "torso_scale": params["torso_scale"],
        "arm_scale": params["arm_scale"],
        "thigh_scale": params["thigh_scale"],
        "shank_scale": params["shank_scale"],
        "pelvis_scale": params["pelvis_scale"],
        "head_scale": params["head_scale"],
        "leg_ratio_after": after_feats["leg_length_ratio"],
        "leg_ratio_target": target["leg_length_ratio"],
    }


def build_weight_configs() -> List[Dict[str, Dict[str, float]]]:
    configs = []

    leg_sets = [6.0, 7.0, 8.0]
    pelvis_sets = [3.5, 4.5, 5.5]
    height_sets = [2.5, 4.0, 5.5]
    reg_global_sets = [1.0, 1.2, 1.5]

    for leg_w, pelvis_w, height_w, reg_g in product(leg_sets, pelvis_sets, height_sets, reg_global_sets):
        cfg = {
            "feature_weights": {
                "height_canonical": height_w,
                "shoulder_width_ratio": 4.0,
                "pelvis_width_ratio": pelvis_w,
                "torso_height_ratio": 4.0,
                "arm_length_ratio": 3.0,
                "thigh_ratio": leg_w - 1.0,
                "shank_ratio": leg_w - 1.0,
                "leg_length_ratio": leg_w,
            },
            "prior_weights": {
                "shoulder_scale_prior": 2.0,
                "head_scale_prior": 2.0,
            },
            "reg_weights": {
                "global_scale": reg_g,
                "shoulder_scale": 0.25,
                "torso_scale": 0.25,
                "arm_scale": 0.20,
                "thigh_scale": 0.15,
                "shank_scale": 0.15,
                "pelvis_scale": 0.20,
                "head_scale": 0.15,
            },
            "balance_weights": {
                "thigh_shank_balance": 0.5,
            },
        }
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-pkl", type=str, required=True)
    parser.add_argument("--group", type=str, default="core", choices=["core", "sub"])
    parser.add_argument("--target-height-cm", type=float, default=120.0)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    configs = build_weight_configs()

    for i, cfg in enumerate(configs, start=1):
        print(f"[INFO] testing config {i}/{len(configs)}")
        try:
            r = evaluate_single(args.adult_pkl, args.group, args.target_height_cm, cfg)
            r["config_id"] = i
            r["weight_config_json"] = json.dumps(cfg, ensure_ascii=False)
            results.append(r)
        except Exception as e:
            results.append({
                "config_id": i,
                "adult_pkl": args.adult_pkl,
                "group": args.group,
                "target_height_cm": args.target_height_cm,
                "score": np.nan,
                "success": False,
                "weight_config_json": json.dumps(cfg, ensure_ascii=False),
                "error": repr(e),
            })

    df = pd.DataFrame(results)
    df.to_csv(outdir / "weight_search_results.csv", index=False)

    ok = df[df["success"] == True].copy()
    ok = ok.sort_values("score")
    ok.to_csv(outdir / "weight_search_results_sorted.csv", index=False)

    if len(ok) > 0:
        best = ok.iloc[0].to_dict()
        with open(outdir / "best_weight_config.json", "w", encoding="utf-8") as f:
            f.write(best["weight_config_json"])
        print("[INFO] best score:", best["score"])
        print("[INFO] saved best_weight_config.json")

    print("[INFO] done")


if __name__ == "__main__":
    main()