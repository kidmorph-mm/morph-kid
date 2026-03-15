from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimize_adult_to_child_v3 import (
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


def evaluate_single(adult_pkl: str, group: str, target_height_cm: float, weight_cfg):
    fit_data = load_pkl(adult_pkl)
    gender = normalize_gender(fit_data.get("gender", "neutral"))

    model = load_smplx_model(
        model_root=MODEL_ROOT,
        gender=gender,
        device="cuda",
        num_betas=int(np.asarray(fit_data["betas"]).shape[-1]),
        kid_template_path=None,
    )

    can_verts, can_joints, _ = build_smplx_output(model, fit_data, canonical=True)
    can_verts, can_joints = center_by_pelvis(can_verts, can_joints, pelvis_idx=JOINT_IDX["pelvis"])

    before_feats = extract_features_from_joints(can_verts, can_joints, JOINT_IDX)
    target = load_target_prototype(group, target_height_cm)

    x0 = get_initial_params(group, target["height_canonical"], before_feats["height_canonical"])
    bounds = get_bounds(group)
    objective = objective_factory(can_verts, can_joints, target, weight_cfg)

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 280, "ftol": 1e-9},
    )

    params = {k: float(v) for k, v in zip(
        ["global_scale", "shoulder_scale", "torso_scale", "arm_scale", "thigh_scale", "shank_scale", "pelvis_scale", "head_scale"],
        result.x.tolist()
    )}

    v2, j2 = apply_child_warp(can_verts, can_joints, params, JOINT_IDX)
    after_feats = extract_features_from_joints(v2, j2, JOINT_IDX)

    # 핵심 오차
    err = {}
    for k in [
        "shoulder_width_ratio",
        "pelvis_width_ratio",
        "torso_height_ratio",
        "arm_length_ratio",
        "thigh_ratio",
        "shank_ratio",
        "leg_length_ratio",
    ]:
        err[k] = abs(after_feats[k] - target[k])

    height_err = abs(after_feats["height_canonical"] - target["height_canonical"]) / target["height_canonical"]

    # 이번엔 pelvis / shoulder / arm를 더 중요하게 반영
    score = (
        1.0 * height_err +
        2.4 * err["shoulder_width_ratio"] +
        3.2 * err["pelvis_width_ratio"] +
        1.6 * err["torso_height_ratio"] +
        2.3 * err["arm_length_ratio"] +
        2.0 * err["thigh_ratio"] +
        2.0 * err["shank_ratio"] +
        2.3 * err["leg_length_ratio"]
    )

    # global shrink only 방지
    score += 0.7 * max(0.0, 0.72 - params["global_scale"])
    score += 0.6 * abs(params["arm_scale"] - 0.88)
    score += 0.8 * abs(params["pelvis_scale"] - 0.80)
    score += 0.4 * abs(params["shoulder_scale"] - 0.92)

    return {
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
        "height_after": after_feats["height_canonical"],
        "height_target": target["height_canonical"],
        "leg_ratio_after": after_feats["leg_length_ratio"],
        "leg_ratio_target": target["leg_length_ratio"],
        "arm_ratio_after": after_feats["arm_length_ratio"],
        "arm_ratio_target": target["arm_length_ratio"],
        "pelvis_ratio_after": after_feats["pelvis_width_ratio"],
        "pelvis_ratio_target": target["pelvis_width_ratio"],
        "shoulder_ratio_after": after_feats["shoulder_width_ratio"],
        "shoulder_ratio_target": target["shoulder_width_ratio"],
    }


def build_weight_configs():
    configs = []

    shoulder_sets = [5.0, 6.0, 7.0]
    pelvis_sets = [6.0, 7.5, 9.0]
    arm_sets = [4.0, 5.0, 6.0]
    height_sets = [2.5, 3.0, 4.0]
    pelvis_reg_sets = [0.04, 0.06, 0.10]
    arm_reg_sets = [0.06, 0.08, 0.12]

    for sw, pw, aw, hw, preg, areg in product(
        shoulder_sets, pelvis_sets, arm_sets, height_sets, pelvis_reg_sets, arm_reg_sets
    ):
        cfg = {
            "feature_weights": {
                "height_canonical": hw,
                "shoulder_width_ratio": sw,
                "pelvis_width_ratio": pw,
                "torso_height_ratio": 4.5,
                "arm_length_ratio": aw,
                "thigh_ratio": 6.0,
                "shank_ratio": 6.0,
                "leg_length_ratio": 7.0,
            },
            "prior_weights": {
                "shoulder_scale_prior": 0.8,
                "head_scale_prior": 2.0,
            },
            "reg_weights": {
                "global_scale": 1.4,
                "shoulder_scale": 0.18,
                "torso_scale": 0.22,
                "arm_scale": areg,
                "thigh_scale": 0.12,
                "shank_scale": 0.12,
                "pelvis_scale": preg,
                "head_scale": 0.12,
            },
            "balance_weights": {
                "thigh_shank_balance": 0.45,
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

    configs = build_weight_configs()
    results = []

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
                "score": np.nan,
                "success": False,
                "error": repr(e),
                "weight_config_json": json.dumps(cfg, ensure_ascii=False),
            })

    df = pd.DataFrame(results)
    df.to_csv(outdir / "weight_search_results.csv", index=False)

    ok = df[df["success"] == True].copy().sort_values("score")
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