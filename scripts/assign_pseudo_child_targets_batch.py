from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


ADULT_CSV = Path("/home/jaeson1012/agora_dataset/data/gt_adults_canonical_ratios.csv")
FINAL_GT_DIR = Path("/home/jaeson1012/agora_dataset/data/final_child_gt")

FEATURE_KEYS = [
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "arm_length_ratio",
    "thigh_ratio",
    "shank_ratio",
    "leg_length_ratio",
]


def load_candidate_prototypes() -> pd.DataFrame:
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
    return df


def prototype_row_to_target(row: pd.Series) -> Dict[str, float]:
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


def get_adult_zone(height_cm: float) -> str:
    if height_cm < 165.0:
        return "small_adult"
    if height_cm < 178.0:
        return "medium_adult"
    return "large_adult"


def filter_candidates_by_zone(
    candidates: pd.DataFrame,
    adult_zone: str,
    mode: str,
) -> pd.DataFrame:
    df = candidates.copy()
    mode = mode.lower()

    if mode == "balanced":
        if adult_zone == "small_adult":
            return df
        if adult_zone == "medium_adult":
            # medium은 너무 작은 core 제외
            return df[
                ((df["group"] == "core") & (df["height_cm_median"] >= 115.0)) |
                (df["group"] == "sub")
            ].copy()
        if adult_zone == "large_adult":
            # large는 sub 우선, core는 거의 상단만 허용
            return df[
                ((df["group"] == "core") & (df["height_cm_median"] >= 125.0)) |
                (df["group"] == "sub")
            ].copy()

    elif mode == "aggressive_child":
        if adult_zone == "small_adult":
            return df
        if adult_zone == "medium_adult":
            return df
        if adult_zone == "large_adult":
            # aggressive라도 large adult는 너무 작은 core는 막음
            return df[
                ((df["group"] == "core") & (df["height_cm_median"] >= 120.0)) |
                (df["group"] == "sub")
            ].copy()

    else:
        raise ValueError("mode must be 'balanced' or 'aggressive_child'")

    return df


def get_mode_config(mode: str) -> Dict[str, Any]:
    mode = mode.lower()

    if mode == "balanced":
        return {
            "ratio_weights": {
                "shoulder_width_ratio": 1.6,
                "pelvis_width_ratio": 1.2,
                "torso_height_ratio": 1.6,
                "arm_length_ratio": 1.0,
                "thigh_ratio": 1.8,
                "shank_ratio": 1.8,
                "leg_length_ratio": 2.2,
            },
            "height_term_weight": 2.0,
            "feasibility_threshold": 0.60,
            "hard_penalty": 8.0,
            "zone_penalties": {
                "small_adult_core": 0.0,
                "small_adult_sub": 0.2,
                "medium_adult_core": 0.0,
                "medium_adult_sub": 0.0,
                "large_adult_core": 0.8,
                "large_adult_sub": 0.0,
            },
            "childness_bonus_weight": 0.0,
            "small_height_bonus_weight": 0.0,
        }

    if mode == "aggressive_child":
        return {
            "ratio_weights": {
                "shoulder_width_ratio": 1.8,
                "pelvis_width_ratio": 1.8,
                "torso_height_ratio": 1.8,
                "arm_length_ratio": 1.2,
                "thigh_ratio": 2.2,
                "shank_ratio": 2.2,
                "leg_length_ratio": 2.8,
            },
            "height_term_weight": 1.4,
            "feasibility_threshold": 0.56,
            "hard_penalty": 5.0,
            "zone_penalties": {
                "small_adult_core": -0.4,
                "small_adult_sub": 0.4,
                "medium_adult_core": -0.2,
                "medium_adult_sub": 0.2,
                "large_adult_core": 0.3,
                "large_adult_sub": 0.0,
            },
            "childness_bonus_weight": 1.6,
            "small_height_bonus_weight": 0.8,
        }

    raise ValueError("mode must be 'balanced' or 'aggressive_child'")


def compute_assignment_score(
    adult_row: pd.Series,
    adult_zone: str,
    target: Dict[str, float],
    mode_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    ratio_weights = mode_cfg["ratio_weights"]

    adult_height_cm = float(adult_row["height_canonical"]) * 100.0
    target_height_cm = target["target_height_cm"]
    scale_ratio = target_height_cm / max(adult_height_cm, 1e-8)

    # 1) height feasibility
    height_term = abs(scale_ratio - 0.72)
    score = mode_cfg["height_term_weight"] * height_term

    if scale_ratio < mode_cfg["feasibility_threshold"]:
        score += mode_cfg["hard_penalty"] * (mode_cfg["feasibility_threshold"] - scale_ratio)

    # 2) ratio distance
    ratio_cost = 0.0
    for k, w in ratio_weights.items():
        ratio_cost += w * abs(float(adult_row[k]) - float(target[k]))
    score += ratio_cost

    # 3) zone prior
    group = target["group"]
    zone_key = f"{adult_zone}_{group}"
    score += float(mode_cfg["zone_penalties"].get(zone_key, 0.0))

    # 4) childness bonus
    childness_bonus = 0.0
    childness_bonus += (0.46 - target["leg_length_ratio"]) * 4.0
    childness_bonus += (0.09 - target["pelvis_width_ratio"]) * 2.0
    childness_bonus += (target["torso_height_ratio"] - 0.25) * 2.0

    small_height_bonus = max(0.0, (140.0 - target_height_cm) / 20.0)

    score -= mode_cfg["childness_bonus_weight"] * childness_bonus
    score -= mode_cfg["small_height_bonus_weight"] * small_height_bonus

    details = {
        "scale_ratio": float(scale_ratio),
        "height_term": float(height_term),
        "ratio_cost": float(ratio_cost),
        "childness_bonus": float(childness_bonus),
        "small_height_bonus": float(small_height_bonus),
    }
    return float(score), details


def assign_single_adult(
    adult_row: pd.Series,
    all_candidates: pd.DataFrame,
    mode: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    adult_height_cm = float(adult_row["height_canonical"]) * 100.0
    adult_zone = get_adult_zone(adult_height_cm)
    mode_cfg = get_mode_config(mode)

    candidates = filter_candidates_by_zone(all_candidates, adult_zone, mode)

    rows = []
    for _, row in candidates.iterrows():
        target = prototype_row_to_target(row)
        score, details = compute_assignment_score(
            adult_row=adult_row,
            adult_zone=adult_zone,
            target=target,
            mode_cfg=mode_cfg,
        )

        rows.append({
            "adult_sample_path": adult_row["sample_path"],
            "adult_zone": adult_zone,
            "group": target["group"],
            "height_bin": target["height_bin"],
            "target_height_cm": target["target_height_cm"],
            "score": score,
            **details,
            **{f"target_{k}": target[k] for k in FEATURE_KEYS},
        })

    ranking = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    best = ranking.iloc[0]

    assigned = {
        "adult_sample_path": adult_row["sample_path"],
        "adult_height_cm": adult_height_cm,
        "adult_zone": adult_zone,
        "assigned_mode": mode,
        "assigned_group": str(best["group"]),
        "assigned_height_bin": str(best["height_bin"]),
        "assigned_target_height_cm": float(best["target_height_cm"]),
        "assignment_score": float(best["score"]),
        "assigned_target_shoulder_width_ratio": float(best["target_shoulder_width_ratio"]),
        "assigned_target_pelvis_width_ratio": float(best["target_pelvis_width_ratio"]),
        "assigned_target_torso_height_ratio": float(best["target_torso_height_ratio"]),
        "assigned_target_arm_length_ratio": float(best["target_arm_length_ratio"]),
        "assigned_target_thigh_ratio": float(best["target_thigh_ratio"]),
        "assigned_target_shank_ratio": float(best["target_shank_ratio"]),
        "assigned_target_leg_length_ratio": float(best["target_leg_length_ratio"]),
    }
    return assigned, ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="balanced", choices=["balanced", "aggressive_child"])
    parser.add_argument("--adult-csv", type=str, default=str(ADULT_CSV))
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    adult_csv = Path(args.adult_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not adult_csv.exists():
        raise FileNotFoundError(adult_csv)

    adults = pd.read_csv(adult_csv)
    candidates = load_candidate_prototypes()

    if args.max_samples is not None:
        adults = adults.head(args.max_samples).copy()

    assigned_rows = []
    ranking_frames = []

    total = len(adults)
    for i, (_, adult_row) in enumerate(adults.iterrows(), start=1):
        assigned, ranking = assign_single_adult(adult_row, candidates, mode=args.mode)
        assigned_rows.append(assigned)
        ranking_frames.append(ranking)

        if i % 100 == 0 or i == total:
            print(f"[INFO] processed {i}/{total}")

    assigned_df = pd.DataFrame(assigned_rows)
    rankings_df = pd.concat(ranking_frames, ignore_index=True)

    assigned_csv = outdir / f"adult_pseudo_labels_{args.mode}.csv"
    ranking_csv = outdir / f"adult_pseudo_label_rankings_{args.mode}.csv"
    zone_summary_csv = outdir / f"adult_zone_summary_{args.mode}.csv"

    assigned_df.to_csv(assigned_csv, index=False)
    rankings_df.to_csv(ranking_csv, index=False)

    zone_summary = (
        assigned_df.groupby(["adult_zone", "assigned_group", "assigned_height_bin"])
        .size()
        .reset_index(name="n")
        .sort_values(["adult_zone", "assigned_group", "assigned_height_bin"])
    )
    zone_summary.to_csv(zone_summary_csv, index=False)

    summary = {
        "mode": args.mode,
        "n_adults": int(len(assigned_df)),
        "adult_zone_counts": assigned_df["adult_zone"].value_counts().to_dict(),
        "assigned_group_counts": assigned_df["assigned_group"].value_counts().to_dict(),
        "assigned_height_bin_counts": assigned_df["assigned_height_bin"].value_counts().to_dict(),
        "mean_assignment_score": float(assigned_df["assignment_score"].mean()),
    }

    with open(outdir / f"adult_pseudo_label_summary_{args.mode}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print("[INFO] saved:", assigned_csv)
    print("[INFO] saved:", ranking_csv)
    print("[INFO] saved:", zone_summary_csv)


if __name__ == "__main__":
    main()