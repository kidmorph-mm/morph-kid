# scripts/make_final_child_gt.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

KID_CSV = Path("/home/jaeson1012/agora_dataset/data/gt_kids_canonical_ratios.csv")
OUT_DIR = Path("/home/jaeson1012/agora_dataset/data/final_child_gt")

CORE_MAX = 125.0
SUB_MAX = 140.0

RATIO_COLS = [
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "upper_arm_ratio",
    "forearm_ratio",
    "arm_length_ratio",
    "thigh_ratio",
    "shank_ratio",
    "leg_length_ratio",
]

ABS_COLS = [
    "height_canonical",
    "shoulder_width",
    "pelvis_width",
    "torso_height",
    "upper_arm",
    "forearm",
    "arm_length",
    "thigh",
    "shank",
    "leg_length",
]

META_COLS = [
    "sample_path",
    "mode",
    "gender",
    "raw_num_betas",
    "model_num_betas",
    "forward_betas_dim",
    "kid_template_path",
]


def assign_group(h_cm: float) -> str:
    if h_cm <= CORE_MAX:
        return "core"
    if h_cm <= SUB_MAX:
        return "sub"
    return "excluded"


def robust_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "feature": c,
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "median": float(s.median()),
            "q10": float(s.quantile(0.10)),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "q90": float(s.quantile(0.90)),
        })
    return pd.DataFrame(rows)


def make_height_bins(df: pd.DataFrame, bin_step_cm: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    h_cm = df["height_canonical"] * 100.0
    h_min = int(np.floor(h_cm.min() / bin_step_cm) * bin_step_cm)
    h_max = int(np.ceil(h_cm.max() / bin_step_cm) * bin_step_cm)

    bins = list(range(h_min, h_max + bin_step_cm, bin_step_cm))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

    tmp = df.copy()
    tmp["height_cm"] = h_cm
    tmp["height_bin"] = pd.cut(tmp["height_cm"], bins=bins, labels=labels, right=True, include_lowest=True)

    rows = []
    for b, g in tmp.groupby("height_bin", observed=False):
        if g.empty:
            continue

        row = {
            "height_bin": str(b),
            "n_samples": int(len(g)),
            "height_cm_mean": float(g["height_cm"].mean()),
            "height_cm_median": float(g["height_cm"].median()),
        }

        for c in RATIO_COLS:
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            row[f"{c}_median"] = float(s.median())
            row[f"{c}_mean"] = float(s.mean())

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not KID_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {KID_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(KID_CSV)

    # 혹시 mode 컬럼이 없으면 추가
    if "mode" not in df.columns:
        df["mode"] = "kids"

    # m -> cm 기준으로 그룹 나누기
    df["height_cm"] = df["height_canonical"] * 100.0
    df["gt_group"] = df["height_cm"].apply(assign_group)

    core_df = df[df["gt_group"] == "core"].copy()
    sub_df = df[df["gt_group"] == "sub"].copy()
    excluded_df = df[df["gt_group"] == "excluded"].copy()

    # 1) 샘플별 CSV 저장
    core_df.to_csv(OUT_DIR / "gt_child_core_samples.csv", index=False)
    sub_df.to_csv(OUT_DIR / "gt_child_sub_samples.csv", index=False)
    excluded_df.to_csv(OUT_DIR / "gt_child_excluded_samples.csv", index=False)

    # 2) 요약 통계 저장
    core_summary = robust_summary(core_df, ABS_COLS + RATIO_COLS)
    sub_summary = robust_summary(sub_df, ABS_COLS + RATIO_COLS)
    excluded_summary = robust_summary(excluded_df, ABS_COLS + RATIO_COLS)

    core_summary.to_csv(OUT_DIR / "gt_child_core_summary.csv", index=False)
    sub_summary.to_csv(OUT_DIR / "gt_child_sub_summary.csv", index=False)
    excluded_summary.to_csv(OUT_DIR / "gt_child_excluded_summary.csv", index=False)

    # 3) prototype 저장
    # core는 전체 median prototype도 저장
    core_proto = {"group": "core", "n_samples": int(len(core_df))}
    for c in RATIO_COLS:
        s = pd.to_numeric(core_df[c], errors="coerce").dropna()
        if len(s) > 0:
            core_proto[f"{c}_median"] = float(s.median())
            core_proto[f"{c}_mean"] = float(s.mean())

    sub_proto = {"group": "sub", "n_samples": int(len(sub_df))}
    for c in RATIO_COLS:
        s = pd.to_numeric(sub_df[c], errors="coerce").dropna()
        if len(s) > 0:
            sub_proto[f"{c}_median"] = float(s.median())
            sub_proto[f"{c}_mean"] = float(s.mean())

    pd.DataFrame([core_proto]).to_csv(OUT_DIR / "gt_child_core_prototype.csv", index=False)
    pd.DataFrame([sub_proto]).to_csv(OUT_DIR / "gt_child_sub_prototype.csv", index=False)

    # 4) 5cm bin prototype 저장
    make_height_bins(core_df, bin_step_cm=5).to_csv(OUT_DIR / "gt_child_core_bin_prototypes_5cm.csv", index=False)
    make_height_bins(sub_df, bin_step_cm=5).to_csv(OUT_DIR / "gt_child_sub_bin_prototypes_5cm.csv", index=False)

    # 5) audit용 개수 저장
    audit = pd.DataFrame([
        {"group": "core", "n_samples": int(len(core_df)), "height_cm_min": float(core_df["height_cm"].min()) if len(core_df) else np.nan, "height_cm_max": float(core_df["height_cm"].max()) if len(core_df) else np.nan},
        {"group": "sub", "n_samples": int(len(sub_df)), "height_cm_min": float(sub_df["height_cm"].min()) if len(sub_df) else np.nan, "height_cm_max": float(sub_df["height_cm"].max()) if len(sub_df) else np.nan},
        {"group": "excluded", "n_samples": int(len(excluded_df)), "height_cm_min": float(excluded_df["height_cm"].min()) if len(excluded_df) else np.nan, "height_cm_max": float(excluded_df["height_cm"].max()) if len(excluded_df) else np.nan},
    ])
    audit.to_csv(OUT_DIR / "gt_child_group_audit.csv", index=False)

    print("[INFO] saved to:", OUT_DIR)
    print("[INFO] core:", len(core_df))
    print("[INFO] sub:", len(sub_df))
    print("[INFO] excluded:", len(excluded_df))


if __name__ == "__main__":
    main()