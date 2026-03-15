#!/usr/bin/env python3
"""
step2_analyze_beta10.py
========================
Analyse beta[10] from the child GT feature CSV produced by step1.

Always produced (existing outputs — unchanged):
  - beta10_summary.json              : mean/std/min/max/percentiles
  - beta10_by_height_bin.csv         : per-height-bin beta10 stats
  - beta_feature_correlations.csv    : corr(beta_i, feature_j) — now includes
                                       any optional head/neck columns present
  - beta_covariance.csv              : full covariance matrix of beta_0..10
  - beta10_role_analysis.csv         : high/low beta10 vs beta_0:9 norm
  - beta_pca_summary.json            : PCA explained variance
  - beta_pca_components.csv          : PCA component matrix

New outputs (added in this version):
  - beta10_partial_correlations.csv  : height-controlled partial correlations
                                       between beta_10 and each ratio feature
  - beta10_regression_summary.json   : R², coefficients, standardised betas
                                       for beta10 ~ height and
                                       beta10 ~ height + proportion features
  - beta10_regression_coefficients.csv : per-feature regression details
  - beta10_group_feature_summary.csv : feature means + Cohen's d across the
                                       four high/low beta10 × norm groups

Optional plots (require matplotlib):
  Existing:
  - beta10_histogram.png
  - beta10_vs_height_scatter.png
  - beta_correlation_heatmap.png
  - beta_pca_variance.png
  - beta_pca_scatter.png
  New:
  - beta10_partial_correlation_bar.png  : simple vs height-controlled corr
  - beta10_residual_vs_feature_scatter.png : beta10 residual vs top features
  - beta10_group_comparison_plot.png    : group means for key features

Optional feature support:
  step2 automatically detects extra columns in the CSV and includes them in
  all analyses if present. High-priority optional features:
    - head_height_ratio
    - head_width_ratio
    - neck_length_ratio
  Other detected optional columns (e.g. head_width_to_shoulder_ratio,
  head_height_to_torso_ratio, torso_depth_ratio) are also included.

Reuses from the pipeline:
  - FEATURE_KEYS : feature column order (from robust_child_shape_opt_upperbody_200)

Usage:
  python scripts/step2_analyze_beta10.py \
      --csv ./analysis_outputs/child_gt_beta_features.csv \
      --outdir ./analysis_outputs

  # Skip plots
  python step2_analyze_beta10.py \\
      --csv ./analysis_outputs/child_gt_beta_features.csv \\
      --outdir ./analysis_outputs \\
      --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ── import feature key order from existing pipeline ─────────────────────────
try:
    from robust_child_shape_opt_upperbody_200 import FEATURE_KEYS
except ImportError:
    # Fallback: same order as in the pipeline file
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
    warnings.warn(
        "Could not import FEATURE_KEYS from pipeline; using hardcoded fallback."
    )

# Features used for correlation with beta (exclude height_canonical, use height_cm)
# This is the core set that is always expected.
RATIO_FEATURES = [
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "arm_length_ratio",
    "thigh_ratio",
    "shank_ratio",
    "leg_length_ratio",
]

# Optional features: included automatically if present in the CSV.
# These must be added to the CSV by the user (e.g. by extending step1 or
# extract_canonical_features.py with head/neck joint computations).
# High-priority:
OPTIONAL_HEAD_NECK_FEATURES = [
    "head_height_ratio",
    "head_width_ratio",
    "neck_length_ratio",
]
# Other optional features that are also auto-detected if present:
OPTIONAL_EXTRA_FEATURES = [
    "head_width_to_shoulder_ratio",
    "head_height_to_torso_ratio",
    "torso_depth_ratio",
    "waist_or_bmi_proxy",
]
# Union of all optional feature names (checked in order; first found = first used)
ALL_OPTIONAL_FEATURES = OPTIONAL_HEAD_NECK_FEATURES + OPTIONAL_EXTRA_FEATURES


def get_analysis_features(df: pd.DataFrame) -> list:
    """
    Return the full list of ratio features to use in analysis:
      RATIO_FEATURES (always) + any OPTIONAL columns that actually exist in df.

    This is the single source of truth for "which features should step2 use."
    """
    present_optional = [f for f in ALL_OPTIONAL_FEATURES if f in df.columns]
    if present_optional:
        print(f"  [optional features detected] {present_optional}")
    return RATIO_FEATURES + present_optional


# Height bins in cm
HEIGHT_BINS = [
    (0,   100, "<100"),
    (100, 110, "100-110"),
    (110, 120, "110-120"),
    (120, 130, "120-130"),
    (130, 140, "130-140"),
    (140, 200, "140+"),
]


def assign_height_bin(h: float) -> str:
    for lo, hi, label in HEIGHT_BINS:
        if lo <= h < hi:
            return label
    return "unknown"


def try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ── analysis functions ───────────────────────────────────────────────────────

def compute_beta10_summary(df: pd.DataFrame) -> dict:
    b10 = df["beta_10"].dropna()
    percs = np.percentile(b10, [5, 10, 25, 50, 75, 90, 95])
    return {
        "n": int(len(b10)),
        "mean": float(b10.mean()),
        "std": float(b10.std()),
        "min": float(b10.min()),
        "max": float(b10.max()),
        "p5":  float(percs[0]),
        "p10": float(percs[1]),
        "p25": float(percs[2]),
        "p50": float(percs[3]),
        "p75": float(percs[4]),
        "p90": float(percs[5]),
        "p95": float(percs[6]),
    }


def compute_height_bin_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["height_bin"] = df["height_cm"].apply(assign_height_bin)

    rows = []
    for lo, hi, label in HEIGHT_BINS:
        mask = (df["height_cm"] >= lo) & (df["height_cm"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        b10 = sub["beta_10"]
        rows.append({
            "height_bin":  label,
            "n":           int(len(sub)),
            "height_cm_mean": float(sub["height_cm"].mean()),
            "height_cm_std":  float(sub["height_cm"].std()),
            "beta10_mean": float(b10.mean()),
            "beta10_std":  float(b10.std()),
            "beta10_min":  float(b10.min()),
            "beta10_max":  float(b10.max()),
            "beta10_p25":  float(np.percentile(b10, 25)),
            "beta10_p50":  float(np.percentile(b10, 50)),
            "beta10_p75":  float(np.percentile(b10, 75)),
        })

    # Also add "overall" row
    b10 = df["beta_10"]
    rows.append({
        "height_bin":  "ALL",
        "n":           int(len(df)),
        "height_cm_mean": float(df["height_cm"].mean()),
        "height_cm_std":  float(df["height_cm"].std()),
        "beta10_mean": float(b10.mean()),
        "beta10_std":  float(b10.std()),
        "beta10_min":  float(b10.min()),
        "beta10_max":  float(b10.max()),
        "beta10_p25":  float(np.percentile(b10, 25)),
        "beta10_p50":  float(np.percentile(b10, 50)),
        "beta10_p75":  float(np.percentile(b10, 75)),
    })
    return pd.DataFrame(rows)


def compute_beta_feature_correlations(df: pd.DataFrame,
                                       ratio_features: list | None = None) -> pd.DataFrame:
    """
    Pearson correlation between each beta_i and:
      - height_cm
      - each feature in ratio_features  (defaults to RATIO_FEATURES + any optional
        features detected by get_analysis_features)

    Pass ratio_features explicitly if you have already called get_analysis_features().
    """
    if ratio_features is None:
        ratio_features = get_analysis_features(df)

    beta_cols = [f"beta_{i}" for i in range(11)]
    target_cols = ["height_cm"] + ratio_features

    rows = []
    for bc in beta_cols:
        if bc not in df.columns:
            continue
        row = {"beta": bc}
        for tc in target_cols:
            if tc not in df.columns:
                row[f"corr_{tc}"] = np.nan
                continue
            valid = df[[bc, tc]].dropna()
            if len(valid) < 3:
                row[f"corr_{tc}"] = np.nan
            else:
                row[f"corr_{tc}"] = float(np.corrcoef(valid[bc], valid[tc])[0, 1])
        rows.append(row)

    return pd.DataFrame(rows)


def compute_beta_covariance(df: pd.DataFrame) -> pd.DataFrame:
    beta_cols = [f"beta_{i}" for i in range(11) if f"beta_{i}" in df.columns]
    mat = df[beta_cols].dropna()
    cov = np.cov(mat.values.T)
    return pd.DataFrame(cov, index=beta_cols, columns=beta_cols)


def compute_beta_pca(df: pd.DataFrame):
    """
    PCA on beta_0..10.
    Returns (explained_variance_ratio, components dataframe, projected dataframe).
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None, None, None

    beta_cols = [f"beta_{i}" for i in range(11) if f"beta_{i}" in df.columns]
    mat = df[beta_cols].dropna().values

    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)

    pca = PCA(n_components=min(11, mat.shape[0]))
    proj = pca.fit_transform(mat_scaled)

    comp_df = pd.DataFrame(
        pca.components_,
        columns=beta_cols,
        index=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    proj_df = pd.DataFrame(
        proj,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    proj_df.index = df[beta_cols].dropna().index

    return pca.explained_variance_ratio_, comp_df, proj_df


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled Cohen's d between two groups (unsigned difference / pooled std)."""
    n_a, n_b = len(a.dropna()), len(b.dropna())
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((n_a - 1) * float(a.std()) ** 2 + (n_b - 1) * float(b.std()) ** 2)
        / (n_a + n_b - 2)
    )
    if pooled_std < 1e-12:
        return float("nan")
    return float(abs(a.mean() - b.mean()) / pooled_std)


def compute_role_analysis(df: pd.DataFrame,
                           ratio_features: list | None = None) -> pd.DataFrame:
    """
    Compare samples where:
      - beta_10 is high but beta_0:9 norm is low  (pure kid axis)
      - beta_10 is low  but beta_0:9 norm is high (shape-only)
      - both high
      - both low

    Reports mean feature values per group.
    Cohen's d is reported for the two most contrasting groups:
      high_b10_low_norm  vs  low_b10_high_norm
    as a proxy for effect size of the beta_10 axis on each feature.
    """
    if ratio_features is None:
        ratio_features = get_analysis_features(df)

    df = df.copy()

    b10_med  = df["beta_10"].median()
    norm_med = df["beta_0to9_norm"].median()

    conditions = {
        "high_b10_low_norm":  (df["beta_10"] >= b10_med) & (df["beta_0to9_norm"] < norm_med),
        "high_b10_high_norm": (df["beta_10"] >= b10_med) & (df["beta_0to9_norm"] >= norm_med),
        "low_b10_low_norm":   (df["beta_10"] < b10_med)  & (df["beta_0to9_norm"] < norm_med),
        "low_b10_high_norm":  (df["beta_10"] < b10_med)  & (df["beta_0to9_norm"] >= norm_med),
    }

    group_data = {label: df[mask] for label, mask in conditions.items()}

    rows = []
    for label, sub in group_data.items():
        if len(sub) == 0:
            continue
        row = {
            "group": label,
            "n": int(len(sub)),
            "beta10_mean": float(sub["beta_10"].mean()),
            "beta10_std":  float(sub["beta_10"].std()),
            "norm_mean":   float(sub["beta_0to9_norm"].mean()),
            "height_cm_mean": float(sub["height_cm"].mean()),
            "height_cm_std":  float(sub["height_cm"].std()),
        }
        for k in ratio_features:
            if k in sub.columns:
                row[f"{k}_mean"] = float(sub[k].mean())
                row[f"{k}_std"]  = float(sub[k].std())
        rows.append(row)

    role_df = pd.DataFrame(rows)

    # Append Cohen's d column for each feature:
    # contrast = high_b10_low_norm vs low_b10_high_norm (most interpretable)
    grp_A = group_data.get("high_b10_low_norm", pd.DataFrame())
    grp_B = group_data.get("low_b10_high_norm",  pd.DataFrame())
    if len(grp_A) >= 2 and len(grp_B) >= 2:
        d_rows = []
        for k in ["height_cm"] + ratio_features:
            if k in grp_A.columns and k in grp_B.columns:
                d = _cohens_d(grp_A[k], grp_B[k])
                d_rows.append({"feature": k, "cohens_d_high_vs_low_b10": d})
        if d_rows:
            effect_df = pd.DataFrame(d_rows).sort_values(
                "cohens_d_high_vs_low_b10", ascending=False
            )
            # Attach effect sizes as extra rows tagged with group="cohens_d"
            for _, er in effect_df.iterrows():
                pass   # effect_df is saved separately in main()

    return role_df, effect_df if (len(grp_A) >= 2 and len(grp_B) >= 2) else pd.DataFrame()



# ── NEW: height-controlled (partial) correlation analysis ────────────────────

def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return residuals of OLS regression y ~ x (intercept included)."""
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def compute_partial_correlations(df: pd.DataFrame,
                                  ratio_features: list | None = None) -> pd.DataFrame:
    """
    Height-controlled partial correlation between beta_10 and each feature.

    Method: regression-residual approach (no extra dependencies needed).
      1. Regress beta_10 on height_cm  → residual_b10
      2. Regress feature   on height_cm  → residual_feat
      3. partial_corr = Pearson(residual_b10, residual_feat)

    This isolates the beta_10 ↔ feature association that is NOT explained by
    shared covariation with height.

    Returns DataFrame with columns:
      feature | pearson_r | partial_r_height_controlled | r_drop
    where r_drop = pearson_r - partial_r  (positive → height mediated)
    """
    if ratio_features is None:
        ratio_features = get_analysis_features(df)

    if "height_cm" not in df.columns or "beta_10" not in df.columns:
        return pd.DataFrame()

    rows = []
    for feat in ratio_features:
        if feat not in df.columns:
            continue
        valid = df[["beta_10", "height_cm", feat]].dropna()
        if len(valid) < 5:
            rows.append({"feature": feat, "pearson_r": np.nan,
                         "partial_r_height_controlled": np.nan, "r_drop": np.nan})
            continue

        b10   = valid["beta_10"].values
        h     = valid["height_cm"].values
        feat_vals = valid[feat].values

        # Simple Pearson
        pearson_r = float(np.corrcoef(b10, feat_vals)[0, 1])

        # Residualise both on height
        resid_b10   = _residualise(b10,       h)
        resid_feat  = _residualise(feat_vals, h)
        partial_r   = float(np.corrcoef(resid_b10, resid_feat)[0, 1])

        rows.append({
            "feature":                    feat,
            "pearson_r":                  pearson_r,
            "partial_r_height_controlled": partial_r,
            "r_drop":                     round(pearson_r - partial_r, 6),
        })

    return pd.DataFrame(rows).sort_values("partial_r_height_controlled",
                                          key=abs, ascending=False)


# ── NEW: regression analysis ─────────────────────────────────────────────────

def _ols_summary(X: np.ndarray, y: np.ndarray,
                 feature_names: list) -> dict:
    """
    Fit OLS y ~ X (X must already include intercept column) via numpy.
    Returns dict with: coefficients, R², and standardised betas.
    Falls back to statsmodels if available for p-values and conf intervals.
    """
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    # Standardised betas (skip intercept column 0)
    std_y = float(y.std())
    std_betas = []
    for i, name in enumerate(feature_names):
        if name == "intercept":
            std_betas.append(float("nan"))
        else:
            std_x = float(X[:, i].std())
            std_betas.append(float(coef[i] * std_x / std_y) if std_y > 1e-12 else float("nan"))

    result = {
        "r2": round(r2, 6),
        "n":  int(len(y)),
        "coefficients": {name: round(float(c), 8)
                         for name, c in zip(feature_names, coef)},
        "standardised_betas": {name: round(v, 6)
                                for name, v in zip(feature_names, std_betas)},
    }

    # Try statsmodels for richer output
    try:
        import statsmodels.api as sm
        res = sm.OLS(y, X).fit()
        pvals = {name: round(float(p), 6)
                 for name, p in zip(feature_names, res.pvalues)}
        result["p_values"] = pvals
        result["r2_adj"]   = round(float(res.rsquared_adj), 6)
    except ImportError:
        pass  # p-values not available; that's fine

    return result


def compute_beta10_regression(df: pd.DataFrame,
                               ratio_features: list | None = None) -> tuple:
    """
    Fit two OLS models predicting beta_10:
      Model 1 (height only):       beta_10 ~ height_cm
      Model 2 (height + features): beta_10 ~ height_cm + ratio_features

    Returns:
      summary_dict  (for beta10_regression_summary.json)
      coef_df       (for beta10_regression_coefficients.csv)
    """
    if ratio_features is None:
        ratio_features = get_analysis_features(df)

    needed = ["beta_10", "height_cm"] + ratio_features
    present = [c for c in needed if c in df.columns]
    valid = df[present].dropna()

    if len(valid) < 5:
        return {}, pd.DataFrame()

    y = valid["beta_10"].values

    # ── Model 1: beta_10 ~ height_cm ──────────────────────────────────
    X1 = np.column_stack([np.ones(len(valid)), valid["height_cm"].values])
    feat_names_1 = ["intercept", "height_cm"]
    m1 = _ols_summary(X1, y, feat_names_1)

    # ── Model 2: beta_10 ~ height_cm + all ratio features ─────────────
    feat_cols_2 = [c for c in ratio_features if c in valid.columns]
    X2 = np.column_stack(
        [np.ones(len(valid)), valid["height_cm"].values]
        + [valid[c].values for c in feat_cols_2]
    )
    feat_names_2 = ["intercept", "height_cm"] + feat_cols_2
    m2 = _ols_summary(X2, y, feat_names_2)

    summary = {
        "model_1_height_only":          m1,
        "model_2_height_and_features":  m2,
        "r2_gain_from_features":        round(m2["r2"] - m1["r2"], 6),
        "note": (
            "r2_gain_from_features > 0.05 suggests proportion features explain "
            "beta_10 variance beyond height alone."
        ),
    }

    # Flat coefficient table for CSV
    rows = []
    for model_name, res in [("height_only", m1), ("height_and_features", m2)]:
        for feat, coef_val in res["coefficients"].items():
            rows.append({
                "model":              model_name,
                "feature":            feat,
                "coefficient":        coef_val,
                "standardised_beta":  res["standardised_betas"].get(feat, np.nan),
                "p_value":            res.get("p_values", {}).get(feat, np.nan),
                "r2":                 res["r2"],
            })
    coef_df = pd.DataFrame(rows)

    return summary, coef_df


# ── NEW: group feature summary with effect sizes ─────────────────────────────

def compute_group_feature_summary(df: pd.DataFrame,
                                   ratio_features: list | None = None) -> pd.DataFrame:
    """
    For each of the four high/low beta_10 × beta_0:9 norm groups, report:
      - mean ± std for height_cm and every ratio feature
      - Cohen's d (high_b10_low_norm vs low_b10_high_norm) per feature

    This is a richer, standalone version of the group section already inside
    compute_role_analysis(), saved separately for easier downstream use.
    """
    if ratio_features is None:
        ratio_features = get_analysis_features(df)

    if "beta_0to9_norm" not in df.columns:
        return pd.DataFrame()

    b10_med  = df["beta_10"].median()
    norm_med = df["beta_0to9_norm"].median()

    groups = {
        "high_b10_low_norm":  df[(df["beta_10"] >= b10_med) & (df["beta_0to9_norm"] < norm_med)],
        "high_b10_high_norm": df[(df["beta_10"] >= b10_med) & (df["beta_0to9_norm"] >= norm_med)],
        "low_b10_low_norm":   df[(df["beta_10"] < b10_med)  & (df["beta_0to9_norm"] < norm_med)],
        "low_b10_high_norm":  df[(df["beta_10"] < b10_med)  & (df["beta_0to9_norm"] >= norm_med)],
    }

    all_features = ["height_cm"] + ratio_features
    rows = []

    for feat in all_features:
        if feat not in df.columns:
            continue
        row = {"feature": feat}
        group_series = {}
        for gname, gdf in groups.items():
            vals = gdf[feat].dropna()
            row[f"{gname}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{gname}_std"]  = float(vals.std())  if len(vals) > 1 else np.nan
            row[f"{gname}_n"]    = int(len(vals))
            group_series[gname] = vals

        # Cohen's d: most informative contrast
        d = _cohens_d(group_series.get("high_b10_low_norm",  pd.Series(dtype=float)),
                      group_series.get("low_b10_high_norm",   pd.Series(dtype=float)))
        row["cohens_d_high_vs_low_b10"] = d
        rows.append(row)

    result = pd.DataFrame(rows)
    if "cohens_d_high_vs_low_b10" in result.columns:
        result = result.sort_values("cohens_d_high_vs_low_b10", key=abs, ascending=False)
    return result


# ── plots ────────────────────────────────────────────────────────────────────

def plot_beta10_histogram(df: pd.DataFrame, outdir: Path, plt):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["beta_10"].dropna(), bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(df["beta_10"].mean(), color="red",    linestyle="--", label=f"mean={df['beta_10'].mean():.3f}")
    ax.axvline(df["beta_10"].median(), color="orange", linestyle=":",  label=f"median={df['beta_10'].median():.3f}")
    ax.set_xlabel("beta_10 (kid axis)")
    ax.set_ylabel("count")
    ax.set_title("Child GT: beta_10 distribution")
    ax.legend()
    fig.tight_layout()
    path = outdir / "beta10_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_beta10_vs_height(df: pd.DataFrame, outdir: Path, plt):
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        df["height_cm"], df["beta_10"],
        c=df["beta_0to9_norm"], cmap="viridis",
        alpha=0.6, s=25, edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("beta[0:9] norm")

    # Trend line
    valid = df[["height_cm", "beta_10"]].dropna()
    m, b = np.polyfit(valid["height_cm"], valid["beta_10"], 1)
    xs = np.linspace(valid["height_cm"].min(), valid["height_cm"].max(), 100)
    ax.plot(xs, m * xs + b, color="red", linewidth=1.5,
            label=f"trend: y={m:.4f}x+{b:.3f}")

    corr = float(np.corrcoef(valid["height_cm"], valid["beta_10"])[0, 1])
    ax.set_xlabel("height (cm)")
    ax.set_ylabel("beta_10")
    ax.set_title(f"Child GT: beta_10 vs height  (r={corr:.3f})")
    ax.legend()
    fig.tight_layout()
    path = outdir / "beta10_vs_height_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_correlation_heatmap(corr_df: pd.DataFrame, outdir: Path, plt):
    # Subset: only beta_10 row and all feature cols
    import matplotlib.colors as mcolors

    row = corr_df[corr_df["beta"] == "beta_10"].copy()
    if len(row) == 0:
        return

    feat_cols = [c for c in row.columns if c.startswith("corr_")]
    vals = row[feat_cols].values[0].reshape(1, -1)
    labels = [c.replace("corr_", "") for c in feat_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 2.5))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels(["beta_10"])
    ax.set_title("Pearson correlation: beta_10 vs features")
    for j, v in enumerate(vals[0]):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    path = outdir / "beta_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_pca_variance(evr: np.ndarray, outdir: Path, plt):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(evr) + 1), evr * 100, color="#4C72B0", edgecolor="white")
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100,
            color="red", marker="o", markersize=4, label="cumulative %")
    ax.set_xlabel("PC")
    ax.set_ylabel("explained variance (%)")
    ax.set_title("PCA of beta_0..10 — explained variance")
    ax.legend()
    ax.set_xticks(range(1, len(evr) + 1))
    fig.tight_layout()
    path = outdir / "beta_pca_variance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_pca_scatter(proj_df: pd.DataFrame, df: pd.DataFrame, outdir: Path, plt):
    if "PC1" not in proj_df.columns or "PC2" not in proj_df.columns:
        return
    merged = proj_df.join(df[["height_cm", "beta_10"]], how="left")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label in [
        (axes[0], "height_cm", "height (cm)"),
        (axes[1], "beta_10",   "beta_10"),
    ]:
        valid = merged[[col, "PC1", "PC2"]].dropna()
        sc = ax.scatter(valid["PC1"], valid["PC2"],
                        c=valid[col], cmap="viridis",
                        alpha=0.6, s=20, edgecolors="none")
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA scatter coloured by {label}")

    fig.tight_layout()
    path = outdir / "beta_pca_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")



# ── NEW plots ────────────────────────────────────────────────────────────────

def plot_partial_correlation_bar(partial_df: pd.DataFrame, outdir: Path, plt):
    """
    Side-by-side bar chart: simple Pearson r vs height-controlled partial r
    for each feature, for beta_10.
    """
    if partial_df.empty or "pearson_r" not in partial_df.columns:
        return

    sub = partial_df.dropna(subset=["pearson_r", "partial_r_height_controlled"])
    if len(sub) == 0:
        return

    features = sub["feature"].tolist()
    x        = np.arange(len(features))
    width    = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(features) * 1.3), 5))
    bars1 = ax.bar(x - width / 2, sub["pearson_r"].values,
                   width, label="Pearson r (simple)", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, sub["partial_r_height_controlled"].values,
                   width, label="Partial r (height-controlled)", color="#DD8452", edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Correlation with beta_10")
    ax.set_title("beta_10 vs features: simple Pearson r vs height-controlled partial r")
    ax.legend()
    fig.tight_layout()
    path = outdir / "beta10_partial_correlation_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_residual_vs_features(df: pd.DataFrame, partial_df: pd.DataFrame,
                               outdir: Path, plt, top_n: int = 4):
    """
    Scatter beta_10_residual (height-removed) vs the top_n features ranked by
    |partial_r_height_controlled|.  Helps visually confirm partial correlations.
    """
    if partial_df.empty or "height_cm" not in df.columns:
        return

    valid_h = df[["beta_10", "height_cm"]].dropna()
    if len(valid_h) < 5:
        return

    b10_resid = _residualise(valid_h["beta_10"].values, valid_h["height_cm"].values)
    resid_idx = valid_h.index

    ranked = partial_df.dropna(subset=["partial_r_height_controlled"]) \
                       .reindex(partial_df["partial_r_height_controlled"].abs()
                                .sort_values(ascending=False).index)
    top_feats = [r for r in ranked["feature"].tolist()
                 if r in df.columns][:top_n]

    if not top_feats:
        return

    ncols = min(2, len(top_feats))
    nrows = (len(top_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).ravel()

    for idx, feat in enumerate(top_feats):
        ax = axes[idx]
        # Align residuals with feature values
        valid = df.loc[resid_idx, [feat]].dropna()
        common_idx = resid_idx.intersection(valid.index)
        if len(common_idx) < 3:
            ax.set_visible(False)
            continue

        b10_r = b10_resid[resid_idx.get_indexer(common_idx)]
        feat_r = _residualise(df.loc[common_idx, feat].values,
                               df.loc[common_idx, "height_cm"].values
                               if "height_cm" in df.columns else np.zeros(len(common_idx)))

        partial_r_val = partial_df.set_index("feature")["partial_r_height_controlled"].get(feat, np.nan)
        ax.scatter(feat_r, b10_r, alpha=0.55, s=20, edgecolors="none", color="#4C72B0")
        m, b = np.polyfit(feat_r, b10_r, 1)
        xs = np.linspace(feat_r.min(), feat_r.max(), 80)
        ax.plot(xs, m * xs + b, color="red", linewidth=1.5)
        ax.set_xlabel(f"{feat} (height-residualised)")
        ax.set_ylabel("beta_10 (height-residualised)")
        ax.set_title(f"partial r = {partial_r_val:+.3f}" if not np.isnan(partial_r_val) else feat)

    for idx in range(len(top_feats), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("beta_10 vs features — both height-residualised  (top partial correlations)",
                 fontsize=10)
    fig.tight_layout()
    path = outdir / "beta10_residual_vs_feature_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_group_comparison(group_summary_df: pd.DataFrame, outdir: Path, plt,
                          top_n: int = 6):
    """
    Grouped bar chart showing the mean of the top_n features (by Cohen's d)
    across the four high/low beta_10 × norm groups.
    """
    if group_summary_df.empty:
        return

    ranked = group_summary_df.dropna(subset=["cohens_d_high_vs_low_b10"]) \
                              .sort_values("cohens_d_high_vs_low_b10", key=abs,
                                           ascending=False)
    top_feats = ranked["feature"].tolist()[:top_n]
    if not top_feats:
        return

    group_cols = [c for c in group_summary_df.columns if c.endswith("_mean")
                  and not c.startswith("height")]
    group_names = [c.replace("_mean", "") for c in group_cols]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    x     = np.arange(len(top_feats))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(top_feats) * 1.8), 5))
    for gi, (gcol, gname, color) in enumerate(zip(group_cols, group_names, colors)):
        vals = group_summary_df.set_index("feature").reindex(top_feats)[gcol].values
        offset = (gi - len(group_cols) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=gname, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("feature mean")
    ax.set_title(f"Group feature comparison — top {top_n} features by Cohen's d")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "beta10_group_comparison_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse beta_10 from child GT feature CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="./analysis_outputs/child_gt_beta_features.csv",
        help="Path to child_gt_beta_features.csv from step1",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./analysis_outputs",
        help="Directory to write outputs (default: ./analysis_outputs)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}\n"
                 "Run step1_extract_child_gt_features.py first.")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples from {csv_path}")

    # Verify expected columns
    missing = [c for c in ["beta_10", "height_cm"] if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing expected columns: {missing}")

    # Detect optional features once; pass to all functions for consistency.
    print("\n[INFO] Detecting available features ...")
    ratio_features = get_analysis_features(df)
    print(f"  Core features  : {RATIO_FEATURES}")
    optional_found = [f for f in ALL_OPTIONAL_FEATURES if f in df.columns]
    if optional_found:
        print(f"  Optional found : {optional_found}")
    else:
        print("  Optional found : (none — head/neck features not in CSV)")

    # ── A. beta_10 summary ───────────────────────────────────────────────
    print("\n[A] beta_10 summary ...")
    summary = compute_beta10_summary(df)
    out = outdir / "beta10_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    print(f"  mean={summary['mean']:.4f}  std={summary['std']:.4f}  "
          f"min={summary['min']:.4f}  max={summary['max']:.4f}")

    # ── B. beta_10 by height bin ─────────────────────────────────────────
    print("\n[B] beta_10 by height bin ...")
    bin_df = compute_height_bin_stats(df)
    out = outdir / "beta10_by_height_bin.csv"
    bin_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(bin_df[["height_bin", "n", "height_cm_mean", "beta10_mean", "beta10_std"]].to_string(index=False))

    # ── C. beta vs feature correlations ─────────────────────────────────
    print("\n[C] beta vs feature correlations ...")
    corr_df = compute_beta_feature_correlations(df, ratio_features=ratio_features)
    out = outdir / "beta_feature_correlations.csv"
    corr_df.to_csv(out, index=False)
    print(f"  Saved: {out}")

    # Print beta_10 row
    row10 = corr_df[corr_df["beta"] == "beta_10"]
    if len(row10) > 0:
        print("\n  beta_10 correlations with features:")
        for col in [c for c in row10.columns if c.startswith("corr_")]:
            val = row10[col].values[0]
            print(f"    {col.replace('corr_',''):40s} {val:+.4f}")

    # ── D. beta covariance ───────────────────────────────────────────────
    print("\n[D] beta covariance ...")
    cov_df = compute_beta_covariance(df)
    out = outdir / "beta_covariance.csv"
    cov_df.to_csv(out)
    print(f"  Saved: {out}")
    print("  Variance per beta (diagonal):")
    for col in cov_df.columns:
        print(f"    {col}: {cov_df.loc[col, col]:.6f}")

    # ── E. Role analysis ─────────────────────────────────────────────────
    print("\n[E] beta_10 role analysis (high/low beta_10 vs beta_0:9 norm) ...")
    if "beta_0to9_norm" in df.columns:
        role_df, effect_df = compute_role_analysis(df, ratio_features=ratio_features)
        out = outdir / "beta10_role_analysis.csv"
        role_df.to_csv(out, index=False)
        print(f"  Saved: {out}")
        print(role_df[["group", "n", "beta10_mean", "norm_mean", "height_cm_mean"]].to_string(index=False))
    else:
        print("  [SKIP] beta_0to9_norm column not found.")
        role_df  = pd.DataFrame()
        effect_df = pd.DataFrame()

    # ── F. PCA ───────────────────────────────────────────────────────────
    print("\n[F] PCA on beta_0..10 ...")
    evr, comp_df, proj_df = compute_beta_pca(df)
    if evr is not None:
        pca_summary = {
            "explained_variance_ratio": evr.tolist(),
            "cumulative_variance_ratio": np.cumsum(evr).tolist(),
        }
        out = outdir / "beta_pca_summary.json"
        with open(out, "w") as f:
            json.dump(pca_summary, f, indent=2)
        print(f"  Saved: {out}")
        print("  Explained variance ratio per PC:")
        for i, v in enumerate(evr):
            print(f"    PC{i+1}: {v:.4f}  (cumulative: {np.cumsum(evr)[i]:.4f})")

        out_comp = outdir / "beta_pca_components.csv"
        comp_df.to_csv(out_comp)
        print(f"  Saved: {out_comp}")

        # Check whether beta_10 contributes strongly to any PC
        print("\n  beta_10 loadings across PCs:")
        if "beta_10" in comp_df.columns:
            for pc in comp_df.index:
                print(f"    {pc}: {comp_df.loc[pc, 'beta_10']:+.4f}")
    else:
        print("  [SKIP] sklearn not available. Install with: pip install scikit-learn")

    # ── G. Existing plots ────────────────────────────────────────────────
    if not args.no_plots:
        plt = try_matplotlib()
        if plt is not None:
            print("\n[G] Generating existing plots ...")
            plot_beta10_histogram(df, outdir, plt)
            plot_beta10_vs_height(df, outdir, plt)
            plot_correlation_heatmap(corr_df, outdir, plt)
            if evr is not None:
                plot_pca_variance(evr, outdir, plt)
                plot_pca_scatter(proj_df, df, outdir, plt)
        else:
            print("\n[G] Plots skipped: matplotlib not available.")

    # ════════════════════════════════════════════════════════════════════
    # NEW SECTIONS (H, I, J) — backward compatible; only add new files
    # ════════════════════════════════════════════════════════════════════

    # ── H. Height-controlled partial correlations ────────────────────────
    print("\n[H] Height-controlled partial correlations for beta_10 ...")
    partial_df = compute_partial_correlations(df, ratio_features=ratio_features)
    if not partial_df.empty:
        out = outdir / "beta10_partial_correlations.csv"
        partial_df.to_csv(out, index=False)
        print(f"  Saved: {out}")
        print("\n  Simple Pearson r  vs  height-controlled partial r  (for beta_10):")
        print(f"  {'feature':38s}  {'pearson_r':>10}  {'partial_r':>10}  {'r_drop':>8}")
        for _, row in partial_df.iterrows():
            print(f"  {row['feature']:38s}  "
                  f"{row['pearson_r']:+10.4f}  "
                  f"{row['partial_r_height_controlled']:+10.4f}  "
                  f"{row['r_drop']:+8.4f}")
        print("\n  NOTE: r_drop > 0 means height explains part of the simple correlation.")
        print("        Partial r near zero means the association vanishes once height")
        print("        is controlled — i.e. beta_10 acts mainly as a height axis.")
        print("        Partial r still large means beta_10 encodes that proportion")
        print("        independently of height (genuine childness axis).")
    else:
        print("  [SKIP] Not enough data or missing height_cm column.")
        partial_df = pd.DataFrame()

    # ── I. Regression analysis ───────────────────────────────────────────
    print("\n[I] Regression analysis: beta_10 ~ height / features ...")
    reg_summary, coef_df = compute_beta10_regression(df, ratio_features=ratio_features)
    if reg_summary:
        out = outdir / "beta10_regression_summary.json"
        with open(out, "w") as f:
            json.dump(reg_summary, f, indent=2)
        print(f"  Saved: {out}")

        out = outdir / "beta10_regression_coefficients.csv"
        coef_df.to_csv(out, index=False)
        print(f"  Saved: {out}")

        m1 = reg_summary.get("model_1_height_only", {})
        m2 = reg_summary.get("model_2_height_and_features", {})
        r2_gain = reg_summary.get("r2_gain_from_features", float("nan"))
        print(f"\n  Model 1 (height only):          R²={m1.get('r2', float('nan')):.4f}")
        print(f"  Model 2 (height + features):    R²={m2.get('r2', float('nan')):.4f}")
        print(f"  R² gain from adding features:   {r2_gain:.4f}")
        if not np.isnan(r2_gain):
            if r2_gain > 0.10:
                print("  → Proportion features add substantial explanatory power beyond height.")
            elif r2_gain > 0.03:
                print("  → Proportion features add modest explanatory power beyond height.")
            else:
                print("  → Height alone explains most of beta_10 variance.")

        # Print standardised betas for model 2 (most informative)
        std_betas = m2.get("standardised_betas", {})
        if std_betas:
            print("\n  Standardised betas (model 2 — height + features):")
            for feat, sb in sorted(std_betas.items(),
                                   key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0,
                                   reverse=True):
                if feat == "intercept":
                    continue
                print(f"    {feat:38s}  {sb:+.4f}")
    else:
        print("  [SKIP] Not enough valid data for regression.")

    # ── J. Group feature summary with Cohen's d ──────────────────────────
    print("\n[J] Group feature summary (beta10 × norm groups, with Cohen's d) ...")
    if "beta_0to9_norm" in df.columns:
        group_feat_df = compute_group_feature_summary(df, ratio_features=ratio_features)
        if not group_feat_df.empty:
            out = outdir / "beta10_group_feature_summary.csv"
            group_feat_df.to_csv(out, index=False)
            print(f"  Saved: {out}")
            print("\n  Top features by Cohen's d (high_b10_low_norm vs low_b10_high_norm):")
            top = group_feat_df.dropna(subset=["cohens_d_high_vs_low_b10"]) \
                               .sort_values("cohens_d_high_vs_low_b10", key=abs, ascending=False)
            print(f"  {'feature':38s}  {'cohens_d':>10}")
            for _, row in top.head(10).iterrows():
                print(f"  {row['feature']:38s}  {row['cohens_d_high_vs_low_b10']:+10.3f}")
    else:
        print("  [SKIP] beta_0to9_norm not in CSV.")
        group_feat_df = pd.DataFrame()

    # ── K. New plots ─────────────────────────────────────────────────────
    if not args.no_plots:
        plt = try_matplotlib()
        if plt is not None:
            print("\n[K] Generating new plots ...")
            if not partial_df.empty:
                plot_partial_correlation_bar(partial_df, outdir, plt)
                plot_residual_vs_features(df, partial_df, outdir, plt)
            if not group_feat_df.empty:
                plot_group_comparison(group_feat_df, outdir, plt)
        else:
            print("\n[K] New plots skipped: matplotlib not available.")

    print("\n[INFO] Step 2 complete. Outputs written to:", outdir)


if __name__ == "__main__":
    main()
