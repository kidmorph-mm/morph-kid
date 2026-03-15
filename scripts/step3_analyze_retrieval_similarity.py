#!/usr/bin/env python3
"""
step3_analyze_retrieval_similarity.py
=======================================
Evaluate the proposed weighted standardised L2 retrieval metric on child GT
data, and optionally compare it against the existing heuristic scorer using
real adult queries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  IMPORTANT: why score_candidate() cannot be evaluated on child-vs-child pairs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  score_candidate(adult_feats, target) is designed for adult-to-child scoring.
  Its core term computes:
      scale_ratio = target_h / adult_h
  and penalises ratios far from ~0.72 (the typical adult→child height shrink).
  It also applies group-specific childness bonuses and sub-bin penalties that
  only make sense when the *query* is an adult and the *target* is a child.

  Applying it to child-vs-child pairs (query=child row, target=child row)
  makes scale_ratio ≈ 1.0 for all pairs, collapsing most of its signal and
  making any comparison with the new metric meaningless and misleading.

  This script therefore supports two modes:

  MODE 1 — child-GT-only analysis (default, no --adult-csv):
      ● pairwise child-child retrieval distance distribution
      ● nearest-neighbour retrieval examples within child GT
      ● feature-std normalisation statistics (σᵢ)
      ● NO heuristic comparison (intentionally absent)

  MODE 2 — adult-query comparison (requires --adult-csv):
      ● for each adult query, retrieve top-k child GT with the new metric
      ● rank the same child GT with score_candidate() (correct usage)
      ● measure Spearman ρ and top-k set overlap on adult→child pairs
      ● adult CSV is produced by running step1 with --gt-dir pointing to
        your adult pkl directory

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Proposed distance:
    d(a, c) = sum_i  w_i * ((f_i(a) - f_i(c)) / sigma_i)^2

where:
    f_i     = ratio feature  (NOT absolute height)
    sigma_i = std of feature i over the child GT population
    w_i     = configurable importance weight

Reuses from the pipeline:
  - FEATURE_KEYS, CANDIDATE_SCORE_WEIGHTS, score_candidate
    from robust_child_shape_opt_upperbody_200.py

Always produced:
  - child_gt_feature_stats.csv         : per-feature mean/std (sigma_i)
  - pairwise_retrieval_distances.npz   : N×N child-child distance matrix
  - pairwise_distance_stats.json       : distribution summary
  - nearest_neighbor_examples.csv      : top-5 NN per child GT sample
  - retrieval_weights_used.json        : weights used in this run

Produced only with --adult-csv (MODE 2):
  - adult_gt_retrieval_results.csv     : per adult query: top-k by new metric
  - adult_heuristic_results.csv        : per adult query: top-k by heuristic
  - adult_rank_agreement.csv           : Spearman rho and top-k overlap per query
  - adult_rank_agreement_summary.json  : mean/std of agreement stats

Optional plots (require matplotlib):
  - pairwise_distance_histogram.png
  - feature_std_bar.png
  - adult_rank_agreement_histogram.png  (MODE 2 only)
  - adult_retrieval_scatter.png         (MODE 2 only)

Usage — MODE 1 (child GT only):
  python step3_analyze_retrieval_similarity.py \\
      --csv ./analysis_outputs/child_gt_beta_features.csv \\
      --outdir ./analysis_outputs

Usage — MODE 2 (adult queries, full comparison):
  python step3_analyze_retrieval_similarity.py \\
      --csv  ./analysis_outputs/child_gt_beta_features.csv \\
      --adult-csv ./analysis_outputs/adult_beta_features.csv \\
      --outdir ./analysis_outputs

  The adult CSV is produced by running step1 on adult pkl files:
    python step1_extract_child_gt_features.py \\
        --gt-dir /path/to/adult_pkls \\
        --outdir ./analysis_outputs/adult \\
        && mv ./analysis_outputs/adult/child_gt_beta_features.csv \\
              ./analysis_outputs/adult_beta_features.csv

  Optional:
    --height-weight 0.15   add weak height penalty (design spec: <= 0.2)
    --weights-json w.json  override importance weights from a JSON file
    --topk 10              number of neighbours to retrieve
    --max-adult-queries 50 cap on queries for the O(A*C) heuristic loop
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Import from the existing pipeline ────────────────────────────────────────
try:
    from robust_child_shape_opt_upperbody_200 import (
        FEATURE_KEYS,
        CANDIDATE_SCORE_WEIGHTS,
        score_candidate,
    )
    _PIPELINE_IMPORTED = True
except ImportError:
    _PIPELINE_IMPORTED = False
    warnings.warn(
        "Could not import from robust_child_shape_opt_upperbody_200.py.\n"
        "MODE 2 (heuristic comparison) will be unavailable.\n"
        "FEATURE_KEYS falling back to hardcoded list."
    )
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
    CANDIDATE_SCORE_WEIGHTS = {}
    score_candidate = None


# ── Retrieval feature set (ratio features only, no absolute height) ───────────
RETRIEVAL_FEATURES = [
    "shoulder_width_ratio",
    "pelvis_width_ratio",
    "torso_height_ratio",
    "arm_length_ratio",
    "leg_length_ratio",
    "thigh_ratio",
    "shank_ratio",
]

# Default importance weights (design spec).
# height_cm weight is 0.0 by default (excluded from retrieval).
DEFAULT_RETRIEVAL_WEIGHTS: Dict[str, float] = {
    "shoulder_width_ratio": 1.3,
    "pelvis_width_ratio":   1.3,
    "torso_height_ratio":   1.4,
    "arm_length_ratio":     1.2,
    "leg_length_ratio":     1.5,
    "thigh_ratio":          1.0,
    "shank_ratio":          1.0,
    "height_cm":            0.0,
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def compute_feature_stats(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Mean, std, min, max of each feature over a dataset."""
    rows = []
    for f in features:
        if f not in df.columns:
            rows.append({"feature": f, "n": 0, "mean": np.nan,
                         "std": np.nan, "min": np.nan, "max": np.nan})
            continue
        vals = df[f].dropna()
        rows.append({
            "feature": f,
            "n":    int(len(vals)),
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "min":  float(vals.min()),
            "max":  float(vals.max()),
        })
    return pd.DataFrame(rows)


def build_feature_matrix(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Return (N, F) float32 matrix; NaN filled with column mean."""
    mat = df[features].copy()
    for col in mat.columns:
        mat[col] = mat[col].fillna(mat[col].mean())
    return mat.values.astype(np.float32)


# ── Distance functions ────────────────────────────────────────────────────────

def compute_pairwise_distances(
    feat_matrix: np.ndarray,   # (N, F)
    sigmas: np.ndarray,        # (F,)
    weights: np.ndarray,       # (F,)
) -> np.ndarray:
    """
    Full N×N symmetric matrix:
        d(i,j) = sum_k  w_k * ((f_k(i) - f_k(j)) / sigma_k)^2
    """
    normed   = feat_matrix / (sigmas[None] + 1e-8)
    weighted = normed * np.sqrt(weights[None])
    sq = np.sum(weighted ** 2, axis=1)
    D  = sq[:, None] + sq[None, :] - 2.0 * (weighted @ weighted.T)
    return np.clip(D, 0.0, None).astype(np.float32)


def compute_cross_distances(
    feat_adult: np.ndarray,    # (A, F)
    feat_child: np.ndarray,    # (C, F)
    sigmas: np.ndarray,        # (F,)  child GT std
    weights: np.ndarray,       # (F,)
) -> np.ndarray:
    """
    A×C cross-distance matrix:
        D[a, c] = weighted standardised L2 between adult a and child GT c.
    Sigma is always from child GT, so distances are calibrated to the
    child GT feature distribution.
    """
    a_norm = feat_adult / (sigmas[None] + 1e-8)
    c_norm = feat_child / (sigmas[None] + 1e-8)
    a_w = a_norm * np.sqrt(weights[None])
    c_w = c_norm * np.sqrt(weights[None])
    sq_a = np.sum(a_w ** 2, axis=1)
    sq_c = np.sum(c_w ** 2, axis=1)
    D = sq_a[:, None] + sq_c[None, :] - 2.0 * (a_w @ c_w.T)
    return np.clip(D, 0.0, None).astype(np.float32)


# ── Heuristic: convert CSV rows to score_candidate() dicts ───────────────────

def row_to_adult_feats(row: pd.Series) -> Dict[str, float]:
    """
    Build adult_feats dict for score_candidate().
    height_canonical is in METRES (matches pipeline convention).
    """
    d: Dict[str, float] = {}
    if "height_canonical" in row.index:
        d["height_canonical"] = float(row["height_canonical"])
    elif "height_cm" in row.index:
        d["height_canonical"] = float(row["height_cm"]) / 100.0
    for k in RETRIEVAL_FEATURES:
        if k in row.index:
            d[k] = float(row[k])
    return d


def row_to_child_target(row: pd.Series) -> Dict:
    """
    Build child target dict for score_candidate().
    group="core" and height_bin="analysis" are placeholder values;
    they are consistent with the prototype CSVs used in the real pipeline
    and sufficient for ranking comparison purposes.
    """
    d = row_to_adult_feats(row)
    d["target_height_cm"] = d.get("height_canonical", 0.0) * 100.0
    d["group"]      = "core"
    d["height_bin"] = "analysis"
    return d


# ── MODE 1: child-child nearest neighbours ───────────────────────────────────

def nearest_neighbors_child(
    D: np.ndarray,
    sample_ids: List[str],
    k: int = 5,
) -> pd.DataFrame:
    rows = []
    N = D.shape[0]
    for i in range(N):
        d_row = D[i].copy()
        d_row[i] = np.inf
        nn_idx = np.argsort(d_row)[:k]
        for rank, j in enumerate(nn_idx, start=1):
            rows.append({
                "query_id": sample_ids[i],
                "nn_rank":  rank,
                "nn_id":    sample_ids[j],
                "distance": float(D[i, j]),
            })
    return pd.DataFrame(rows)


# ── MODE 2: adult-query retrieval + heuristic comparison ─────────────────────

def retrieve_adult_topk(
    D_cross: np.ndarray,
    adult_ids: List[str],
    child_ids: List[str],
    child_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Top-k child GT per adult query by new metric."""
    rows = []
    for ai in range(len(adult_ids)):
        d_row = D_cross[ai]
        top_k = np.argsort(d_row)[:k]
        for rank, ci in enumerate(top_k, start=1):
            r = {
                "adult_id":    adult_ids[ai],
                "nn_rank":     rank,
                "child_id":    child_ids[ci],
                "new_distance": float(d_row[ci]),
            }
            for feat in RETRIEVAL_FEATURES + ["height_cm"]:
                if feat in child_df.columns:
                    r[f"child_{feat}"] = float(child_df.iloc[ci].get(feat, np.nan))
            rows.append(r)
    return pd.DataFrame(rows)


def heuristic_rank_all_children(
    adult_row: pd.Series,
    child_df: pd.DataFrame,
) -> List[Tuple[int, float]]:
    """
    Score every child GT sample against one adult query with score_candidate().
    Returns list of (child_index, score) sorted ascending (lower = more similar).
    This is the correct usage: adult query, child GT targets.
    """
    adult_feats = row_to_adult_feats(adult_row)
    scored = []
    for ci, (_, child_row) in enumerate(child_df.iterrows()):
        target = row_to_child_target(child_row)
        try:
            s = float(score_candidate(adult_feats, target))
        except Exception:
            s = float("inf")
        scored.append((ci, s))
    scored.sort(key=lambda x: x[1])
    return scored


def compute_rank_agreement(
    D_cross: np.ndarray,
    adult_df: pd.DataFrame,
    adult_ids: List[str],
    child_df: pd.DataFrame,
    child_ids: List[str],
    k: int,
    max_adults: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each adult query (up to max_adults):
      - full child ranking by new metric
      - full child ranking by heuristic (score_candidate, correct usage)
      - Spearman rho over full ranking
      - top-k set overlap

    Returns (agreement_df, heuristic_topk_df).
    """
    n_adults = min(len(adult_ids), max_adults)
    agreement_rows = []
    heuristic_rows = []
    n_children = len(child_ids)

    for ai in range(n_adults):
        adult_row = adult_df.iloc[ai]
        adult_id  = adult_ids[ai]

        # New metric full ranking
        new_order = np.argsort(D_cross[ai])   # child indices sorted by distance

        # Heuristic full ranking
        heur_scored  = heuristic_rank_all_children(adult_row, child_df)
        heur_order   = [ci for ci, _ in heur_scored]
        heur_scores  = {ci: s for ci, s in heur_scored}

        # Build rank arrays indexed by child index
        new_rank_arr  = np.empty(n_children, dtype=np.float32)
        heur_rank_arr = np.empty(n_children, dtype=np.float32)
        for r, ci in enumerate(new_order):
            new_rank_arr[ci] = r
        for r, ci in enumerate(heur_order):
            heur_rank_arr[ci] = r

        # Spearman rho
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(new_rank_arr, heur_rank_arr)
            rho, pval = float(rho), float(pval)
        except ImportError:
            d2   = float(np.sum((new_rank_arr - heur_rank_arr) ** 2))
            rho  = float(1.0 - 6.0 * d2 / (n_children * (n_children ** 2 - 1)))
            pval = float("nan")

        # Top-k overlap
        new_top_k  = set(new_order[:k].tolist())
        heur_top_k = set(heur_order[:k])
        overlap    = len(new_top_k & heur_top_k)

        agreement_rows.append({
            "adult_id":               adult_id,
            "spearman_rho":           rho,
            "spearman_pval":          pval,
            f"top{k}_overlap":        overlap,
            f"top{k}_overlap_frac":   float(overlap) / k,
        })

        # Save heuristic top-k for this adult
        for rank, ci in enumerate(heur_order[:k], start=1):
            r = {
                "adult_id":        adult_id,
                "nn_rank":         rank,
                "child_id":        child_ids[ci],
                "heuristic_score": float(heur_scores[ci]),
            }
            for feat in RETRIEVAL_FEATURES + ["height_cm"]:
                if feat in child_df.columns:
                    r[f"child_{feat}"] = float(child_df.iloc[ci].get(feat, np.nan))
            heuristic_rows.append(r)

    return pd.DataFrame(agreement_rows), pd.DataFrame(heuristic_rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_distance_histogram(D: np.ndarray, outdir: Path, plt, title_suffix: str = ""):
    upper = D[np.triu_indices_from(D, k=1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(upper, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(float(np.median(upper)), color="red",    linestyle="--",
               label=f"median={float(np.median(upper)):.3f}")
    ax.axvline(float(upper.mean()),     color="orange", linestyle=":",
               label=f"mean={float(upper.mean()):.3f}")
    ax.set_xlabel("Retrieval distance (weighted standardised L2)")
    ax.set_ylabel("pair count")
    ax.set_title(f"Pairwise retrieval distance distribution{title_suffix}")
    ax.legend()
    fig.tight_layout()
    path = outdir / "pairwise_distance_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_feature_std(stats_df: pd.DataFrame, outdir: Path, plt):
    sub = stats_df.dropna(subset=["std"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(sub))
    ax.bar(x, sub["std"].values, color="#4C72B0", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["feature"].tolist(), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("std over child GT")
    ax.set_title("Per-feature std (σᵢ) — denominator for retrieval distance normalisation")
    fig.tight_layout()
    path = outdir / "feature_std_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_rank_agreement_histogram(agree_df: pd.DataFrame, outdir: Path, plt, k: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, xlabel, title in [
        (axes[0], "spearman_rho",          "Spearman ρ",
         "Rank agreement: Spearman ρ (full child ranking)"),
        (axes[1], f"top{k}_overlap_frac",  f"Top-{k} overlap fraction",
         f"Top-{k} set overlap fraction"),
    ]:
        vals = agree_df[col].dropna()
        ax.hist(vals, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.axvline(float(vals.mean()), color="red", linestyle="--",
                   label=f"mean={float(vals.mean()):.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("adult query count")
        ax.set_title(title)
        ax.legend()
    fig.suptitle(
        "New metric vs heuristic score_candidate() — rank agreement on adult→child pairs",
        fontsize=10,
    )
    fig.tight_layout()
    path = outdir / "adult_rank_agreement_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_adult_retrieval_scatter(
    adult_df: pd.DataFrame,
    child_df: pd.DataFrame,
    D_cross: np.ndarray,
    adult_ids: List[str],
    child_ids: List[str],
    outdir: Path,
    plt,
    n_adults: int = 8,
    k: int = 10,
):
    """
    For the first n_adults queries, scatter child height vs torso_height_ratio
    for the top-k retrieved, with adult value shown as a horizontal reference.
    """
    feat = "torso_height_ratio"
    if feat not in adult_df.columns or feat not in child_df.columns:
        return
    n = min(n_adults, len(adult_ids))
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).ravel()
    for idx in range(n):
        ax    = axes[idx]
        ai    = idx
        adult_val = float(adult_df.iloc[ai].get(feat, np.nan))
        top_k_ci  = np.argsort(D_cross[ai])[:k]
        child_vals   = [float(child_df.iloc[ci].get(feat, np.nan))   for ci in top_k_ci]
        child_heights= [float(child_df.iloc[ci].get("height_cm", np.nan)) for ci in top_k_ci]
        dists        = D_cross[ai][top_k_ci]
        sc = ax.scatter(child_heights, child_vals, c=dists, cmap="viridis_r",
                        s=50, edgecolors="k", linewidths=0.4)
        ax.axhline(adult_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"adult {feat[:8]}={adult_val:.3f}")
        ax.set_xlabel("child height_cm", fontsize=8)
        ax.set_ylabel(feat, fontsize=8)
        ax.set_title(f"Query: {adult_ids[ai][:18]}", fontsize=8)
        ax.legend(fontsize=7)
    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.colorbar(sc, ax=axes[n - 1], label="distance")
    fig.suptitle(f"Top-{k} retrieved child GT per adult query  (colour = distance)", fontsize=10)
    fig.tight_layout()
    path = outdir / "adult_retrieval_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse retrieval similarity for child GT data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="./analysis_outputs/child_gt_beta_features.csv",
        help="Path to child_gt_beta_features.csv from step1",
    )
    parser.add_argument(
        "--adult-csv",
        default=None,
        help=(
            "Adult feature CSV (same columns as child CSV) produced by running "
            "step1 on adult pkl files.  Required for MODE 2 (heuristic comparison)."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="./analysis_outputs",
        help="Output directory (default: ./analysis_outputs)",
    )
    parser.add_argument(
        "--weights-json",
        default=None,
        help="JSON file overriding DEFAULT_RETRIEVAL_WEIGHTS.",
    )
    parser.add_argument(
        "--height-weight",
        type=float,
        default=0.0,
        help="Weight for height_cm (default: 0.0 = excluded; design spec recommends <= 0.2)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of nearest neighbours to retrieve (default: 10)",
    )
    parser.add_argument(
        "--max-adult-queries",
        type=int,
        default=50,
        help="Cap on adult queries for rank-agreement loop (default: 50)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    child_csv = Path(args.csv)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not child_csv.exists():
        sys.exit(f"[ERROR] Child GT CSV not found: {child_csv}\n"
                 "Run step1_extract_child_gt_features.py first.")

    child_df  = pd.read_csv(child_csv)
    child_ids = child_df["sample_id"].astype(str).tolist()
    print(f"[INFO] Loaded {len(child_df)} child GT samples from {child_csv}")

    # Determine mode
    mode2 = args.adult_csv is not None
    if mode2:
        adult_path = Path(args.adult_csv)
        if not adult_path.exists():
            sys.exit(f"[ERROR] Adult CSV not found: {adult_path}")
        adult_df  = pd.read_csv(adult_path)
        adult_ids = adult_df["sample_id"].astype(str).tolist()
        print(f"[INFO] Loaded {len(adult_df)} adult samples from {adult_path}")
        print("[INFO] MODE 2: adult-query retrieval + heuristic comparison enabled")
    else:
        adult_df  = None
        adult_ids = []
        print("[INFO] MODE 1: child-GT-only analysis")
        print("[INFO] NOTE:  heuristic comparison requires --adult-csv (see docstring)")

    # ── Weights ──────────────────────────────────────────────────────────
    weights_dict = dict(DEFAULT_RETRIEVAL_WEIGHTS)
    weights_dict["height_cm"] = args.height_weight

    if args.weights_json:
        with open(args.weights_json) as f:
            weights_dict.update(json.load(f))
        print(f"[INFO] Weight overrides loaded from {args.weights_json}")

    features_in_use = [f for f in RETRIEVAL_FEATURES if f in child_df.columns]
    if args.height_weight > 0.0 and "height_cm" in child_df.columns:
        features_in_use.append("height_cm")

    print(f"[INFO] Retrieval features ({len(features_in_use)}): {features_in_use}")

    weights_used = {f: weights_dict.get(f, 1.0) for f in features_in_use}
    out = outdir / "retrieval_weights_used.json"
    with open(out, "w") as f:
        json.dump(weights_used, f, indent=2)
    print(f"[INFO] Weights saved: {out}")

    # ── A. Feature stats (sigma_i) ───────────────────────────────────────
    print("\n[A] Child GT feature statistics (σᵢ) ...")
    stats_df = compute_feature_stats(child_df, features_in_use)
    out = outdir / "child_gt_feature_stats.csv"
    stats_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(stats_df.to_string(index=False))

    sigmas  = stats_df.set_index("feature")["std"].reindex(features_in_use).values.astype(np.float32)
    w_array = np.array([weights_used[f] for f in features_in_use], dtype=np.float32)
    child_feat_mat = build_feature_matrix(child_df, features_in_use)

    # ── B. Child-child pairwise distances ────────────────────────────────
    N = len(child_ids)
    print(f"\n[B] Computing {N}×{N} child-child pairwise distances ...")
    D_child = compute_pairwise_distances(child_feat_mat, sigmas, w_array)
    out_npz = outdir / "pairwise_retrieval_distances.npz"
    np.savez_compressed(str(out_npz), D=D_child, sample_ids=np.array(child_ids))
    print(f"  Saved: {out_npz}")

    upper = D_child[np.triu_indices_from(D_child, k=1)]
    dist_stats = {
        "n_pairs": int(len(upper)),
        "mean":  float(upper.mean()),
        "std":   float(upper.std()),
        "min":   float(upper.min()),
        "p5":    float(np.percentile(upper, 5)),
        "p25":   float(np.percentile(upper, 25)),
        "p50":   float(np.percentile(upper, 50)),
        "p75":   float(np.percentile(upper, 75)),
        "p95":   float(np.percentile(upper, 95)),
        "max":   float(upper.max()),
    }
    out = outdir / "pairwise_distance_stats.json"
    with open(out, "w") as f:
        json.dump(dist_stats, f, indent=2)
    print(f"  Saved: {out}")
    print(f"  Distance summary: mean={dist_stats['mean']:.4f}  "
          f"p50={dist_stats['p50']:.4f}  p95={dist_stats['p95']:.4f}")

    # ── C. Nearest-neighbour retrieval within child GT ───────────────────
    print("\n[C] Nearest-neighbour retrieval within child GT (k=5) ...")
    nn_df = nearest_neighbors_child(D_child, child_ids, k=5)
    out = outdir / "nearest_neighbor_examples.csv"
    nn_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    for qid in child_ids[:3]:
        sub = nn_df[nn_df["query_id"] == qid]
        print(f"\n  Query: {qid}")
        print(sub[["nn_rank", "nn_id", "distance"]].to_string(index=False))

    # ── D. MODE 2: adult-query retrieval + heuristic comparison ──────────
    if mode2:
        if not _PIPELINE_IMPORTED or score_candidate is None:
            print("\n[D] [SKIP] score_candidate not importable — "
                  "ensure pipeline is on PYTHONPATH.")
        else:
            print(f"\n[D] MODE 2: adult→child retrieval + heuristic comparison ...")
            A = len(adult_ids)
            print(f"  {A} adult queries  ×  {N} child GT targets")

            adult_feat_mat = build_feature_matrix(adult_df, features_in_use)
            D_cross = compute_cross_distances(adult_feat_mat, child_feat_mat, sigmas, w_array)
            print(f"  Cross-distance matrix shape: {D_cross.shape}")

            # New metric top-k results
            retrieval_df = retrieve_adult_topk(
                D_cross, adult_ids, child_ids, child_df, k=args.topk
            )
            out = outdir / "adult_gt_retrieval_results.csv"
            retrieval_df.to_csv(out, index=False)
            print(f"  Saved: {out}")

            # Rank agreement
            n_q = min(args.max_adult_queries, A)
            print(f"\n  Rank agreement: {n_q} adult queries, "
                  f"each scored against {N} child samples with score_candidate() ...")
            agree_df, heur_topk_df = compute_rank_agreement(
                D_cross=D_cross,
                adult_df=adult_df,
                adult_ids=adult_ids,
                child_df=child_df,
                child_ids=child_ids,
                k=args.topk,
                max_adults=n_q,
            )

            out = outdir / "adult_rank_agreement.csv"
            agree_df.to_csv(out, index=False)
            print(f"  Saved: {out}")

            out = outdir / "adult_heuristic_results.csv"
            heur_topk_df.to_csv(out, index=False)
            print(f"  Saved: {out}")

            rho_mean = float(agree_df["spearman_rho"].mean())
            rho_std  = float(agree_df["spearman_rho"].std())
            ov_col   = f"top{args.topk}_overlap_frac"
            ov_mean  = float(agree_df[ov_col].mean())
            ov_std   = float(agree_df[ov_col].std())

            summary = {
                "n_adult_queries": int(len(agree_df)),
                "n_child_gt":      N,
                "topk":            args.topk,
                "spearman_rho_mean": rho_mean,
                "spearman_rho_std":  rho_std,
                f"top{args.topk}_overlap_frac_mean": ov_mean,
                f"top{args.topk}_overlap_frac_std":  ov_std,
            }
            out = outdir / "adult_rank_agreement_summary.json"
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved: {out}")

            print(f"\n  ── Rank agreement summary ──")
            print(f"  Spearman ρ:       mean={rho_mean:.4f}  std={rho_std:.4f}")
            print(f"  Top-{args.topk} overlap: mean={ov_mean:.3f}  std={ov_std:.3f}")

            if rho_mean < 0.4 or ov_mean < 0.4:
                print("  ⚠  Low agreement — metrics rank children very differently "
                      "for adult queries. Strongly supports replacing the heuristic.")
            elif rho_mean < 0.65 or ov_mean < 0.6:
                print("  ~  Partial agreement — metrics partially align but surface "
                      "meaningfully different candidates. "
                      "Consider the new metric as a principled replacement.")
            else:
                print("  ✓  High agreement — new metric is broadly consistent with "
                      "the heuristic on adult→child pairs.")

            if not args.no_plots:
                plt = try_matplotlib()
                if plt is not None:
                    plot_rank_agreement_histogram(agree_df, outdir, plt, k=args.topk)
                    plot_adult_retrieval_scatter(
                        adult_df, child_df, D_cross,
                        adult_ids, child_ids,
                        outdir, plt, n_adults=8, k=args.topk,
                    )

    # ── E. Plots (always) ────────────────────────────────────────────────
    if not args.no_plots:
        plt = try_matplotlib()
        if plt is not None:
            print("\n[E] Generating plots ...")
            plot_distance_histogram(
                D_child, outdir, plt,
                title_suffix=" (child GT × child GT)",
            )
            plot_feature_std(stats_df, outdir, plt)
        else:
            print("\n[E] Plots skipped: matplotlib not available.")

    print("\n[INFO] Step 3 complete. Outputs written to:", outdir)


if __name__ == "__main__":
    main()
