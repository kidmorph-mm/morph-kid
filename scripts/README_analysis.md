# Child GT Analysis Pipeline

Analyses `beta[10]` and the proposed retrieval similarity metric for the
adult-to-child SMPL-X shape transformation project.

---

## Overview

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `step1_extract_child_gt_features.py` | Load child GT `.pkl` files, run SMPL-X kid in canonical pose, extract betas + features | `child_gt_beta_features.csv` |
| `step2_analyze_beta10.py` | Analyse `beta[10]`: stats, height-bin breakdown, feature correlations, covariance, PCA | `beta10_summary.json`, `beta10_by_height_bin.csv`, `beta_feature_correlations.csv`, `beta_covariance.csv`, plots |
| `step3_analyze_retrieval_similarity.py` | Evaluate weighted standardised L2 retrieval metric on child GT; optionally compare against `score_candidate()` using adult queries | `child_gt_feature_stats.csv`, `pairwise_retrieval_distances.npz`, `nearest_neighbor_examples.csv`, comparison outputs (MODE 2 only) |

---

## ⚠ Important: why heuristic comparison requires adult queries

`score_candidate(adult_feats, target)` in the existing pipeline is designed
specifically for **adult-to-child** scoring. Its core term is:

```python
scale_ratio = target_h / adult_h   # expected to be ~0.72 for adult→child
```

It penalises scale ratios far from ~0.72 and applies group-specific childness
bonuses that only make sense when the query is an adult and the target is a
child prototype.

**Applying it to child-vs-child pairs** (query = child GT row, target = child
GT row) makes `scale_ratio ≈ 1.0` for every pair, collapsing the heuristic's
main signal and making any comparison with the new metric invalid and
misleading.

For this reason, **step3 supports two modes**:

| Mode | `--adult-csv` | Heuristic comparison |
|------|--------------|---------------------|
| MODE 1 (default) | not provided | intentionally absent |
| MODE 2 | adult feature CSV | ✓ `score_candidate()` called on adult→child pairs |

---

## Required inputs

| Input | Default path | Notes |
|-------|-------------|-------|
| Child GT `.pkl` files | `FINAL_GT_DIR` from pipeline | Each `.pkl`: `betas` (10 or 11 dims) + optional `gender` |
| SMPL-X kid template | `DEFAULT_KID_TEMPLATE` from pipeline | `.npy` file |
| `robust_child_shape_opt_upperbody_200.py` | same directory | constants, model factories, `score_candidate` |
| `extract_canonical_features.py` | same directory | `extract_features_from_joints(verts, joints, JOINT_IDX)` — **single source of truth for all features; extended here for head/neck; imported by both the main pipeline and step1** |

Place all scripts in the same directory as the existing pipeline files:

```
your_project/
  robust_child_shape_opt_upperbody_200.py   ← existing pipeline
  extract_canonical_features.py             ← existing feature module
  step1_extract_child_gt_features.py
  step2_analyze_beta10.py
  step3_analyze_retrieval_similarity.py
```

---

## Usage

### Step 1 — Extract child GT features

```bash
python step1_extract_child_gt_features.py \
    --gt-dir /home/jaeson1012/agora_dataset/data/final_child_gt \
    --kid-template /home/jaeson1012/agora_dataset/models/smplx_kid_template.npy \
    --outdir ./analysis_outputs

# Quick sanity check (10 samples)
python step1_extract_child_gt_features.py \
    --gt-dir /home/jaeson1012/agora_dataset/data/final_child_gt \
    --outdir ./analysis_outputs \
    --limit 10
```

**Output:** `./analysis_outputs/child_gt_beta_features.csv`

#### Shared extractor contract

All feature computation goes through the **one shared function**:

```
extract_canonical_features.py :: extract_features_from_joints(verts, joints, joint_idx)
```

This is the same module imported by `robust_child_shape_opt_upperbody_200.py`.
Do **not** add feature logic directly to step1 — extend `extract_canonical_features.py`
so that both the pipeline and step1 stay in sync automatically.

#### Core columns (always present)

| Column | Source | Definition |
|--------|--------|------------|
| `sample_id` | filename stem | — |
| `gender` | pkl | normalised |
| `raw_beta_dim` | pkl | original beta array length |
| `beta_0to9_norm` | computed | `‖β₀…β₉‖₂` |
| `beta_0` … `beta_10` | pkl | SMPL-X shape + kid axis |
| `height_canonical` | joint-based | `joints[15,Y] − min(joints[7,Y], joints[8,Y])` in metres |
| `shoulder_width_ratio` | joint-based | bilateral shoulder distance / height |
| `pelvis_width_ratio` | joint-based | hip joint distance / height |
| `torso_height_ratio` | joint-based | pelvis→neck / height |
| `arm_length_ratio` | joint-based | mean shoulder→wrist / height |
| `thigh_ratio` | joint-based | mean hip→knee / height |
| `shank_ratio` | joint-based | mean knee→ankle / height |
| `leg_length_ratio` | joint-based | mean hip→ankle / height |
| `height_cm` | derived | `height_canonical × 100` |

#### Head / neck columns (present when `verts` is non-None)

These are written by step1 only when `extract_features_from_joints` returns them.
They are detected automatically by step2 — no flags needed.

| Column | Source | Definition | Limitation |
|--------|--------|------------|------------|
| `head_height_ratio` | **vertex-based** | `(max Y of verts above neck joint) − neck_joint_Y` / height | Assumes canonical T-pose; neck joint (12) is anatomical anchor |
| `head_width_ratio` | **vertex-based** | X-span of skull-cap verts (upper 50% of head region) / height | Upper-half filter excludes arm verts in T-pose; less reliable in other poses |
| `neck_length_ratio` | **joint-based** | `‖joints[9] − joints[12]‖` (spine3→neck) / height | SMPL-X "neck" joint is mid-cervical, not C7/T1; treat as proxy |
| `head_width_to_shoulder_ratio` | derived | `head_width / shoulder_width` | Inherits vertex-based head_width limitation |
| `head_features_used_fallback` | derived | `1` if the fallback path was taken for head feature computation (fewer than 10 verts above neck joint); `0` otherwise. Use this column to exclude or flag approximate samples in step2 analysis. |

If vertex data is unavailable (e.g. `build_output_from_model` returns `None`
for verts), head/neck columns are absent from that row and will be NaN in the
CSV. step2 handles NaN gracefully.

---

### Step 2 — Analyse beta[10]

```bash
python step2_analyze_beta10.py \
    --csv ./analysis_outputs/child_gt_beta_features.csv \
    --outdir ./analysis_outputs

# Without plots
python step2_analyze_beta10.py \
    --csv ./analysis_outputs/child_gt_beta_features.csv \
    --outdir ./analysis_outputs \
    --no-plots
```

**Existing outputs (always produced, filenames unchanged):**

| File | Contents |
|------|---------|
| `beta10_summary.json` | mean, std, min, max, p5–p95 |
| `beta10_by_height_bin.csv` | per-bin (100-110, …, 140+) beta10 stats |
| `beta_feature_correlations.csv` | Pearson `corr(beta_i, feature_j)` — now includes optional head/neck columns if present |
| `beta_covariance.csv` | 11×11 covariance matrix of `beta_0..10` |
| `beta10_role_analysis.csv` | high/low `beta_10` × high/low `beta[0:9]` norm groups — now includes std and optional features |
| `beta_pca_summary.json` | explained variance, loadings |
| `beta_pca_components.csv` | PCA component matrix |

**New outputs:**

| File | Contents |
|------|---------|
| `beta10_partial_correlations.csv` | Height-controlled partial correlation between `beta_10` and each feature. Columns: `feature`, `pearson_r`, `partial_r_height_controlled`, `r_drop`. |
| `beta10_regression_summary.json` | Two OLS models: `beta10 ~ height` and `beta10 ~ height + features`. Reports R², R² gain, standardised betas, p-values (if statsmodels available). |
| `beta10_regression_coefficients.csv` | Per-feature coefficient table across both models. |
| `beta10_group_feature_summary.csv` | Feature means ± std across four high/low groups, plus Cohen's d for the most contrasting pair. |

**Existing plots (unchanged):**
`beta10_histogram.png`, `beta10_vs_height_scatter.png`, `beta_correlation_heatmap.png`, `beta_pca_variance.png`, `beta_pca_scatter.png`

**New plots:**

| File | Contents |
|------|---------|
| `beta10_partial_correlation_bar.png` | Side-by-side bar: simple Pearson r vs height-controlled partial r per feature |
| `beta10_residual_vs_feature_scatter.png` | Scatter of height-residualised `beta_10` vs top features by partial correlation |
| `beta10_group_comparison_plot.png` | Grouped bars for feature means across the four high/low groups |

#### Optional feature support

step2 automatically detects extra columns in the CSV and includes them in all
analyses. No flags are needed — just ensure the columns exist in the CSV.

**High-priority optional features** (produced by extending `extract_canonical_features.py`):
- `head_height_ratio`
- `head_width_ratio`
- `neck_length_ratio`

**Other detected optional features:**
`head_width_to_shoulder_ratio`, `head_height_to_torso_ratio`, `torso_depth_ratio`, `waist_or_bmi_proxy`

If none are present, step2 runs normally on the core feature set.

#### ⚠ Why simple correlation is not enough

A non-zero `corr(beta_10, torso_height_ratio)` could mean either:
- beta_10 directly encodes child-like torso proportions, **or**
- beta_10 and torso_height_ratio are both correlated with height, creating
  a spurious association

The height-controlled partial correlation (`beta10_partial_correlations.csv`)
separates these cases. A feature with `partial_r ≈ 0` but large `pearson_r`
is height-mediated — beta_10 acts mainly as a height axis for that feature.
A feature with large `partial_r` even after controlling for height is genuinely
encoded by beta_10 as a childness/proportion axis.

The regression R² gain (`beta10_regression_summary.json: r2_gain_from_features`)
provides a complementary answer: if adding proportion features substantially
increases R² beyond height alone, beta_10 encodes more than just height.

---

### Step 3 — Retrieval similarity analysis

#### MODE 1 — child GT only (always available)

```bash
python step3_analyze_retrieval_similarity.py \
    --csv ./analysis_outputs/child_gt_beta_features.csv \
    --outdir ./analysis_outputs

# With weak height penalty
python step3_analyze_retrieval_similarity.py \
    --csv ./analysis_outputs/child_gt_beta_features.csv \
    --outdir ./analysis_outputs \
    --height-weight 0.15

# Override weights from a JSON file
python step3_analyze_retrieval_similarity.py \
    --csv ./analysis_outputs/child_gt_beta_features.csv \
    --outdir ./analysis_outputs \
    --weights-json my_weights.json
```

**Outputs (MODE 1):**

| File | Contents |
|------|---------|
| `child_gt_feature_stats.csv` | per-feature mean/std (the σᵢ denominators) |
| `retrieval_weights_used.json` | weights wᵢ actually used |
| `pairwise_retrieval_distances.npz` | N×N child-child distance matrix + sample_ids |
| `pairwise_distance_stats.json` | mean/std/percentiles of all pairwise distances |
| `nearest_neighbor_examples.csv` | top-5 NN for every child GT sample |
| `pairwise_distance_histogram.png` | |
| `feature_std_bar.png` | σᵢ bar chart |

#### MODE 2 — adult queries + heuristic comparison

First, produce an adult feature CSV using the same step1 script:

```bash
# Extract features from adult pkl files
python step1_extract_child_gt_features.py \
    --gt-dir /path/to/adult_pkl_dir \
    --outdir ./analysis_outputs/adult_tmp

mv ./analysis_outputs/adult_tmp/child_gt_beta_features.csv \
   ./analysis_outputs/adult_beta_features.csv
```

Then run step3 with `--adult-csv`:

```bash
python step3_analyze_retrieval_similarity.py \
    --csv        ./analysis_outputs/child_gt_beta_features.csv \
    --adult-csv  ./analysis_outputs/adult_beta_features.csv \
    --outdir     ./analysis_outputs \
    --topk 10 \
    --max-adult-queries 50
```

**Additional outputs (MODE 2):**

| File | Contents |
|------|---------|
| `adult_gt_retrieval_results.csv` | top-k child GT per adult query by new metric |
| `adult_heuristic_results.csv` | top-k child GT per adult query by `score_candidate()` |
| `adult_rank_agreement.csv` | Spearman ρ and top-k overlap per adult query |
| `adult_rank_agreement_summary.json` | mean/std agreement summary |
| `adult_rank_agreement_histogram.png` | |
| `adult_retrieval_scatter.png` | per-query retrieval visualisation |

**Interpreting rank agreement:**

| Spearman ρ | Top-k overlap | Interpretation |
|-----------|--------------|----------------|
| < 0.4 | < 0.4 | Metrics diverge strongly → supports replacing heuristic |
| 0.4–0.65 | 0.4–0.6 | Partial agreement → new metric surfaces different candidates |
| > 0.65 | > 0.6 | Broadly consistent → new metric is a principled equivalent |

---

## What the analysis answers

| Question | Where to look |
|----------|--------------|
| Does `beta[10]` mainly track height, or does it encode proportions independently? | `beta10_partial_correlations.csv` (`partial_r_height_controlled`) + `beta10_regression_summary.json` (`r2_gain_from_features`) |
| Are head/neck features more strongly tied to `beta[10]` than torso/limb features? | `beta10_partial_correlations.csv` — compare `partial_r` for `head_*` vs `torso_height_ratio`/`leg_length_ratio` |
| Does `beta[10]` mainly encode childness, height, limb-to-torso proportions, or a mix? | `beta_feature_correlations.csv` row `beta_10`, `beta10_by_height_bin.csv`, `beta10_role_analysis.csv` |
| Should `beta[10]` use a height-conditioned prior? | `beta10_by_height_bin.csv` — if `beta10_mean` varies significantly across height bins, use a bin-specific prior |
| Which features are most informative for beta[10] prior design? | `beta10_group_feature_summary.csv` (Cohen's d column) + `beta10_regression_coefficients.csv` (standardised betas) |
| Is the heuristic scorer too ad hoc? | `adult_rank_agreement_summary.json` (step3 MODE 2) |
| Which child GT samples are retrieved for a given adult? | `adult_gt_retrieval_results.csv` (step3 MODE 2) or `nearest_neighbor_examples.csv` (step3 MODE 1) |

---

## Extending this code

**Height-conditioned `beta[10]` prior:**
Use `beta10_by_height_bin.csv` bin means as the `REG["kid_axis"]` prior target
in `objective_factory()`, conditioned on the adult's estimated child height bin.

**k-NN child prototype retrieval:**
Load `pairwise_retrieval_distances.npz`, then use `compute_cross_distances()`
from step3 to score a new adult query against all child GT rows.

**Prototype mean beta initialisation:**
For each height bin, compute `mean(beta_0..10)` over that bin's samples from
`child_gt_beta_features.csv` and use it as `kid_beta_init` in `build_inits()`.

---

## Notes

- `height_canonical` in the pipeline is in **metres** (e.g. `1.20` = 120 cm).
  The analysis CSV adds `height_cm` for human-readable output and binning.
- Retrieval uses only **ratio features** by default (`--height-weight 0.0`),
  matching the design spec. Pass `--height-weight 0.15` for a weak height
  penalty (design spec recommends ≤ 0.2).
- σᵢ are always computed from the **child GT** distribution, so distances
  are calibrated to child proportional variability regardless of whether the
  query is a child or adult sample.
