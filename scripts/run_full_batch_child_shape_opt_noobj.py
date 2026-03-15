from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_CSV = Path("/home/jaeson1012/agora_dataset/runs/pseudo_labels_balanced/adult_pseudo_labels_balanced.csv")
DEFAULT_OPT_SCRIPT = Path("/home/jaeson1012/agora_dataset/scripts/optimize_child_shape_from_balanced_label.py")
DEFAULT_OUTROOT = Path("/home/jaeson1012/agora_dataset/runs/full_batch_child_shape_opt_noobj")


def safe_get(dct, *keys, default=""):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", type=str, default=str(DEFAULT_LABEL_CSV))
    parser.add_argument("--opt-script", type=str, default=str(DEFAULT_OPT_SCRIPT))
    parser.add_argument("--outroot", type=str, default=str(DEFAULT_OUTROOT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    label_csv = Path(args.label_csv)
    opt_script = Path(args.opt_script)
    outroot = Path(args.outroot)

    if not label_csv.exists():
        raise FileNotFoundError(f"label csv not found: {label_csv}")
    if not opt_script.exists():
        raise FileNotFoundError(f"opt script not found: {opt_script}")

    outroot.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(label_csv)
    if "adult_sample_path" not in df.columns:
        raise KeyError("adult_sample_path column not found in label csv")

    paths = df["adult_sample_path"].dropna().astype(str).tolist()
    paths = [p for p in paths if Path(p).exists()]

    if args.limit is not None:
        paths = paths[:args.limit]

    results = []
    total = len(paths)

    for i, adult_pkl in enumerate(paths, start=1):
        stem = Path(adult_pkl).stem
        case_outdir = outroot / f"{i:05d}_{stem}"
        case_outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(opt_script),
            "--adult-pkl", adult_pkl,
            "--balanced-label-csv", str(label_csv),
            "--outdir", str(case_outdir),
            "--no-obj",
        ]

        print(f"[INFO] ({i}/{total}) {stem}")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        report_path = case_outdir / f"{stem}_optimization_report.json"
        feature_csv = case_outdir / f"{stem}_feature_compare.csv"

        row = {
            "idx": i,
            "adult_sample_path": adult_pkl,
            "stem": stem,
            "status": "ok" if proc.returncode == 0 and report_path.exists() and feature_csv.exists() else "fail",
            "returncode": proc.returncode,
            "report_path": str(report_path) if report_path.exists() else "",
            "feature_csv": str(feature_csv) if feature_csv.exists() else "",
            "stdout_tail": proc.stdout[-1000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
        }

        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)

                row["optimizer_success"] = report.get("optimizer_success", "")
                row["optimizer_fun"] = report.get("optimizer_fun", "")
                row["optimizer_message"] = report.get("optimizer_message", "")
                row["best_global_scale"] = report.get("best_global_scale", "")

                row["target_group"] = safe_get(report, "balanced_target", "group")
                row["target_height_cm"] = safe_get(report, "balanced_target", "target_height_cm")
                row["target_height_bin"] = safe_get(report, "balanced_target", "source_height_bin")

                before_feats = report.get("before_feats", {})
                after_feats = report.get("after_feats", {})
                target_feats = report.get("balanced_target", {})

                keys = [
                    "height_canonical",
                    "shoulder_width_ratio",
                    "pelvis_width_ratio",
                    "torso_height_ratio",
                    "arm_length_ratio",
                    "thigh_ratio",
                    "shank_ratio",
                    "leg_length_ratio",
                ]

                for k in keys:
                    b = before_feats.get(k, "")
                    a = after_feats.get(k, "")
                    t = target_feats.get(k, "")
                    row[f"before_{k}"] = b
                    row[f"after_{k}"] = a
                    row[f"target_{k}"] = t

                    if b != "" and a != "" and t != "":
                        row[f"abs_err_before_{k}"] = abs(float(b) - float(t))
                        row[f"abs_err_after_{k}"] = abs(float(a) - float(t))
                        row[f"improve_{k}"] = float(abs(float(b) - float(t)) - abs(float(a) - float(t)))
            except Exception as e:
                row["report_parse_error"] = repr(e)

        results.append(row)

        if i % 100 == 0 or i == total:
            pd.DataFrame(results).to_csv(outroot / "batch_results_partial.csv", index=False)

    results_df = pd.DataFrame(results)
    results_csv = outroot / "batch_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "total": int(len(results_df)),
        "ok": int((results_df["status"] == "ok").sum()),
        "fail": int((results_df["status"] == "fail").sum()),
        "optimizer_success_true": int((results_df["optimizer_success"] == True).sum()) if "optimizer_success" in results_df.columns else 0,
        "optimizer_success_false": int((results_df["optimizer_success"] == False).sum()) if "optimizer_success" in results_df.columns else 0,
    }

    summary_json = outroot / "batch_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print(f"[INFO] results csv: {results_csv}")
    print(f"[INFO] summary json: {summary_json}")


if __name__ == "__main__":
    main()