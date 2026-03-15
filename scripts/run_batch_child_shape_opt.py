from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_CSV = Path("/home/jaeson1012/agora_dataset/runs/pseudo_labels_balanced/adult_pseudo_labels_balanced.csv")
DEFAULT_OPT_SCRIPT = Path("/home/jaeson1012/agora_dataset/scripts/optimize_child_shape_from_balanced_label.py")
DEFAULT_OUTROOT = Path("/home/jaeson1012/agora_dataset/runs/batch_child_shape_opt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", type=str, default=str(DEFAULT_LABEL_CSV))
    parser.add_argument("--opt-script", type=str, default=str(DEFAULT_OPT_SCRIPT))
    parser.add_argument("--outroot", type=str, default=str(DEFAULT_OUTROOT))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
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

    if len(paths) == 0:
        raise RuntimeError("No valid adult sample paths found")

    n = min(args.n, len(paths))
    rng = random.Random(args.seed)
    sampled = rng.sample(paths, n)

    sample_df = pd.DataFrame({"adult_sample_path": sampled})
    sample_csv = outroot / "sampled_adults.csv"
    sample_df.to_csv(sample_csv, index=False)

    results = []

    for i, adult_pkl in enumerate(sampled, start=1):
        stem = Path(adult_pkl).stem
        case_outdir = outroot / f"{i:03d}_{stem}"
        case_outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(opt_script),
            "--adult-pkl", adult_pkl,
            "--balanced-label-csv", str(label_csv),
            "--outdir", str(case_outdir),
        ]

        print(f"[INFO] ({i}/{n}) running: {stem}")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        report_path = case_outdir / f"{stem}_optimization_report.json"
        feature_csv = case_outdir / f"{stem}_feature_compare.csv"
        child_obj = case_outdir / f"{stem}_child_posed.obj"

        status = "ok" if proc.returncode == 0 and report_path.exists() else "fail"

        row = {
            "idx": i,
            "adult_sample_path": adult_pkl,
            "stem": stem,
            "status": status,
            "returncode": proc.returncode,
            "report_path": str(report_path) if report_path.exists() else "",
            "feature_csv": str(feature_csv) if feature_csv.exists() else "",
            "child_posed_obj": str(child_obj) if child_obj.exists() else "",
            "stdout_tail": proc.stdout[-1000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
        }

        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)

                row["optimizer_success"] = report.get("optimizer_success", "")
                row["optimizer_fun"] = report.get("optimizer_fun", "")
                row["target_group"] = report.get("balanced_target", {}).get("group", "")
                row["target_height_cm"] = report.get("balanced_target", {}).get("target_height_cm", "")

                before_feats = report.get("before_feats", {})
                after_feats = report.get("after_feats", {})
                target_feats = report.get("balanced_target", {})

                for k in [
                    "height_canonical",
                    "shoulder_width_ratio",
                    "pelvis_width_ratio",
                    "arm_length_ratio",
                    "leg_length_ratio",
                ]:
                    row[f"before_{k}"] = before_feats.get(k, "")
                    row[f"after_{k}"] = after_feats.get(k, "")
                    row[f"target_{k}"] = target_feats.get(k, "")
            except Exception as e:
                row["report_parse_error"] = repr(e)

        results.append(row)

    results_df = pd.DataFrame(results)
    results_csv = outroot / "batch_results.csv"
    results_df.to_csv(results_csv, index=False)

    ok_df = results_df[results_df["status"] == "ok"].copy()
    summary = {
        "requested_n": args.n,
        "actual_n": n,
        "ok": int((results_df["status"] == "ok").sum()),
        "fail": int((results_df["status"] == "fail").sum()),
        "seed": args.seed,
    }

    summary_json = outroot / "batch_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] done")
    print(f"[INFO] sampled csv: {sample_csv}")
    print(f"[INFO] results csv: {results_csv}")
    print(f"[INFO] summary json: {summary_json}")


if __name__ == "__main__":
    main()