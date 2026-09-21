#!/usr/bin/env python3
"""
Collect the zero-shot results of evaluation-pipeline-2025 into one CSV.

Walks a directory of experiments, scores the BLiMP, BLiMP Supplement and EWoK
prediction files of each run against the gold data, and writes one row per run.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd


def compute_blimp_accuracy(pred_file: Path, data_dir: Path) -> dict:
    """Positional matching: pred[i].pred == gold[i].sentence_good"""
    with open(pred_file) as f:
        preds = json.load(f)
    correct, total = 0, 0
    for paradigm, val in preds.items():
        pred_list = val["predictions"]
        gold_file = data_dir / f"{paradigm}.jsonl"
        if not gold_file.exists():
            continue
        golds = [json.loads(l) for l in open(gold_file)]
        for p, g in zip(pred_list, golds):
            good = g.get("sentence_good", "")
            total += 1
            if p["pred"].strip() == good.strip():
                correct += 1
    return {"overall": correct / total if total else None, "n": total}


def compute_ewok_accuracy(pred_file: Path, data_dir: Path) -> dict:
    """EWoK: pred should match Target1 (the congruent target)."""
    with open(pred_file) as f:
        preds = json.load(f)
    correct, total = 0, 0
    for paradigm, val in preds.items():
        pred_list = val["predictions"]
        gold_file = data_dir / f"{paradigm}.jsonl"
        if not gold_file.exists():
            continue
        golds = [json.loads(l) for l in open(gold_file)]
        for p, g in zip(pred_list, golds):
            target1 = g.get("Target1", "")
            total += 1
            if p["pred"].strip() == target1.strip():
                correct += 1
    return {"overall": correct / total if total else None, "n": total}


def parse_experiment_name(folder_name: str) -> dict:
    name = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", folder_name)
    name = re.sub(r"_10m$", "", name)
    if name.startswith("selective_entropy_"):
        rest = name[len("selective_entropy_"):]
        parts = rest.split("_", 1)
        return {"mode": "selective_kd", "entropy_threshold": float(parts[0]),
                "architecture": parts[1] if len(parts) > 1 else "unknown"}
    elif name.startswith("full_kd_"):
        return {"mode": "full_kd", "entropy_threshold": None, "architecture": name[len("full_kd_"):]}
    elif name.startswith("scratch_"):
        return {"mode": "scratch", "entropy_threshold": None, "architecture": name[len("scratch_"):]}
    return {"mode": "unknown", "entropy_threshold": None, "architecture": name}


def collect_results(experiments_dir: Path, eval_data_dir: Path) -> list:
    blimp_data = eval_data_dir / "blimp_filtered"
    supp_data  = eval_data_dir / "supplement_filtered"
    ewok_data  = eval_data_dir / "ewok_filtered"
    rows = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        eval_2025_dir = exp_dir / "eval_2025"
        if not eval_2025_dir.exists():
            continue
        pred_files = list(eval_2025_dir.rglob("predictions.json"))
        if not pred_files:
            continue

        meta = parse_experiment_name(exp_dir.name)
        row = dict(experiment=exp_dir.name, **meta,
                   blimp_acc=None, supplement_acc=None, ewok_acc=None)

        for pf in pred_files:
            ps = str(pf)
            if "/blimp/blimp_" in ps:
                r = compute_blimp_accuracy(pf, blimp_data)
                row["blimp_acc"] = round(r["overall"], 4) if r["overall"] is not None else None
            elif "/blimp/supplement_" in ps:
                r = compute_blimp_accuracy(pf, supp_data)
                row["supplement_acc"] = round(r["overall"], 4) if r["overall"] is not None else None
            elif "/ewok/" in ps:
                r = compute_ewok_accuracy(pf, ewok_data)
                row["ewok_acc"] = round(r["overall"], 4) if r["overall"] is not None else None

        b = f"{row['blimp_acc']:.3f}" if row['blimp_acc'] else "N/A"
        s = f"{row['supplement_acc']:.3f}" if row['supplement_acc'] else "N/A"
        e = f"{row['ewok_acc']:.3f}" if row['ewok_acc'] else "N/A"
        print(f"  {exp_dir.name}: BLiMP={b}  Supp={s}  EWoK={e}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", required=True, type=Path)
    parser.add_argument("--eval_data_dir", required=True, type=Path)
    parser.add_argument("--output_csv", default="results_2025_zero_shot.csv", type=Path)
    args = parser.parse_args()

    rows = collect_results(args.experiments_dir, args.eval_data_dir)
    if not rows:
        print("No results found!")
        return
    df = pd.DataFrame(rows).sort_values(["architecture", "mode", "entropy_threshold"])
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(df)} rows → {args.output_csv}")
    print(df[["architecture","mode","entropy_threshold","blimp_acc","supplement_acc","ewok_acc"]].to_string(index=False))

if __name__ == "__main__":
    main()
