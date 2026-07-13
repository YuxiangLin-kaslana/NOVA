#!/usr/bin/env python3
"""Collect SigLA/NOVA paper experiment JSONs into compact tables.

The runner JSONs intentionally keep rich per-seed records. This script builds a
single audit-friendly summary for paper figures and for checking missing runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def add(rows: list[dict[str, Any]], experiment: str, group: str, method: str, metric: str,
        values: Iterable[float], source: Path) -> None:
    m, s, n = mean_std(values)
    rows.append({
        "experiment": experiment,
        "group": group,
        "method": method,
        "metric": metric,
        "mean": m,
        "std": s,
        "n": n,
        "source": source.name,
    })


def collect_detection(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    per_seed = data.get("per_seed", [])
    for method in ("frozen", "anomaly_transformer", "memstream", "bootstrap"):
        for metric in ("nov_recall", "nov_classacc", "f1", "prec", "rec"):
            add(rows, "synthetic_detection_naming", data.get("novel", ""), method, metric,
                (r.get(method, {}).get(metric, float("nan")) for r in per_seed), path)
    add(rows, "synthetic_detection_naming", data.get("novel", ""), "bootstrap", "llm_rate",
        (r.get("llm_rate", float("nan")) for r in per_seed), path)
    add(rows, "synthetic_detection_naming", data.get("novel", ""), "bootstrap", "grew",
        (r.get("grew", float("nan")) for r in per_seed), path)


def collect_multidata(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    per_entity = data.get("per_entity") or data.get("per_machine") or {}
    dataset = data.get("dataset", "SMD")
    flat: dict[str, dict[str, list[float]]] = {}
    for entity, recs in per_entity.items():
        for method in ("frozen", "anomaly_transformer", "memstream", "bootstrap"):
            for metric in ("nov_recall", "nov_classacc", "f1", "prec", "rec"):
                vals = [r.get(method, {}).get(metric, float("nan")) for r in recs]
                add(rows, "real_background_injection", f"{dataset}:{entity}", method, metric, vals, path)
                flat.setdefault(method, {}).setdefault(metric, []).extend(vals)
        add(rows, "real_background_injection", f"{dataset}:{entity}", "bootstrap", "llm_rate",
            (r.get("llm_rate", float("nan")) for r in recs), path)
    for method, metrics in flat.items():
        for metric, vals in metrics.items():
            add(rows, "real_background_injection", f"{dataset}:ALL", method, metric, vals, path)


def collect_ew(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    per_seed = data.get("per_seed", [])
    for method in ("frozen", "bootstrap", "anomaly_transformer", "memstream"):
        add(rows, "typed_early_warning", data.get("novel", ""), method, "typed_ew_recall",
            (r.get(method, {}).get("typed", {}).get("ew_recall", float("nan")) for r in per_seed), path)
        add(rows, "typed_early_warning", data.get("novel", ""), method, "typed_lead_mean",
            (r.get(method, {}).get("typed", {}).get("lead_mean", float("nan")) for r in per_seed), path)
        add(rows, "typed_early_warning", data.get("novel", ""), method, "type_far",
            (r.get(method, {}).get("type_far", float("nan")) for r in per_seed), path)
        add(rows, "typed_early_warning", data.get("novel", ""), method, "binary_ew_recall",
            (r.get(method, {}).get("binary", {}).get("ew_recall", float("nan")) for r in per_seed), path)
    add(rows, "typed_early_warning", data.get("novel", ""), "bootstrap", "grew",
        (r.get("grew", float("nan")) for r in per_seed), path)


def collect_drift(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    per_seed = data.get("per_seed", [])
    group = data.get("bg", "")
    for method in ("frozen", "memstream", "ours"):
        add(rows, "drift_vs_novel", group, method, "drift_fa",
            (r.get("drift_FA", {}).get(method, float("nan")) for r in per_seed), path)
        add(rows, "drift_vs_novel", group, method, "nov_recall",
            (r.get("nov_recall", {}).get(method, float("nan")) for r in per_seed), path)
    add(rows, "drift_vs_novel", group, "ours", "name_acc",
        (r.get("ours_name_acc", float("nan")) for r in per_seed), path)
    add(rows, "drift_vs_novel", group, "ours", "grew",
        (r.get("grew", float("nan")) for r in per_seed), path)


def collect_robust(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    flat: dict[str, list[float]] = {}
    for entity, recs in data.items():
        for metric in ("rob_fa", "rob_fa_late", "rob_spur", "t2_drift_fa", "t2_recall",
                       "t2_name", "t2_grew", "t2_commit", "t2_recalib"):
            vals = [r.get(metric, float("nan")) for r in recs]
            add(rows, "guarded_update_robustness", entity, "robust_ours", metric, vals, path)
            flat.setdefault(metric, []).extend(vals)
        if recs and "orig_fa" in recs[0]:
            add(rows, "guarded_update_robustness", entity, "original_ours", "orig_fa",
                [recs[0].get("orig_fa", float("nan"))], path)
            add(rows, "guarded_update_robustness", entity, "original_ours", "orig_spur",
                [recs[0].get("orig_spur", float("nan"))], path)
    for metric, vals in flat.items():
        add(rows, "guarded_update_robustness", "ALL", "robust_ours", metric, vals, path)


def collect_generic_flat(path: Path, data: Any, rows: list[dict[str, Any]]) -> None:
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if isinstance(value, (int, float)):
            add(rows, path.stem, "", "summary", key, [float(value)], path)
        elif isinstance(value, dict):
            for metric, mv in value.items():
                if isinstance(mv, (int, float)):
                    add(rows, path.stem, key, "summary", metric, [float(mv)], path)


def collect_file(path: Path, rows: list[dict[str, Any]]) -> None:
    data = load_json(path)
    name = path.name
    if name.startswith("sota_detection_compare"):
        collect_detection(path, data, rows)
    elif name.startswith("sota_multidata_compare"):
        collect_multidata(path, data, rows)
    elif name.startswith("sota_ew_compare"):
        collect_ew(path, data, rows)
    elif name.startswith("drift_vs_novel"):
        collect_drift(path, data, rows)
    elif name.startswith("robust_multi"):
        collect_robust(path, data, rows)
    else:
        collect_generic_flat(path, data, rows)


def format_pct(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{100 * x:.1f}%"


def format_value(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if abs(x) <= 1:
        return format_pct(x)
    return f"{x:.3f}"


def write_markdown(rows: list[dict[str, Any]], path: Path, tag: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# SigLA/NOVA Paper Results Summary\n\n")
        f.write(f"Tag filter: `{tag or 'ALL'}`\n\n")
        f.write("| Experiment | Group | Method | Metric | Mean | Std | N | Source |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['experiment']} | {r['group']} | {r['method']} | {r['metric']} | "
                f"{format_value(r['mean'])} | "
                f"{format_value(r['std'])} | "
                f"{r['n']} | {r['source']} |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="Only collect JSON files whose names contain this tag.")
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    ap.add_argument("--out-prefix", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.runs_dir.glob("*.json"))
    if args.tag:
        files = [p for p in files if args.tag in p.name]
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            collect_file(path, rows)
        except Exception as exc:  # keep one malformed file from hiding completed runs
            rows.append({
                "experiment": "COLLECT_ERROR",
                "group": "",
                "method": "",
                "metric": str(exc),
                "mean": float("nan"),
                "std": float("nan"),
                "n": 0,
                "source": path.name,
            })

    prefix = args.out_prefix or (args.runs_dir / f"paper_results_summary_{args.tag or 'all'}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path = prefix.with_suffix(".json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "group", "method", "metric", "mean", "std", "n", "source"])
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    write_markdown(rows, md_path, args.tag)
    print(f"files={len(files)} rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
