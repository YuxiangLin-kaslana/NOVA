#!/usr/bin/env python3
"""Collect tagged P3 SigLA/NOVA experiment outputs.

This collector is intentionally tolerant: it records missing files instead of
crashing, so it can run with an afterany Slurm dependency and still explain what
finished, what failed to produce output, and which metrics are available.
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
DEFAULT_TAG = "p3_20260708_132402"


def expected_files(tag: str) -> list[tuple[str, str]]:
    return [
        (f"sota_multidata_compare_msl_{tag}_msl_entity_sweep.json", "msl_entity_sweep"),
        (f"native_binary_bridge_{tag}_native_smd_psm.json", "native_binary_smd_psm"),
        (f"native_binary_bridge_{tag}_native_msl.json", "native_binary_msl"),
        (f"native_binary_bridge_{tag}_native_smap.json", "native_binary_smap"),
        (f"hparam_sweep_{tag}_llm_sanity_synth.json", "llm_hparam_synth"),
        (f"hparam_sweep_{tag}_llm_sanity_smd.json", "llm_hparam_smd"),
        (f"hparam_sweep_{tag}_llm_sanity_msl.json", "llm_hparam_msl"),
        (f"backbone_openvocab_{tag}_backbone.json", "backbone_openvocab"),
        (f"ablation_namer_{tag}_ablation.json", "namer_ablation_synth"),
        (f"ablation_namer_1-1_{tag}_ablation.json", "namer_ablation_smd_1-1"),
        (f"ablation_namer_2-5_{tag}_ablation.json", "namer_ablation_smd_2-5"),
        (f"openvocab_namer_{tag}_openvocab_namer.json", "openvocab_namer_synth"),
        (f"openvocab_namer_1-1_{tag}_openvocab_namer.json", "openvocab_namer_smd_1-1"),
        (f"openvocab_namer_2-5_{tag}_openvocab_namer.json", "openvocab_namer_smd_2-5"),
        (f"openvocab_loop_result_{tag}_openvocab_loop.json", "openvocab_loop_cost"),
        (f"openvocab_multi_result_{tag}_openvocab_multi.json", "openvocab_multi_cost"),
    ]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def mean_std(values: Iterable[Any]) -> tuple[float | None, float | None, int]:
    arr = np.asarray([v for v in (clean_float(x) for x in values) if v is not None], dtype=float)
    if arr.size == 0:
        return None, None, 0
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def add(
    rows: list[dict[str, Any]],
    experiment: str,
    group: str,
    method: str,
    metric: str,
    mean: Any,
    std: Any,
    n: int,
    source: Path,
) -> None:
    rows.append(
        {
            "experiment": experiment,
            "group": group,
            "method": method,
            "metric": metric,
            "mean": clean_float(mean),
            "std": clean_float(std),
            "n": int(n),
            "source": source.name,
        }
    )


def add_values(
    rows: list[dict[str, Any]],
    experiment: str,
    group: str,
    method: str,
    metric: str,
    values: Iterable[Any],
    source: Path,
) -> None:
    mean, std, n = mean_std(values)
    add(rows, experiment, group, method, metric, mean, std, n, source)


def add_metric_summary(
    rows: list[dict[str, Any]],
    experiment: str,
    group: str,
    method: str,
    metric: str,
    value: Any,
    source: Path,
    n: int,
) -> None:
    if isinstance(value, dict) and ("mean" in value or "std" in value):
        add(rows, experiment, group, method, metric, value.get("mean"), value.get("std"), n, source)
    else:
        add(rows, experiment, group, method, metric, value, 0.0, n, source)


def collect_multidata(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    per_entity = data.get("per_entity") or data.get("per_machine") or {}
    dataset = str(data.get("dataset", "SMD"))
    flat: dict[tuple[str, str], list[Any]] = {}
    for entity, recs in per_entity.items():
        if not isinstance(recs, list):
            continue
        group = f"{dataset}:{entity}"
        for rec in recs:
            for method, metrics in rec.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    if clean_float(value) is None:
                        continue
                    flat.setdefault((method, metric), []).append(value)
        for method, metrics in _method_metric_values(recs).items():
            for metric, values in metrics.items():
                add_values(rows, "real_background_injection", group, method, metric, values, path)
        if any("llm_rate" in rec for rec in recs):
            vals = [rec.get("llm_rate") for rec in recs]
            add_values(rows, "real_background_injection", group, "bootstrap", "llm_rate", vals, path)
            flat.setdefault(("bootstrap", "llm_rate"), []).extend(vals)
    for (method, metric), values in flat.items():
        add_values(rows, "real_background_injection", f"{dataset}:ALL", method, metric, values, path)


def _method_metric_values(recs: list[dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for rec in recs:
        for method, metrics in rec.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                if clean_float(value) is None:
                    continue
                out.setdefault(method, {}).setdefault(metric, []).append(value)
    return out


def collect_summary_list(path: Path, data: dict[str, Any], rows: list[dict[str, Any]], experiment: str) -> None:
    n = int(data.get("nseed", 0) or 0)
    for item in data.get("summary", []):
        if not isinstance(item, dict):
            continue
        method = str(item.get("method", ""))
        if "q" in item:
            method = f"{method}@q={item['q']}"
        group = str(data.get("background") or data.get("dataset") or "all")
        for metric, value in item.get("metrics", {}).items():
            add_metric_summary(rows, experiment, group, method, str(metric), value, path, n)


def collect_native(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    n = int(data.get("nseed", 0) or 0)
    for key, result in data.get("results", {}).items():
        if not isinstance(result, dict):
            continue
        for item in result.get("summary", []):
            method = str(item.get("method", ""))
            if "q" in item:
                method = f"{method}@q={item['q']}"
            for metric, value in item.get("metrics", {}).items():
                add_metric_summary(rows, "native_binary_bridge", str(key), method, str(metric), value, path, n)
        add(rows, "native_binary_bridge", str(key), "dataset", "anomaly_ratio",
            result.get("anomaly_ratio"), 0.0, 1, path)
        add(rows, "native_binary_bridge", str(key), "dataset", "n_events",
            result.get("n_events"), 0.0, 1, path)


def collect_hparam(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    n = int(data.get("nseed", 0) or 0)
    group = str(data.get("background") or "unknown")
    for item in data.get("per_config", []):
        cfg = item.get("config", {}) if isinstance(item, dict) else {}
        method = str(cfg.get("name") or cfg)
        for metric, value in item.get("summary", {}).items():
            add_metric_summary(rows, "hparam_sanity", group, method, str(metric), value, path, n)


def collect_backbone(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for key, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        model, _, novel = str(key).partition("/")
        for metric, value in metrics.items():
            add(rows, "backbone_openvocab", novel, model, str(metric), value, 0.0, 1, path)


def collect_ablation(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    bg = str(data.get("bg") or "synthetic")
    per_seed = data.get("per_seed", [])
    concepts: dict[str, list[Any]] = {}
    for rec in per_seed:
        if not isinstance(rec, dict):
            continue
        for concept, values in rec.items():
            if isinstance(values, list) and len(values) >= 2:
                concepts.setdefault(f"{concept}:rule", []).append(values[0])
                concepts.setdefault(f"{concept}:llm", []).append(values[1])
    for key, values in concepts.items():
        concept, _, method = key.partition(":")
        metric = "normal_misname_rate" if concept == "__normal_misname__" else "name_acc"
        add_values(rows, "namer_ablation", bg, method, metric, values, path)


def collect_openvocab_namer(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    bg = str(data.get("bg") or "synthetic")
    buckets: dict[tuple[str, str, str], list[Any]] = {}
    for rec in data.get("per_seed", []):
        if not isinstance(rec, dict):
            continue
        for concept, metrics in rec.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                method = "rule" if metric == "rule" else "llm"
                metric_name = "name_acc" if metric == "rule" else str(metric).replace("llm_", "")
                buckets.setdefault((concept, method, metric_name), []).append(value)
    for (concept, method, metric), values in buckets.items():
        add_values(rows, "openvocab_namer", f"{bg}:{concept}", method, metric, values, path)


def collect_openvocab_loop(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    n = int(data.get("nseed", 0) or 0)
    for metric, value in data.items():
        if metric in {"nseed", "per_seed", "nov_curve", "llm_curve"}:
            continue
        add_metric_summary(rows, "openvocab_loop_cost", "sequential", "summary", str(metric), value, path, n)


def collect_openvocab_multi(path: Path, data: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    n = int(data.get("nseed", 0) or 0)
    for metric in ("frozen_lastseg", "bootstrap_lastseg", "llm_overall"):
        add_metric_summary(rows, "openvocab_multi_cost", "multi", "summary", metric, data.get(metric), path, n)
    for novel, metrics in data.get("per_novel", {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if isinstance(value, list):
                continue
            add(rows, "openvocab_multi_cost", str(novel), "bootstrap", str(metric), value, 0.0, n, path)


def collect_generic(path: Path, data: Any, rows: list[dict[str, Any]]) -> None:
    if not isinstance(data, dict):
        return
    n = int(data.get("nseed", 1) or 1)
    for key, value in data.items():
        if key in {"per_seed", "rows", "results", "summary"}:
            continue
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                add_metric_summary(rows, path.stem, str(key), "summary", str(metric), metric_value, path, n)
        elif clean_float(value) is not None:
            add(rows, path.stem, "", "summary", str(key), value, 0.0, n, path)


def collect_file(path: Path, rows: list[dict[str, Any]]) -> None:
    data = load_json(path)
    name = path.name
    if name.startswith("sota_multidata_compare"):
        collect_multidata(path, data, rows)
    elif name.startswith("native_binary_bridge"):
        collect_native(path, data, rows)
    elif name.startswith("hparam_sweep"):
        collect_hparam(path, data, rows)
    elif name.startswith("threshold_strategy_sweep"):
        collect_summary_list(path, data, rows, "threshold_strategy")
    elif name.startswith("backbone_openvocab"):
        collect_backbone(path, data, rows)
    elif name.startswith("ablation_namer"):
        collect_ablation(path, data, rows)
    elif name.startswith("openvocab_namer"):
        collect_openvocab_namer(path, data, rows)
    elif name.startswith("openvocab_loop_result"):
        collect_openvocab_loop(path, data, rows)
    elif name.startswith("openvocab_multi_result"):
        collect_openvocab_multi(path, data, rows)
    else:
        collect_generic(path, data, rows)


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    x = float(value)
    if abs(x) <= 1:
        return f"{100 * x:.1f}%"
    return f"{x:.3f}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "group", "method", "metric", "mean", "std", "n", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], audit: list[dict[str, Any]], path: Path, tag: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# P3 Experiment Results Audit\n\n")
        f.write(f"Tag: `{tag}`\n\n")
        n_present = sum(1 for item in audit if item["exists"])
        f.write(f"Expected files present: {n_present}/{len(audit)}\n\n")
        f.write("## Expected Files\n\n")
        f.write("| Status | Experiment | File |\n")
        f.write("|---|---|---|\n")
        for item in audit:
            status = "present" if item["exists"] else "missing"
            f.write(f"| {status} | {item['experiment']} | `{item['file']}` |\n")
        f.write("\n## Metrics\n\n")
        f.write("| Experiment | Group | Method | Metric | Mean | Std | N | Source |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['experiment']} | {row['group']} | {row['method']} | {row['metric']} | "
                f"{format_value(row['mean'])} | {format_value(row['std'])} | "
                f"{row['n']} | `{row['source']}` |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for filename, experiment in expected_files(args.tag):
        path = args.runs_dir / filename
        exists = path.exists()
        audit.append({"file": filename, "experiment": experiment, "exists": exists})
        if not exists:
            continue
        try:
            collect_file(path, rows)
        except Exception as exc:
            rows.append(
                {
                    "experiment": "COLLECT_ERROR",
                    "group": experiment,
                    "method": "",
                    "metric": str(exc),
                    "mean": None,
                    "std": None,
                    "n": 0,
                    "source": path.name,
                }
            )

    prefix = args.out_prefix or (args.runs_dir / f"p3_results_summary_{args.tag}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    write_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "audit": audit, "rows": rows}, f, indent=2, ensure_ascii=False)
    write_markdown(rows, audit, md_path, args.tag)

    missing = [item["file"] for item in audit if not item["exists"]]
    print(f"present={len(audit) - len(missing)}/{len(audit)} rows={len(rows)}")
    if missing:
        print("missing:")
        for name in missing:
            print(f"  {name}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
