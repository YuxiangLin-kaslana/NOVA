#!/usr/bin/env python3
"""Supervised representation ceiling for the many-type composite benchmark.

This is an oracle diagnostic, not a deployable method.  It asks whether the
current 14-dimensional evidence representation can separate the generated
subtypes when discovery labels and, optionally, the true component pair are
provided.  The result determines whether more online-memory tuning is useful.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import sigla_exp.longtail_bench as LT  # noqa: E402
from sota_compare.run_longtail_prequential import (  # noqa: E402
    ExperimentConfig,
    StreamSample,
    bootstrap_ci,
    common_route,
    component_key,
    composite_order,
    fast_features,
    file_hash,
    prepare_base,
    provenance,
    sanitize,
    transform,
)


def config_for_base(seeds: int, seed_start: int) -> ExperimentConfig:
    return ExperimentConfig(
        seeds=seeds,
        seed_start=seed_start,
        ks=(20, 100),
        radius_scales=(1.0,),
        default_radius_scale=1.0,
        train_per_type=8,
        normal_train_n=64,
        normal_stats_n=80,
        score_cal_n=80,
        discovery_repeats=2,
        reuse_repeats=5,
        discovery_normal_per_k=0.0,
        discovery_known_per_k=0.0,
        reuse_normal_per_k=0.0,
        reuse_known_per_k=0.0,
        known_radius_quantile=0.90,
        score_quantile=0.95,
        component_threshold=1.6,
        merge_factor=1.15,
        guard_confirm_k=2,
        guard_reuse_margin=0.85,
        complexity_k=20,
        bootstrap_samples=5000,
        include_complexity=False,
    )


def nearest_label(vector: np.ndarray, prototypes: dict[str, np.ndarray], allowed: set[str] | None = None) -> str:
    names = allowed if allowed is not None else set(prototypes)
    return min(names, key=lambda name: float(np.linalg.norm(vector - prototypes[name])))


def separation_auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return float("nan")
    wins = 0.0
    total = 0
    for own in positive:
        for other in negative:
            wins += own < other
            wins += 0.5 * (own == other)
            total += 1
    return wins / total


def run_seed_k(
    seed: int,
    novel_specs: list[dict[str, Any]],
    discovery_values: tuple[int, ...],
    reuse_per_type: int,
    config: ExperimentConfig,
    known_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = prepare_base(seed, config, known_specs)
    rng = np.random.default_rng(900_000 + 10_000 * seed + len(novel_specs))
    max_discovery = max(discovery_values)
    labels = [str(spec["name"]) for spec in novel_specs]
    spec_by_name = {str(spec["name"]): spec for spec in novel_specs}
    pair_by_name = {
        str(spec["name"]): tuple(sorted(str(component) for component in spec["components"]))
        for spec in novel_specs
    }

    discovery: dict[str, list[np.ndarray]] = defaultdict(list)
    for label in labels:
        spec = spec_by_name[label]
        for _ in range(max_discovery):
            raw = fast_features(LT.make_window(spec, rng), base.ev_mu, base.ev_sd)
            discovery[label].append(transform(raw, base.scale_mu, base.scale_sd))

    reuse: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label in labels:
        spec = spec_by_name[label]
        for _ in range(reuse_per_type):
            raw = fast_features(LT.make_window(spec, rng), base.ev_mu, base.ev_sd)
            reuse.append((label, raw, transform(raw, base.scale_mu, base.scale_sd)))
    rng.shuffle(reuse)

    rows = []
    for discovery_n in discovery_values:
        prototypes = {
            label: np.mean(discovery[label][:discovery_n], axis=0) for label in labels
        }
        pair_labels: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for label in labels:
            pair_labels[pair_by_name[label]].add(label)

        global_correct = []
        component_correct = []
        gate_global_correct = []
        candidate_correct = []
        candidate_flags = []
        detector_flags = []
        known_absorbed = []
        own_distances = []
        other_distances = []
        for label, raw, feature in reuse:
            global_prediction = nearest_label(feature, prototypes)
            component_prediction = nearest_label(feature, prototypes, pair_labels[pair_by_name[label]])
            global_correct.append(global_prediction == label)
            component_correct.append(component_prediction == label)
            own_distances.append(float(np.linalg.norm(feature - prototypes[label])))
            other_distances.append(
                min(float(np.linalg.norm(feature - prototypes[name])) for name in labels if name != label)
            )

            sample = StreamSample(
                phase="locked_reuse",
                true_label=label,
                raw_feature=raw,
                feature=feature,
                key=component_key(raw, config.component_threshold),
                novel_rank=None,
            )
            route = common_route(sample, base)
            candidate_flags.append(route["candidate"])
            detector_flags.append(route["anomaly"])
            known_absorbed.append(route["anomaly"] and not route["candidate"])
            gate_global_correct.append(route["candidate"] and global_prediction == label)
            if route["candidate"]:
                candidate_correct.append(global_prediction == label)

        rows.append(
            {
                "seed": seed,
                "novel_n": len(novel_specs),
                "discovery_per_type": discovery_n,
                "reuse_per_type": reuse_per_type,
                "global_centroid_accuracy": float(np.mean(global_correct)),
                "oracle_component_accuracy": float(np.mean(component_correct)),
                "observed_gate_global_accuracy": float(np.mean(gate_global_correct)),
                "candidate_conditioned_global_accuracy": float(np.mean(candidate_correct)) if candidate_correct else 0.0,
                "candidate_recall": float(np.mean(candidate_flags)),
                "detector_recall": float(np.mean(detector_flags)),
                "known_absorption_rate": float(np.mean(known_absorbed)),
                "distance_separation_auc": separation_auc(own_distances, other_distances),
                "own_distance_mean": float(np.mean(own_distances)),
                "nearest_other_distance_mean": float(np.mean(other_distances)),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    metrics = [
        "global_centroid_accuracy",
        "oracle_component_accuracy",
        "observed_gate_global_accuracy",
        "candidate_conditioned_global_accuracy",
        "candidate_recall",
        "detector_recall",
        "known_absorption_rate",
        "distance_separation_auc",
    ]
    output = []
    for novel_n in sorted({row["novel_n"] for row in rows}):
        for discovery_n in sorted({row["discovery_per_type"] for row in rows}):
            group = [
                row
                for row in rows
                if row["novel_n"] == novel_n and row["discovery_per_type"] == discovery_n
            ]
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                low, high = bootstrap_ci(
                    values,
                    bootstrap_samples,
                    seed=novel_n * 10_000 + discovery_n * 100 + len(metric),
                )
                output.append(
                    {
                        "novel_n": novel_n,
                        "discovery_per_type": discovery_n,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "n": len(values),
                    }
                )
    return output


def get(summary: list[dict[str, Any]], novel_n: int, discovery_n: int, metric: str) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if row["novel_n"] == novel_n
        and row["discovery_per_type"] == discovery_n
        and row["metric"] == metric
    )


def plot(summary: list[dict[str, Any]], ks: tuple[int, ...], discovery_values: tuple[int, ...], output: Path) -> None:
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 3.8), squeeze=False)
    for axis, novel_n in zip(axes[0], ks):
        for metric, label, color in [
            ("global_centroid_accuracy", "Global supervised centroid", "#4c566a"),
            ("oracle_component_accuracy", "Oracle component routing", "#2a8c67"),
            ("observed_gate_global_accuracy", "Observed gate + global", "#b06c2f"),
        ]:
            means = [get(summary, novel_n, value, metric)["mean"] for value in discovery_values]
            stds = [get(summary, novel_n, value, metric)["std"] for value in discovery_values]
            axis.errorbar(discovery_values, means, yerr=stds, marker="o", capsize=3, label=label, color=color)
        axis.set_title(f"K={novel_n} novel composite types")
        axis.set_xlabel("Labeled discovery samples per type")
        axis.set_ylabel("Locked exact accuracy")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Representation ceiling before online memory", fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_report(
    output: Path,
    summary: list[dict[str, Any]],
    ks: tuple[int, ...],
    discovery_values: tuple[int, ...],
    seeds: int,
    seed_start: int,
    result_path: Path,
    figure_path: Path,
) -> None:
    lines = [
        "# Many-Type Representation Ceiling Audit",
        "",
        "This is an oracle diagnostic, not an online method or LLM result. True discovery labels build supervised centroids; oracle-component routing additionally reveals the true family pair.",
        "",
        f"Seeds: `{seed_start}..{seed_start + seeds - 1}`; reuse samples/type: `5`.",
        "",
        "| K | Discovery/type | Global centroid | Oracle component | Observed gate + global | Candidate recall | Known absorption | Distance AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for novel_n in ks:
        for discovery_n in discovery_values:
            values = {
                metric: get(summary, novel_n, discovery_n, metric)["mean"]
                for metric in (
                    "global_centroid_accuracy",
                    "oracle_component_accuracy",
                    "observed_gate_global_accuracy",
                    "candidate_recall",
                    "known_absorption_rate",
                    "distance_separation_auc",
                )
            }
            lines.append(
                f"| {novel_n} | {discovery_n} | {values['global_centroid_accuracy']:.1%} | "
                f"{values['oracle_component_accuracy']:.1%} | {values['observed_gate_global_accuracy']:.1%} | "
                f"{values['candidate_recall']:.1%} | {values['known_absorption_rate']:.1%} | "
                f"{values['distance_separation_auc']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- If the global supervised ceiling is low, online radius or merge tuning cannot solve the task.",
            "- The gap from global to oracle-component routing measures the value of a reliable coarse key.",
            "- The gap from global to observed-gate accuracy measures the detector/known-rejection bottleneck.",
            "- This audit still uses the synthetic generator and cannot establish external generalization.",
            "",
            f"Result JSON: `{result_path}`",
            f"Figure: `{figure_path}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_int(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--ks", default="20,100")
    parser.add_argument("--discovery-values", default="2,5,10")
    parser.add_argument("--reuse-per-type", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "manytype_oracle_ceiling_2026-07-09",
    )
    args = parser.parse_args()
    ks = parse_csv_int(args.ks)
    discovery_values = parse_csv_int(args.discovery_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = config_for_base(args.seeds, args.seed_start)
    catalog = LT.generate_taxonomy(216)
    known_specs = [spec for spec in catalog if len(spec["components"]) == 1][:6]
    composite_specs = [spec for spec in catalog if len(spec["components"]) == 2]
    rows = []
    started = time.time()
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        ordered = composite_order(seed, composite_specs)
        for novel_n in ks:
            print(f"seed={seed} K={novel_n}", flush=True)
            rows.extend(
                run_seed_k(
                    seed,
                    ordered[:novel_n],
                    discovery_values,
                    args.reuse_per_type,
                    config,
                    known_specs,
                )
            )
    summary = summarize(rows, args.bootstrap_samples)
    source_provenance = provenance()
    source_provenance["source_sha256"][str(Path(__file__).relative_to(REPO))] = file_hash(Path(__file__))
    result_path = args.output_dir / "manytype_oracle_ceiling_result.json"
    figure_path = args.output_dir / "manytype_oracle_ceiling.png"
    report_path = args.output_dir / "manytype_oracle_ceiling_report.md"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.time() - started,
        "config": {
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "ks": ks,
            "discovery_values": discovery_values,
            "reuse_per_type": args.reuse_per_type,
            "bootstrap_samples": args.bootstrap_samples,
            "base": asdict(config),
        },
        "provenance": source_provenance,
        "rows": rows,
        "summary": summary,
    }
    result_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    plot(summary, ks, discovery_values, figure_path)
    build_report(
        report_path,
        summary,
        ks,
        discovery_values,
        args.seeds,
        args.seed_start,
        result_path,
        figure_path,
    )
    print(f"saved -> {result_path}", flush=True)
    print(f"saved -> {figure_path}", flush=True)
    print(f"saved -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
