#!/usr/bin/env python3
"""Two-stage memory isolation for the many-type follow-up.

The previous follow-up showed that ``observable72`` raises the supervised
K=100 ceiling, but online locked memory still performs poorly.  This runner
holds the representation and gate fixed, then separates three effects:

1. reuse-gate misses,
2. discovery-query misses,
3. online assignment/prototype pollution during discovery.

It is still a controlled synthetic diagnostic.  Discovery labels are oracle
labels used only on authorized discovery-query branches.
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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from sigla_exp.prequential_memory import MemoryConfig, OnlinePrototypeMemory  # noqa: E402
from sota_compare.run_representation_gate_followup import (  # noqa: E402
    FollowupConfig,
    GateConfig,
    build_feature_bundle,
    build_seed_dataset,
    fit_feature_context,
    gate_thresholds,
    memory_radius,
    observed_component_key,
    route_with_gate,
    sanitize,
    scale_bundle,
    stable_hash,
)


FEATURE = "observable72"
GATE = GateConfig(score_quantile=0.95, known_quantile=0.90, split_known=True)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def base_config(args: argparse.Namespace) -> FollowupConfig:
    if args.smoke:
        return FollowupConfig(
            dev_seeds=(),
            holdout_seeds=(30,),
            ks=(20,),
            discovery_values=(2,),
            reuse_per_type=2,
            normal_stats_n=20,
            normal_train_n=24,
            normal_cal_n=20,
            train_per_known_type=3,
            cal_per_known_type=3,
            bootstrap_samples=200,
            memory_discovery=2,
            memory_radius_quantile=0.90,
            memory_component_threshold=1.6,
        )
    return FollowupConfig(
        dev_seeds=(),
        holdout_seeds=parse_csv_ints(args.seeds),
        ks=(args.k,),
        discovery_values=(args.discovery,),
        reuse_per_type=args.reuse_per_type,
        normal_stats_n=args.normal_stats_n,
        normal_train_n=args.normal_train_n,
        normal_cal_n=args.normal_cal_n,
        train_per_known_type=args.train_per_known_type,
        cal_per_known_type=args.cal_per_known_type,
        bootstrap_samples=args.bootstrap_samples,
        memory_discovery=args.discovery,
        memory_radius_quantile=args.memory_radius_quantile,
        memory_component_threshold=1.6,
    )


def centroids(samples: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {label: np.mean(vectors, axis=0) for label, vectors in samples.items() if vectors}


def nearest(vector: np.ndarray, prototypes: dict[str, np.ndarray]) -> tuple[str, float]:
    label = min(prototypes, key=lambda name: float(np.linalg.norm(vector - prototypes[name])))
    return label, float(np.linalg.norm(vector - prototypes[label]))


def radius_from_samples(samples: dict[str, list[np.ndarray]], prototypes: dict[str, np.ndarray], quantile: float) -> float:
    distances = [
        float(np.linalg.norm(vector - prototypes[label]))
        for label, vectors in samples.items()
        if label in prototypes
        for vector in vectors
    ]
    if not distances:
        return float("inf")
    return max(1e-6, float(np.quantile(distances, quantile)))


def summarize_predictions(rows: list[dict[str, Any]], novel_n: int) -> dict[str, Any]:
    locked = [row for row in rows if row["phase"] == "locked_reuse"]
    discovery = [row for row in rows if row["phase"] == "discovery"]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in locked:
        by_type[row["true_label"]].append(row)
    per_type = {
        label: float(np.mean([item["correct"] for item in items]))
        for label, items in by_type.items()
    }
    typed = [row for row in locked if row["pred_label"] not in {"normal", "unknown"}]
    return {
        "locked_macro_accuracy": float(np.mean(list(per_type.values()))) if per_type else 0.0,
        "locked_micro_accuracy": float(np.mean([row["correct"] for row in locked])) if locked else 0.0,
        "locked_type_coverage": len(
            {label for label, items in by_type.items() if any(item["correct"] for item in items)}
        )
        / novel_n,
        "locked_unknown_rate": float(np.mean([row["pred_label"] == "unknown" for row in locked])) if locked else 0.0,
        "locked_candidate_recall": float(np.mean([row["candidate"] for row in locked])) if locked else 0.0,
        "typed_prediction_precision": float(np.mean([row["correct"] for row in typed])) if typed else 0.0,
        "annotation_queries": int(sum(row["queried"] for row in discovery)),
        "discovery_candidate_rate": float(np.mean([row["candidate"] for row in discovery])) if discovery else 0.0,
        "discovery_type_coverage": len(
            {row["true_label"] for row in discovery if row["queried"]}
        )
        / novel_n,
    }


def route_candidate(
    phase_policy: str,
    vector: np.ndarray,
    score: float,
    thresholds: dict[str, Any],
) -> tuple[bool, str]:
    if phase_policy == "oracle":
        return True, "unknown"
    route = route_with_gate(vector, score, thresholds)
    return bool(route["candidate"]), str(route["pred_label"])


def run_two_stage(
    scaled: Any,
    thresholds: dict[str, Any],
    discovery_policy: str,
    reuse_policy: str,
    use_radius: bool,
    radius_quantile: float,
) -> dict[str, Any]:
    queried: dict[str, list[np.ndarray]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    feature = scaled.feature
    for label in sorted(scaled.discovery):
        for index, vector in enumerate(scaled.discovery[label]):
            candidate, route_label = route_candidate(
                discovery_policy,
                vector,
                float(feature.scores["discovery"][label][index]),
                thresholds,
            )
            if candidate:
                queried[label].append(vector)
            events.append(
                {
                    "phase": "discovery",
                    "true_label": label,
                    "candidate": candidate,
                    "queried": candidate,
                    "pred_label": label if candidate else route_label,
                    "correct": candidate,
                }
            )
    prototypes = centroids(queried)
    radius = radius_from_samples(queried, prototypes, radius_quantile) if use_radius else float("inf")
    for (label, vector), (_, score) in zip(scaled.reuse, feature.scores["reuse"]):
        candidate, route_label = route_candidate(reuse_policy, vector, float(score), thresholds)
        pred_label = route_label
        distance = None
        if candidate:
            if prototypes:
                pred_label, distance = nearest(vector, prototypes)
                if distance > radius:
                    pred_label = "unknown"
            else:
                pred_label = "unknown"
        events.append(
            {
                "phase": "locked_reuse",
                "true_label": label,
                "candidate": candidate,
                "queried": False,
                "pred_label": pred_label,
                "correct": pred_label == label,
                "distance": distance,
            }
        )
    metrics = summarize_predictions(events, scaled.feature.dataset.novel_n)
    metrics.update(
        {
            "arm": (
                f"two_stage_{discovery_policy}_disc_{reuse_policy}_reuse_"
                f"{'radius' if use_radius else 'nearest'}"
            ),
            "prototype_types": len(prototypes),
            "prototype_radius": radius,
            "radius_enabled": use_radius,
            "discovery_policy": discovery_policy,
            "reuse_policy": reuse_policy,
        }
    )
    return metrics


def shuffled_discovery_items(scaled: Any) -> list[tuple[str, int, np.ndarray]]:
    items = [
        (label, index, vector)
        for label in sorted(scaled.discovery)
        for index, vector in enumerate(scaled.discovery[label])
    ]
    rng = np.random.default_rng(930_000 + scaled.feature.dataset.seed + scaled.feature.dataset.novel_n)
    rng.shuffle(items)
    return items


def run_budgeted_two_stage(
    scaled: Any,
    thresholds: dict[str, Any],
    query_budget: int,
    reuse_policy: str,
) -> dict[str, Any]:
    queried: dict[str, list[np.ndarray]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    feature = scaled.feature
    queries = 0
    for label, index, vector in shuffled_discovery_items(scaled):
        candidate, route_label = route_candidate(
            "gated",
            vector,
            float(feature.scores["discovery"][label][index]),
            thresholds,
        )
        do_query = bool(candidate and queries < query_budget)
        if do_query:
            queried[label].append(vector)
            queries += 1
        events.append(
            {
                "phase": "discovery",
                "true_label": label,
                "candidate": candidate,
                "queried": do_query,
                "pred_label": label if do_query else route_label,
                "correct": do_query,
            }
        )
    prototypes = centroids(queried)
    for (label, vector), (_, score) in zip(scaled.reuse, feature.scores["reuse"]):
        candidate, route_label = route_candidate(reuse_policy, vector, float(score), thresholds)
        pred_label = route_label
        distance = None
        if candidate:
            if prototypes:
                pred_label, distance = nearest(vector, prototypes)
            else:
                pred_label = "unknown"
        events.append(
            {
                "phase": "locked_reuse",
                "true_label": label,
                "candidate": candidate,
                "queried": False,
                "pred_label": pred_label,
                "correct": pred_label == label,
                "distance": distance,
            }
        )
    metrics = summarize_predictions(events, scaled.feature.dataset.novel_n)
    metrics.update(
        {
            "arm": f"two_stage_gated_budget{query_budget}_{reuse_policy}_reuse_nearest",
            "prototype_types": len(prototypes),
            "prototype_radius": float("inf"),
            "radius_enabled": False,
            "discovery_policy": f"gated_budget{query_budget}",
            "reuse_policy": reuse_policy,
            "query_budget": query_budget,
        }
    )
    return metrics


def memory_state_hash(memory: OnlinePrototypeMemory) -> str:
    return stable_hash(memory.state())


def run_online_flat(
    scaled: Any,
    thresholds: dict[str, Any],
    discovery_policy: str,
    reuse_policy: str,
    radius: float,
) -> dict[str, Any]:
    memory = OnlinePrototypeMemory(MemoryConfig(name="flat", hierarchical=False, radius=radius))
    feature = scaled.feature
    events: list[dict[str, Any]] = []
    step = 0
    for label in sorted(scaled.discovery):
        for index, vector in enumerate(scaled.discovery[label]):
            candidate, route_label = route_candidate(
                discovery_policy,
                vector,
                float(feature.scores["discovery"][label][index]),
                thresholds,
            )
            pred_label = route_label
            queried = False
            if candidate:
                key = observed_component_key(
                    feature.dataset.discovery[label][index],
                    feature.context,
                    threshold=1.6,
                )
                decision = memory.process(vector, key, label, True, step)
                pred_label = decision.pred_label
                queried = decision.queried
            events.append(
                {
                    "phase": "discovery",
                    "true_label": label,
                    "candidate": candidate,
                    "queried": queried,
                    "pred_label": pred_label,
                    "correct": pred_label == label,
                }
            )
            step += 1
    state_before = memory_state_hash(memory)
    for index, ((label, vector), (_, score)) in enumerate(zip(scaled.reuse, feature.scores["reuse"])):
        candidate, route_label = route_candidate(reuse_policy, vector, float(score), thresholds)
        pred_label = route_label
        distance = None
        if candidate:
            key = observed_component_key(
                feature.dataset.reuse[index][1],
                feature.context,
                threshold=1.6,
            )
            decision = memory.predict_locked(vector, key)
            pred_label = decision.pred_label
            distance = decision.distance
        events.append(
            {
                "phase": "locked_reuse",
                "true_label": label,
                "candidate": candidate,
                "queried": False,
                "pred_label": pred_label,
                "correct": pred_label == label,
                "distance": distance,
            }
        )
    state_after = memory_state_hash(memory)
    if state_before != state_after:
        raise AssertionError("online flat memory mutated during locked reuse")
    metrics = summarize_predictions(events, scaled.feature.dataset.novel_n)
    metrics.update(
        {
            "arm": f"online_flat_{discovery_policy}_disc_{reuse_policy}_reuse",
            "prototype_types": memory.committed_count,
            "prototype_radius": radius,
            "radius_enabled": True,
            "discovery_policy": discovery_policy,
            "reuse_policy": reuse_policy,
            "active_vocab": memory.active_count,
            "historical_clusters": memory.historical_clusters,
            "locked_state_unchanged": True,
        }
    )
    return metrics


def run_seed(seed: int, config: FollowupConfig, budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    dataset = build_seed_dataset(seed, config.ks[0], config)
    context = fit_feature_context(dataset.normal_stats)
    bundle = build_feature_bundle(dataset, FEATURE, context)
    scaled = scale_bundle(bundle, config.memory_discovery, scaler_scope="base_discovery")
    thresholds = gate_thresholds(scaled, GATE)
    online_radius = memory_radius(scaled, config.memory_radius_quantile)
    rows = [
        run_online_flat(scaled, thresholds, "gated", "gated", online_radius),
        run_online_flat(scaled, thresholds, "oracle", "oracle", online_radius),
    ]
    for discovery_policy, reuse_policy in (
        ("gated", "gated"),
        ("oracle", "gated"),
        ("gated", "oracle"),
        ("oracle", "oracle"),
    ):
        for use_radius in (False, True):
            rows.append(
                run_two_stage(
                    scaled,
                    thresholds,
                    discovery_policy,
                    reuse_policy,
                    use_radius,
                    config.memory_radius_quantile,
                )
            )
    for budget in budgets:
        rows.append(run_budgeted_two_stage(scaled, thresholds, budget, reuse_policy="gated"))
        rows.append(run_budgeted_two_stage(scaled, thresholds, budget, reuse_policy="oracle"))
    for row in rows:
        row.update(
            {
                "seed": seed,
                "novel_n": config.ks[0],
                "feature": FEATURE,
                "gate": GATE.name,
                "discovery_per_type": config.memory_discovery,
                "reuse_per_type": config.reuse_per_type,
                "online_radius": online_radius,
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["arm"]].append(row)
    output: list[dict[str, Any]] = []
    for arm, group in sorted(groups.items()):
        metrics = sorted(
            name
            for name, value in group[0].items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and name not in {"seed", "novel_n", "discovery_per_type", "reuse_per_type"}
        )
        for metric in metrics:
            values = [float(row[metric]) for row in group if np.isfinite(float(row[metric]))]
            if not values:
                continue
            mean, std = mean_std(values)
            low, high = bootstrap_ci(values, bootstrap_samples, seed=int(stable_hash([arm, metric])[:8], 16))
            output.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": len(values),
                }
            )
    return output


def lookup(summary: list[dict[str, Any]], arm: str, metric: str) -> dict[str, Any] | None:
    for row in summary:
        if row["arm"] == arm and row["metric"] == metric:
            return row
    return None


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    arms = [
        "online_flat_gated_disc_gated_reuse",
        "online_flat_oracle_disc_oracle_reuse",
        "two_stage_gated_disc_gated_reuse_nearest",
        "two_stage_oracle_disc_gated_reuse_nearest",
        "two_stage_gated_disc_oracle_reuse_nearest",
        "two_stage_oracle_disc_oracle_reuse_nearest",
    ]
    labels = [
        "Online flat\nreal gate",
        "Online flat\noracle cand.",
        "Two-stage\nreal gate",
        "Two-stage\noracle discovery",
        "Two-stage\noracle reuse",
        "Two-stage\noracle both",
    ]
    values = [lookup(summary, arm, "locked_macro_accuracy")["mean"] for arm in arms]
    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.bar(labels, values, color=["#4c566a", "#6b7280", "#2f6b9a", "#287f8e", "#2a8c67", "#b06c2f"])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Locked macro accuracy")
    axis.set_title("Memory isolation after observable72")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_report(output: Path, result_path: Path, figure_path: Path, summary: list[dict[str, Any]], config: FollowupConfig) -> None:
    arms = [
        "online_flat_gated_disc_gated_reuse",
        "online_flat_oracle_disc_oracle_reuse",
        "two_stage_gated_disc_gated_reuse_nearest",
        "two_stage_gated_disc_gated_reuse_radius",
        "two_stage_oracle_disc_gated_reuse_nearest",
        "two_stage_gated_disc_oracle_reuse_nearest",
        "two_stage_oracle_disc_oracle_reuse_nearest",
        "two_stage_oracle_disc_oracle_reuse_radius",
    ]
    lines = [
        "# Two-Stage Memory Isolation Follow-up",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"K={config.ks[0]}, D/type={config.memory_discovery}, reuse/type={config.reuse_per_type}, seeds={list(config.holdout_seeds)}.",
        "",
        "Representation/gate fixed to `observable72 + split_score0.95_known0.90`.",
        "",
        "| Arm | Locked macro | Type coverage | Unknown rate | Candidate recall | Discovery type coverage | Queries | Prototype types |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        rows = {
            metric: lookup(summary, arm, metric)
            for metric in (
                "locked_macro_accuracy",
                "locked_type_coverage",
                "locked_unknown_rate",
                "locked_candidate_recall",
                "discovery_type_coverage",
                "annotation_queries",
                "prototype_types",
            )
        }
        if rows["locked_macro_accuracy"] is None:
            continue
        lines.append(
            f"| {arm} | {pct(rows['locked_macro_accuracy']['mean'])} | "
            f"{pct(rows['locked_type_coverage']['mean'])} | "
            f"{pct(rows['locked_unknown_rate']['mean'])} | "
            f"{pct(rows['locked_candidate_recall']['mean'])} | "
            f"{pct(rows['discovery_type_coverage']['mean'])} | "
            f"{rows['annotation_queries']['mean']:.1f} | "
            f"{rows['prototype_types']['mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Budgeted Gated Two-Stage Sweep",
            "",
            "| Arm | Locked macro | Type coverage | Candidate recall | Discovery type coverage | Queries | Prototype types |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    budget_arms = sorted(
        {
            row["arm"]
            for row in summary
            if row["arm"].startswith("two_stage_gated_budget")
            and row["metric"] == "locked_macro_accuracy"
        },
        key=lambda arm: (
            int(arm.split("_budget", 1)[1].split("_", 1)[0]),
            "oracle" in arm,
        ),
    )
    for arm in budget_arms:
        rows = {
            metric: lookup(summary, arm, metric)
            for metric in (
                "locked_macro_accuracy",
                "locked_type_coverage",
                "locked_candidate_recall",
                "discovery_type_coverage",
                "annotation_queries",
                "prototype_types",
            )
        }
        lines.append(
            f"| {arm} | {pct(rows['locked_macro_accuracy']['mean'])} | "
            f"{pct(rows['locked_type_coverage']['mean'])} | "
            f"{pct(rows['locked_candidate_recall']['mean'])} | "
            f"{pct(rows['discovery_type_coverage']['mean'])} | "
            f"{rows['annotation_queries']['mean']:.1f} | "
            f"{rows['prototype_types']['mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `online_flat_*` keeps the original online assignment behavior.",
            "- `two_stage_*` removes discovery-time autonomous reuse and builds prototypes only after discovery.",
            "- `oracle_disc` gives every discovery sample an authorized label; `oracle_reuse` sends every locked novel sample to memory.",
            "- The gap between `online_flat_oracle_disc_oracle_reuse` and `two_stage_oracle_disc_oracle_reuse` measures online assignment/prototype pollution.",
            "- The gap between `two_stage_gated_disc_gated_reuse` and `two_stage_oracle_disc_oracle_reuse` measures the combined candidate/gate loss.",
            "",
            f"Result JSON: `{result_path}`",
            f"Figure: `{figure_path}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = base_config(args)
    budgets = (5,) if args.smoke else parse_csv_ints(args.budgets)
    started = time.time()
    rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in config.holdout_seeds:
        print(f"[seed] {seed}", flush=True)
        rows.extend(run_seed(seed, config, budgets))
    summary = summarize(rows, config.bootstrap_samples)
    result_path = args.output_dir / "two_stage_memory_isolation_result.json"
    figure_path = args.output_dir / "two_stage_memory_isolation.png"
    report_path = args.output_dir / "two_stage_memory_isolation_report.md"
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.time() - started,
        "config": asdict(config),
        "feature": FEATURE,
        "gate": GATE.name,
        "budgets": budgets,
        "rows": rows,
        "summary": summary,
    }
    result_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    plot(summary, figure_path)
    build_report(report_path, result_path, figure_path, summary, config)
    print(f"saved -> {result_path}", flush=True)
    print(f"saved -> {figure_path}", flush=True)
    print(f"saved -> {report_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="30,31,32,33,34")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--discovery", type=int, default=10)
    parser.add_argument("--reuse-per-type", type=int, default=5)
    parser.add_argument("--normal-stats-n", type=int, default=40)
    parser.add_argument("--normal-train-n", type=int, default=40)
    parser.add_argument("--normal-cal-n", type=int, default=40)
    parser.add_argument("--train-per-known-type", type=int, default=6)
    parser.add_argument("--cal-per-known-type", type=int, default=6)
    parser.add_argument("--memory-radius-quantile", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--budgets", default="35,75,150,300,600")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "two_stage_memory_isolation_2026-07-09",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
