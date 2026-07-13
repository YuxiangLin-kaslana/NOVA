#!/usr/bin/env python3
"""Streaming sklearn BIRCH plus the fair pilot's shared deterministic namer.

This is a standard sklearn.cluster.Birch implementation adapted to the
repository's six-dimensional structured-evidence stream.  It is an adjacent
streaming-clustering baseline, not a reproduction of a time-series GCD paper.

The online order is strict: predict from the old CF tree, decide whether the
current point needs a semantic query, issue that query, update BIRCH, and only
then attach the returned name to a newly formed subcluster.  A queried window
is never counted as autonomous reuse.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.cluster import Birch


CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
sys.path.insert(0, str(CODE))

import scripts.exp_detection_tie as DT  # noqa: E402
import sigla_exp.ovbench as CB  # noqa: E402
import sota_compare.realbench as RB  # noqa: E402
import sota_compare.run_fair_openvocab_ablation as FAIR  # noqa: E402


ANOMALY = "anomaly"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Config:
    seeds: int
    seed_start: int
    backgrounds: tuple[str, ...]
    normal_stats_n: int
    train_per_class: int
    normal_train_n: int
    normal_cal_n: int
    warm_n: int
    post_n: int
    namer_threshold: float
    calibration_quantile: float
    training_radius_quantile: float
    birch_threshold_quantile: float
    branching_factor: int


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def provenance() -> dict[str, Any]:
    dependencies = [
        Path(__file__).resolve(),
        CODE / "sota_compare" / "run_fair_openvocab_ablation.py",
        CODE / "sigla_exp" / "ovbench.py",
        CODE / "sota_compare" / "realbench.py",
    ]
    dirty = git_value("status", "--short", "--untracked-files=all")
    return {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_dirty": int(bool(dirty and dirty != "unavailable")),
        "git_status_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(REPO)): file_hash(path) for path in dependencies
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "birch_class": "sklearn.cluster.Birch",
        "command": [sys.executable, *sys.argv],
    }


def make_known_training(
    rng: np.random.Generator, n_per_class: int
) -> tuple[list[np.ndarray], list[str]]:
    """Reproduce the fair runner's labeled training-window generation exactly."""
    windows: list[np.ndarray] = []
    labels: list[str] = []
    for label in DT.BASE_VOCAB:
        for _ in range(n_per_class):
            windows.append(FAIR.make_window(label, rng))
            labels.append(label)
    return windows, labels


def class_radius(
    features: np.ndarray, labels: list[str], quantile: float
) -> float:
    distances: list[float] = []
    label_array = np.asarray(labels)
    for label in DT.BASE_VOCAB:
        rows = features[label_array == label]
        center = rows.mean(axis=0)
        distances.extend(float(np.linalg.norm(row - center)) for row in rows)
    return float(np.quantile(distances, quantile))


def calibrated_threshold(
    train_features: np.ndarray,
    train_labels: list[str],
    extra_normal_features: np.ndarray,
    normal_cal_features: np.ndarray,
    config: Config,
) -> tuple[float, float, dict[str, float]]:
    """Use only training and held-out normal calibration data, never stream data."""
    label_array = np.asarray(train_labels)
    normal_fit = np.vstack(
        [train_features[label_array == DT.NORMAL], extra_normal_features]
    )
    normal_center = normal_fit.mean(axis=0)
    normal_cal_distances = np.linalg.norm(
        normal_cal_features - normal_center[None, :], axis=1
    )
    known_radius = class_radius(
        train_features, train_labels, config.training_radius_quantile
    )
    birch_threshold = class_radius(
        train_features, train_labels, config.birch_threshold_quantile
    )
    normal_radius = float(
        np.quantile(normal_cal_distances, config.calibration_quantile)
    )
    threshold = max(0.05, known_radius, normal_radius)
    return threshold, max(0.05, birch_threshold), {
        "known_training_radius": known_radius,
        "normal_calibration_radius": normal_radius,
        "selected_candidate_radius": threshold,
        "selected_birch_cf_threshold": max(0.05, birch_threshold),
    }


def majority_cluster_labels(
    assignments: np.ndarray, labels: list[str], n_clusters: int
) -> tuple[dict[int, str | None], float]:
    buckets: dict[int, Counter[str]] = defaultdict(Counter)
    for assignment, label in zip(assignments, labels):
        buckets[int(assignment)][label] += 1
    mapping: dict[int, str | None] = {}
    correct = 0
    total = 0
    label_rank = {label: index for index, label in enumerate(DT.BASE_VOCAB)}
    for cluster in range(n_clusters):
        counts = buckets.get(cluster, Counter())
        if not counts:
            mapping[cluster] = None
            continue
        label = sorted(
            counts, key=lambda item: (-counts[item], label_rank.get(item, 10_000), item)
        )[0]
        mapping[cluster] = label
        correct += counts[label]
        total += sum(counts.values())
    purity = correct / total if total else 0.0
    return mapping, float(purity)


def carry_labels(
    old_centers: np.ndarray,
    old_labels: dict[int, str | None],
    new_centers: np.ndarray,
) -> tuple[dict[int, str | None], set[int]]:
    """Match old and updated CF centers one-to-one; leave new centers unlabeled."""
    mapping = {index: None for index in range(len(new_centers))}
    if not len(old_centers) or not len(new_centers):
        return mapping, set(range(len(new_centers)))
    costs = np.linalg.norm(
        old_centers[:, None, :] - new_centers[None, :, :], axis=2
    )
    old_indices, new_indices = linear_sum_assignment(costs)
    matched_new: set[int] = set()
    for old_index, new_index in zip(old_indices.tolist(), new_indices.tolist()):
        mapping[new_index] = old_labels.get(old_index)
        matched_new.add(new_index)
    return mapping, set(range(len(new_centers))) - matched_new


def run_birch_stream(
    initial_features: np.ndarray,
    initial_labels: list[str],
    windows_evidence: list[dict[str, float]],
    stream_features: np.ndarray,
    mu: dict[str, float],
    sd: dict[str, float],
    candidate_radius: float,
    birch_threshold: float,
    config: Config,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    model = Birch(
        threshold=birch_threshold,
        branching_factor=config.branching_factor,
        n_clusters=None,
        compute_labels=True,
    )
    model.fit(initial_features)
    initial_assignments = model.predict(initial_features)
    cluster_labels, initial_purity = majority_cluster_labels(
        initial_assignments, initial_labels, len(model.subcluster_centers_)
    )

    rule_config = type("RuleConfig", (), {
        "namer_mode": "rule",
        "namer_threshold": config.namer_threshold,
    })()
    namer, _ = FAIR.make_namer(rule_config, "")
    predictions: list[str] = []
    events: list[dict[str, Any]] = []
    vocab = set(DT.BASE_VOCAB)
    semantic_attach_conflicts = 0
    semantic_attachments = 0

    for step, (row, feature) in enumerate(zip(windows_evidence, stream_features)):
        vector = feature.reshape(1, -1)
        old_centers = np.asarray(model.subcluster_centers_, dtype=np.float64).copy()
        old_labels = dict(cluster_labels)

        # 1. Predict and route using only the state available before this window.
        pre_cluster = int(model.predict(vector)[0])
        pre_distance = float(np.linalg.norm(feature - old_centers[pre_cluster]))
        remembered_label = old_labels.get(pre_cluster)
        candidate = remembered_label is None or pre_distance > candidate_radius
        prediction_before_query = UNKNOWN if candidate else str(remembered_label)

        # 2. Query semantics before any current-window model update.
        queried = bool(candidate)
        discovery_name = namer(row, feature, mu, sd) if queried else None
        if discovery_name is not None:
            discovery_name = str(discovery_name)
            vocab.add(discovery_name)
        final_prediction = (
            discovery_name if queried and discovery_name is not None
            else ANOMALY if queried
            else prediction_before_query
        )

        # 3. Update BIRCH only after prediction and the optional query are logged.
        model.partial_fit(vector)
        new_centers = np.asarray(model.subcluster_centers_, dtype=np.float64).copy()
        cluster_labels, unmatched = carry_labels(old_centers, old_labels, new_centers)
        post_cluster = int(model.predict(vector)[0])

        # 4. A returned semantic name can label an updated/new subcluster only now.
        attached = False
        attach_conflict = False
        if queried and discovery_name is not None:
            if unmatched:
                attach_index = min(
                    unmatched, key=lambda index: float(np.linalg.norm(feature - new_centers[index]))
                )
            else:
                attach_index = post_cluster
            current = cluster_labels.get(attach_index)
            if current in {None, discovery_name}:
                cluster_labels[attach_index] = discovery_name
                attached = True
                semantic_attachments += 1
            else:
                attach_conflict = True
                semantic_attach_conflicts += 1

        autonomous_reuse = bool(
            not queried and prediction_before_query not in {DT.NORMAL, UNKNOWN, ANOMALY}
        )
        predictions.append(str(final_prediction))
        events.append(
            {
                "step": step,
                "operation_order": [
                    "predict_old_tree",
                    "candidate_decision",
                    "query_if_candidate",
                    "partial_fit_current_window",
                    "attach_semantics_after_update",
                ],
                "prediction_before_query": prediction_before_query,
                "queried": queried,
                "candidate": candidate,
                "discovery_name": discovery_name,
                "autonomous_reuse": autonomous_reuse,
                "pre_cluster": pre_cluster,
                "pre_distance": pre_distance,
                "pre_cluster_label": remembered_label,
                "post_cluster": post_cluster,
                "cluster_count_before": len(old_centers),
                "cluster_count_after": len(new_centers),
                "semantic_attached_after_update": attached,
                "semantic_attach_conflict": attach_conflict,
                "query_before_model_update": queried,
                "route": "birch_candidate_query" if queried else "birch_cluster_reuse",
            }
        )

    spurious = sorted(vocab - set(DT.BASE_VOCAB) - {DT.NOVEL})
    return predictions, events, {
        "initial_subclusters": int(len(set(initial_assignments.tolist()))),
        "initial_cluster_purity": initial_purity,
        "active_prototypes": int(len(model.subcluster_centers_)),
        "final_vocab_size": len(vocab),
        "final_vocab_labels": sorted(vocab),
        "spurious_vocab_labels": spurious,
        "spurious_vocab_count": len(spurious),
        "grew_novel": int(DT.NOVEL in vocab),
        "semantic_attachments": semantic_attachments,
        "semantic_attach_conflicts": semantic_attach_conflicts,
        "memory_reuses_total": int(sum(event["autonomous_reuse"] for event in events)),
        "queried_autonomous_overlap_count": int(
            sum(event["queried"] and event["autonomous_reuse"] for event in events)
        ),
        "operation_order_trace_sha256": stable_hash(
            [event["operation_order"] for event in events]
        ),
        "birch_cf_threshold": birch_threshold,
        "candidate_radius": candidate_radius,
    }


def custom_metrics(
    predictions: list[str],
    events: list[dict[str, Any]],
    truths: list[str],
    onset: int,
) -> dict[str, Any]:
    post_predictions = predictions[onset:]
    post_events = events[onset:]
    post_truths = truths[onset:]
    novel = [i for i, truth in enumerate(post_truths) if truth == DT.NOVEL]
    known = [i for i, truth in enumerate(post_truths) if truth in DT.KNOWN_ANOM]
    normal = [i for i, truth in enumerate(post_truths) if truth == DT.NORMAL]
    novel_queries = [i for i in novel if post_events[i]["queried"]]
    first_query = novel_queries[0] if novel_queries else None
    correct_query_names = [
        i for i in novel_queries if post_events[i]["discovery_name"] == DT.NOVEL
    ]
    correct_discoveries = [
        i for i in correct_query_names
        if post_events[i]["semantic_attached_after_update"]
    ]
    first_correct = correct_discoveries[0] if correct_discoveries else None
    until = [i for i in novel if first_correct is None or i <= first_correct]
    future = [i for i in novel if first_correct is not None and i > first_correct]
    future_reuse = [i for i in future if post_events[i]["autonomous_reuse"]]
    return {
        "novel_candidate_recall": float(
            np.mean([post_events[i]["candidate"] for i in novel])
        ),
        "novel_candidate_recall_until_correct_discovery": float(
            np.mean([post_events[i]["candidate"] for i in until])
        ),
        "novel_candidate_or_correct_reuse_recall": float(
            np.mean(
                [
                    post_events[i]["candidate"]
                    or post_events[i]["prediction_before_query"] == DT.NOVEL
                    for i in novel
                ]
            )
        ),
        "first_novel_query_post_index": first_query,
        "first_novel_query_name": (
            post_events[first_query]["discovery_name"] if first_query is not None else None
        ),
        "first_novel_query_name_correct": (
            int(post_events[first_query]["discovery_name"] == DT.NOVEL)
            if first_query is not None
            else None
        ),
        "eventual_correct_novel_query_name": int(bool(correct_query_names)),
        "eventual_correct_novel_discovery": int(bool(correct_discoveries)),
        "correct_discovery_occurred": int(bool(correct_discoveries)),
        "correct_discovery_post_index": first_correct,
        "correct_discovery_novel_ordinal": (
            novel.index(first_correct) + 1 if first_correct is not None else None
        ),
        "post_discovery_novel_n": len(future),
        "post_discovery_future_reuse_count": len(future_reuse),
        "post_discovery_future_reuse_rate": (
            len(future_reuse) / len(future) if future else None
        ),
        "post_discovery_future_reuse_accuracy": (
            float(np.mean([post_predictions[i] == DT.NOVEL for i in future_reuse]))
            if future_reuse
            else None
        ),
        "preupdate_known_typed_accuracy": float(
            np.mean(
                [post_events[i]["prediction_before_query"] == post_truths[i] for i in known]
            )
        ),
        "preupdate_normal_false_alarm_rate": float(
            np.mean(
                [post_events[i]["prediction_before_query"] != DT.NORMAL for i in normal]
            )
        ),
        "candidate_precision_any_anomaly": (
            float(
                np.mean(
                    [post_truths[i] != DT.NORMAL for i, event in enumerate(post_events) if event["candidate"]]
                )
            )
            if any(event["candidate"] for event in post_events)
            else None
        ),
        "candidate_count_post": int(sum(event["candidate"] for event in post_events)),
    }


def run_seed(
    background: dict[str, Any], seed: int, config: Config
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng_stats = np.random.default_rng(100_000 + seed)
    rng_train = np.random.default_rng(200_000 + seed)
    rng_unsup = np.random.default_rng(300_000 + seed)
    rng_stream = np.random.default_rng(400_000 + seed)

    mu, sd = CB.normal_stats(rng_stats, n=config.normal_stats_n)
    train_windows, train_labels = make_known_training(rng_train, config.train_per_class)
    normal_train = [CB.make_window(None, rng_unsup) for _ in range(config.normal_train_n)]
    normal_cal = [CB.make_window(None, rng_unsup) for _ in range(config.normal_cal_n)]
    stream_windows, truths, onset = FAIR.build_stream(rng_stream, config)

    _, train_features = FAIR.evidence_and_z(train_windows, mu, sd)
    _, normal_train_features = FAIR.evidence_and_z(normal_train, mu, sd)
    _, normal_cal_features = FAIR.evidence_and_z(normal_cal, mu, sd)
    stream_evidence, stream_features = FAIR.evidence_and_z(stream_windows, mu, sd)
    candidate_radius, birch_threshold, calibration = calibrated_threshold(
        train_features,
        train_labels,
        normal_train_features,
        normal_cal_features,
        config,
    )
    initial_features = np.vstack([train_features, normal_train_features])
    initial_labels = [*train_labels, *([DT.NORMAL] * len(normal_train_features))]

    predictions, events, extra = run_birch_stream(
        initial_features,
        initial_labels,
        stream_evidence,
        stream_features,
        mu,
        sd,
        candidate_radius,
        birch_threshold,
        config,
    )
    extra.update(calibration)
    row = FAIR.metrics(predictions, events, truths, onset, extra)
    row.update(custom_metrics(predictions, events, truths, onset))
    row.update(
        {
            "background": background["name"],
            "background_kind": background["kind"],
            "seed": seed,
            "method": "sklearn_birch_shared_deterministic_namer",
            "implementation_status": "standard_sklearn_birch_adapted_to_streaming_evidence",
            "official_external_implementation": True,
            "paper_specific_time_series_gcd_reproduction": False,
            "namer_call_rate_post": row["namer_calls_post"] / config.post_n,
        }
    )

    stream_manifest = {
        "background": background,
        "seed": seed,
        "onset": onset,
        "labels": truths,
        "window_digest": hashlib.sha256(np.stack(stream_windows).tobytes()).hexdigest(),
        "normal_train_digest": hashlib.sha256(np.stack(normal_train).tobytes()).hexdigest(),
        "normal_calibration_digest": hashlib.sha256(np.stack(normal_cal).tobytes()).hexdigest(),
        "score_detector_calibration_split_shared": True,
    }
    manifest = {
        "background": background["name"],
        "seed": seed,
        "sha256": stable_hash(stream_manifest),
        "onset": onset,
        "post_label_counts": dict(Counter(truths[onset:])),
        "window_digest": stream_manifest["window_digest"],
        "normal_train_digest": stream_manifest["normal_train_digest"],
        "normal_calibration_digest": stream_manifest["normal_calibration_digest"],
    }
    return row, manifest


SUMMARY_METRICS = [
    "binary_f1",
    "novel_detection_recall",
    "novel_candidate_recall",
    "novel_candidate_recall_until_correct_discovery",
    "novel_candidate_or_correct_reuse_recall",
    "novel_typed_accuracy_including_queries",
    "first_occurrence_pre_update_detection",
    "first_occurrence_pre_update_typed_correct",
    "first_occurrence_queried",
    "first_novel_query_name_correct",
    "eventual_correct_novel_query_name",
    "eventual_correct_novel_discovery",
    "correct_discovery_novel_ordinal",
    "post_discovery_future_reuse_accuracy",
    "post_discovery_future_reuse_rate",
    "post_discovery_future_reuse_count",
    "known_typed_accuracy",
    "preupdate_known_typed_accuracy",
    "normal_false_alarm_rate",
    "preupdate_normal_false_alarm_rate",
    "namer_calls_warm",
    "namer_calls_post",
    "namer_calls_total",
    "candidate_count_post",
    "candidate_precision_any_anomaly",
    "spurious_vocab_count",
    "active_prototypes",
    "initial_cluster_purity",
    "semantic_attach_conflicts",
    "birch_cf_threshold",
    "candidate_radius",
]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["background"]].append(row)
    output: list[dict[str, Any]] = []
    for background, group in sorted(groups.items()):
        for metric in SUMMARY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ]
            if values:
                output.append(
                    {
                        "background": background,
                        "method": "sklearn_birch_shared_deterministic_namer",
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }
                )
    return output


def lookup(summary: list[dict[str, Any]], background: str, metric: str) -> float | None:
    for row in summary:
        if row["background"] == background and row["metric"] == metric:
            return float(row["mean"])
    return None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def compare_fair_manifests(
    manifests: list[dict[str, Any]], reference_path: Path, config: Config
) -> dict[str, Any]:
    if not reference_path.exists():
        return {
            "reference": str(reference_path),
            "checked": False,
            "reason": "reference result not found",
        }
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    compatible = (
        reference.get("config", {}).get("warm_n") == config.warm_n
        and reference.get("config", {}).get("post_n") == config.post_n
        and reference.get("config", {}).get("normal_train_n") == config.normal_train_n
        and reference.get("config", {}).get("normal_cal_n") == config.normal_cal_n
        and reference.get("config", {}).get("cnn_train_per_class") == config.train_per_class
    )
    if not compatible:
        return {
            "reference": str(reference_path),
            "checked": False,
            "reason": "smoke/nonmatching dimensions",
        }
    expected = {
        (item["background"], int(item["seed"])): item["sha256"]
        for item in reference.get("manifests", [])
    }
    matches = [
        expected.get((item["background"], int(item["seed"]))) == item["sha256"]
        for item in manifests
    ]
    return {
        "reference": str(reference_path),
        "checked": True,
        "all_stream_and_split_manifests_match": bool(matches and all(matches)),
        "matched": int(sum(matches)),
        "total": len(matches),
    }


def integrity_checks(
    rows: list[dict[str, Any]], manifests: list[dict[str, Any]], fair_match: dict[str, Any], config: Config
) -> dict[str, Any]:
    expected_rows = len(config.backgrounds) * config.seeds
    query_reuse_disjoint = all(
        row["post_discovery_future_reuse_count"] <= row["post_discovery_novel_n"]
        and row["namer_calls_total"] == row["namer_calls_warm"] + row["namer_calls_post"]
        and row["queried_autonomous_overlap_count"] == 0
        for row in rows
    )
    bounds = all(
        0 <= row["namer_calls_warm"] <= config.warm_n
        and 0 <= row["namer_calls_post"] <= config.post_n
        for row in rows
    )
    checks = {
        "expected_rows": len(rows) == expected_rows,
        "manifest_count": len(manifests) == expected_rows,
        "query_and_reuse_accounting_disjoint": query_reuse_disjoint,
        "query_counts_phase_bounded": bounds,
        "predict_query_update_order_encoded_per_event": True,
        "fair_stream_manifest_match": (
            fair_match.get("all_stream_and_split_manifests_match", True)
            if fair_match.get("checked")
            else None
        ),
        "rows": len(rows),
    }
    required = [value for value in checks.values() if isinstance(value, bool)]
    if not all(required):
        raise AssertionError(f"integrity failure: {checks}")
    return checks


def build_report(payload: dict[str, Any], path: Path) -> None:
    config = payload["config"]
    summary = payload["summary"]
    lines = [
        "# sklearn BIRCH + Shared Deterministic Namer",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Scope and Status",
        "",
        f"This baseline uses the standard `sklearn.cluster.Birch` implementation from scikit-learn {payload['provenance']['sklearn']}, with `n_clusters=None`. It is adapted to the same six-dimensional structured-evidence stream as the fair pilot. It is not an implementation or claimed reproduction of a time-series generalized-category-discovery paper.",
        "",
        "The semantic interface is exactly the fair pilot's deterministic evidence-argmax namer. It is evidence-aligned and is not an LLM result.",
        "",
        "## Data and Calibration",
        "",
        f"- Backgrounds: `{config['backgrounds']}`; the SMD condition is controlled injection into normal SMD 1-1 background, not native typed faults.",
        f"- Seeds: {config['seed_start']} through {config['seed_start'] + config['seeds'] - 1}.",
        f"- Known labeled training: {config['train_per_class']} windows/class for `{DT.BASE_VOCAB}`.",
        f"- Additional normal training/calibration: {config['normal_train_n']}/{config['normal_cal_n']} independent windows.",
        f"- Stream: {config['warm_n']} warm + {config['post_n']} post-onset windows.",
        f"- BIRCH branching factor: {config['branching_factor']}; the CF threshold is the q{100 * config['birch_threshold_quantile']:.0f} within-known-class training radius. The separate semantic-candidate radius is the maximum of the q{100 * config['training_radius_quantile']:.0f} within-known-class radius and q{100 * config['calibration_quantile']:.0f} held-out-normal radius. No stream/test window calibrates either value.",
        "- The exact stream, normal-train split, and normal-calibration split are digest-compared with the completed fair pilot.",
        "",
        "## Strict Online Protocol",
        "",
        "For each window: (1) predict using the old CF tree and old cluster-name map; (2) declare a candidate if its old cluster is unnamed or its distance exceeds the calibrated candidate radius; (3) call the shared namer if needed; (4) call `partial_fit`; (5) align old/new CF centers and attach a returned name only after the update. A queried window has `autonomous_reuse=false` by construction. The BIRCH CF threshold remains the standard clustering hyperparameter and is distinct from the semantic-candidate radius.",
        "",
        "Known cluster names are initialized by majority label assignment on the labeled training split. This gives BIRCH the same opportunity as prototype baselines to retain known types; novel labels are absent from training.",
        "",
        "## Five-Seed Results",
        "",
        "| Background | Novel candidate recall | Candidate or correct reuse recall | First pre-update typed | First novel query name correct | Any correct query name | Eventual stored discovery | Future autonomous reuse acc. | Future autonomous reuse coverage | Queries post | Known typed retention | Normal FAR | Spurious vocab |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for background in config["backgrounds"]:
        lines.append(
            f"| {background} | {pct(lookup(summary, background, 'novel_candidate_recall'))} | "
            f"{pct(lookup(summary, background, 'novel_candidate_or_correct_reuse_recall'))} | "
            f"{pct(lookup(summary, background, 'first_occurrence_pre_update_typed_correct'))} | "
            f"{pct(lookup(summary, background, 'first_novel_query_name_correct'))} | "
            f"{pct(lookup(summary, background, 'eventual_correct_novel_query_name'))} | "
            f"{pct(lookup(summary, background, 'eventual_correct_novel_discovery'))} | "
            f"{pct(lookup(summary, background, 'post_discovery_future_reuse_accuracy'))} | "
            f"{pct(lookup(summary, background, 'post_discovery_future_reuse_rate'))} | "
            f"{num(lookup(summary, background, 'namer_calls_post'))} | "
            f"{pct(lookup(summary, background, 'known_typed_accuracy'))} | "
            f"{pct(lookup(summary, background, 'normal_false_alarm_rate'))} | "
            f"{num(lookup(summary, background, 'spurious_vocab_count'))} |"
        )
    lines.extend(
        [
            "",
            "`Eventual stored discovery` requires both a correct novel query name and successful post-update attachment to a CF subcluster. `Future autonomous reuse` is conditional on that stored discovery and includes only later novel windows with `queried=false`. The BIRCH tree and CF centers continue to update online, so this is autonomous online-updated reuse, not locked-state reuse.",
            "",
            "### Supporting Metrics",
            "",
            "| Background | Novel detection recall | Typed incl. query | Candidate recall until discovery | Candidate precision | Queries warm | Initial cluster purity | Final CF subclusters | Attach conflicts | CF threshold | Candidate radius |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for background in config["backgrounds"]:
        lines.append(
            f"| {background} | {pct(lookup(summary, background, 'novel_detection_recall'))} | "
            f"{pct(lookup(summary, background, 'novel_typed_accuracy_including_queries'))} | "
            f"{pct(lookup(summary, background, 'novel_candidate_recall_until_correct_discovery'))} | "
            f"{pct(lookup(summary, background, 'candidate_precision_any_anomaly'))} | "
            f"{num(lookup(summary, background, 'namer_calls_warm'))} | "
            f"{pct(lookup(summary, background, 'initial_cluster_purity'))} | "
            f"{num(lookup(summary, background, 'active_prototypes'))} | "
            f"{num(lookup(summary, background, 'semantic_attach_conflicts'))} | "
            f"{num(lookup(summary, background, 'birch_cf_threshold'))} | "
            f"{num(lookup(summary, background, 'candidate_radius'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "1. BIRCH clusters structured statistics, not raw time-series windows or learned embeddings. The evidence design is type-aligned, so the deterministic namer can be unusually strong.",
            "2. This is a Level-1 constrained naming pilot. The held-out label is absent from training but remains available to the shared rule namer through the statistic-to-concept mapping.",
            "3. Initial cluster semantics use supervised known labels. Query counts therefore cover novel/ambiguous online semantics, not initial vocabulary construction.",
            "4. Cluster identities are carried across `partial_fit` by one-to-one minimum-distance matching of old and new CF centers. A genuinely new center is unnamed until the current query is attached after update.",
            "5. `Future autonomous reuse accuracy` is undefined in seeds without a correct discovery; its aggregate `n` can therefore be smaller than five and must be read with eventual discovery and coverage.",
            "6. This baseline is a useful streaming-clustering control, but it does not replace official OpenMax, TGCD, continual-NCD, or learned-representation baselines.",
            "",
            "## Integrity Checks",
            "",
            *[f"- `{key}`: `{value}`" for key, value in payload["integrity"].items()],
            "",
            "## Artifacts",
            "",
            f"- Runner: `{Path(__file__).resolve()}`",
            f"- Result JSON: `{payload['artifacts']['result_json']}`",
            f"- Report: `{payload['artifacts']['report_markdown']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "birch_shared_namer_2026-07-09",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--backgrounds", default="synthetic,SMD:1-1")
    parser.add_argument("--normal-stats-n", type=int, default=60)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--normal-train-n", type=int, default=64)
    parser.add_argument("--normal-cal-n", type=int, default=48)
    parser.add_argument("--warm-n", type=int, default=60)
    parser.add_argument("--post-n", type=int, default=120)
    parser.add_argument("--namer-threshold", type=float, default=2.0)
    parser.add_argument("--calibration-quantile", type=float, default=0.95)
    parser.add_argument("--training-radius-quantile", type=float, default=0.90)
    parser.add_argument("--birch-threshold-quantile", type=float, default=0.50)
    parser.add_argument("--branching-factor", type=int, default=50)
    parser.add_argument(
        "--reference-fair-result",
        type=Path,
        default=REPO
        / "docs"
        / "fair_openvocab_ablation_2026-07-09"
        / "fair_openvocab_ablation_result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        seeds=args.seeds,
        seed_start=args.seed_start,
        backgrounds=tuple(
            item.strip() for item in args.backgrounds.split(",") if item.strip()
        ),
        normal_stats_n=args.normal_stats_n,
        train_per_class=args.train_per_class,
        normal_train_n=args.normal_train_n,
        normal_cal_n=args.normal_cal_n,
        warm_n=args.warm_n,
        post_n=args.post_n,
        namer_threshold=args.namer_threshold,
        calibration_quantile=args.calibration_quantile,
        training_radius_quantile=args.training_radius_quantile,
        birch_threshold_quantile=args.birch_threshold_quantile,
        branching_factor=args.branching_factor,
    )
    if config.seeds < 1 or config.warm_n < 1 or config.post_n < 1:
        raise ValueError("seeds, warm_n, and post_n must be positive")
    if not 0 < config.calibration_quantile < 1:
        raise ValueError("calibration quantile must be in (0, 1)")
    if not 0 < config.training_radius_quantile < 1:
        raise ValueError("training radius quantile must be in (0, 1)")
    if not 0 < config.birch_threshold_quantile < 1:
        raise ValueError("BIRCH threshold quantile must be in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    backgrounds: list[dict[str, Any]] = []
    try:
        for name in config.backgrounds:
            background = FAIR.activate_background(name)
            backgrounds.append(background)
            print(f"[background] {name}: {background['kind']}", flush=True)
            for seed in range(config.seed_start, config.seed_start + config.seeds):
                row, manifest = run_seed(background, seed, config)
                rows.append(row)
                manifests.append(manifest)
                print(
                    f"  [seed {seed}] cand={row['novel_candidate_recall']:.1%} "
                    f"discover={row['eventual_correct_novel_discovery']:.0%} "
                    f"reuse={row['post_discovery_future_reuse_accuracy']} "
                    f"queries={row['namer_calls_post']} firstPreTyped="
                    f"{row['first_occurrence_pre_update_typed_correct']}",
                    flush=True,
                )
    finally:
        RB.deactivate()

    summary = summarize(rows)
    fair_match = compare_fair_manifests(manifests, args.reference_fair_result, config)
    integrity = integrity_checks(rows, manifests, fair_match, config)
    result_json = args.output_dir / "birch_shared_namer_result.json"
    report_md = args.output_dir / "birch_shared_namer_report.md"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": asdict(config),
        "method": {
            "name": "sklearn_birch_shared_deterministic_namer",
            "birch": "sklearn.cluster.Birch",
            "n_clusters": None,
            "cf_threshold_selection": "training within-class distance quantile",
            "implementation": "standard sklearn implementation",
            "adaptation": "online structured-evidence clustering with post-update semantic attachment",
            "paper_specific_time_series_gcd_reproduction": False,
            "shared_namer": "fair-pilot deterministic evidence argmax rule",
        },
        "protocol": {
            "known_vocab": DT.BASE_VOCAB,
            "known_anomalies": DT.KNOWN_ANOM,
            "novel": DT.NOVEL,
            "prediction_order": [
                "predict_old_tree",
                "candidate_decision",
                "query_if_candidate",
                "partial_fit_current_window",
                "attach_semantics_after_update",
            ],
            "query_window_counted_as_future_reuse": False,
            "tree_updates_during_future_reuse": True,
            "locked_state_reuse": False,
            "stream_shared_with_fair_pilot": True,
            "real_background_is_controlled_injection": True,
            "native_typed_real_faults": False,
            "naming_level": "Level 1 constrained evidence-aligned naming",
        },
        "backgrounds": backgrounds,
        "provenance": provenance(),
        "fair_manifest_comparison": fair_match,
        "integrity": integrity,
        "manifests": manifests,
        "rows": rows,
        "summary": summary,
        "warnings": [
            "BIRCH operates on hand-designed structured evidence, not raw windows or learned embeddings.",
            "The deterministic namer has access to a statistic-to-concept mapping and is not an LLM.",
            "This is a Level-1 constrained naming pilot, not free-form semantic induction.",
            "SMD uses controlled injections on normal real background, not native typed faults.",
            "Future reuse is conditional and online-updated, not locked-state reuse.",
            "This standard sklearn baseline does not represent an official time-series GCD method.",
        ],
        "artifacts": {
            "result_json": str(result_json),
            "report_markdown": str(report_md),
            "runner": str(Path(__file__).resolve()),
        },
    }
    result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_report(payload, report_md)
    print(f"saved result -> {result_json}")
    print(f"saved report -> {report_md}")


if __name__ == "__main__":
    main()
