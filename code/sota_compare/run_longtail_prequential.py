#!/usr/bin/env python3
"""Leakage-free many-type compositional memory experiment.

This runner replaces retrospective cluster-majority evaluation with an explicit
discovery/query phase followed by a frozen, query-free reuse phase.  It keeps a
fixed six-type known ontology, treats K as the number of novel types, isolates
compositional complexity from K, and compares all prototype arms with the same
detector, known-class gate, feature space, and radius scale.

The queried label is a controlled oracle used to isolate memory behavior.  It
must not be described as an LLM result.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import sigla_exp.longtail_bench as LT  # noqa: E402
import sigla_exp.ovbench as OV  # noqa: E402
from sigla_exp.prequential_birch import OnlineBirchMemory  # noqa: E402
from sigla_exp.prequential_memory import MemoryConfig, OnlinePrototypeMemory  # noqa: E402


METHODS = ("flat", "hier", "hier_fallback", "hier_merge", "hier_guard", "birch")
PRIMARY_METRICS = (
    "locked_macro_accuracy",
    "locked_micro_accuracy",
    "locked_type_coverage",
    "locked_unknown_rate",
    "bcubed_f1",
    "ari",
    "nmi",
    "annotation_queries",
    "query_rate_discovery",
    "active_vocab",
    "historical_clusters",
    "singleton_fraction",
    "normal_typed_far",
    "known_to_novel_rate",
)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: int
    seed_start: int
    ks: tuple[int, ...]
    radius_scales: tuple[float, ...]
    default_radius_scale: float
    train_per_type: int
    normal_train_n: int
    normal_stats_n: int
    score_cal_n: int
    discovery_repeats: int
    reuse_repeats: int
    discovery_normal_per_k: float
    discovery_known_per_k: float
    reuse_normal_per_k: float
    reuse_known_per_k: float
    known_radius_quantile: float
    score_quantile: float
    component_threshold: float
    merge_factor: float
    guard_confirm_k: int
    guard_reuse_margin: float
    complexity_k: int
    bootstrap_samples: int
    include_complexity: bool


@dataclass
class BaseState:
    ev_mu: dict[str, float]
    ev_sd: dict[str, float]
    scale_mu: np.ndarray
    scale_sd: np.ndarray
    centroids: dict[str, np.ndarray]
    known_radius: float
    score_threshold: float
    known_names: tuple[str, ...]


@dataclass
class StreamSample:
    phase: str
    true_label: str
    raw_feature: np.ndarray
    feature: np.ndarray
    key: tuple[str, ...]
    novel_rank: int | None


def parse_csv_int(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_float(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(sanitize(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance() -> dict[str, Any]:
    source_paths = [
        Path(__file__),
        ROOT / "sigla_exp" / "prequential_memory.py",
        ROOT / "sigla_exp" / "prequential_birch.py",
        ROOT / "sigla_exp" / "longtail_bench.py",
        ROOT / "sigla_exp" / "ovbench.py",
    ]
    dirty = git_value("status", "--short", "--untracked-files=all")
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "unavailable"
    return {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_status_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "source_sha256": {str(path.relative_to(REPO)): file_hash(path) for path in source_paths},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn_version,
    }


def vectorized_local_step(x: np.ndarray, window: int = 10) -> float:
    views = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
    medians = np.median(views, axis=-1)
    differences = medians[window:] - medians[:-window]
    return float(np.max(np.abs(differences)))


def vectorized_step_location_and_sign(x: np.ndarray, window: int = 10) -> tuple[float, float]:
    views = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
    medians = np.median(views, axis=-1)
    differences = medians[window:] - medians[:-window]
    flat_index = int(np.argmax(np.abs(differences)))
    time_index, dim_index = np.unravel_index(flat_index, differences.shape)
    best = float(differences[time_index, dim_index])
    location = (window + time_index) / LT.WIN
    return float(location), float(np.sign(best))


def vectorized_variance_localization(x: np.ndarray, kernel: int = 7, segments: int = 5) -> float:
    smooth = np.stack(
        [np.convolve(x[:, dim], np.ones(kernel) / kernel, mode="same") for dim in range(x.shape[1])],
        axis=1,
    )
    residual = x - smooth
    scales = []
    for segment in np.array_split(residual, segments, axis=0):
        median = np.median(segment, axis=0)
        scales.append(1.4826 * np.median(np.abs(segment - median), axis=0) + 1e-6)
    scale_array = np.stack(scales)
    ratios = np.max(scale_array, axis=0) / (np.median(scale_array, axis=0) + 1e-6)
    return float(np.max(ratios))


def vectorized_linear_r2(x: np.ndarray) -> float:
    time_index = np.arange(x.shape[0], dtype=np.float64)
    centered_time = time_index - time_index.mean()
    centered_x = x - x.mean(axis=0, keepdims=True)
    slopes = (centered_time[:, None] * centered_x).sum(axis=0) / np.square(centered_time).sum()
    fitted = x.mean(axis=0, keepdims=True) + centered_time[:, None] * slopes
    residual = np.square(x - fitted).sum(axis=0)
    total = np.square(centered_x).sum(axis=0) + 1e-6
    return float(np.max(np.clip(1.0 - residual / total, 0.0, 1.0)))


def average_absolute_correlation(segment: np.ndarray) -> float:
    mask = segment.std(axis=0) > 1e-6
    if int(mask.sum()) < 2:
        return 1.0
    centered = segment[:, mask] - segment[:, mask].mean(axis=0, keepdims=True)
    gram = centered.T @ centered
    norms = np.sqrt(np.diag(gram))
    correlation = gram / (np.outer(norms, norms) + 1e-12)
    count = correlation.shape[0]
    return float((np.abs(correlation).sum() - count) / (count * (count - 1)))


def vectorized_decorr(x: np.ndarray, window: int = 33) -> float:
    values = [
        average_absolute_correlation(x[start : start + window])
        for start in range(0, x.shape[0] - window + 1, window // 2)
    ]
    return float(1.0 - min(values))


def vectorized_corr_location(x: np.ndarray, window: int = 25) -> float:
    starts = list(range(0, x.shape[0] - window + 1, max(1, window // 2)))
    values = [average_absolute_correlation(x[start : start + window]) for start in starts]
    return float((starts[int(np.argmin(values))] + window / 2) / x.shape[0])


def fast_evidence(x: np.ndarray) -> dict[str, float]:
    mean = x.mean(0)
    variance = x.var(0) + 1e-9
    kurtosis = float(np.max(((x - mean) ** 4).mean(0) / variance**2 - 3.0))
    values = {
        "kurtosis": kurtosis,
        "local_step": vectorized_local_step(x),
        "spectral_peak": OV._spectral_peak(x),
        "var_localiz": vectorized_variance_localization(x),
        "lin_r2": vectorized_linear_r2(x),
        "decorr": vectorized_decorr(x),
    }
    return {key: float(value) if np.isfinite(value) else 0.0 for key, value in values.items()}


def fast_features(x: np.ndarray, mu: dict[str, float], sd: dict[str, float]) -> np.ndarray:
    evidence = fast_evidence(x)
    z = np.asarray([(evidence[name] - mu[name]) / (sd[name] + 1e-9) for name in LT.STATS], dtype=np.float32)
    z = np.clip(z, -2.0, 10.0)
    step_location, step_sign = vectorized_step_location_and_sign(x)
    extra = np.asarray(
        [
            LT._spike_location(x),
            step_location,
            LT._variance_location(x),
            vectorized_corr_location(x),
            LT._spectral_freq(x),
            LT._scope_estimate(x),
            LT._slope_sign(x),
            step_sign,
        ],
        dtype=np.float32,
    )
    return np.concatenate([z, extra]).astype(np.float32)


def validate_fast_features() -> dict[str, float]:
    rng = np.random.default_rng(913)
    mu, sd = LT.normal_stats(np.random.default_rng(914), n=20)
    maximum = 0.0
    for spec in [None, *LT.generate_taxonomy(6)]:
        window = LT.make_window(spec, rng)
        slow = LT.features(window, mu, sd)
        fast = fast_features(window, mu, sd)
        maximum = max(maximum, float(np.max(np.abs(slow - fast))))
    if maximum > 1e-5:
        raise AssertionError(f"fast feature implementation diverged: max_abs={maximum}")
    return {"max_abs_difference": maximum}


def normal_stats(rng: np.random.Generator, n: int) -> tuple[dict[str, float], dict[str, float]]:
    evidence = [fast_evidence(LT.make_window(None, rng)) for _ in range(n)]
    mu = {name: float(np.mean([row[name] for row in evidence])) for name in LT.STATS}
    sd = {name: float(np.std([row[name] for row in evidence]) + 1e-6) for name in LT.STATS}
    return mu, sd


def fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return features.mean(axis=0), features.std(axis=0) + 1e-6


def transform(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def centroids(features: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for feature, label in zip(features, labels):
        buckets[label].append(feature)
    return {label: np.mean(rows, axis=0) for label, rows in buckets.items()}


def nearest(vector: np.ndarray, centers: dict[str, np.ndarray]) -> tuple[str, float]:
    label = min(centers, key=lambda name: float(np.linalg.norm(vector - centers[name])))
    return label, float(np.linalg.norm(vector - centers[label]))


def prepare_base(seed: int, config: ExperimentConfig, known_specs: list[dict[str, Any]]) -> BaseState:
    rng = np.random.default_rng(100_000 + seed)
    ev_mu, ev_sd = normal_stats(rng, config.normal_stats_n)
    known_names = tuple(str(spec["name"]) for spec in known_specs)

    raw_features: list[np.ndarray] = []
    labels: list[str] = []
    for _ in range(config.normal_train_n):
        raw_features.append(fast_features(LT.make_window(None, rng), ev_mu, ev_sd))
        labels.append("normal")
    for spec in known_specs:
        for _ in range(config.train_per_type):
            raw_features.append(fast_features(LT.make_window(spec, rng), ev_mu, ev_sd))
            labels.append(str(spec["name"]))

    raw_array = np.stack(raw_features)
    scale_mu, scale_sd = fit_scaler(raw_array)
    scaled = transform(raw_array, scale_mu, scale_sd)
    centers = centroids(scaled, labels)
    distances = [float(np.linalg.norm(row - centers[label])) for row, label in zip(scaled, labels)]
    known_radius = float(np.quantile(distances, config.known_radius_quantile))

    calibration_scores = []
    for _ in range(config.score_cal_n):
        feature = fast_features(LT.make_window(None, rng), ev_mu, ev_sd)
        calibration_scores.append(LT.anomaly_score(feature))
    score_threshold = float(np.quantile(calibration_scores, config.score_quantile))
    return BaseState(
        ev_mu=ev_mu,
        ev_sd=ev_sd,
        scale_mu=scale_mu,
        scale_sd=scale_sd,
        centroids=centers,
        known_radius=known_radius,
        score_threshold=score_threshold,
        known_names=known_names,
    )


def component_key(raw_feature: np.ndarray, threshold: float) -> tuple[str, ...]:
    components = LT.component_signature(raw_feature, top=2, threshold=threshold)
    return tuple(sorted(components)) if components else ("unknown",)


def composite_order(seed: int, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(200_000 + seed)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        groups[tuple(sorted(str(item) for item in spec["components"]))].append(spec)
    pairs = sorted(groups)
    rng.shuffle(pairs)
    for pair in pairs:
        rng.shuffle(groups[pair])
    ordered: list[dict[str, Any]] = []
    for round_index in range(max(len(group) for group in groups.values())):
        rotated = pairs[round_index % len(pairs) :] + pairs[: round_index % len(pairs)]
        for pair in rotated:
            if round_index < len(groups[pair]):
                ordered.append(groups[pair][round_index])
    if len(ordered) != len(specs):
        raise AssertionError("composite manifest construction lost specifications")
    return ordered


def single_order(seed: int, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(300_000 + seed)
    indices = rng.permutation(len(specs))
    return [specs[int(index)] for index in indices]


def make_sample(
    phase: str,
    label: str,
    spec: dict[str, Any] | None,
    rank: int | None,
    rng: np.random.Generator,
    base: BaseState,
    config: ExperimentConfig,
) -> StreamSample:
    window = LT.make_window(spec, rng)
    raw_feature = fast_features(window, base.ev_mu, base.ev_sd)
    feature = transform(raw_feature, base.scale_mu, base.scale_sd)
    return StreamSample(
        phase=phase,
        true_label=label,
        raw_feature=raw_feature,
        feature=feature,
        key=component_key(raw_feature, config.component_threshold),
        novel_rank=rank,
    )


def build_stream(
    seed: int,
    track: str,
    novel_specs: list[dict[str, Any]],
    known_specs: list[dict[str, Any]],
    base: BaseState,
    config: ExperimentConfig,
) -> tuple[list[StreamSample], dict[str, Any]]:
    rng = np.random.default_rng(400_000 + 10_000 * seed + int(stable_hash(track)[:6], 16) % 10_000 + len(novel_specs))
    novel_by_name = {str(spec["name"]): spec for spec in novel_specs}
    known_by_name = {str(spec["name"]): spec for spec in known_specs}
    novel_rank = {name: rank for rank, name in enumerate(novel_by_name)}

    def descriptors(
        phase: str,
        novel_repeats: int,
        normal_count: int,
        known_count: int,
    ) -> list[tuple[str, dict[str, Any] | None, int | None]]:
        rows: list[tuple[str, dict[str, Any] | None, int | None]] = []
        for name, spec in novel_by_name.items():
            rows.extend((name, spec, novel_rank[name]) for _ in range(novel_repeats))
        rows.extend(("normal", None, None) for _ in range(normal_count))
        known_names = list(known_by_name)
        for _ in range(known_count):
            name = str(rng.choice(known_names))
            rows.append((name, known_by_name[name], None))
        rng.shuffle(rows)
        return rows

    novel_n = len(novel_specs)
    discovery = descriptors(
        "discovery",
        config.discovery_repeats,
        int(round(config.discovery_normal_per_k * novel_n)),
        int(round(config.discovery_known_per_k * novel_n)),
    )
    locked = descriptors(
        "locked_reuse",
        config.reuse_repeats,
        int(round(config.reuse_normal_per_k * novel_n)),
        int(round(config.reuse_known_per_k * novel_n)),
    )

    stream = [
        make_sample("discovery", label, spec, rank, rng, base, config)
        for label, spec, rank in discovery
    ]
    stream.extend(
        make_sample("locked_reuse", label, spec, rank, rng, base, config)
        for label, spec, rank in locked
    )
    manifest = {
        "track": track,
        "seed": seed,
        "known": [str(spec["name"]) for spec in known_specs],
        "novel": [
            {
                "rank": rank,
                "name": str(spec["name"]),
                "components": list(spec["components"]),
                "loc": spec["loc"],
                "scope": spec["scope"],
                "severity": spec["severity"],
            }
            for rank, spec in enumerate(novel_specs)
        ],
        "discovery_n": len(discovery),
        "locked_reuse_n": len(locked),
    }
    manifest["sha256"] = stable_hash(manifest)
    return stream, manifest


def common_route(sample: StreamSample, base: BaseState) -> dict[str, Any]:
    score = LT.anomaly_score(sample.raw_feature)
    anomaly = score > base.score_threshold
    if not anomaly:
        return {
            "candidate": False,
            "pred_label": "normal",
            "route": "normal",
            "score": score,
            "known_distance": None,
            "anomaly": False,
        }
    known_label, known_distance = nearest(sample.feature, base.centroids)
    if known_label in base.known_names and known_distance <= base.known_radius:
        return {
            "candidate": False,
            "pred_label": known_label,
            "route": "known",
            "score": score,
            "known_distance": known_distance,
            "anomaly": True,
        }
    return {
        "candidate": True,
        "pred_label": "unknown",
        "route": "memory_candidate",
        "score": score,
        "known_distance": known_distance,
        "anomaly": True,
    }


def make_method(method: str, radius: float, config: ExperimentConfig) -> Any:
    if method == "birch":
        return OnlineBirchMemory(threshold=radius)
    memory_config = MemoryConfig(
        name=method,
        hierarchical=method != "flat",
        radius=radius,
        merge_radius=radius * config.merge_factor if method in {"hier_merge", "hier_guard"} else None,
        confirm_k=config.guard_confirm_k if method == "hier_guard" else 1,
        reuse_margin=config.guard_reuse_margin if method == "hier_guard" else 1.0,
        fallback_global=method in {"hier_fallback", "hier_merge", "hier_guard"},
        block_label_conflict=method in {"hier_merge", "hier_guard"},
    )
    return OnlinePrototypeMemory(memory_config)


def memory_process(
    memory: Any,
    method: str,
    sample: StreamSample,
    novel_names: set[str],
    step: int,
) -> Any:
    if method == "birch":
        return memory.process(
            sample.feature,
            oracle_label=sample.true_label,
            commit_eligible=sample.true_label in novel_names,
            step=step,
        )
    return memory.process(
        sample.feature,
        sample.key,
        oracle_label=sample.true_label,
        commit_eligible=sample.true_label in novel_names,
        step=step,
    )


def memory_locked_predict(memory: Any, method: str, sample: StreamSample) -> Any:
    if method == "birch":
        return memory.predict_locked(sample.feature)
    return memory.predict_locked(sample.feature, sample.key)


def state_hash(memory: Any) -> str:
    return stable_hash(memory.state())


def run_method(
    method: str,
    radius_scale: float,
    stream: list[StreamSample],
    manifest: dict[str, Any],
    base: BaseState,
    config: ExperimentConfig,
    keep_trace: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    radius = base.known_radius * radius_scale
    memory = make_method(method, radius, config)
    novel_names = {row["name"] for row in manifest["novel"]}
    trace: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    locked_hash_before: str | None = None

    for step, sample in enumerate(stream):
        if sample.phase == "locked_reuse" and locked_hash_before is None:
            locked_hash_before = state_hash(memory)
        started = time.perf_counter()
        route = common_route(sample, base)
        action = route["route"]
        pred_label = route["pred_label"]
        cluster_id = None
        queried = False
        created = False
        autonomous_reuse = False
        distance = None
        active_clusters = memory.active_count
        historical_clusters = memory.historical_clusters
        merges = 0

        if route["candidate"]:
            if sample.phase == "discovery":
                decision = memory_process(memory, method, sample, novel_names, step)
            else:
                decision = memory_locked_predict(memory, method, sample)
            action = decision.action
            pred_label = decision.pred_label
            cluster_id = decision.cluster_id
            queried = decision.queried
            created = decision.created
            autonomous_reuse = decision.autonomous_reuse
            distance = decision.distance
            active_clusters = decision.active_clusters
            historical_clusters = decision.historical_clusters
            merges = decision.merges
        elapsed = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed)
        row = {
            "step": step,
            "phase": sample.phase,
            "true_label": sample.true_label,
            "novel_rank": sample.novel_rank,
            "key": list(sample.key),
            "candidate": route["candidate"],
            "anomaly": route["anomaly"],
            "score": route["score"],
            "known_distance": route["known_distance"],
            "action": action,
            "pred_label": pred_label,
            "prediction_before_reveal": "unknown" if queried else pred_label,
            "revealed_label": sample.true_label if queried else None,
            "cluster_id": cluster_id,
            "queried": queried,
            "created": created,
            "autonomous_reuse": autonomous_reuse,
            "correct": pred_label == sample.true_label,
            "distance": distance,
            "active_clusters": active_clusters,
            "historical_clusters": historical_clusters,
            "merges": merges,
        }
        trace.append(row)

    locked_hash_after = state_hash(memory)
    if locked_hash_before is None:
        raise AssertionError("stream did not contain a locked reuse phase")
    if locked_hash_before != locked_hash_after:
        raise AssertionError(f"{method} mutated memory during locked reuse")
    metrics = evaluate_trace(trace, novel_names, memory, latency_ms)
    metrics.update(
        {
            "method": method,
            "radius_scale": radius_scale,
            "radius": radius,
            "manifest_sha256": manifest["sha256"],
            "locked_state_unchanged": True,
            "locked_state_sha256": locked_hash_after,
        }
    )
    if keep_trace:
        metrics["final_memory_state"] = memory.state()
    return metrics, trace if keep_trace else []


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if len(array) else float("nan")


def comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def contingency_metrics(true_labels: list[str], cluster_labels: list[str]) -> dict[str, float]:
    if not true_labels:
        return {"bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0, "ari": 0.0, "nmi": 0.0}
    cells = Counter(zip(true_labels, cluster_labels))
    true_counts = Counter(true_labels)
    cluster_counts = Counter(cluster_labels)
    n = len(true_labels)

    bc_precision = sum(count * count / cluster_counts[cluster] for (label, cluster), count in cells.items()) / n
    bc_recall = sum(count * count / true_counts[label] for (label, cluster), count in cells.items()) / n
    bc_f1 = 2 * bc_precision * bc_recall / (bc_precision + bc_recall) if bc_precision + bc_recall else 0.0

    sum_cells = sum(comb2(count) for count in cells.values())
    sum_true = sum(comb2(count) for count in true_counts.values())
    sum_clusters = sum(comb2(count) for count in cluster_counts.values())
    total_pairs = comb2(n)
    expected = sum_true * sum_clusters / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (sum_true + sum_clusters)
    ari = (sum_cells - expected) / (maximum - expected) if maximum != expected else 1.0

    mutual_information = 0.0
    for (label, cluster), count in cells.items():
        mutual_information += (count / n) * math.log((count * n) / (true_counts[label] * cluster_counts[cluster]))
    true_entropy = -sum((count / n) * math.log(count / n) for count in true_counts.values())
    cluster_entropy = -sum((count / n) * math.log(count / n) for count in cluster_counts.values())
    nmi = mutual_information / math.sqrt(true_entropy * cluster_entropy) if true_entropy and cluster_entropy else 1.0
    return {
        "bcubed_precision": float(bc_precision),
        "bcubed_recall": float(bc_recall),
        "bcubed_f1": float(bc_f1),
        "ari": float(ari),
        "nmi": float(nmi),
    }


def evaluate_trace(
    trace: list[dict[str, Any]],
    novel_names: set[str],
    memory: Any,
    latency_ms: list[float],
) -> dict[str, Any]:
    discovery = [row for row in trace if row["phase"] == "discovery"]
    locked = [row for row in trace if row["phase"] == "locked_reuse"]
    novel_locked = [row for row in locked if row["true_label"] in novel_names]
    normal_locked = [row for row in locked if row["true_label"] == "normal"]
    known_locked = [row for row in locked if row["true_label"] not in novel_names and row["true_label"] != "normal"]

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in novel_locked:
        by_type[row["true_label"]].append(row)
    per_type_accuracy = {
        label: safe_mean(float(row["correct"]) for row in rows) for label, rows in by_type.items()
    }
    covered = {label for label, rows in by_type.items() if any(row["correct"] for row in rows)}
    autonomous = [row for row in novel_locked if row["autonomous_reuse"]]

    cluster_labels = []
    true_labels = []
    for row in novel_locked:
        assignment = row["cluster_id"]
        if assignment is None:
            assignment = f"route:{row['pred_label']}"
        cluster_labels.append(str(assignment))
        true_labels.append(str(row["true_label"]))
    clustering = contingency_metrics(true_labels, cluster_labels)

    final_novel_predictions = [row for row in locked if row["pred_label"] in novel_names]
    normal_typed_far = safe_mean(row["pred_label"] in novel_names for row in normal_locked)
    known_to_novel = safe_mean(row["pred_label"] in novel_names for row in known_locked)
    metrics = {
        "locked_novel_n": len(novel_locked),
        "locked_macro_accuracy": safe_mean(per_type_accuracy.values()),
        "locked_micro_accuracy": safe_mean(row["correct"] for row in novel_locked),
        "locked_type_coverage": len(covered) / len(novel_names) if novel_names else 0.0,
        "locked_unknown_rate": safe_mean(row["pred_label"] == "unknown" for row in novel_locked),
        "locked_autonomous_reuse_rate": len(autonomous) / len(novel_locked) if novel_locked else 0.0,
        "locked_conditional_reuse_accuracy": safe_mean(row["correct"] for row in autonomous),
        "novel_detection_recall": safe_mean(row["anomaly"] for row in novel_locked),
        "novel_candidate_recall": safe_mean(row["candidate"] for row in novel_locked),
        "normal_anomaly_far": safe_mean(row["anomaly"] for row in normal_locked),
        "normal_typed_far": normal_typed_far,
        "known_to_novel_rate": known_to_novel,
        "typed_prediction_precision": safe_mean(row["correct"] for row in final_novel_predictions),
        "annotation_queries": sum(row["queried"] for row in discovery),
        "query_rate_discovery": safe_mean(row["queried"] for row in discovery),
        "candidate_rate_discovery": safe_mean(row["candidate"] for row in discovery),
        "query_created_clusters": sum(row["created"] for row in discovery),
        "active_vocab": memory.active_count,
        "committed_vocab": memory.committed_count,
        "historical_clusters": memory.historical_clusters,
        "singleton_fraction": memory.singleton_fraction,
        "merge_precision": memory.merge_precision,
        "merge_count": len(getattr(memory, "merge_events", [])),
        "latency_p50_ms": float(np.percentile(latency_ms, 50)),
        "latency_p95_ms": float(np.percentile(latency_ms, 95)),
        "per_type_accuracy": per_type_accuracy,
        **clustering,
    }
    return metrics


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


def summarize(rows: list[dict[str, Any]], config: ExperimentConfig) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["track"], row["novel_n"], row["method"], row["radius_scale"])
        groups[key].append(row)
    summary: list[dict[str, Any]] = []
    for (track, novel_n, method, radius_scale), group in sorted(groups.items()):
        metrics = sorted(
            key
            for key, value in group[0].items()
            if isinstance(value, (int, float)) and key not in {"seed", "novel_n", "radius_scale", "radius"}
        )
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None and np.isfinite(row[metric])]
            if not values:
                continue
            mean, std = mean_std(values)
            low, high = bootstrap_ci(values, config.bootstrap_samples, seed=int(stable_hash([track, novel_n, method, metric])[:8], 16))
            summary.append(
                {
                    "track": track,
                    "novel_n": novel_n,
                    "method": method,
                    "radius_scale": radius_scale,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": len(values),
                }
            )
    return summary


def exact_sign_flip_p(differences: list[float]) -> float:
    observed = abs(float(np.mean(differences)))
    n = len(differences)
    if not n:
        return float("nan")
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        value = abs(float(np.mean(np.asarray(differences) * np.asarray(signs))))
        extreme += value >= observed - 1e-12
        total += 1
    return extreme / total


def paired_comparisons(rows: list[dict[str, Any]], config: ExperimentConfig) -> list[dict[str, Any]]:
    contrasts = (
        ("hier", "flat"),
        ("hier_fallback", "hier"),
        ("hier_merge", "hier_fallback"),
        ("hier_guard", "hier_merge"),
        ("birch", "flat"),
    )
    comparisons = []
    for track in sorted({row["track"] for row in rows}):
        for novel_n in sorted({row["novel_n"] for row in rows if row["track"] == track}):
            subset = [
                row
                for row in rows
                if row["track"] == track
                and row["novel_n"] == novel_n
                and row["radius_scale"] == config.default_radius_scale
            ]
            by_method_seed = {(row["method"], row["seed"]): row for row in subset}
            for first, second in contrasts:
                seeds = sorted(
                    seed
                    for method, seed in by_method_seed
                    if method == first and (second, seed) in by_method_seed
                )
                if not seeds:
                    continue
                differences = [
                    by_method_seed[(first, seed)]["locked_macro_accuracy"]
                    - by_method_seed[(second, seed)]["locked_macro_accuracy"]
                    for seed in seeds
                ]
                low, high = bootstrap_ci(
                    differences,
                    config.bootstrap_samples,
                    seed=int(stable_hash([track, novel_n, first, second])[:8], 16),
                )
                comparisons.append(
                    {
                        "track": track,
                        "novel_n": novel_n,
                        "contrast": f"{first}-{second}",
                        "metric": "locked_macro_accuracy",
                        "mean_delta": float(np.mean(differences)),
                        "median_delta": float(np.median(differences)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "exact_sign_flip_p": exact_sign_flip_p(differences),
                        "seed_differences": differences,
                    }
                )
    ordered = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["exact_sign_flip_p"])
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * comparisons[index]["exact_sign_flip_p"])
        running = max(running, adjusted)
        comparisons[index]["holm_p"] = running
    return comparisons


def summary_value(
    summary: list[dict[str, Any]],
    track: str,
    novel_n: int,
    method: str,
    radius_scale: float,
    metric: str,
) -> dict[str, Any] | None:
    for row in summary:
        if (
            row["track"] == track
            and row["novel_n"] == novel_n
            and row["method"] == method
            and row["radius_scale"] == radius_scale
            and row["metric"] == metric
        ):
            return row
    return None


def plot_results(summary: list[dict[str, Any]], config: ExperimentConfig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "flat": "#4c566a",
        "hier": "#2f6b9a",
        "hier_fallback": "#287f8e",
        "hier_merge": "#2a8c67",
        "hier_guard": "#b06c2f",
        "birch": "#7b4f9d",
    }
    labels = {
        "flat": "Flat prototypes",
        "hier": "Component-key memory",
        "hier_fallback": "+ global fallback",
        "hier_merge": "+ merge",
        "hier_guard": "+ merge + guard",
        "birch": "Online BIRCH",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    ks = list(config.ks)
    for method in METHODS:
        means, stds = [], []
        for novel_n in ks:
            row = summary_value(
                summary,
                "scale_composite",
                novel_n,
                method,
                config.default_radius_scale,
                "locked_macro_accuracy",
            )
            means.append(float("nan") if row is None else row["mean"])
            stds.append(0.0 if row is None else row["std"])
        axes[0].errorbar(ks, means, yerr=stds, marker="o", capsize=2, color=colors[method], label=labels[method])
    axes[0].set_title("Frozen future reuse")
    axes[0].set_xlabel("Novel type count K")
    axes[0].set_ylabel("Macro exact accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)

    for method in METHODS:
        means = []
        for novel_n in ks:
            row = summary_value(
                summary,
                "scale_composite",
                novel_n,
                method,
                config.default_radius_scale,
                "locked_type_coverage",
            )
            means.append(float("nan") if row is None else row["mean"])
        axes[1].plot(ks, means, marker="o", color=colors[method], label=labels[method])
    axes[1].set_title("Correctly reusable type coverage")
    axes[1].set_xlabel("Novel type count K")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)

    max_k = max(ks)
    for method in METHODS:
        xs, ys = [], []
        for scale in config.radius_scales:
            query = summary_value(summary, "scale_composite", max_k, method, scale, "annotation_queries")
            accuracy = summary_value(summary, "scale_composite", max_k, method, scale, "locked_macro_accuracy")
            if query is not None and accuracy is not None:
                xs.append(query["mean"])
                ys.append(accuracy["mean"])
        axes[2].plot(xs, ys, marker="o", color=colors[method], label=labels[method])
    axes[2].set_title(f"Radius operating curve (K={max_k})")
    axes[2].set_xlabel("Discovery annotation queries")
    axes[2].set_ylabel("Locked macro accuracy")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Corrected prequential many-type composite memory experiment", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def report_table(
    summary: list[dict[str, Any]],
    track: str,
    ks: Iterable[int],
    methods: Iterable[str],
    scale: float,
) -> list[str]:
    lines = [
        "| Track | K | Method | Locked macro acc. | Type coverage | Candidate recall | Unknown rate | Queries | Committed/active vocab | B-cubed F1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for novel_n in ks:
        for method in methods:
            values = {
                metric: summary_value(summary, track, novel_n, method, scale, metric)
                for metric in (
                    "locked_macro_accuracy",
                    "locked_type_coverage",
                    "locked_unknown_rate",
                    "novel_candidate_recall",
                    "annotation_queries",
                    "active_vocab",
                    "committed_vocab",
                    "bcubed_f1",
                )
            }
            if values["locked_macro_accuracy"] is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        track,
                        str(novel_n),
                        method,
                        pct(values["locked_macro_accuracy"]["mean"]),
                        pct(values["locked_type_coverage"]["mean"]),
                        pct(values["novel_candidate_recall"]["mean"]),
                        pct(values["locked_unknown_rate"]["mean"]),
                        f"{values['annotation_queries']['mean']:.1f}",
                        f"{values['committed_vocab']['mean']:.1f}/{values['active_vocab']['mean']:.1f}",
                        pct(values["bcubed_f1"]["mean"]),
                    ]
                )
                + " |"
            )
    return lines


def build_report(
    output: Path,
    result_json: Path,
    trace_path: Path,
    figure_path: Path,
    config: ExperimentConfig,
    summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> None:
    lines = [
        "# Corrected Prequential Many-Type Composite Memory Experiment",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Scope",
        "",
        "This experiment replaces the previous retrospective cluster-majority metric. It uses a controlled annotation oracle only during discovery, then freezes every memory for a locked reuse phase with no query and no update. The result isolates memory reuse; it is not an LLM naming experiment.",
        "",
        "Key protocol changes:",
        "",
        "- Six fixed known base families for every K and seed.",
        "- K means the number of novel types; the scaling track uses only two-family composites.",
        "- Every type has exactly the configured discovery and future-reuse opportunities.",
        "- All methods share the detector, known gate, features, stream, and radius scale.",
        "- No generator `spec_signature`, no test-label majority map, and no query in locked reuse.",
        "- The component key is derived from observed evidence only; location/scope/severity are intentionally excluded because the current generator does not implement those fields consistently for every composite.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(sanitize(asdict(config)), indent=2),
        "```",
        "",
        "## Main Scaling Result",
        "",
        *report_table(summary, "scale_composite", config.ks, METHODS, config.default_radius_scale),
        "",
    ]
    if config.include_complexity:
        lines.extend(
            [
                "## Complexity Control (fixed K)",
                "",
                *report_table(
                    summary,
                    "complexity_single",
                    [config.complexity_k],
                    METHODS,
                    config.default_radius_scale,
                ),
                "",
                *report_table(
                    summary,
                    "complexity_composite",
                    [config.complexity_k],
                    METHODS,
                    config.default_radius_scale,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Paired Contrasts",
            "",
        "| Track | K | Contrast | Mean delta | 95% bootstrap CI | Exact p | Holm p |",
        "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['track']} | {row['novel_n']} | {row['contrast']} | {pct(row['mean_delta'])} | "
            f"[{pct(row['ci95_low'])}, {pct(row['ci95_high'])}] | {row['exact_sign_flip_p']:.4f} | {row['holm_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These are pilot-scale paired tests. With five seeds, the exact sign-flip test has coarse resolution; effect sizes and intervals are more informative than a binary significance claim.",
            "",
            "## Integrity Checks",
            "",
            f"- Fast feature maximum absolute difference: `{integrity['fast_features']['max_abs_difference']:.3g}`.",
            f"- Locked memories unchanged: `{integrity['all_locked_states_unchanged']}`.",
            f"- Paired stream manifest agreement across methods/scales: `{integrity['paired_manifest_agreement']}`.",
            f"- Result JSON: `{result_json}`.",
            f"- Default-scale trace: `{trace_path}`.",
            f"- Figure: `{figure_path}`.",
            "",
            "## Interpretation Rules",
            "",
            "1. `locked_macro_accuracy` is the primary reuse metric. UNKNOWN and undiscovered types count as incorrect.",
            "2. Queried discovery windows are never counted as reuse.",
            "3. `annotation_queries` are controlled oracle labels, not LLM/API calls.",
            "4. Radius sweeps are operating curves, not matched-query comparisons; raw method rows have different query counts.",
            "5. `hier_guard` is only a confirmation/margin heuristic in this pilot, not the full do-no-harm guard from the paper draft.",
            "6. Every type has balanced exposure. This is a many-type compositional scaling study, not a frequency long-tail experiment; a separate Zipf stream is still required.",
            "7. This pilot still uses evidence-aligned synthetic mechanisms. External typed/OOD data and a chronological real-background split remain required before an AAAI main claim.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fast_validation = validate_fast_features()
    catalog = LT.generate_taxonomy(216)
    known_specs = [spec for spec in catalog if len(spec["components"]) == 1][:6]
    single_specs = [spec for spec in catalog[6:] if len(spec["components"]) == 1]
    composite_specs = [spec for spec in catalog if len(spec["components"]) == 2]
    if len(known_specs) != 6 or len(single_specs) != 30 or len(composite_specs) != 180:
        raise AssertionError(
            f"unexpected taxonomy sizes: known={len(known_specs)} single={len(single_specs)} composite={len(composite_specs)}"
        )
    if max(config.ks) > len(composite_specs):
        raise ValueError("requested K exceeds the composite taxonomy pool")
    if config.complexity_k > min(len(single_specs), len(composite_specs)):
        raise ValueError("complexity_k exceeds an available taxonomy pool")

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    started = time.time()
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        print(f"[base] seed={seed}", flush=True)
        base = prepare_base(seed, config, known_specs)
        composite_manifest = composite_order(seed, composite_specs)
        single_manifest = single_order(seed, single_specs)
        datasets: list[tuple[str, list[dict[str, Any]]]] = [
            ("scale_composite", composite_manifest[:novel_n]) for novel_n in config.ks
        ]
        if config.include_complexity:
            datasets.extend(
                [
                    ("complexity_single", single_manifest[: config.complexity_k]),
                    ("complexity_composite", composite_manifest[: config.complexity_k]),
                ]
            )
        for track, novel_specs in datasets:
            novel_n = len(novel_specs)
            print(f"[stream] seed={seed} track={track} K={novel_n}", flush=True)
            stream, manifest = build_stream(seed, track, novel_specs, known_specs, base, config)
            manifests.append(manifest)
            for radius_scale in config.radius_scales:
                for method in METHODS:
                    keep_trace = radius_scale == config.default_radius_scale
                    metrics, trace = run_method(
                        method,
                        radius_scale,
                        stream,
                        manifest,
                        base,
                        config,
                        keep_trace,
                    )
                    metrics.update({"seed": seed, "track": track, "novel_n": novel_n})
                    rows.append(metrics)
                    if keep_trace:
                        traces.append(
                            {
                                "seed": seed,
                                "track": track,
                                "novel_n": novel_n,
                                "method": method,
                                "radius_scale": radius_scale,
                                "manifest_sha256": manifest["sha256"],
                                "events": trace,
                            }
                        )
            default_rows = [
                row
                for row in rows
                if row["seed"] == seed
                and row["track"] == track
                and row["novel_n"] == novel_n
                and row["radius_scale"] == config.default_radius_scale
            ]
            print(
                "  "
                + " | ".join(
                    f"{row['method']} reuse={row['locked_macro_accuracy']:.1%} "
                    f"cov={row['locked_type_coverage']:.1%} q={row['annotation_queries']}"
                    for row in default_rows
                ),
                flush=True,
            )

    summary = summarize(rows, config)
    comparisons = paired_comparisons(rows, config)
    manifest_groups: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    for row in rows:
        manifest_groups[(row["seed"], row["track"], row["novel_n"])].add(row["manifest_sha256"])
    integrity = {
        "fast_features": fast_validation,
        "all_locked_states_unchanged": all(row["locked_state_unchanged"] for row in rows),
        "paired_manifest_agreement": all(len(values) == 1 for values in manifest_groups.values()),
        "manifest_count": len(manifests),
    }
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.time() - started,
        "config": asdict(config),
        "provenance": provenance(),
        "integrity": integrity,
        "taxonomy": {
            "known_n": len(known_specs),
            "single_pool_n": len(single_specs),
            "composite_pool_n": len(composite_specs),
        },
        "manifests": manifests,
        "rows": rows,
        "summary": summary,
        "paired_comparisons": comparisons,
    }

    result_json = output_dir / "manytype_prequential_result.json"
    trace_path = output_dir / "manytype_prequential_trace.json.gz"
    figure_path = output_dir / "manytype_prequential_results.png"
    report_path = output_dir / "manytype_prequential_report.md"
    result_json.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(sanitize({"schema_version": 1, "traces": traces}), handle)
    plot_results(summary, config, figure_path)
    build_report(
        report_path,
        result_json,
        trace_path,
        figure_path,
        config,
        summary,
        comparisons,
        integrity,
    )
    print(f"saved result -> {result_json}", flush=True)
    print(f"saved trace  -> {trace_path}", flush=True)
    print(f"saved figure -> {figure_path}", flush=True)
    print(f"saved report -> {report_path}", flush=True)
    return payload


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        return ExperimentConfig(
            seeds=1,
            seed_start=0,
            ks=(6,),
            radius_scales=(1.0,),
            default_radius_scale=1.0,
            train_per_type=3,
            normal_train_n=24,
            normal_stats_n=20,
            score_cal_n=20,
            discovery_repeats=2,
            reuse_repeats=2,
            discovery_normal_per_k=0.5,
            discovery_known_per_k=0.25,
            reuse_normal_per_k=0.5,
            reuse_known_per_k=0.25,
            known_radius_quantile=0.90,
            score_quantile=0.95,
            component_threshold=1.6,
            merge_factor=1.15,
            guard_confirm_k=2,
            guard_reuse_margin=0.85,
            complexity_k=6,
            bootstrap_samples=500,
            include_complexity=False,
        )
    scales = parse_csv_float(args.radius_scales)
    if args.default_radius_scale not in scales:
        raise ValueError("default radius scale must be included in --radius-scales")
    return ExperimentConfig(
        seeds=args.seeds,
        seed_start=args.seed_start,
        ks=parse_csv_int(args.ks),
        radius_scales=scales,
        default_radius_scale=args.default_radius_scale,
        train_per_type=args.train_per_type,
        normal_train_n=args.normal_train_n,
        normal_stats_n=args.normal_stats_n,
        score_cal_n=args.score_cal_n,
        discovery_repeats=args.discovery_repeats,
        reuse_repeats=args.reuse_repeats,
        discovery_normal_per_k=0.5,
        discovery_known_per_k=0.25,
        reuse_normal_per_k=1.0,
        reuse_known_per_k=0.5,
        known_radius_quantile=0.90,
        score_quantile=0.95,
        component_threshold=1.6,
        merge_factor=1.15,
        guard_confirm_k=2,
        guard_reuse_margin=0.85,
        complexity_k=args.complexity_k,
        bootstrap_samples=args.bootstrap_samples,
        include_complexity=not args.no_complexity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--ks", default="10,25,50,100")
    parser.add_argument("--radius-scales", default="0.7,1.0,1.3")
    parser.add_argument("--default-radius-scale", type=float, default=1.0)
    parser.add_argument("--train-per-type", type=int, default=8)
    parser.add_argument("--normal-train-n", type=int, default=64)
    parser.add_argument("--normal-stats-n", type=int, default=80)
    parser.add_argument("--score-cal-n", type=int, default=80)
    parser.add_argument("--discovery-repeats", type=int, default=2)
    parser.add_argument("--reuse-repeats", type=int, default=3)
    parser.add_argument("--complexity-k", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--no-complexity", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "longtail_prequential_2026-07-09",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print(json.dumps(sanitize(asdict(config)), indent=2), flush=True)
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
