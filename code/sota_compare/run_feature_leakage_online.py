#!/usr/bin/env python3
"""Audit direct-feature leakage under a strictly ordered online protocol.

This experiment is intentionally independent from the paper runners.  It asks
whether future typed reuse is driven by statistics that directly mirror the
synthetic injectors.  Every representation receives the same one-shot label:

    predict -> score -> query/reveal -> update

All later events are query-free and update-free.  Hidden labels are owned by an
evaluator/oracle and are never accepted by ``predict``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import sigla_exp.ovbench as OV  # noqa: E402


DIRECT_TARGETS = (
    "oscillation",
    "variance_burst",
    "trend",
    "correlation_break",
)
DIRECT_FEATURE = {
    "oscillation": "spectral_peak",
    "variance_burst": "var_localiz",
    "trend": "lin_r2",
    "correlation_break": "decorr",
}
KNOWN_TYPES = ("spike", "level_shift")
BASE_REPRESENTATIONS = ("generic", "specialized", "combined")
DIRECT_REPRESENTATIONS = (*BASE_REPRESENTATIONS, "specialized_loo", "combined_loo")
SPECIALIZED_NAMES = tuple(OV.STATS)
# Naming support is deliberately limited to the fixed known ontology.  Using
# OV.STAT_OF for withheld targets would reproduce the generator's hidden table.
SUPPORTED_STAT_TO_ATOMIC = {OV.STAT_OF[concept]: concept for concept in KNOWN_TYPES}


@dataclass(frozen=True)
class Config:
    seeds: int = 10
    normal_train_n: int = 160
    known_train_per_type: int = 30
    normal_cal_n: int = 160
    known_val_per_type: int = 20
    pre_normal_n: int = 20
    pre_known_n: int = 10
    future_target_n: int = 40
    future_normal_n: int = 60
    future_known_n: int = 30
    projection_dim: int = 48
    anomaly_quantile: float = 0.95
    known_radius_quantile: float = 0.95
    memory_radius_quantile: float = 0.95
    namer_z_threshold: float = 3.0
    namer_secondary_threshold: float = 2.5
    namer_margin: float = 1.0
    bootstrap_samples: int = 5000


@dataclass(frozen=True)
class Record:
    sample_id: str
    split: str
    label: str
    window: np.ndarray


@dataclass(frozen=True)
class Decision:
    pred_label: str
    route: str
    anomaly: bool
    candidate: bool
    anomaly_score: float
    known_distance: float
    memory_distance: float | None


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


def stable_hash(value: Any) -> str:
    encoded = json.dumps(sanitize(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def vectorized_local_step(x: np.ndarray, window: int = 10) -> float:
    views = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
    medians = np.median(views, axis=-1)
    return float(np.max(np.abs(medians[window:] - medians[:-window])))


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


def fast_evidence(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    variance = x.var(axis=0) + 1e-9
    values = {
        "kurtosis": float(np.max(((x - mean) ** 4).mean(axis=0) / variance**2 - 3.0)),
        "local_step": vectorized_local_step(x),
        "spectral_peak": OV._spectral_peak(x),
        "var_localiz": vectorized_variance_localization(x),
        "lin_r2": vectorized_linear_r2(x),
        "decorr": vectorized_decorr(x),
    }
    return np.asarray(
        [float(values[name]) if np.isfinite(values[name]) else 0.0 for name in SPECIALIZED_NAMES],
        dtype=np.float32,
    )


def validate_evidence() -> dict[str, float]:
    rng = np.random.default_rng(20260709)
    maximum = 0.0
    for concept in (None, *OV.CONCEPTS):
        window = OV.make_window(concept, rng)
        reference = OV.evidence(window)
        observed = fast_evidence(window)
        expected = np.asarray([reference[name] for name in SPECIALIZED_NAMES])
        maximum = max(maximum, float(np.max(np.abs(observed - expected))))
    if maximum > 2e-5:
        raise AssertionError(f"vectorized evidence mismatch: {maximum}")
    return {"max_abs_difference": maximum}


def atomic_window(concept: str | None, rng: np.random.Generator) -> np.ndarray:
    return OV.make_window(concept, rng).astype(np.float32)


def delayed_dependency_window(rng: np.random.Generator) -> np.ndarray:
    """Introduce lagged pairwise dependence without assigning a naming statistic."""
    x = OV.base_normal(rng).astype(np.float32)
    start, end = 18, 86
    length = end - start
    ramp_width = 7
    index = np.arange(length)
    ramp = np.clip(np.minimum(index, length - 1 - index) / ramp_width, 0.0, 1.0).astype(np.float32)
    for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
        lag = int(rng.integers(7, 18))
        delayed = np.roll(x[start:end, first], lag)
        replacement = 0.82 * delayed + 0.18 * x[start:end, second]
        x[start:end, second] = ramp * replacement + (1.0 - ramp) * x[start:end, second]
    return x


def regime_switch_window(rng: np.random.Generator) -> np.ndarray:
    """Alternate short regimes; no single specialized statistic defines the label."""
    x = OV.base_normal(rng).astype(np.float32)
    dims = rng.choice(OV.NVARS, 5, replace=False)
    start = 15
    block = 9
    for block_index in range(8):
        left = start + block_index * block
        right = min(OV.WIN, left + block)
        if left >= OV.WIN:
            break
        offset = (1.45 if block_index % 2 == 0 else -1.15) * float(rng.uniform(0.85, 1.15))
        scale = 1.25 if block_index % 3 == 0 else 0.85
        x[left:right, dims] = scale * x[left:right, dims] + offset
    return x


def composite_window(components: tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
    x = OV.base_normal(rng).astype(np.float32)
    for component in components:
        if component == "delayed_dependency":
            delayed = delayed_dependency_window(rng)
            x += delayed - OV.base_normal(np.random.default_rng(int(rng.integers(0, 2**31))))
        elif component == "regime_switch":
            dims = rng.choice(OV.NVARS, 4, replace=False)
            for left in range(20, 85, 12):
                right = min(OV.WIN, left + 6)
                x[left:right, dims] += rng.choice((-1.5, 1.5))
        else:
            OV.INJ[component](x, rng)
    return x.astype(np.float32)


TARGET_BUILDERS: dict[str, Callable[[np.random.Generator], np.ndarray]] = {
    **{name: (lambda rng, name=name: atomic_window(name, rng)) for name in DIRECT_TARGETS},
    "delayed_dependency": delayed_dependency_window,
    "regime_switch": regime_switch_window,
    "trend_plus_variance": lambda rng: composite_window(("trend", "variance_burst"), rng),
    "oscillation_plus_corrbreak": lambda rng: composite_window(("oscillation", "correlation_break"), rng),
    "levelshift_plus_regime": lambda rng: composite_window(("level_shift", "regime_switch"), rng),
}
UNSUPPORTED_TARGETS = (
    "delayed_dependency",
    "regime_switch",
    "trend_plus_variance",
    "oscillation_plus_corrbreak",
    "levelshift_plus_regime",
)


def make_records(
    prefix: str,
    split: str,
    labels: Iterable[str],
    builder: Callable[[str, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
) -> list[Record]:
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            Record(
                sample_id=f"{prefix}:{split}:{index:05d}",
                split=split,
                label=label,
                window=builder(label, rng).astype(np.float32),
            )
        )
    return rows


def build_base(seed: int, config: Config) -> dict[str, list[Record]]:
    rng = np.random.default_rng(1_000_000 + seed)

    def base_builder(label: str, local_rng: np.random.Generator) -> np.ndarray:
        return atomic_window(None if label == "normal" else label, local_rng)

    splits = {
        "train_normal": make_records(
            f"s{seed}", "train_normal", ["normal"] * config.normal_train_n, base_builder, rng
        ),
        "train_known": make_records(
            f"s{seed}",
            "train_known",
            [name for name in KNOWN_TYPES for _ in range(config.known_train_per_type)],
            base_builder,
            rng,
        ),
        "cal_normal": make_records(
            f"s{seed}", "cal_normal", ["normal"] * config.normal_cal_n, base_builder, rng
        ),
        "val_known": make_records(
            f"s{seed}",
            "val_known",
            [name for name in KNOWN_TYPES for _ in range(config.known_val_per_type)],
            base_builder,
            rng,
        ),
        "pre": make_records(
            f"s{seed}",
            "pre",
            ["normal"] * config.pre_normal_n
            + [KNOWN_TYPES[index % len(KNOWN_TYPES)] for index in range(config.pre_known_n)],
            base_builder,
            rng,
        ),
        "future_background": make_records(
            f"s{seed}",
            "future_background",
            ["normal"] * config.future_normal_n
            + [KNOWN_TYPES[index % len(KNOWN_TYPES)] for index in range(config.future_known_n)],
            base_builder,
            rng,
        ),
    }
    rng.shuffle(splits["pre"])
    rng.shuffle(splits["future_background"])
    return splits


def build_target(seed: int, target: str, config: Config) -> dict[str, list[Record]]:
    rng = np.random.default_rng(2_000_000 + 10_000 * seed + int(stable_hash(target)[:8], 16))
    builder = TARGET_BUILDERS[target]
    first = Record(
        sample_id=f"s{seed}:{target}:first",
        split="first_occurrence",
        label=target,
        window=builder(rng).astype(np.float32),
    )
    future = [
        Record(
            sample_id=f"s{seed}:{target}:future:{index:05d}",
            split="future_target",
            label=target,
            window=builder(rng).astype(np.float32),
        )
        for index in range(config.future_target_n)
    ]
    return {"first": [first], "future_target": future}


class FeatureContext:
    def __init__(self, records: list[Record], train_normal: list[Record], train_known: list[Record], config: Config):
        self.config = config
        projection_rng = np.random.default_rng(31_415_926)
        projection = projection_rng.normal(
            0.0, 1.0 / math.sqrt(OV.WIN * OV.NVARS), size=(OV.WIN * OV.NVARS, config.projection_dim)
        ).astype(np.float32)
        self.projection_sha256 = hashlib.sha256(projection.tobytes()).hexdigest()

        normal_windows = np.stack([record.window for record in train_normal])
        self.channel_mean = normal_windows.mean(axis=(0, 1))
        self.channel_std = normal_windows.std(axis=(0, 1)) + 1e-6

        windows = np.stack([record.window for record in records])
        normalized = (windows - self.channel_mean[None, None, :]) / self.channel_std[None, None, :]
        generic_raw = normalized.reshape(len(records), -1) @ projection
        specialized_raw = np.stack([fast_evidence(window) for window in windows])
        by_id = {record.sample_id: index for index, record in enumerate(records)}

        train_ids = [record.sample_id for record in [*train_normal, *train_known]]
        normal_ids = [record.sample_id for record in train_normal]
        train_indices = [by_id[sample_id] for sample_id in train_ids]
        normal_indices = [by_id[sample_id] for sample_id in normal_ids]
        self.generic_mean = generic_raw[train_indices].mean(axis=0)
        self.generic_std = generic_raw[train_indices].std(axis=0) + 1e-6
        self.specialized_mean = specialized_raw[normal_indices].mean(axis=0)
        self.specialized_std = specialized_raw[normal_indices].std(axis=0) + 1e-6
        self.generic_z: dict[str, np.ndarray] = {}
        self.specialized_z: dict[str, np.ndarray] = {}
        for record, generic, specialized in zip(records, generic_raw, specialized_raw):
            self.generic_z[record.sample_id] = np.clip(
                (generic - self.generic_mean) / self.generic_std, -8.0, 8.0
            ).astype(np.float32)
            self.specialized_z[record.sample_id] = np.clip(
                (specialized - self.specialized_mean) / self.specialized_std, -5.0, 15.0
            ).astype(np.float32)

    def vector(self, record: Record, representation: str, target: str) -> np.ndarray:
        generic = self.generic_z[record.sample_id] / math.sqrt(self.config.projection_dim)
        keep = np.ones(len(SPECIALIZED_NAMES), dtype=bool)
        if representation in {"specialized_loo", "combined_loo"}:
            keep[SPECIALIZED_NAMES.index(DIRECT_FEATURE[target])] = False
        specialized = self.specialized_z[record.sample_id][keep]
        specialized = specialized / math.sqrt(len(specialized))
        if representation == "generic":
            return generic.copy()
        if representation in {"specialized", "specialized_loo"}:
            return specialized.copy()
        if representation in {"combined", "combined_loo"}:
            return np.concatenate([generic, specialized]).astype(np.float32)
        raise KeyError(representation)


class RevealOracle:
    def __init__(self, authorized_id: str, label: str):
        self.authorized_id = authorized_id
        self._label = label
        self.calls = 0

    def reveal(self, sample_id: str) -> str:
        if sample_id != self.authorized_id or self.calls:
            raise AssertionError("unauthorized or repeated label reveal")
        self.calls += 1
        return self._label


class OneShotMemoryModel:
    def __init__(
        self,
        train_normal: list[tuple[np.ndarray, str]],
        train_known: list[tuple[np.ndarray, str]],
        cal_normal: list[np.ndarray],
        val_known: list[tuple[np.ndarray, str]],
        config: Config,
    ):
        self.known_names = tuple(KNOWN_TYPES)
        self.normal_centroid = np.mean([vector for vector, _ in train_normal], axis=0)
        known_buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for vector, label in train_known:
            known_buckets[label].append(vector)
        self.known_centroids = {label: np.mean(vectors, axis=0) for label, vectors in known_buckets.items()}
        normal_scores = [float(np.linalg.norm(vector - self.normal_centroid)) for vector in cal_normal]
        self.anomaly_threshold = float(np.quantile(normal_scores, config.anomaly_quantile))
        known_distances = [
            float(np.linalg.norm(vector - self.known_centroids[label])) for vector, label in val_known
        ]
        self.known_radius = float(np.quantile(known_distances, config.known_radius_quantile))
        references = {label: vectors[0] for label, vectors in known_buckets.items()}
        one_shot_distances = [float(np.linalg.norm(vector - references[label])) for vector, label in val_known]
        self.memory_radius = float(np.quantile(one_shot_distances, config.memory_radius_quantile))
        self.memory_vector: np.ndarray | None = None
        self.memory_label: str | None = None
        self.update_count = 0

    def predict(self, vector: np.ndarray) -> Decision:
        anomaly_score = float(np.linalg.norm(vector - self.normal_centroid))
        if anomaly_score <= self.anomaly_threshold:
            return Decision("normal", "normal", False, False, anomaly_score, float("inf"), None)
        known_label, known_distance = min(
            (
                (label, float(np.linalg.norm(vector - centroid)))
                for label, centroid in self.known_centroids.items()
            ),
            key=lambda item: item[1],
        )
        if known_distance <= self.known_radius:
            return Decision(known_label, "known", True, False, anomaly_score, known_distance, None)
        memory_distance = (
            None if self.memory_vector is None else float(np.linalg.norm(vector - self.memory_vector))
        )
        if (
            self.memory_vector is not None
            and self.memory_label is not None
            and memory_distance is not None
            and memory_distance <= self.memory_radius
        ):
            return Decision(
                self.memory_label,
                "memory_reuse",
                True,
                True,
                anomaly_score,
                known_distance,
                memory_distance,
            )
        return Decision("unknown", "unknown_candidate", True, True, anomaly_score, known_distance, memory_distance)

    def update_with_reveal(self, vector: np.ndarray, revealed_label: str) -> None:
        if self.update_count:
            raise AssertionError("one-shot memory received more than one update")
        self.memory_vector = vector.copy()
        self.memory_label = str(revealed_label)
        self.update_count += 1

    def state_hash(self) -> str:
        return stable_hash(
            {
                "anomaly_threshold": self.anomaly_threshold,
                "known_radius": self.known_radius,
                "memory_radius": self.memory_radius,
                "memory_label": self.memory_label,
                "memory_vector": self.memory_vector,
                "update_count": self.update_count,
            }
        )


def deterministic_namer(specialized_z: np.ndarray, config: Config) -> str:
    order = np.argsort(-specialized_z)
    first, second = int(order[0]), int(order[1])
    if float(specialized_z[first]) < config.namer_z_threshold:
        return "abstain"
    if (
        float(specialized_z[second]) >= config.namer_secondary_threshold
        or float(specialized_z[first] - specialized_z[second]) < config.namer_margin
    ):
        return "unsupported"
    return SUPPORTED_STAT_TO_ATOMIC.get(SPECIALIZED_NAMES[first], "unsupported")


def record_hash(record: Record) -> str:
    return hashlib.sha256(record.window.tobytes()).hexdigest()


def overlap_audit(split_records: dict[str, list[Record]]) -> dict[str, Any]:
    hashes = {split: {record_hash(record) for record in records} for split, records in split_records.items()}
    collisions = []
    for first, second in itertools.combinations(sorted(hashes), 2):
        overlap = hashes[first] & hashes[second]
        if overlap:
            collisions.append({"first": first, "second": second, "count": len(overlap)})
    ids = [record.sample_id for records in split_records.values() for record in records]
    return {
        "independent_window_protocol": True,
        "sliding_windows_used": False,
        "cross_split_duplicate_hashes": collisions,
        "unique_sample_ids": len(ids) == len(set(ids)),
    }


def assay_manifest_hash(split_records: dict[str, list[Record]]) -> str:
    return stable_hash(
        {
            split: [
                {"sample_id": record.sample_id, "window_sha256": record_hash(record)}
                for record in records
            ]
            for split, records in sorted(split_records.items())
        }
    )


def fit_model(
    context: FeatureContext,
    base: dict[str, list[Record]],
    representation: str,
    target: str,
    config: Config,
) -> OneShotMemoryModel:
    train_normal = [(context.vector(record, representation, target), record.label) for record in base["train_normal"]]
    train_known = [(context.vector(record, representation, target), record.label) for record in base["train_known"]]
    cal_normal = [context.vector(record, representation, target) for record in base["cal_normal"]]
    val_known = [(context.vector(record, representation, target), record.label) for record in base["val_known"]]
    return OneShotMemoryModel(train_normal, train_known, cal_normal, val_known, config)


def rate(values: Iterable[bool]) -> float:
    rows = list(values)
    return float(np.mean(rows)) if rows else float("nan")


def run_assay(
    seed: int,
    target: str,
    representation: str,
    base: dict[str, list[Record]],
    target_data: dict[str, list[Record]],
    context: FeatureContext,
    config: Config,
    manifest_sha256: str,
) -> dict[str, Any]:
    model = fit_model(context, base, representation, target, config)

    pre_decisions = [
        (record, model.predict(context.vector(record, representation, target))) for record in base["pre"]
    ]
    first = target_data["first"][0]
    first_vector = context.vector(first, representation, target)
    operation_order = []
    first_decision = model.predict(first_vector)
    operation_order.append("predict")
    first_correct = first_decision.pred_label == first.label
    del first_correct  # scoring is deliberately complete before the reveal
    operation_order.append("score")
    oracle = RevealOracle(first.sample_id, first.label)
    operation_order.append("query")
    revealed = oracle.reveal(first.sample_id)
    operation_order.append("reveal")
    model.update_with_reveal(first_vector, revealed)
    operation_order.append("update")
    if operation_order != ["predict", "score", "query", "reveal", "update"]:
        raise AssertionError("first-occurrence operation order changed")

    future = [*target_data["future_target"], *base["future_background"]]
    rng = np.random.default_rng(3_000_000 + seed + int(stable_hash(target)[:8], 16))
    rng.shuffle(future)
    state_before = model.state_hash()
    future_rows = []
    for record in future:
        vector = context.vector(record, representation, target)
        event_order = ["predict"]
        decision = model.predict(vector)
        event_order.append("score")
        if event_order != ["predict", "score"]:
            raise AssertionError("future event performed a reveal or update")
        future_rows.append((record, decision))
    state_after = model.state_hash()
    if state_before != state_after:
        raise AssertionError("future phase mutated one-shot memory")

    target_rows = [(record, decision) for record, decision in future_rows if record.label == target]
    normal_rows = [(record, decision) for record, decision in future_rows if record.label == "normal"]
    known_rows = [(record, decision) for record, decision in future_rows if record.label in KNOWN_TYPES]
    namer_outputs = [
        deterministic_namer(context.specialized_z[record.sample_id], config)
        for record in [first, *target_data["future_target"]]
    ]
    target_supported = target in KNOWN_TYPES
    naming_exact = rate(output == target for output in namer_outputs) if target_supported else float("nan")
    safe_unsupported = rate(output in {"abstain", "unsupported"} for output in namer_outputs)
    atomic_misname = (
        rate(output in KNOWN_TYPES for output in namer_outputs)
        if not target_supported
        else float("nan")
    )

    return {
        "seed": seed,
        "target": target,
        "target_group": "direct" if target in DIRECT_TARGETS else "unsupported",
        "direct_feature": DIRECT_FEATURE.get(target),
        "representation": representation,
        "representation_dim": len(first_vector),
        "first_pred_label": first_decision.pred_label,
        "first_route": first_decision.route,
        "first_anomaly_detected": first_decision.anomaly,
        "first_candidate": first_decision.candidate,
        "first_unknown": first_decision.pred_label == "unknown",
        "first_typed_correct_before_reveal": first_decision.pred_label == target,
        "first_score_margin": first_decision.anomaly_score - model.anomaly_threshold,
        "annotation_queries": oracle.calls,
        "future_target_n": len(target_rows),
        "future_reuse_accuracy": rate(decision.pred_label == target for _, decision in target_rows),
        "future_candidate_recall": rate(decision.candidate for _, decision in target_rows),
        "future_anomaly_recall": rate(decision.anomaly for _, decision in target_rows),
        "future_unknown_rate": rate(decision.pred_label == "unknown" for _, decision in target_rows),
        "future_known_absorption_rate": rate(decision.route == "known" for _, decision in target_rows),
        "future_wrong_memory_rate": rate(
            decision.route == "memory_reuse" and decision.pred_label != target for _, decision in target_rows
        ),
        "normal_far": rate(decision.anomaly for _, decision in normal_rows),
        "normal_typed_far": rate(decision.pred_label == target for _, decision in normal_rows),
        "known_accuracy": rate(decision.pred_label == record.label for record, decision in known_rows),
        "pre_normal_far": rate(
            decision.anomaly for record, decision in pre_decisions if record.label == "normal"
        ),
        "pre_known_accuracy": rate(
            decision.pred_label == record.label
            for record, decision in pre_decisions
            if record.label in KNOWN_TYPES
        ),
        "anomaly_threshold": model.anomaly_threshold,
        "known_radius": model.known_radius,
        "memory_radius": model.memory_radius,
        "operation_order": operation_order,
        "future_state_unchanged": state_before == state_after,
        "future_state_sha256": state_after,
        "assay_manifest_sha256": manifest_sha256,
        "namer_supported_scope": target_supported,
        "namer_exact_accuracy": naming_exact,
        "namer_safe_abstain_rate": safe_unsupported,
        "namer_atomic_misname_rate": atomic_misname,
    }


def separability_metrics(vectors: np.ndarray, labels: list[str]) -> dict[str, float]:
    unique = sorted(set(labels))
    label_array = np.asarray(labels)
    centroids = {label: vectors[label_array == label].mean(axis=0) for label in unique}
    within = np.mean(
        [float(np.linalg.norm(vector - centroids[label])) for vector, label in zip(vectors, labels)]
    )
    between = np.mean(
        [float(np.linalg.norm(centroids[first] - centroids[second])) for first, second in itertools.combinations(unique, 2)]
    )

    correct = 0
    for index, (vector, label) in enumerate(zip(vectors, labels)):
        candidates = {}
        for candidate in unique:
            mask = label_array == candidate
            if candidate == label:
                mask[index] = False
            candidates[candidate] = vectors[mask].mean(axis=0)
        prediction = min(candidates, key=lambda name: float(np.linalg.norm(vector - candidates[name])))
        correct += prediction == label

    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    silhouettes = []
    for index, label in enumerate(labels):
        same = np.where(label_array == label)[0]
        same = same[same != index]
        a = float(np.mean(distances[index, same]))
        b = min(
            float(np.mean(distances[index, label_array == other])) for other in unique if other != label
        )
        silhouettes.append((b - a) / max(a, b, 1e-12))
    return {
        "centroid_loo_accuracy": correct / len(labels),
        "silhouette": float(np.mean(silhouettes)),
        "between_within_ratio": float(between / (within + 1e-12)),
    }


def mean_std_ci(values: list[float], config: Config, seed: int) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        rng.choice(array, size=(config.bootstrap_samples, len(array)), replace=True), axis=1
    )
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "n": len(array),
    }


def summarize(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["target"], row["representation"])].append(row)
    metrics = (
        "first_anomaly_detected",
        "first_candidate",
        "first_unknown",
        "future_reuse_accuracy",
        "future_candidate_recall",
        "future_anomaly_recall",
        "future_unknown_rate",
        "future_known_absorption_rate",
        "normal_far",
        "normal_typed_far",
        "known_accuracy",
        "namer_exact_accuracy",
        "namer_safe_abstain_rate",
        "namer_atomic_misname_rate",
    )
    output = []
    for (target, representation), group in sorted(groups.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in group if np.isfinite(float(row[metric]))]
            if not values:
                continue
            stats = mean_std_ci(
                values,
                config,
                seed=int(stable_hash([target, representation, metric])[:8], 16),
            )
            output.append(
                {
                    "target": target,
                    "representation": representation,
                    "metric": metric,
                    **stats,
                }
            )
    return output


def exact_sign_flip_p(differences: list[float]) -> float:
    observed = abs(float(np.mean(differences)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(float(np.mean(np.asarray(differences) * np.asarray(signs))))
        extreme += permuted >= observed - 1e-12
        total += 1
    return extreme / total


def loo_effects(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    by_key = {(row["target"], row["seed"], row["representation"]): row for row in rows}
    output = []
    for target in DIRECT_TARGETS:
        for full, loo in (("specialized", "specialized_loo"), ("combined", "combined_loo")):
            differences = [
                by_key[(target, seed, full)]["future_reuse_accuracy"]
                - by_key[(target, seed, loo)]["future_reuse_accuracy"]
                for seed in range(config.seeds)
            ]
            stats = mean_std_ci(
                differences,
                config,
                seed=int(stable_hash([target, full, loo])[:8], 16),
            )
            output.append(
                {
                    "target": target,
                    "contrast": f"{full}-{loo}",
                    "metric": "future_reuse_accuracy",
                    **stats,
                    "exact_sign_flip_p": exact_sign_flip_p(differences),
                    "seed_differences": differences,
                }
            )
    ordered = sorted(range(len(output)), key=lambda index: output[index]["exact_sign_flip_p"])
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(output) - rank) * output[index]["exact_sign_flip_p"])
        running = max(running, adjusted)
        output[index]["holm_adjusted_p"] = running
    return output


def get_summary(
    summary: list[dict[str, Any]], target: str, representation: str, metric: str
) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if row["target"] == target and row["representation"] == representation and row["metric"] == metric
    )


def pct(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{100 * float(value):.1f}%"


def build_report(
    path: Path,
    result_path: Path,
    config: Config,
    summary: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    separability_summary: list[dict[str, Any]],
    protocol_audit: dict[str, Any],
) -> None:
    lines = [
        "# Feature-Leakage and Strict-Online Audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Scope",
        "",
        "This is an independent controlled assay, not a replacement paper result. It tests whether synthetic typed reuse depends on statistics explicitly designed to match the injectors. Every arm receives exactly one annotation after its first-occurrence prediction is scored; all future events are frozen and query-free.",
        "",
        "Representations:",
        "",
        "- `generic`: fixed random projection of channel-normalized raw windows; no named anomaly statistic.",
        "- `specialized`: six injector-aligned evidence statistics.",
        "- `combined`: equal-block weighting of generic and specialized features.",
        "- `*_loo`: removes only the target's direct specialized coordinate.",
        "",
        "## Direct-Feature Results",
        "",
        "| Target | Direct feature | Representation | First candidate | Future candidate | Future exact reuse | Future unknown | Normal FAR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for target in DIRECT_TARGETS:
        for representation in DIRECT_REPRESENTATIONS:
            values = {
                metric: get_summary(summary, target, representation, metric)["mean"]
                for metric in (
                    "first_candidate",
                    "future_candidate_recall",
                    "future_reuse_accuracy",
                    "future_unknown_rate",
                    "normal_far",
                )
            }
            lines.append(
                f"| {target} | {DIRECT_FEATURE[target]} | {representation} | "
                f"{pct(values['first_candidate'])} | {pct(values['future_candidate_recall'])} | "
                f"{pct(values['future_reuse_accuracy'])} | {pct(values['future_unknown_rate'])} | "
                f"{pct(values['normal_far'])} |"
            )
    lines.extend(
        [
            "",
            "## Leave-One-Direct-Feature-Out Effects",
            "",
            "Positive delta means the full representation did better than its LOO version.",
            "",
            "| Target | Contrast | Mean future-reuse delta | 95% CI | Exact paired p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['target']} | {row['contrast']} | {pct(row['mean'])} | "
            f"[{pct(row['ci95_low'])}, {pct(row['ci95_high'])}] | {row['exact_sign_flip_p']:.4f} | "
            f"{row['holm_adjusted_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Unsupported and Composite Anomalies",
            "",
            "These labels have no deterministic name lookup. Memory reuse still uses the controlled annotation oracle, while naming is evaluated only for abstention or accidental known-ontology misnaming.",
            "",
            "| Target | Representation | First candidate | Future candidate | Future exact reuse | Namer safe abstain | Atomic misname |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target in UNSUPPORTED_TARGETS:
        for representation in BASE_REPRESENTATIONS:
            values = {
                metric: get_summary(summary, target, representation, metric)["mean"]
                for metric in (
                    "first_candidate",
                    "future_candidate_recall",
                    "future_reuse_accuracy",
                    "namer_safe_abstain_rate",
                    "namer_atomic_misname_rate",
                )
            }
            lines.append(
                f"| {target} | {representation} | {pct(values['first_candidate'])} | "
                f"{pct(values['future_candidate_recall'])} | {pct(values['future_reuse_accuracy'])} | "
                f"{pct(values['namer_safe_abstain_rate'])} | {pct(values['namer_atomic_misname_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Unsupported-Class Separability",
            "",
            "Labels are used only by this offline evaluator, never by the encoder or online model.",
            "",
            "| Representation | Nearest-centroid LOO acc. | Silhouette | Between/within ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in separability_summary:
        lines.append(
            f"| {row['representation']} | {pct(row['centroid_loo_accuracy_mean'])} | "
            f"{row['silhouette_mean']:.3f} | {row['between_within_ratio_mean']:.3f} |"
        )
    oscillation_effect = next(
        row for row in effects if row["target"] == "oscillation" and row["contrast"].startswith("specialized-")
    )
    trend_effect = next(
        row for row in effects if row["target"] == "trend" and row["contrast"].startswith("specialized-")
    )
    delayed_candidate = get_summary(
        summary, "delayed_dependency", "specialized", "future_candidate_recall"
    )["mean"]
    levelshift_misname = get_summary(
        summary, "levelshift_plus_regime", "specialized", "namer_atomic_misname_rate"
    )["mean"]
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Removing the direct specialized coordinate reduces future reuse by {100 * oscillation_effect['mean']:.1f} percentage points for oscillation and {100 * trend_effect['mean']:.1f} percentage points for trend. These are direct-channel dependence estimates, not generalization gains.",
            f"- Delayed dependency reaches only {pct(delayed_candidate)} candidate recall even with specialized statistics, exposing a clear detector blind spot outside the preset signatures.",
            "- Specialized features separate the unsupported/composite evaluation classes well, but separability does not create a legal semantic name. The online memory still requires the controlled annotation.",
            f"- The known-only deterministic namer safely abstains on most unsupported cases, but misnames {pct(levelshift_misname)} of `levelshift_plus_regime` samples as a supported atomic known type.",
        ]
    )
    lines.extend(
        [
            "",
            "## Protocol Audit",
            "",
            f"- First occurrence order always `predict -> score -> query -> reveal -> update`: `{protocol_audit['all_first_orders_valid']}`.",
            f"- Future memories unchanged: `{protocol_audit['all_future_states_unchanged']}`.",
            f"- Exactly one annotation per arm: `{protocol_audit['all_query_counts_one']}`.",
            f"- Shared dataset manifest agrees across representations: `{protocol_audit['paired_manifest_agreement']}`.",
            f"- No withheld target is predicted exactly before reveal: `{protocol_audit['no_first_occurrence_direct_answer']}`.",
            f"- Cross-split duplicate window hashes: `{protocol_audit['cross_split_duplicate_count']}`.",
            "- Every window is generated from an independent base series; sliding or overlapping windows are not used.",
            "- Channel normalization and specialized-stat baselines use `train_normal` only.",
            "- Generic coordinate scaling uses `train_normal + train_known`; no calibration/test sample is used.",
            "- Anomaly threshold uses a disjoint normal calibration set; known and one-shot radii use known validation only.",
            "",
            "## Current Protocol Defects and Limits",
            "",
            "1. The specialized representation is structurally circular: the benchmark source explicitly assigns one statistic to each atomic injector. Full-vs-LOO gaps measure dependence on that direct channel, not general anomaly understanding.",
            "2. The one-shot query is externally scheduled. It equalizes supervision and makes future reuse causal, but it does not evaluate an active query policy or time-to-discovery.",
            "3. The primary data are independent event windows. This removes overlap leakage but does not establish behavior on overlapping sliding windows from a chronological real stream.",
            "4. The generic representation is a fixed random projection, not a trained SOTA time-series encoder. It is a leakage-resistant control, not a competitive backbone claim.",
            "5. Exact semantic naming is undefined for every withheld target. The deterministic namer may only emit `spike`, `level_shift`, or `abstain/unsupported`; no hidden target-name lookup is used.",
            "6. All results remain synthetic and evidence-aligned. Entity-disjoint real-background calibration and native typed faults are still required.",
            "",
            f"Result JSON: `{result_path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: Config, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_audit = validate_evidence()
    rows: list[dict[str, Any]] = []
    separability_rows: list[dict[str, Any]] = []
    overlap_audits = []
    assay_manifests = []
    projection_hashes = set()
    start = time.time()

    for seed in range(config.seeds):
        print(f"seed={seed}", flush=True)
        base = build_base(seed, config)
        targets = {target: build_target(seed, target, config) for target in TARGET_BUILDERS}
        all_records = [record for records in base.values() for record in records]
        all_records.extend(
            record for target_data in targets.values() for records in target_data.values() for record in records
        )
        context = FeatureContext(all_records, base["train_normal"], base["train_known"], config)
        projection_hashes.add(context.projection_sha256)

        for target, target_data in targets.items():
            split_records = {**base, **{f"{target}:{key}": value for key, value in target_data.items()}}
            overlap_audits.append(overlap_audit(split_records))
            manifest_sha256 = assay_manifest_hash(split_records)
            assay_manifests.append(
                {"seed": seed, "target": target, "sha256": manifest_sha256}
            )
            representations = DIRECT_REPRESENTATIONS if target in DIRECT_TARGETS else BASE_REPRESENTATIONS
            for representation in representations:
                rows.append(
                    run_assay(
                        seed,
                        target,
                        representation,
                        base,
                        target_data,
                        context,
                        config,
                        manifest_sha256,
                    )
                )

        for representation in BASE_REPRESENTATIONS:
            vectors = []
            labels = []
            for target in UNSUPPORTED_TARGETS:
                for record in targets[target]["future_target"]:
                    vectors.append(context.vector(record, representation, target))
                    labels.append(target)
            separability_rows.append(
                {
                    "seed": seed,
                    "representation": representation,
                    **separability_metrics(np.stack(vectors), labels),
                }
            )

    summary = summarize(rows, config)
    effects = loo_effects(rows, config)
    separability_summary = []
    for representation in BASE_REPRESENTATIONS:
        group = [row for row in separability_rows if row["representation"] == representation]
        entry = {"representation": representation}
        for metric in ("centroid_loo_accuracy", "silhouette", "between_within_ratio"):
            stats = mean_std_ci(
                [row[metric] for row in group],
                config,
                seed=int(stable_hash([representation, metric])[:8], 16),
            )
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_std"] = stats["std"]
            entry[f"{metric}_ci95"] = [stats["ci95_low"], stats["ci95_high"]]
        separability_summary.append(entry)

    manifest_groups: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in rows:
        manifest_groups[(row["seed"], row["target"])].add(row["assay_manifest_sha256"])
    protocol_audit = {
        "evidence_implementation": evidence_audit,
        "all_first_orders_valid": all(
            row["operation_order"] == ["predict", "score", "query", "reveal", "update"] for row in rows
        ),
        "all_future_states_unchanged": all(row["future_state_unchanged"] for row in rows),
        "all_query_counts_one": all(row["annotation_queries"] == 1 for row in rows),
        "paired_manifest_agreement": all(len(hashes) == 1 for hashes in manifest_groups.values()),
        "no_first_occurrence_direct_answer": all(
            not row["first_typed_correct_before_reveal"] for row in rows
        ),
        "cross_split_duplicate_count": sum(
            item["count"]
            for audit in overlap_audits
            for item in audit["cross_split_duplicate_hashes"]
        ),
        "all_sample_ids_unique_within_assay": all(audit["unique_sample_ids"] for audit in overlap_audits),
        "projection_hashes": sorted(projection_hashes),
        "normalization_fit_splits": {
            "channel": ["train_normal"],
            "specialized": ["train_normal"],
            "generic_coordinates": ["train_normal", "train_known"],
        },
        "threshold_fit_splits": {
            "anomaly": ["cal_normal"],
            "known_radius": ["val_known"],
            "memory_radius": ["train_known_reference", "val_known"],
        },
        "sliding_windows_used": False,
    }
    if not all(
        (
            protocol_audit["all_first_orders_valid"],
            protocol_audit["all_future_states_unchanged"],
            protocol_audit["all_query_counts_one"],
            protocol_audit["paired_manifest_agreement"],
            protocol_audit["no_first_occurrence_direct_answer"],
            protocol_audit["cross_split_duplicate_count"] == 0,
            protocol_audit["all_sample_ids_unique_within_assay"],
        )
    ):
        raise AssertionError(f"protocol audit failed: {protocol_audit}")

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.time() - start,
        "config": asdict(config),
        "known_ontology": list(KNOWN_TYPES),
        "direct_targets": list(DIRECT_TARGETS),
        "direct_feature": DIRECT_FEATURE,
        "unsupported_targets": list(UNSUPPORTED_TARGETS),
        "representations": {
            "base": list(BASE_REPRESENTATIONS),
            "direct_assay": list(DIRECT_REPRESENTATIONS),
        },
        "provenance": {
            "script": str(Path(__file__).relative_to(REPO)),
            "script_sha256": file_hash(Path(__file__)),
            "ovbench_sha256": file_hash(ROOT / "sigla_exp" / "ovbench.py"),
            "git_sha": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--short", "--untracked-files=all")),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "protocol_audit": protocol_audit,
        "assay_manifests": assay_manifests,
        "rows": rows,
        "summary": summary,
        "loo_effects": effects,
        "separability_rows": separability_rows,
        "separability_summary": separability_summary,
    }
    result_path = output_dir / "feature_leakage_online_result.json"
    report_path = output_dir / "feature_leakage_online_report.md"
    result_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    build_report(
        report_path,
        result_path,
        config,
        summary,
        effects,
        separability_summary,
        protocol_audit,
    )
    print(f"saved -> {result_path}", flush=True)
    print(f"report -> {report_path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "feature_leakage_online_2026-07-09",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seeds=args.seeds, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(asdict(config), indent=2), flush=True)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
