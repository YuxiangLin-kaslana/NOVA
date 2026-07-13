#!/usr/bin/env python3
"""Strict cumulative component-chain experiment for typed anomaly reuse.

The chain is deliberately compact and auditable:

detector -> +UNKNOWN gate -> +namer (no reuse) -> +semantic memory
         -> +confirm-2 guard -> +class-balanced ridge replay proxy

Queried-window naming is never counted as reuse.  The future phase has no
queries and no updates, and every arm sees the same windows in the same order.
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import sigla_exp.ovbench as OV  # noqa: E402
from sota_compare.run_feature_leakage_online import (  # noqa: E402
    fast_evidence,
    file_hash,
    sanitize,
    stable_hash,
    validate_evidence,
)


KNOWN_TYPES = ("spike", "level_shift")
NOVEL_TYPES = ("oscillation", "variance_burst", "trend", "correlation_break")
ALL_ATOMIC_TYPES = (*KNOWN_TYPES, *NOVEL_TYPES)
STATS = tuple(OV.STATS)
STAT_TO_TYPE = {stat: concept for concept, stat in OV.STAT_OF.items()}
ARMS = (
    "detector",
    "gate_unknown",
    "namer_no_reuse",
    "semantic_memory",
    "confirm2_guard",
    "balanced_replay_proxy",
)


@dataclass(frozen=True)
class Config:
    seeds: int = 10
    normal_train_n: int = 200
    known_train_per_type: int = 30
    normal_cal_n: int = 200
    known_val_per_type: int = 20
    normal_guard_n: int = 100
    normal_guard_benign_per_pattern: int = 3
    discovery_novel_per_type: int = 4
    discovery_known_per_type: int = 8
    discovery_normal_n: int = 30
    discovery_benign_patterns: int = 8
    future_novel_per_type: int = 20
    future_known_per_type: int = 20
    future_normal_n: int = 64
    future_benign_per_pattern: int = 2
    anomaly_quantile: float = 0.95
    known_radius_quantile: float = 0.95
    memory_radius_quantile: float = 0.95
    namer_z_threshold: float = 2.5
    namer_margin: float = 0.4
    guard_confirm_k: int = 2
    guard_normal_typed_far_max: float = 0.02
    replay_radius_factor: float = 1.5
    replay_ridge: float = 0.1
    bootstrap_samples: int = 5000


@dataclass(frozen=True)
class RawEvent:
    event_id: str
    phase: str
    true_label: str
    is_normal: bool
    kind: str
    window: np.ndarray


@dataclass(frozen=True)
class Event:
    event_id: str
    phase: str
    true_label: str
    is_normal: bool
    kind: str
    vector: np.ndarray


@dataclass(frozen=True)
class Prediction:
    pred_label: str
    route: str
    anomaly: bool
    candidate: bool
    known_distance: float | None
    memory_distance: float | None


@dataclass
class MemoryEntry:
    entry_id: str
    label: str
    vector: np.ndarray
    support: int
    committed: bool
    source_event_ids: list[str] = field(default_factory=list)


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def atomic_window(label: str | None, rng: np.random.Generator) -> np.ndarray:
    return OV.make_window(label, rng).astype(np.float32)


def benign_pattern_window(pattern: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic operating changes that are declared normal by this protocol."""
    x = OV.base_normal(rng).astype(np.float32)
    mode = pattern % 8
    if mode < 3:
        frequency = 16 + 3 * mode
        time_index = np.arange(OV.WIN, dtype=np.float32)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        wave = (0.75 + 0.08 * mode) * np.sin(2.0 * np.pi * frequency * time_index / OV.WIN + phase)
        dims = np.asarray([(2 * mode + offset) % OV.NVARS for offset in range(3)])
        x[:, dims] += wave[:, None]
    elif mode < 6:
        start = 18 + 7 * (mode - 3)
        end = min(OV.WIN, start + 22 + 3 * mode)
        dims = np.asarray([(mode + offset * 3) % OV.NVARS for offset in range(3)])
        x[start:end, dims] += rng.normal(0.0, 0.65 + 0.08 * mode, (end - start, len(dims)))
    else:
        dims = np.asarray([(mode + offset * 2) % OV.NVARS for offset in range(4)])
        ramp = np.linspace(0.0, 1.2 + 0.15 * (mode - 6), OV.WIN, dtype=np.float32)
        x[:, dims] += ramp[:, None]
    return x.astype(np.float32)


def make_raw_event(
    seed: int,
    phase: str,
    index: int,
    label: str,
    is_normal: bool,
    kind: str,
    window: np.ndarray,
) -> RawEvent:
    return RawEvent(
        event_id=f"s{seed}:{phase}:{index:05d}:{kind}",
        phase=phase,
        true_label=label,
        is_normal=is_normal,
        kind=kind,
        window=window.astype(np.float32),
    )


def build_raw_data(seed: int, config: Config) -> dict[str, list[RawEvent]]:
    rng = np.random.default_rng(5_000_000 + seed)
    output: dict[str, list[RawEvent]] = defaultdict(list)

    def add(split: str, label: str, count: int, is_normal: bool, kind: str) -> None:
        for _ in range(count):
            index = len(output[split])
            window = atomic_window(None if is_normal else label, rng)
            output[split].append(make_raw_event(seed, split, index, label, is_normal, kind, window))

    add("train_normal", "normal", config.normal_train_n, True, "base_normal")
    for label in KNOWN_TYPES:
        add("train_known", label, config.known_train_per_type, False, f"known:{label}")
    add("cal_normal", "normal", config.normal_cal_n, True, "base_normal")
    for label in KNOWN_TYPES:
        add("val_known", label, config.known_val_per_type, False, f"known:{label}")
    add("normal_guard", "normal", config.normal_guard_n, True, "base_normal")
    for pattern in range(config.discovery_benign_patterns):
        for _ in range(config.normal_guard_benign_per_pattern):
            index = len(output["normal_guard"])
            output["normal_guard"].append(
                make_raw_event(
                    seed,
                    "normal_guard",
                    index,
                    "normal",
                    True,
                    f"guard_benign:{pattern}",
                    benign_pattern_window(pattern, rng),
                )
            )

    discovery: list[RawEvent] = []
    for label in NOVEL_TYPES:
        for _ in range(config.discovery_novel_per_type):
            index = len(discovery)
            discovery.append(
                make_raw_event(seed, "discovery", index, label, False, f"novel:{label}", atomic_window(label, rng))
            )
    for label in KNOWN_TYPES:
        for _ in range(config.discovery_known_per_type):
            index = len(discovery)
            discovery.append(
                make_raw_event(seed, "discovery", index, label, False, f"known:{label}", atomic_window(label, rng))
            )
    for _ in range(config.discovery_normal_n):
        index = len(discovery)
        discovery.append(
            make_raw_event(seed, "discovery", index, "normal", True, "base_normal", atomic_window(None, rng))
        )
    for pattern in range(config.discovery_benign_patterns):
        index = len(discovery)
        discovery.append(
            make_raw_event(
                seed,
                "discovery",
                index,
                "normal",
                True,
                f"benign:{pattern}",
                benign_pattern_window(pattern, rng),
            )
        )
    rng.shuffle(discovery)
    output["discovery"] = discovery

    future: list[RawEvent] = []
    for label in NOVEL_TYPES:
        for _ in range(config.future_novel_per_type):
            index = len(future)
            future.append(
                make_raw_event(seed, "future", index, label, False, f"novel:{label}", atomic_window(label, rng))
            )
    for label in KNOWN_TYPES:
        for _ in range(config.future_known_per_type):
            index = len(future)
            future.append(
                make_raw_event(seed, "future", index, label, False, f"known:{label}", atomic_window(label, rng))
            )
    for _ in range(config.future_normal_n):
        index = len(future)
        future.append(
            make_raw_event(seed, "future", index, "normal", True, "base_normal", atomic_window(None, rng))
        )
    for pattern in range(config.discovery_benign_patterns):
        for _ in range(config.future_benign_per_pattern):
            index = len(future)
            future.append(
                make_raw_event(
                    seed,
                    "future",
                    index,
                    "normal",
                    True,
                    f"benign:{pattern}",
                    benign_pattern_window(pattern, rng),
                )
            )
    rng.shuffle(future)
    output["future"] = future
    return dict(output)


def fit_evidence_transform(raw: dict[str, list[RawEvent]]) -> tuple[dict[str, Event], dict[str, Any]]:
    all_rows = [event for rows in raw.values() for event in rows]
    raw_evidence = {event.event_id: fast_evidence(event.window) for event in all_rows}
    normal = np.stack([raw_evidence[event.event_id] for event in raw["train_normal"]])
    mean = normal.mean(axis=0)
    std = normal.std(axis=0) + 1e-6
    transformed = {}
    for event in all_rows:
        vector = np.clip((raw_evidence[event.event_id] - mean) / std, -5.0, 15.0).astype(np.float32)
        transformed[event.event_id] = Event(
            event_id=event.event_id,
            phase=event.phase,
            true_label=event.true_label,
            is_normal=event.is_normal,
            kind=event.kind,
            vector=vector,
        )
    return transformed, {
        "mean": mean,
        "std": std,
        "stat_names": list(STATS),
        "fit_split": "train_normal",
    }


class FrontEnd:
    def __init__(self, events: dict[str, Event], raw: dict[str, list[RawEvent]], config: Config):
        self.config = config
        train_normal = [events[event.event_id] for event in raw["train_normal"]]
        train_known = [events[event.event_id] for event in raw["train_known"]]
        cal_normal = [events[event.event_id] for event in raw["cal_normal"]]
        val_known = [events[event.event_id] for event in raw["val_known"]]
        self.normal_centroid = np.mean([event.vector for event in train_normal], axis=0)
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for event in train_known:
            buckets[event.true_label].append(event.vector)
        self.known_train = {label: np.stack(rows) for label, rows in buckets.items()}
        self.known_centroids = {label: rows.mean(axis=0) for label, rows in self.known_train.items()}
        cal_scores = [float(np.max(event.vector)) for event in cal_normal]
        self.anomaly_threshold = float(np.quantile(cal_scores, config.anomaly_quantile))
        known_distances = [
            float(np.linalg.norm(event.vector - self.known_centroids[event.true_label]))
            for event in val_known
        ]
        self.known_radius = float(np.quantile(known_distances, config.known_radius_quantile))
        references = {label: rows[0] for label, rows in self.known_train.items()}
        pair_distances = [
            float(np.linalg.norm(event.vector - references[event.true_label])) for event in val_known
        ]
        self.memory_radius = float(np.quantile(pair_distances, config.memory_radius_quantile))

    def base_route(self, vector: np.ndarray) -> tuple[bool, str, float]:
        anomaly = float(np.max(vector)) > self.anomaly_threshold
        distances = {
            label: float(np.linalg.norm(vector - centroid))
            for label, centroid in self.known_centroids.items()
        }
        known_label = min(distances, key=distances.get)
        return anomaly, known_label, distances[known_label]


def evidence_namer(vector: np.ndarray, config: Config) -> str:
    order = np.argsort(-vector)
    first, second = int(order[0]), int(order[1])
    if float(vector[first]) < config.namer_z_threshold:
        return "unsupported"
    if float(vector[first] - vector[second]) < config.namer_margin:
        return "unsupported"
    return STAT_TO_TYPE[STATS[first]]


class BalancedRidgeProxy:
    """Small class-balanced linear proxy; this is not the full NOVA replay model."""

    def __init__(self, ridge: float):
        self.ridge = float(ridge)
        self.classes: tuple[str, ...] = ()
        self.weights: np.ndarray | None = None
        self.centroids: dict[str, np.ndarray] = {}

    def fit(self, vectors: list[np.ndarray], labels: list[str]) -> None:
        if not vectors:
            self.classes = ()
            self.weights = None
            self.centroids = {}
            return
        x = np.stack(vectors).astype(np.float64)
        self.classes = tuple(sorted(set(labels)))
        counts = Counter(labels)
        sample_weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
        sample_weights *= len(labels) / sample_weights.sum()
        x_bias = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        y = np.zeros((len(x), len(self.classes)), dtype=np.float64)
        class_index = {label: index for index, label in enumerate(self.classes)}
        for row, label in enumerate(labels):
            y[row, class_index[label]] = 1.0
        weighted_x = x_bias * sample_weights[:, None]
        regularizer = self.ridge * np.eye(x_bias.shape[1])
        regularizer[-1, -1] = 0.0
        self.weights = np.linalg.solve(x_bias.T @ weighted_x + regularizer, weighted_x.T @ y)
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for vector, label in zip(vectors, labels):
            buckets[label].append(vector)
        self.centroids = {label: np.mean(rows, axis=0) for label, rows in buckets.items()}

    def predict(self, vector: np.ndarray) -> tuple[str | None, float, float]:
        if self.weights is None or not self.classes:
            return None, float("nan"), float("inf")
        vector_bias = np.concatenate([vector.astype(np.float64), np.ones(1)])
        scores = vector_bias @ self.weights
        order = np.argsort(-scores)
        label = self.classes[int(order[0])]
        margin = float(scores[int(order[0])] - scores[int(order[1])]) if len(order) > 1 else float(scores[0])
        distance = float(np.linalg.norm(vector - self.centroids[label]))
        return label, margin, distance

    def state(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "weights": self.weights,
            "centroids": self.centroids,
            "ridge": self.ridge,
        }


class ChainArm:
    def __init__(
        self,
        name: str,
        front_end: FrontEnd,
        known_train_events: list[Event],
        normal_guard_events: list[Event],
        config: Config,
    ):
        self.name = name
        self.level = ARMS.index(name)
        self.front_end = front_end
        self.config = config
        self.entries: dict[str, MemoryEntry] = {}
        self.next_entry = 0
        self.query_count = 0
        self.query_examples: list[tuple[np.ndarray, str, str]] = []
        self.normal_guard_vectors = [event.vector for event in normal_guard_events]
        self.replay = BalancedRidgeProxy(config.replay_ridge)
        self.known_train_events = known_train_events
        if self.level >= ARMS.index("balanced_replay_proxy"):
            self._refit_replay()

    @property
    def has_gate(self) -> bool:
        return self.level >= ARMS.index("gate_unknown")

    @property
    def has_namer(self) -> bool:
        return self.level >= ARMS.index("namer_no_reuse")

    @property
    def has_memory(self) -> bool:
        return self.level >= ARMS.index("semantic_memory")

    @property
    def guarded(self) -> bool:
        return self.level >= ARMS.index("confirm2_guard")

    @property
    def has_replay(self) -> bool:
        return self.level >= ARMS.index("balanced_replay_proxy")

    def _active_entries(self) -> list[MemoryEntry]:
        return [entry for entry in self.entries.values() if entry.committed]

    def _nearest_entry(
        self, vector: np.ndarray, committed_only: bool, label: str | None = None
    ) -> tuple[MemoryEntry | None, float]:
        candidates = [
            entry
            for entry in self.entries.values()
            if (entry.committed or not committed_only) and (label is None or entry.label == label)
        ]
        if not candidates:
            return None, float("inf")
        entry = min(candidates, key=lambda item: float(np.linalg.norm(vector - item.vector)))
        return entry, float(np.linalg.norm(vector - entry.vector))

    def predict(self, vector: np.ndarray) -> Prediction:
        anomaly, known_label, known_distance = self.front_end.base_route(vector)
        if not anomaly:
            return Prediction("normal", "normal", False, False, known_distance, None)
        if not self.has_gate:
            return Prediction("anomaly", "detector", True, False, known_distance, None)
        if known_distance <= self.front_end.known_radius:
            return Prediction(known_label, "known", True, False, known_distance, None)
        if self.has_memory:
            entry, memory_distance = self._nearest_entry(vector, committed_only=True)
            if entry is not None and memory_distance <= self.front_end.memory_radius:
                return Prediction(entry.label, "semantic_memory", True, True, known_distance, memory_distance)
        else:
            memory_distance = None
        if self.has_replay:
            replay_label, margin, replay_distance = self.replay.predict(vector)
            if (
                replay_label in NOVEL_TYPES
                and replay_label in {entry.label for entry in self._active_entries()}
                and margin > 0.0
                and replay_distance <= self.config.replay_radius_factor * self.front_end.memory_radius
            ):
                return Prediction(
                    replay_label,
                    "balanced_replay_proxy",
                    True,
                    True,
                    known_distance,
                    replay_distance,
                )
        return Prediction("unknown", "unknown_candidate", True, True, known_distance, memory_distance)

    def should_query(self, prediction: Prediction, phase: str) -> bool:
        return phase == "discovery" and self.has_namer and prediction.route == "unknown_candidate"

    def _new_entry(self, vector: np.ndarray, label: str, event_id: str, committed: bool) -> MemoryEntry:
        entry_id = f"entry_{self.next_entry:04d}"
        self.next_entry += 1
        entry = MemoryEntry(entry_id, label, vector.copy(), 1, committed, [event_id])
        self.entries[entry_id] = entry
        return entry

    def _passes_normal_guard(self, vector: np.ndarray) -> bool:
        hits = 0
        for guard_vector in self.normal_guard_vectors:
            anomaly, _, known_distance = self.front_end.base_route(guard_vector)
            if (
                anomaly
                and known_distance > self.front_end.known_radius
                and float(np.linalg.norm(guard_vector - vector)) <= self.front_end.memory_radius
            ):
                hits += 1
        return hits / len(self.normal_guard_vectors) <= self.config.guard_normal_typed_far_max

    def consume_name(self, vector: np.ndarray, proposed_name: str, event_id: str) -> str | None:
        self.query_count += 1
        if not self.has_memory or proposed_name not in NOVEL_TYPES:
            return None
        self.query_examples.append((vector.copy(), proposed_name, event_id))
        if not self.guarded:
            entry = self._new_entry(vector, proposed_name, event_id, committed=True)
            return entry.entry_id

        entry, distance = self._nearest_entry(vector, committed_only=False, label=proposed_name)
        if entry is None or entry.committed or distance > self.front_end.memory_radius:
            entry = self._new_entry(vector, proposed_name, event_id, committed=False)
            return entry.entry_id
        entry.support += 1
        eta = 1.0 / entry.support
        entry.vector = (1.0 - eta) * entry.vector + eta * vector
        entry.source_event_ids.append(event_id)
        if entry.support >= self.config.guard_confirm_k and self._passes_normal_guard(entry.vector):
            entry.committed = True
            if self.has_replay:
                self._refit_replay()
        return entry.entry_id

    def _refit_replay(self) -> None:
        active_labels = {entry.label for entry in self._active_entries()}
        vectors = [event.vector for event in self.known_train_events]
        labels = [event.true_label for event in self.known_train_events]
        for vector, proposed_name, _ in self.query_examples:
            if proposed_name in active_labels:
                vectors.append(vector)
                labels.append(proposed_name)
        self.replay.fit(vectors, labels)

    def state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query_count": self.query_count,
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "label": entry.label,
                    "vector": entry.vector,
                    "support": entry.support,
                    "committed": entry.committed,
                    "source_event_ids": entry.source_event_ids,
                }
                for entry in sorted(self.entries.values(), key=lambda item: item.entry_id)
            ],
            "replay": self.replay.state() if self.has_replay else None,
        }

    def state_hash(self) -> str:
        return stable_hash(self.state())


def window_hash(event: RawEvent) -> str:
    return hashlib.sha256(event.window.tobytes()).hexdigest()


def stream_manifest(raw: dict[str, list[RawEvent]]) -> dict[str, Any]:
    manifest = {
        split: [
            {
                "event_id": event.event_id,
                "window_sha256": window_hash(event),
                "kind": event.kind,
            }
            for event in rows
        ]
        for split, rows in sorted(raw.items())
    }
    return {"sha256": stable_hash(manifest), "splits": manifest}


def overlap_audit(raw: dict[str, list[RawEvent]]) -> dict[str, Any]:
    hashes = {split: {window_hash(event) for event in rows} for split, rows in raw.items()}
    overlaps = []
    for first, second in itertools.combinations(sorted(hashes), 2):
        count = len(hashes[first] & hashes[second])
        if count:
            overlaps.append({"first": first, "second": second, "count": count})
    ids = [event.event_id for rows in raw.values() for event in rows]
    return {
        "cross_split_overlaps": overlaps,
        "unique_event_ids": len(ids) == len(set(ids)),
        "sliding_windows_used": False,
    }


def macro_accuracy(rows: list[dict[str, Any]], labels: tuple[str, ...]) -> float:
    values = []
    for label in labels:
        subset = [row for row in rows if row["true_label"] == label]
        values.append(float(np.mean([row["pred_label"] == label for row in subset])))
    return float(np.mean(values))


def run_arm(
    seed: int,
    arm_name: str,
    front_end: FrontEnd,
    events: dict[str, Event],
    raw: dict[str, list[RawEvent]],
    manifest_sha256: str,
    config: Config,
) -> dict[str, Any]:
    known_train = [events[event.event_id] for event in raw["train_known"]]
    normal_guard = [events[event.event_id] for event in raw["normal_guard"]]
    arm = ChainArm(arm_name, front_end, known_train, normal_guard, config)
    truth_by_id = {
        event.event_id: event.true_label for rows in raw.values() for event in rows
    }
    discovery_log = []
    query_operation_orders = []
    for raw_event in raw["discovery"]:
        event = events[raw_event.event_id]
        operations = ["predict"]
        prediction = arm.predict(event.vector)
        operations.append("score")
        queried = arm.should_query(prediction, event.phase)
        proposed_name = None
        entry_id = None
        if queried:
            operations.append("query")
            proposed_name = evidence_namer(event.vector, config)
            operations.append("reveal_name")
            entry_id = arm.consume_name(event.vector, proposed_name, event.event_id)
            operations.append("update")
            query_operation_orders.append(operations)
        discovery_log.append(
            {
                "event_id": event.event_id,
                "true_label": event.true_label,
                "is_normal": event.is_normal,
                "kind": event.kind,
                "pred_before_query": prediction.pred_label,
                "route_before_query": prediction.route,
                "queried": queried,
                "discovery_name": proposed_name,
                "entry_id": entry_id,
                "operations": operations,
            }
        )

    state_before_future = arm.state_hash()
    future_log = []
    for raw_event in raw["future"]:
        event = events[raw_event.event_id]
        operations = ["predict"]
        prediction = arm.predict(event.vector)
        operations.append("score")
        future_log.append(
            {
                "event_id": event.event_id,
                "true_label": event.true_label,
                "is_normal": event.is_normal,
                "kind": event.kind,
                "pred_label": prediction.pred_label,
                "route": prediction.route,
                "anomaly": prediction.anomaly,
                "candidate": prediction.candidate,
                "operations": operations,
            }
        )
    state_after_future = arm.state_hash()
    if state_before_future != state_after_future:
        raise AssertionError(f"{arm_name} mutated during future phase")

    queried_novel = [
        row for row in discovery_log if row["queried"] and row["true_label"] in NOVEL_TYPES
    ]
    future_novel = [row for row in future_log if row["true_label"] in NOVEL_TYPES]
    future_known = [row for row in future_log if row["true_label"] in KNOWN_TYPES]
    future_normal = [row for row in future_log if row["is_normal"]]
    future_base_normal = [row for row in future_normal if row["kind"] == "base_normal"]
    future_benign = [row for row in future_normal if row["kind"].startswith("benign:")]
    active_entries = [entry for entry in arm.entries.values() if entry.committed]
    spurious = 0
    for entry in active_entries:
        source_truth = [truth_by_id[event_id] for event_id in entry.source_event_ids]
        if any(label != entry.label for label in source_truth):
            spurious += 1

    named_correct_types = {
        row["true_label"]
        for row in queried_novel
        if row["discovery_name"] == row["true_label"]
    }
    return {
        "seed": seed,
        "arm": arm_name,
        "manifest_sha256": manifest_sha256,
        "queries": sum(row["queried"] for row in discovery_log),
        "queried_novel_n": len(queried_novel),
        "discovery_name_accuracy": (
            float(np.mean([row["discovery_name"] == row["true_label"] for row in queried_novel]))
            if queried_novel
            else float("nan")
        ),
        "discovery_named_type_coverage": len(named_correct_types) / len(NOVEL_TYPES),
        "queried_current_direct_answer_n": sum(
            row["queried"] and row["discovery_name"] == row["true_label"] for row in discovery_log
        ),
        "future_novel_n": len(future_novel),
        "future_reuse_macro_accuracy": macro_accuracy(future_novel, NOVEL_TYPES),
        "future_reuse_micro_accuracy": float(
            np.mean([row["pred_label"] == row["true_label"] for row in future_novel])
        ),
        "future_unknown_rate": float(np.mean([row["pred_label"] == "unknown" for row in future_novel])),
        "future_known_absorption_rate": float(
            np.mean([row["route"] == "known" for row in future_novel])
        ),
        "future_memory_route_rate": float(
            np.mean(
                [
                    row["route"] in {"semantic_memory", "balanced_replay_proxy"}
                    for row in future_novel
                ]
            )
        ),
        "known_detection_retention": float(np.mean([row["anomaly"] for row in future_known])),
        "known_typed_retention": float(
            np.mean([row["pred_label"] == row["true_label"] for row in future_known])
        ),
        "normal_far": float(np.mean([row["pred_label"] != "normal" for row in future_normal])),
        "base_normal_far": float(
            np.mean([row["pred_label"] != "normal" for row in future_base_normal])
        ),
        "benign_alarm_far": float(
            np.mean([row["pred_label"] != "normal" for row in future_benign])
        ),
        "normal_typed_far": float(
            np.mean([row["pred_label"] in NOVEL_TYPES for row in future_normal])
        ),
        "benign_typed_far": float(
            np.mean([row["pred_label"] in NOVEL_TYPES for row in future_benign])
        ),
        "active_vocab": len(active_entries),
        "pending_vocab": sum(not entry.committed for entry in arm.entries.values()),
        "spurious_vocab": spurious,
        "future_state_unchanged": state_before_future == state_after_future,
        "future_state_sha256": state_after_future,
        "all_query_orders_valid": all(
            operations == ["predict", "score", "query", "reveal_name", "update"]
            for operations in query_operation_orders
        ),
        "all_future_orders_valid": all(row["operations"] == ["predict", "score"] for row in future_log),
        "replay_is_compact_proxy": arm_name == "balanced_replay_proxy",
        "final_state": arm.state(),
    }


def mean_std_ci(values: list[float], config: Config, seed: int) -> dict[str, Any]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    samples = np.mean(
        rng.choice(array, size=(config.bootstrap_samples, len(array)), replace=True), axis=1
    )
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n": len(array),
    }


def summarize(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    metrics = (
        "queries",
        "discovery_name_accuracy",
        "discovery_named_type_coverage",
        "future_reuse_macro_accuracy",
        "future_reuse_micro_accuracy",
        "future_unknown_rate",
        "future_known_absorption_rate",
        "future_memory_route_rate",
        "known_detection_retention",
        "known_typed_retention",
        "normal_far",
        "base_normal_far",
        "benign_alarm_far",
        "normal_typed_far",
        "benign_typed_far",
        "active_vocab",
        "pending_vocab",
        "spurious_vocab",
    )
    output = []
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        for metric in metrics:
            values = [float(row[metric]) for row in group if np.isfinite(float(row[metric]))]
            if not values:
                continue
            output.append(
                {
                    "arm": arm,
                    "metric": metric,
                    **mean_std_ci(
                        values,
                        config,
                        seed=int(stable_hash([arm, metric])[:8], 16),
                    ),
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


def paired_chain_effects(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    by_key = {(row["seed"], row["arm"]): row for row in rows}
    output = []
    for later, earlier in zip(ARMS[1:], ARMS[:-1]):
        differences = [
            by_key[(seed, later)]["future_reuse_macro_accuracy"]
            - by_key[(seed, earlier)]["future_reuse_macro_accuracy"]
            for seed in range(config.seeds)
        ]
        output.append(
            {
                "contrast": f"{later}-{earlier}",
                "metric": "future_reuse_macro_accuracy",
                **mean_std_ci(
                    differences,
                    config,
                    seed=int(stable_hash([later, earlier])[:8], 16),
                ),
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


def get(summary: list[dict[str, Any]], arm: str, metric: str) -> dict[str, Any]:
    return next(row for row in summary if row["arm"] == arm and row["metric"] == metric)


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
    audit: dict[str, Any],
) -> None:
    lines = [
        "# Strict Component-Chain Experiment",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Scope",
        "",
        "This independent experiment separates queried-window discovery naming from later unqueried reuse. All arms share the same synthetic streams, evidence features, detector threshold, known gate, and deterministic evidence namer. The future phase is frozen.",
        "",
        "The final replay arm is a **class-balanced weighted-ridge compact proxy**. It is not the full NOVA classifier, grow-head training, or production replay implementation.",
        "",
        "## Cumulative Arms",
        "",
        "1. `detector`: binary normal/anomaly output.",
        "2. `gate_unknown`: adds known centroids and UNKNOWN rejection.",
        "3. `namer_no_reuse`: queries an evidence namer for discovery candidates, but stores nothing.",
        "4. `semantic_memory`: immediately stores valid novel names and one-shot prototypes.",
        "5. `confirm2_guard`: requires two consistent same-name matches and checks an independent normal guard set.",
        "6. `balanced_replay_proxy`: adds class-balanced weighted-ridge replay after guarded commit.",
        "",
        "## Main Results",
        "",
        "| Arm | Queries | Discovery-name acc. | Named-type coverage | Future reuse | Known typed retention | Base/all normal FAR | Benign typed FAR | Active/pending vocab | Spurious vocab |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = {
            metric: get(summary, arm, metric)["mean"]
            for metric in (
                "queries",
                "future_reuse_macro_accuracy",
                "discovery_named_type_coverage",
                "known_typed_retention",
                "normal_far",
                "base_normal_far",
                "benign_typed_far",
                "active_vocab",
                "pending_vocab",
                "spurious_vocab",
            )
        }
        naming = next(
            (row for row in summary if row["arm"] == arm and row["metric"] == "discovery_name_accuracy"),
            None,
        )
        lines.append(
            f"| {arm} | {values['queries']:.1f} | {pct(None if naming is None else naming['mean'])} | "
            f"{pct(values['discovery_named_type_coverage'])} | {pct(values['future_reuse_macro_accuracy'])} | "
            f"{pct(values['known_typed_retention'])} | "
            f"{pct(values['base_normal_far'])}/{pct(values['normal_far'])} | "
            f"{pct(values['benign_typed_far'])} | "
            f"{values['active_vocab']:.1f}/{values['pending_vocab']:.1f} | {values['spurious_vocab']:.1f} |"
        )
    lines.extend(
        [
            "",
            "`discovery_name_accuracy` uses only queried novel windows. It is the namer's current answer and is never included in future reuse. `future_reuse` uses only query-free future events.",
            "",
            "## Incremental Future-Reuse Effects",
            "",
            "| Contrast | Mean delta | 95% CI | Exact paired p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['contrast']} | {pct(row['mean'])} | "
            f"[{pct(row['ci95_low'])}, {pct(row['ci95_high'])}] | {row['exact_sign_flip_p']:.4f} | "
            f"{row['holm_adjusted_p']:.4f} |"
        )
    memory = get(summary, "semantic_memory", "future_reuse_macro_accuracy")["mean"]
    guard = get(summary, "confirm2_guard", "future_reuse_macro_accuracy")["mean"]
    replay = get(summary, "balanced_replay_proxy", "future_reuse_macro_accuracy")["mean"]
    memory_spurious = get(summary, "semantic_memory", "spurious_vocab")["mean"]
    guard_spurious = get(summary, "confirm2_guard", "spurious_vocab")["mean"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Immediate semantic memory obtains {pct(memory)} future exact reuse, but creates {memory_spurious:.2f} spurious prototypes per seed on average.",
            f"- Confirm-2/guard changes future reuse to {pct(guard)} and spurious prototypes to {guard_spurious:.2f}; the trade-off must be reported rather than hiding rejected/pending clusters.",
            f"- The compact balanced-replay proxy reaches {pct(replay)} future reuse. This number cannot be attributed to the full NOVA replay implementation.",
            "- Detector and gate arms may retain binary anomaly detection or UNKNOWN behavior, but they have no mechanism for exact novel-type reuse.",
            "- `spurious_vocab` uses a strict evaluator-only definition: any committed prototype with at least one queried source whose hidden label disagrees with its semantic name is contaminated.",
            "",
            "## Protocol Audit",
            "",
            f"- At least five seeds: `{config.seeds >= 5}` (`n={config.seeds}`).",
            f"- Shared stream manifest across all arms: `{audit['paired_manifest_agreement']}`.",
            f"- Query order always `predict -> score -> query -> reveal_name -> update`: `{audit['all_query_orders_valid']}`.",
            f"- Future order always `predict -> score`, with no query/update: `{audit['all_future_orders_valid']}`.",
            f"- Future state hashes unchanged: `{audit['all_future_states_unchanged']}`.",
            f"- Cross-split duplicate window hashes: `{audit['cross_split_duplicate_count']}`.",
            "- Evidence normalization uses `train_normal`; anomaly calibration uses disjoint `cal_normal`; radii use `val_known`; guard uses separate `normal_guard`.",
            "",
            "## Current Defects and Limits",
            "",
            "1. The deterministic namer is generator-aligned: it maps the largest specialized evidence statistic to the benchmark's atomic name. It is not an LLM or open-ended semantic evaluation.",
            "2. Confirm-2 plus a normal-set contamination check is a compact guard, not the complete deployed do-no-harm mechanism, horizon policy, or rollback system.",
            "3. The replay arm is a class-balanced weighted-ridge proxy trained on known examples and confirmed pseudo-labels. It must not be reported as the complete NOVA replay/grow-head model.",
            "4. Benign operating patterns are synthetic and declared normal by protocol. Spurious-vocabulary results depend on this safety definition.",
            "5. Windows are independent rather than chronological overlapping windows. The experiment avoids overlap leakage but does not validate a real sliding-window deployment.",
            "6. The benchmark remains evidence-aligned synthetic data with only two known and four novel atomic concepts.",
            "",
            f"Result JSON: `{result_path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: Config, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_audit = validate_evidence()
    rows = []
    manifests = []
    overlap_audits = []
    start = time.time()
    for seed in range(config.seeds):
        print(f"seed={seed}", flush=True)
        raw = build_raw_data(seed, config)
        transformed, transform_meta = fit_evidence_transform(raw)
        front_end = FrontEnd(transformed, raw, config)
        manifest = stream_manifest(raw)
        manifests.append({"seed": seed, **manifest})
        overlap_audits.append(overlap_audit(raw))
        for arm_name in ARMS:
            rows.append(
                run_arm(
                    seed,
                    arm_name,
                    front_end,
                    transformed,
                    raw,
                    manifest["sha256"],
                    config,
                )
            )

    summary = summarize(rows, config)
    effects = paired_chain_effects(rows, config)
    manifest_groups: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        manifest_groups[row["seed"]].add(row["manifest_sha256"])
    audit = {
        "evidence_implementation": evidence_audit,
        "paired_manifest_agreement": all(len(values) == 1 for values in manifest_groups.values()),
        "all_query_orders_valid": all(row["all_query_orders_valid"] for row in rows),
        "all_future_orders_valid": all(row["all_future_orders_valid"] for row in rows),
        "all_future_states_unchanged": all(row["future_state_unchanged"] for row in rows),
        "cross_split_duplicate_count": sum(
            overlap["count"]
            for audit_row in overlap_audits
            for overlap in audit_row["cross_split_overlaps"]
        ),
        "all_event_ids_unique": all(row["unique_event_ids"] for row in overlap_audits),
        "sliding_windows_used": False,
        "normalization_fit_split": "train_normal",
        "anomaly_calibration_split": "cal_normal",
        "radius_calibration_split": "val_known",
        "guard_split": "normal_guard",
    }
    if not all(
        (
            audit["paired_manifest_agreement"],
            audit["all_query_orders_valid"],
            audit["all_future_orders_valid"],
            audit["all_future_states_unchanged"],
            audit["cross_split_duplicate_count"] == 0,
            audit["all_event_ids_unique"],
        )
    ):
        raise AssertionError(f"protocol audit failed: {audit}")

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.time() - start,
        "config": asdict(config),
        "arms": list(ARMS),
        "known_types": list(KNOWN_TYPES),
        "novel_types": list(NOVEL_TYPES),
        "protocol_audit": audit,
        "provenance": {
            "script": str(Path(__file__).relative_to(REPO)),
            "script_sha256": file_hash(Path(__file__)),
            "feature_audit_script_sha256": file_hash(
                ROOT / "sota_compare" / "run_feature_leakage_online.py"
            ),
            "ovbench_sha256": file_hash(ROOT / "sigla_exp" / "ovbench.py"),
            "git_sha": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--short", "--untracked-files=all")),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "manifests": manifests,
        "rows": rows,
        "summary": summary,
        "paired_chain_effects": effects,
        "replay_disclaimer": "class-balanced weighted-ridge compact proxy; not full NOVA replay",
    }
    result_path = output_dir / "component_chain_strict_result.json"
    report_path = output_dir / "component_chain_strict_report.md"
    result_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    build_report(report_path, result_path, config, summary, effects, audit)
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
        default=REPO / "docs" / "component_chain_strict_2026-07-09",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seeds=args.seeds, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(asdict(config), indent=2), flush=True)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
