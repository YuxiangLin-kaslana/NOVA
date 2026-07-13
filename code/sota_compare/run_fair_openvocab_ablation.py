#!/usr/bin/env python3
"""Fair detector+namer baselines and NOVA mechanism ablations.

This runner is intentionally self-contained and conservative about method
provenance.  The repository's MemStream and Anomaly Transformer classes are
compact proxies/reimplementations, not the authors' official releases.  Every
detector is evaluated both alone and with the exact same semantic namer.  The
NOVA ablations share one frozen CNN anomaly decision stream, one evidence
representation, and one namer.

The real-background condition uses controlled injections into normal training
backgrounds.  It is not an evaluation on native typed real faults.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import scripts.exp_detection_tie as DT  # noqa: E402
import sigla_exp.ovbench as CB  # noqa: E402
from sota_compare.baselines import AnomalyTransformer, MemStream  # noqa: E402
import sota_compare.realbench as RB  # noqa: E402


ANOMALY = "anomaly"
UNKNOWN = "unknown"
CLUSTER_PREFIX = "cluster:"
STAT_TO_CONCEPT = {stat: concept for concept, stat in CB.STAT_OF.items()}


@dataclass(frozen=True)
class Config:
    seeds: int
    seed_start: int
    backgrounds: tuple[str, ...]
    normal_stats_n: int
    cnn_train_per_class: int
    cnn_epochs: int
    normal_train_n: int
    normal_cal_n: int
    unsup_epochs: int
    memstream_emb_dim: int
    anomaly_transformer_d_model: int
    anomaly_transformer_heads: int
    anomaly_transformer_layers: int
    warm_n: int
    post_n: int
    score_quantile: float
    namer_mode: str
    namer_threshold: float
    novelty_threshold: float
    prototype_radius_quantile: float
    prototype_radius_scale: float


METHODS: dict[str, dict[str, Any]] = {
    "cnn_detector_only": {
        "family": "detector_only",
        "implementation_status": "repository_native_compact_cnn",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "none",
    },
    "cnn_shared_namer_flag_only": {
        "family": "detector_plus_shared_namer",
        "implementation_status": "repository_native_compact_cnn_flag_only_plus_shared_namer",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "shared",
    },
    "cnn_shared_namer_flag_or_novelty_gate": {
        "family": "detector_plus_shared_namer",
        "implementation_status": "repository_native_compact_cnn_or_shared_novelty_gate_plus_shared_namer",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "shared",
    },
    "memstream_proxy_detector_only": {
        "family": "detector_only",
        "implementation_status": "repository_tiny_compact_ae_memory_proxy",
        "proxy_of": "MemStream (Bhatia et al., WWW 2022)",
        "official_external_implementation": False,
        "semantic_namer": "none",
    },
    "memstream_proxy_shared_namer_flag_only": {
        "family": "detector_plus_shared_namer",
        "implementation_status": "repository_tiny_compact_ae_memory_proxy_flag_only_plus_shared_namer",
        "proxy_of": "MemStream (Bhatia et al., WWW 2022)",
        "official_external_implementation": False,
        "semantic_namer": "shared",
    },
    "memstream_proxy_shared_namer_flag_or_novelty_gate": {
        "family": "detector_plus_shared_namer",
        "implementation_status": "repository_tiny_compact_ae_memory_proxy_or_shared_novelty_gate_plus_shared_namer",
        "proxy_of": "MemStream (Bhatia et al., WWW 2022)",
        "official_external_implementation": False,
        "semantic_namer": "shared",
    },
    "anomaly_transformer_proxy_detector_only": {
        "family": "detector_only",
        "implementation_status": "repository_tiny_compact_reimplementation_proxy",
        "proxy_of": "Anomaly Transformer (Xu et al., ICLR 2022)",
        "official_external_implementation": False,
        "semantic_namer": "none",
    },
    "anomaly_transformer_proxy_shared_namer_flag_only": {
        "family": "detector_plus_shared_namer",
        "implementation_status": "repository_tiny_compact_reimplementation_proxy_flag_only_plus_shared_namer",
        "proxy_of": "Anomaly Transformer (Xu et al., ICLR 2022)",
        "official_external_implementation": False,
        "semantic_namer": "shared",
    },
    "anomaly_transformer_proxy_shared_namer_flag_or_novelty_gate": {
        "family": "detector_plus_shared_namer",
        "implementation_status": "repository_tiny_compact_reimplementation_proxy_or_shared_novelty_gate_plus_shared_namer",
        "proxy_of": "Anomaly Transformer (Xu et al., ICLR 2022)",
        "official_external_implementation": False,
        "semantic_namer": "shared",
    },
    "nova_memory_reference": {
        "family": "nova_memory_mechanism",
        "implementation_status": "controlled_frozen_detector_evidence_keyed_semantic_memory_no_replay",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "shared_on_memory_miss",
        "replay": False,
    },
    "nova_no_growth": {
        "family": "nova_ablation",
        "implementation_status": "controlled_ablation_growth_disabled",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "shared_but_new_labels_rejected",
        "replay": False,
    },
    "nova_no_reuse": {
        "family": "nova_ablation",
        "implementation_status": "controlled_ablation_memory_reuse_disabled",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "shared_on_every_detected_anomaly",
        "replay": False,
    },
    "nearest_prototype": {
        "family": "nova_ablation",
        "implementation_status": "controlled_global_nearest_prototype_baseline",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "shared_on_radius_miss",
        "replay": False,
    },
    "clustering_no_semantic_verify": {
        "family": "nova_ablation",
        "implementation_status": "controlled_online_radius_clustering_without_semantics",
        "proxy_of": None,
        "official_external_implementation": False,
        "semantic_namer": "none",
        "replay": False,
    },
}


@dataclass
class Prototype:
    vector: np.ndarray
    label: str
    key: str
    support: int = 1

    def update(self, vector: np.ndarray) -> None:
        self.support += 1
        eta = 1.0 / self.support
        self.vector = (1.0 - eta) * self.vector + eta * vector


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance() -> dict[str, Any]:
    dependencies = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "exp_detection_tie.py",
        ROOT / "sigla_exp" / "ovbench.py",
        ROOT / "sota_compare" / "baselines.py",
        ROOT / "sota_compare" / "realbench.py",
    ]
    dirty = git_value("status", "--short", "--untracked-files=all")
    return {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_dirty": int(bool(dirty and dirty != "unavailable")),
        "git_status_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "source_sha256": {str(path.relative_to(REPO)): file_hash(path) for path in dependencies},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(DT.device),
        "command": [sys.executable, *sys.argv],
    }


def activate_background(name: str) -> dict[str, Any]:
    RB.deactivate()
    if name == "synthetic":
        return {
            "name": name,
            "kind": "controlled_synthetic",
            "dataset": None,
            "entity": None,
            "native_typed_faults": False,
        }
    parts = name.split(":")
    if len(parts) != 2:
        raise ValueError(f"real background must be DATASET:ENTITY, got {name!r}")
    dataset, entity = parts
    values = RB.activate(entity=entity, dataset=dataset)
    return {
        "name": name,
        "kind": "controlled_injection_on_real_normal_background",
        "dataset": dataset.upper(),
        "entity": entity,
        "background_length": int(len(values)),
        "native_typed_faults": False,
    }


def make_window(label: str, rng: np.random.Generator) -> np.ndarray:
    return CB.make_window(None if label == DT.NORMAL else label, rng)


def build_stream(rng: np.random.Generator, config: Config) -> tuple[list[np.ndarray], list[str], int]:
    labels: list[str] = []
    for _ in range(config.warm_n):
        label = DT.NORMAL if rng.random() < 0.5 else str(rng.choice(DT.KNOWN_ANOM))
        labels.append(label)
    onset = len(labels)
    post_anomalies = [*DT.KNOWN_ANOM, DT.NOVEL]
    for _ in range(config.post_n):
        label = DT.NORMAL if rng.random() < 0.5 else str(rng.choice(post_anomalies))
        labels.append(label)
    return [make_window(label, rng) for label in labels], labels, onset


def evidence_and_z(
    windows: list[np.ndarray], mu: dict[str, float], sd: dict[str, float]
) -> tuple[list[dict[str, float]], np.ndarray]:
    evidence = [CB.evidence(window) for window in windows]
    features = np.asarray(
        [[(row[name] - mu[name]) / (sd[name] + 1e-9) for name in CB.STATS] for row in evidence],
        dtype=np.float32,
    )
    return evidence, np.clip(features, -2.0, 10.0)


def evidence_key(feature: np.ndarray, threshold: float) -> str:
    index = int(np.argmax(feature))
    return STAT_TO_CONCEPT[CB.STATS[index]] if float(feature[index]) >= threshold else UNKNOWN


def novelty_gate_flags(
    evidence: list[dict[str, float]],
    mu: dict[str, float],
    sd: dict[str, float],
    threshold: float,
) -> list[bool]:
    known_stats = {CB.STAT_OF[concept] for concept in DT.KNOWN_ANOM}
    flags: list[bool] = []
    for row in evidence:
        deviations = {
            name: abs((row[name] - mu[name]) / (sd[name] + 1e-9)) for name in CB.STATS
        }
        dominant = max(deviations, key=deviations.get)
        flags.append(dominant not in known_stats and deviations[dominant] > threshold)
    return flags


def make_namer(config: Config, api_key: str) -> tuple[Callable[..., str | None], dict[str, Any]]:
    if config.namer_mode == "llm" and not api_key:
        raise RuntimeError("--namer-mode llm requires OPENAI_API_KEY")

    def name(evidence: dict[str, float], feature: np.ndarray, mu: dict[str, float], sd: dict[str, float]) -> str | None:
        if config.namer_mode == "llm":
            result = CB.gpt_recognize_top1(evidence, api_key, mu, sd)
            return None if result in {None, "__ERROR__"} else str(result)
        index = int(np.argmax(feature))
        if float(feature[index]) < config.namer_threshold:
            return None
        return STAT_TO_CONCEPT.get(CB.STATS[index])

    metadata = {
        "mode": config.namer_mode,
        "shared_across_all_namer_arms": True,
        "evidence": list(CB.STATS),
        "threshold": config.namer_threshold,
        "external_api": config.namer_mode == "llm",
        "description": (
            "CB.gpt_recognize_top1 with the repository prompt"
            if config.namer_mode == "llm"
            else "deterministic argmax evidence-rule proxy matching the current prompt's decision rule"
        ),
    }
    return name, metadata


def train_cnn(
    seed: int, config: Config, rng: np.random.Generator
) -> tuple[Any, list[np.ndarray], list[str]]:
    torch.manual_seed(10_000 + seed)
    model = DT.make_detector(len(DT.BASE_VOCAB))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    windows: list[np.ndarray] = []
    labels: list[str] = []
    targets: list[np.ndarray] = []
    for label in DT.BASE_VOCAB:
        for _ in range(config.cnn_train_per_class):
            windows.append(make_window(label, rng))
            labels.append(label)
            targets.append(DT.onehot(DT.BASE_VOCAB.index(label), len(DT.BASE_VOCAB)))
    DT.train_on(model, optimizer, windows, targets, epochs=config.cnn_epochs)
    return model, windows, labels


def cnn_predictions(model: Any, windows: list[np.ndarray]) -> tuple[list[str], list[bool]]:
    probabilities = DT.proba(model, windows)
    predictions = [DT.BASE_VOCAB[int(index)] for index in np.argmax(probabilities, axis=1)]
    return predictions, [prediction != DT.NORMAL for prediction in predictions]


def fit_unsupervised_detectors(
    seed: int,
    config: Config,
    normal_train: list[np.ndarray],
    normal_cal: list[np.ndarray],
    windows: list[np.ndarray],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    constructors = {
        "memstream_proxy": lambda: MemStream(
            CB.WIN,
            CB.NVARS,
            DT.device,
            emb_dim=config.memstream_emb_dim,
            epochs=config.unsup_epochs,
            seed=seed,
        ),
        "anomaly_transformer_proxy": lambda: AnomalyTransformer(
            CB.WIN,
            CB.NVARS,
            DT.device,
            d_model=config.anomaly_transformer_d_model,
            n_heads=config.anomaly_transformer_heads,
            n_layers=config.anomaly_transformer_layers,
            epochs=config.unsup_epochs,
            seed=seed,
        ),
    }
    for offset, (name, constructor) in enumerate(constructors.items()):
        torch.manual_seed(20_000 + 100 * seed + offset)
        model = constructor()
        model.fit(normal_train)
        calibration = model.score_stream(normal_cal, update=False)
        threshold = float(np.quantile(calibration, config.score_quantile))
        scores = model.score_stream(windows, update=True)
        outputs[name] = {
            "flags": [bool(value > threshold) for value in scores],
            "threshold": threshold,
            "score_mean": float(np.mean(scores)),
        }
    return outputs


def shared_namer_predictions(
    base_predictions: list[str],
    detector_flags: list[bool],
    novelty_flags: list[bool],
    route_mode: str,
    evidence: list[dict[str, float]],
    features: np.ndarray,
    mu: dict[str, float],
    sd: dict[str, float],
    namer: Callable[..., str | None],
) -> tuple[list[str], list[dict[str, Any]]]:
    predictions: list[str] = []
    events: list[dict[str, Any]] = []
    if route_mode not in {"flag_only", "flag_or_novelty_gate"}:
        raise ValueError(f"unknown route_mode={route_mode}")
    for base_prediction, detector_flag, novelty_flag, row, feature in zip(
        base_predictions, detector_flags, novelty_flags, evidence, features
    ):
        route = detector_flag or (route_mode == "flag_or_novelty_gate" and novelty_flag)
        if not route:
            predictions.append(base_prediction)
            events.append(
                {
                    "prediction_before_query": base_prediction,
                    "queried": False,
                    "discovery_name": None,
                    "autonomous_reuse": False,
                    "route": "no_query",
                }
            )
            continue
        label = namer(row, feature, mu, sd)
        final = label if label is not None else ANOMALY
        predictions.append(final)
        events.append(
            {
                "prediction_before_query": base_prediction,
                "queried": True,
                "discovery_name": label,
                "autonomous_reuse": False,
                "route": route_mode,
            }
        )
    return predictions, events


def detector_events(predictions: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "prediction_before_query": prediction,
            "queried": False,
            "discovery_name": None,
            "autonomous_reuse": False,
            "route": "detector_only",
        }
        for prediction in predictions
    ]


def prototype_setup(
    train_features: np.ndarray,
    train_labels: list[str],
    config: Config,
) -> tuple[list[Prototype], float]:
    prototypes: list[Prototype] = []
    distances: list[float] = []
    for label in DT.KNOWN_ANOM:
        rows = train_features[np.asarray([item == label for item in train_labels])]
        center = rows.mean(axis=0)
        prototypes.append(Prototype(center.copy(), label, label, support=len(rows)))
        distances.extend(float(np.linalg.norm(row - center)) for row in rows)
    radius = float(np.quantile(distances, config.prototype_radius_quantile))
    return prototypes, radius * config.prototype_radius_scale


def nearest(feature: np.ndarray, prototypes: list[Prototype]) -> tuple[Prototype | None, float]:
    if not prototypes:
        return None, float("inf")
    distances = [float(np.linalg.norm(feature - prototype.vector)) for prototype in prototypes]
    index = int(np.argmin(distances))
    return prototypes[index], distances[index]


def run_semantic_memory(
    method: str,
    flags: list[bool],
    evidence: list[dict[str, float]],
    features: np.ndarray,
    initial_prototypes: list[Prototype],
    radius: float,
    mu: dict[str, float],
    sd: dict[str, float],
    config: Config,
    namer: Callable[..., str | None],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    prototypes = [Prototype(p.vector.copy(), p.label, p.key, p.support) for p in initial_prototypes]
    initial_vocab = set(DT.BASE_VOCAB)
    vocab = set(initial_vocab)
    predictions: list[str] = []
    events: list[dict[str, Any]] = []
    calls = 0
    reuses = 0

    for flag, row, feature in zip(flags, evidence, features):
        if not flag:
            predictions.append(DT.NORMAL)
            events.append(
                {
                    "prediction_before_query": DT.NORMAL,
                    "queried": False,
                    "discovery_name": None,
                    "autonomous_reuse": False,
                    "route": "common_route_normal",
                }
            )
            continue

        key = evidence_key(feature, config.namer_threshold)
        if method == "nova_no_reuse":
            calls += 1
            label = namer(row, feature, mu, sd)
            if label is not None:
                vocab.add(label)
                predictions.append(label)
            else:
                predictions.append(ANOMALY)
            events.append(
                {
                    "prediction_before_query": UNKNOWN,
                    "queried": True,
                    "discovery_name": label,
                    "autonomous_reuse": False,
                    "route": "query_without_reuse",
                }
            )
            continue

        candidates = prototypes
        if method in {"nova_memory_reference", "nova_no_growth"}:
            candidates = [prototype for prototype in prototypes if prototype.key == key]
        match, distance = nearest(feature, candidates)
        if match is not None and distance <= radius:
            predictions.append(match.label)
            match.update(feature)
            reuses += 1
            events.append(
                {
                    "prediction_before_query": match.label,
                    "queried": False,
                    "discovery_name": None,
                    "autonomous_reuse": True,
                    "route": "prototype_reuse",
                }
            )
            continue

        calls += 1
        label = namer(row, feature, mu, sd)
        if label is None:
            predictions.append(ANOMALY)
            events.append(
                {
                    "prediction_before_query": UNKNOWN,
                    "queried": True,
                    "discovery_name": None,
                    "autonomous_reuse": False,
                    "route": "query_unresolved",
                }
            )
            continue
        if method == "nova_no_growth" and label not in initial_vocab:
            predictions.append(ANOMALY)
            events.append(
                {
                    "prediction_before_query": UNKNOWN,
                    "queried": True,
                    "discovery_name": label,
                    "autonomous_reuse": False,
                    "route": "query_growth_rejected",
                }
            )
            continue

        vocab.add(label)
        prototypes.append(Prototype(feature.copy(), label, key))
        predictions.append(label)
        events.append(
            {
                "prediction_before_query": UNKNOWN,
                "queried": True,
                "discovery_name": label,
                "autonomous_reuse": False,
                "route": "query_create",
            }
        )

    return predictions, events, {
        "namer_calls_total": calls,
        "memory_reuses_total": reuses,
        "final_vocab_size": len(vocab),
        "final_vocab_labels": sorted(vocab),
        "spurious_vocab_labels": sorted(vocab - initial_vocab - {DT.NOVEL}),
        "spurious_vocab_count": len(vocab - initial_vocab - {DT.NOVEL}),
        "active_prototypes": len(prototypes),
        "grew_novel": int(DT.NOVEL in vocab),
        "cluster_purity": None,
    }


def run_unverified_clustering(
    flags: list[bool], features: np.ndarray, radius: float
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    centers: list[np.ndarray] = []
    support: list[int] = []
    predictions: list[str] = []
    events: list[dict[str, Any]] = []
    for flag, feature in zip(flags, features):
        if not flag:
            predictions.append(DT.NORMAL)
            events.append(
                {
                    "prediction_before_query": DT.NORMAL,
                    "queried": False,
                    "discovery_name": None,
                    "autonomous_reuse": False,
                    "route": "common_route_normal",
                }
            )
            continue
        if not centers:
            centers.append(feature.copy())
            support.append(1)
            predictions.append(f"{CLUSTER_PREFIX}0")
            events.append(
                {
                    "prediction_before_query": f"{CLUSTER_PREFIX}0",
                    "queried": False,
                    "discovery_name": None,
                    "autonomous_reuse": False,
                    "route": "cluster_create_unverified",
                }
            )
            continue
        distances = [float(np.linalg.norm(feature - center)) for center in centers]
        index = int(np.argmin(distances))
        if distances[index] > radius:
            index = len(centers)
            centers.append(feature.copy())
            support.append(1)
            reused = False
        else:
            support[index] += 1
            eta = 1.0 / support[index]
            centers[index] = (1.0 - eta) * centers[index] + eta * feature
            reused = True
        predictions.append(f"{CLUSTER_PREFIX}{index}")
        events.append(
            {
                "prediction_before_query": f"{CLUSTER_PREFIX}{index}",
                "queried": False,
                "discovery_name": None,
                # Opaque cluster assignment is not semantic reuse.
                "autonomous_reuse": False,
                "cluster_reuse": reused,
                "route": "cluster_reuse_unverified" if reused else "cluster_create_unverified",
            }
        )
    return predictions, events, {
        "namer_calls_total": 0,
        "memory_reuses_total": int(sum(max(0, value - 1) for value in support)),
        "final_vocab_size": len(DT.BASE_VOCAB),
        "active_prototypes": len(centers),
        "grew_novel": 0,
    }


def weighted_cluster_purity(predictions: list[str], truths: list[str]) -> float | None:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction, truth in zip(predictions, truths):
        if prediction.startswith(CLUSTER_PREFIX) and truth != DT.NORMAL:
            buckets[prediction][truth] += 1
    total = sum(sum(counter.values()) for counter in buckets.values())
    if not total:
        return None
    return float(sum(max(counter.values()) for counter in buckets.values()) / total)


def metrics(
    predictions: list[str],
    events: list[dict[str, Any]],
    truths: list[str],
    onset: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if not (len(predictions) == len(events) == len(truths)):
        raise AssertionError("prediction/event/truth ledgers must have equal length")
    if any(event["queried"] and event["autonomous_reuse"] for event in events):
        raise AssertionError("a current-window naming query cannot also be counted as autonomous reuse")
    namer_calls_total = int(sum(bool(event["queried"]) for event in events))
    namer_calls_warm = int(sum(bool(event["queried"]) for event in events[:onset]))
    predictions = predictions[onset:]
    events = events[onset:]
    truths = truths[onset:]
    predicted_anomaly = [prediction != DT.NORMAL for prediction in predictions]
    true_anomaly = [truth != DT.NORMAL for truth in truths]
    tp = sum(pred and true for pred, true in zip(predicted_anomaly, true_anomaly))
    fp = sum(pred and not true for pred, true in zip(predicted_anomaly, true_anomaly))
    fn = sum((not pred) and true for pred, true in zip(predicted_anomaly, true_anomaly))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    novel_indices = [index for index, truth in enumerate(truths) if truth == DT.NOVEL]
    known_indices = [index for index, truth in enumerate(truths) if truth in DT.KNOWN_ANOM]
    normal_indices = [index for index, truth in enumerate(truths) if truth == DT.NORMAL]
    if not novel_indices:
        raise AssertionError("post-onset stream contains no novel occurrence")
    first_novel_index = novel_indices[0]
    first_event = events[first_novel_index]
    naming_query_indices = [
        index
        for index in novel_indices
        if bool(events[index]["queried"]) and events[index]["discovery_name"] is not None
    ]
    first_naming_query_index = naming_query_indices[0] if naming_query_indices else None
    correct_discovery_indices = [
        index
        for index in naming_query_indices
        if events[index]["discovery_name"] == DT.NOVEL
    ]
    correct_discovery_index = (
        correct_discovery_indices[0] if correct_discovery_indices else None
    )
    future_novel_indices = (
        [index for index in novel_indices if index > correct_discovery_index]
        if correct_discovery_index is not None
        else []
    )
    future_reuse_indices = [
        index for index in future_novel_indices if bool(events[index]["autonomous_reuse"])
    ]
    future_query_indices = [
        index for index in future_novel_indices if bool(events[index]["queried"])
    ]
    future_reuse_accuracy = (
        float(np.mean([predictions[index] == DT.NOVEL for index in future_reuse_indices]))
        if future_reuse_indices
        else None
    )
    future_query_name_accuracy = (
        float(
            np.mean(
                [events[index]["discovery_name"] == DT.NOVEL for index in future_query_indices]
            )
        )
        if future_query_indices
        else None
    )
    out = {
        "post_n": len(truths),
        "binary_f1": float(f1),
        "binary_precision": float(precision),
        "binary_recall": float(recall),
        "novel_detection_recall": float(np.mean([predicted_anomaly[index] for index in novel_indices])),
        "novel_typed_accuracy_including_queries": float(
            np.mean([predictions[index] == DT.NOVEL for index in novel_indices])
        ),
        "known_typed_accuracy": float(
            np.mean([predictions[index] == truths[index] for index in known_indices])
        ),
        "normal_false_alarm_rate": float(
            np.mean([predicted_anomaly[index] for index in normal_indices])
        ),
        "overall_typed_accuracy": float(
            np.mean([prediction == truth for prediction, truth in zip(predictions, truths)])
        ),
        "cluster_purity": weighted_cluster_purity(predictions, truths),
        "first_occurrence_post_index": first_novel_index,
        "first_occurrence_pre_update_prediction": first_event["prediction_before_query"],
        "first_occurrence_pre_update_detection": int(
            first_event["prediction_before_query"] != DT.NORMAL
        ),
        "first_occurrence_pre_update_typed_correct": int(
            first_event["prediction_before_query"] == DT.NOVEL
        ),
        "first_occurrence_queried": int(first_event["queried"]),
        "first_naming_query_post_index": first_naming_query_index,
        "discovery_name": (
            events[first_naming_query_index]["discovery_name"]
            if first_naming_query_index is not None
            else None
        ),
        "discovery_name_correct": (
            int(events[first_naming_query_index]["discovery_name"] == DT.NOVEL)
            if first_naming_query_index is not None
            else None
        ),
        "correct_discovery_occurred": int(correct_discovery_index is not None),
        "correct_discovery_post_index": correct_discovery_index,
        "correct_discovery_novel_ordinal": (
            novel_indices.index(correct_discovery_index) + 1
            if correct_discovery_index is not None
            else None
        ),
        "post_discovery_novel_n": len(future_novel_indices),
        "post_discovery_future_typed_accuracy_including_queries": (
            float(np.mean([predictions[index] == DT.NOVEL for index in future_novel_indices]))
            if future_novel_indices
            else None
        ),
        "post_discovery_future_reuse_count": len(future_reuse_indices),
        "post_discovery_future_reuse_rate": (
            len(future_reuse_indices) / len(future_novel_indices)
            if future_novel_indices
            else None
        ),
        "post_discovery_future_reuse_accuracy": future_reuse_accuracy,
        "post_discovery_future_query_count": len(future_query_indices),
        "post_discovery_future_query_name_accuracy": future_query_name_accuracy,
        "namer_calls_total": namer_calls_total,
        "namer_calls_warm": namer_calls_warm,
        "namer_calls_post": int(sum(bool(event["queried"]) for event in events)),
    }
    out.update(extra)
    return out


def run_seed(
    background: dict[str, Any], seed: int, config: Config, api_key: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng_stats = np.random.default_rng(100_000 + seed)
    rng_train = np.random.default_rng(200_000 + seed)
    rng_unsup = np.random.default_rng(300_000 + seed)
    rng_stream = np.random.default_rng(400_000 + seed)
    mu, sd = CB.normal_stats(rng_stats, n=config.normal_stats_n)
    namer, namer_metadata = make_namer(config, api_key)

    cnn, cnn_train_windows, cnn_train_labels = train_cnn(seed, config, rng_train)
    normal_train = [CB.make_window(None, rng_unsup) for _ in range(config.normal_train_n)]
    normal_cal = [CB.make_window(None, rng_unsup) for _ in range(config.normal_cal_n)]
    windows, truths, onset = build_stream(rng_stream, config)
    evidence, features = evidence_and_z(windows, mu, sd)
    shared_novelty_flags = novelty_gate_flags(
        evidence, mu, sd, config.novelty_threshold
    )
    train_evidence, train_features = evidence_and_z(cnn_train_windows, mu, sd)
    del train_evidence

    cnn_closed_predictions, cnn_flags = cnn_predictions(cnn, windows)
    unsupervised = fit_unsupervised_detectors(
        seed, config, normal_train, normal_cal, windows
    )

    rows: list[dict[str, Any]] = []

    def add(
        method: str,
        predictions: list[str],
        events: list[dict[str, Any]],
        extra: dict[str, Any],
    ) -> None:
        metadata = METHODS[method]
        row = metrics(predictions, events, truths, onset, extra)
        row.update(
            {
                "background": background["name"],
                "background_kind": background["kind"],
                "seed": seed,
                "method": method,
                "implementation_status": metadata["implementation_status"],
                "proxy_of": metadata["proxy_of"],
                "official_external_implementation": metadata["official_external_implementation"],
                "namer_call_rate_post": row["namer_calls_post"] / config.post_n,
            }
        )
        rows.append(row)

    add(
        "cnn_detector_only",
        cnn_closed_predictions,
        detector_events(cnn_closed_predictions),
        {
            "final_vocab_size": len(DT.BASE_VOCAB),
            "active_prototypes": 0,
            "grew_novel": 0,
            "memory_reuses_total": 0,
        },
    )
    for route_mode in ("flag_only", "flag_or_novelty_gate"):
        cnn_named, cnn_events = shared_namer_predictions(
            cnn_closed_predictions,
            cnn_flags,
            shared_novelty_flags,
            route_mode,
            evidence,
            features,
            mu,
            sd,
            namer,
        )
        add(
            f"cnn_shared_namer_{route_mode}",
            cnn_named,
            cnn_events,
            {
                "routing": route_mode,
                "final_vocab_size": len(DT.BASE_VOCAB),
                "active_prototypes": 0,
                "grew_novel": 0,
                "memory_reuses_total": 0,
            },
        )

    for prefix in ("memstream_proxy", "anomaly_transformer_proxy"):
        flags = unsupervised[prefix]["flags"]
        detector_predictions = [ANOMALY if flag else DT.NORMAL for flag in flags]
        add(
            f"{prefix}_detector_only",
            detector_predictions,
            detector_events(detector_predictions),
            {
                "detector_threshold": unsupervised[prefix]["threshold"],
                "final_vocab_size": 0,
                "active_prototypes": 0,
                "grew_novel": 0,
                "memory_reuses_total": 0,
            },
        )
        for route_mode in ("flag_only", "flag_or_novelty_gate"):
            named, named_events = shared_namer_predictions(
                detector_predictions,
                flags,
                shared_novelty_flags,
                route_mode,
                evidence,
                features,
                mu,
                sd,
                namer,
            )
            add(
                f"{prefix}_shared_namer_{route_mode}",
                named,
                named_events,
                {
                    "routing": route_mode,
                    "detector_threshold": unsupervised[prefix]["threshold"],
                    "final_vocab_size": 0,
                    "active_prototypes": 0,
                    "grew_novel": 0,
                    "memory_reuses_total": 0,
                },
            )

    initial_prototypes, radius = prototype_setup(train_features, cnn_train_labels, config)
    common_memory_route = [
        detector or novel for detector, novel in zip(cnn_flags, shared_novelty_flags)
    ]
    for method in (
        "nova_memory_reference",
        "nova_no_growth",
        "nova_no_reuse",
        "nearest_prototype",
    ):
        predictions, memory_events, extra = run_semantic_memory(
            method,
            common_memory_route,
            evidence,
            features,
            initial_prototypes,
            radius,
            mu,
            sd,
            config,
            namer,
        )
        # Query accounting is recomputed from the immutable per-event ledger.
        extra.pop("namer_calls_total", None)
        extra["routing"] = "cnn_flag_or_shared_novelty_gate"
        extra["prototype_radius"] = radius
        add(method, predictions, memory_events, extra)

    cluster_predictions, cluster_events, cluster_extra = run_unverified_clustering(
        common_memory_route, features, radius
    )
    cluster_extra["routing"] = "cnn_flag_or_shared_novelty_gate"
    cluster_extra["prototype_radius"] = radius
    add(
        "clustering_no_semantic_verify",
        cluster_predictions,
        cluster_events,
        cluster_extra,
    )

    stream_manifest = {
        "background": background,
        "seed": seed,
        "onset": onset,
        "labels": truths,
        "window_digest": hashlib.sha256(np.stack(windows).tobytes()).hexdigest(),
        "normal_train_digest": hashlib.sha256(np.stack(normal_train).tobytes()).hexdigest(),
        "normal_calibration_digest": hashlib.sha256(np.stack(normal_cal).tobytes()).hexdigest(),
        "score_detector_calibration_split_shared": True,
    }
    return rows, {
        "background": background["name"],
        "seed": seed,
        "sha256": stable_hash(stream_manifest),
        "onset": onset,
        "post_label_counts": dict(Counter(truths[onset:])),
        "namer": namer_metadata,
        "prototype_radius": radius,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_to_summarize = [
        "binary_f1",
        "binary_precision",
        "binary_recall",
        "novel_detection_recall",
        "novel_typed_accuracy_including_queries",
        "known_typed_accuracy",
        "normal_false_alarm_rate",
        "overall_typed_accuracy",
        "namer_calls_post",
        "namer_calls_warm",
        "namer_calls_total",
        "namer_call_rate_post",
        "first_occurrence_pre_update_detection",
        "first_occurrence_pre_update_typed_correct",
        "first_occurrence_queried",
        "discovery_name_correct",
        "correct_discovery_occurred",
        "correct_discovery_novel_ordinal",
        "post_discovery_future_typed_accuracy_including_queries",
        "post_discovery_future_reuse_count",
        "post_discovery_future_reuse_rate",
        "post_discovery_future_reuse_accuracy",
        "post_discovery_future_query_count",
        "post_discovery_future_query_name_accuracy",
        "final_vocab_size",
        "spurious_vocab_count",
        "active_prototypes",
        "grew_novel",
        "cluster_purity",
    ]
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["background"], row["method"])].append(row)
    for (background, method), group in sorted(groups.items()):
        for metric_name in metrics_to_summarize:
            values = [
                float(row[metric_name])
                for row in group
                if row.get(metric_name) is not None and np.isfinite(float(row[metric_name]))
            ]
            if not values:
                continue
            summary.append(
                {
                    "background": background,
                    "method": method,
                    "metric": metric_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    return summary


def integrity_checks(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[(row["background"], int(row["seed"]))][row["method"]] = row
    expected_groups = len(config.backgrounds) * config.seeds
    if len(groups) != expected_groups:
        raise AssertionError(f"expected {expected_groups} background/seed groups, got {len(groups)}")
    detector_pairs = [
        ("cnn_detector_only", "cnn_shared_namer_flag_only"),
        ("memstream_proxy_detector_only", "memstream_proxy_shared_namer_flag_only"),
        (
            "anomaly_transformer_proxy_detector_only",
            "anomaly_transformer_proxy_shared_namer_flag_only",
        ),
    ]
    detector_invariance = True
    complete_methods = True
    no_reuse_is_query_only = True
    query_bounds = True
    for methods in groups.values():
        complete_methods &= set(methods) == set(METHODS)
        for detector_only, flag_only in detector_pairs:
            for metric_name in (
                "binary_f1",
                "binary_precision",
                "binary_recall",
                "novel_detection_recall",
                "normal_false_alarm_rate",
            ):
                detector_invariance &= bool(
                    np.isclose(methods[detector_only][metric_name], methods[flag_only][metric_name])
                )
        no_reuse_is_query_only &= methods["nova_no_reuse"]["post_discovery_future_reuse_count"] == 0
        for row in methods.values():
            query_bounds &= 0 <= row["namer_calls_warm"] <= config.warm_n
            query_bounds &= 0 <= row["namer_calls_post"] <= config.post_n
            query_bounds &= row["namer_calls_total"] == row["namer_calls_warm"] + row["namer_calls_post"]
    checks = {
        "complete_method_matrix": bool(complete_methods),
        "flag_only_preserves_detector_binary_decisions": bool(detector_invariance),
        "no_reuse_arm_has_zero_autonomous_future_reuse": bool(no_reuse_is_query_only),
        "query_counts_are_phase_exact_and_bounded": bool(query_bounds),
        "background_seed_groups": len(groups),
        "rows": len(rows),
    }
    if not all(value for key, value in checks.items() if isinstance(value, bool)):
        raise AssertionError(f"integrity check failed: {checks}")
    return checks


def lookup(summary: list[dict[str, Any]], background: str, method: str, metric_name: str) -> float | None:
    for row in summary:
        if row["background"] == background and row["method"] == method and row["metric"] == metric_name:
            return float(row["mean"])
    return None


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def build_report(payload: dict[str, Any], output: Path) -> None:
    config = payload["config"]
    summary = payload["summary"]
    methods = payload["methods"]
    lines = [
        "# Fair Open-Vocabulary Baselines and NOVA Mechanism Ablations",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "This is a small reproducible mechanism pilot. All detector+namer arms use the same evidence and shared namer. NOVA mechanism arms use the same frozen CNN anomaly decisions, evidence features, stream order, and prototype radius.",
        "",
        "The real-background condition is a controlled anomaly injection into normal SMD background. It does not contain native typed real-fault ground truth.",
        "",
        "## Implementation Disclosure",
        "",
        "| Method | Family | Implementation status | External official implementation | Shared semantic namer | Replay |",
        "|---|---|---|---:|---|---:|",
    ]
    for method, metadata in methods.items():
        lines.append(
            f"| `{method}` | {metadata['family']} | {metadata['implementation_status']}"
            f"{('; proxy of ' + metadata['proxy_of']) if metadata['proxy_of'] else ''} | "
            f"{'yes' if metadata['official_external_implementation'] else 'no'} | {metadata['semantic_namer']} | "
            f"{'yes' if metadata.get('replay') else 'no'} |"
        )

    lines.extend(
        [
            "",
            "The MemStream result uses the repository's AE reconstruction-memory proxy, and the Anomaly Transformer result uses the repository's compact reimplementation. Neither number is an official-model result or an external SOTA reproduction.",
            "",
            "The default shared namer is a deterministic evidence-argmax rule matching the current prompt's decision rule. It is not an LLM result. Use `--namer-mode llm` to call the repository's common LLM namer for every relevant arm.",
            "",
            "## Protocol",
            "",
            f"- Known vocabulary: `{DT.BASE_VOCAB}`.",
            f"- Held-out novel type: `{DT.NOVEL}`.",
            f"- Stream: {config['warm_n']} warm windows, then {config['post_n']} evaluation windows; both preserve the existing 50% normal / 50% anomaly protocol.",
            f"- Seeds: {config['seed_start']} through {config['seed_start'] + config['seeds'] - 1}.",
            f"- CNN pilot budget: {config['cnn_train_per_class']} windows/class, {config['cnn_epochs']} epochs.",
            f"- Score-proxy budget: {config['normal_train_n']} normal training windows, {config['normal_cal_n']} calibration windows, {config['unsup_epochs']} epochs.",
            f"- MemStream proxy embedding: {config['memstream_emb_dim']}; Anomaly Transformer proxy: d_model={config['anomaly_transformer_d_model']}, heads={config['anomaly_transformer_heads']}, layers={config['anomaly_transformer_layers']}.",
            f"- Namer mode: `{config['namer_mode']}`.",
            f"- Shared novelty gate: dominant non-known evidence deviation > {config['novelty_threshold']:.2f}.",
            "- All score detectors use the same held-out normal calibration windows; each score family receives its own q95 value in its native score scale.",
            "- Detection is prediction != `normal`; detector-only scalar baselines emit the sentinel `anomaly` and therefore do not receive typed credit.",
            "- `flag_only` and `flag_or_novelty_gate` routing variants use the same detector outputs and same namer; only the routing Boolean changes.",
            "- The first novel arrival locks `first_occurrence_pre_update`; the first actual naming query records `discovery_name`; future reuse starts only after the first correct discovery name and requires `queried=false`.",
            "",
            "## Results",
        ]
    )
    for background in config["backgrounds"]:
        lines.extend(
            [
                "",
                f"### {background}",
                "",
                "| Method | Binary F1 | Novel recall | Typed incl. queries | First pre-update typed | First queried name correct | Correct discovery rate | Future reuse acc. | Future reuse rate | Queries post |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            calls = lookup(summary, background, method, "namer_calls_post")
            lines.append(
                f"| `{method}` | {percent(lookup(summary, background, method, 'binary_f1'))} | "
                f"{percent(lookup(summary, background, method, 'novel_detection_recall'))} | "
                f"{percent(lookup(summary, background, method, 'novel_typed_accuracy_including_queries'))} | "
                f"{percent(lookup(summary, background, method, 'first_occurrence_pre_update_typed_correct'))} | "
                f"{percent(lookup(summary, background, method, 'discovery_name_correct'))} | "
                f"{percent(lookup(summary, background, method, 'correct_discovery_occurred'))} | "
                f"{percent(lookup(summary, background, method, 'post_discovery_future_reuse_accuracy'))} | "
                f"{percent(lookup(summary, background, method, 'post_discovery_future_reuse_rate'))} | "
                f"{('n/a' if calls is None else f'{calls:.1f}')} |"
            )
        lines.extend(
            [
                "",
                "| Method | Queries warm | Queries post | Active prototypes | Final vocab | Spurious vocab | Correct discovery ordinal | Cluster purity |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            warm_calls = lookup(summary, background, method, "namer_calls_warm")
            post_calls = lookup(summary, background, method, "namer_calls_post")
            active = lookup(summary, background, method, "active_prototypes")
            vocab = lookup(summary, background, method, "final_vocab_size")
            spurious = lookup(summary, background, method, "spurious_vocab_count")
            ordinal = lookup(summary, background, method, "correct_discovery_novel_ordinal")
            purity = lookup(summary, background, method, "cluster_purity")
            lines.append(
                f"| `{method}` | {('n/a' if warm_calls is None else f'{warm_calls:.1f}')} | "
                f"{('n/a' if post_calls is None else f'{post_calls:.1f}')} | "
                f"{('n/a' if active is None else f'{active:.1f}')} | "
                f"{('n/a' if vocab is None else f'{vocab:.1f}')} | "
                f"{('n/a' if spurious is None else f'{spurious:.1f}')} | "
                f"{('n/a' if ordinal is None else f'{ordinal:.1f}')} | {percent(purity)} |"
            )

    lines.extend(["", "## Pilot Observations", ""])
    for background in config["backgrounds"]:
        cnn_flag = lookup(summary, background, "cnn_shared_namer_flag_only", "novel_detection_recall")
        cnn_gate = lookup(
            summary,
            background,
            "cnn_shared_namer_flag_or_novelty_gate",
            "novel_detection_recall",
        )
        mem_flag = lookup(
            summary,
            background,
            "memstream_proxy_shared_namer_flag_only",
            "novel_detection_recall",
        )
        mem_gate = lookup(
            summary,
            background,
            "memstream_proxy_shared_namer_flag_or_novelty_gate",
            "novel_detection_recall",
        )
        at_flag = lookup(
            summary,
            background,
            "anomaly_transformer_proxy_shared_namer_flag_only",
            "novel_detection_recall",
        )
        at_gate = lookup(
            summary,
            background,
            "anomaly_transformer_proxy_shared_namer_flag_or_novelty_gate",
            "novel_detection_recall",
        )
        reference_calls = lookup(summary, background, "nova_memory_reference", "namer_calls_post")
        no_reuse_calls = lookup(summary, background, "nova_no_reuse", "namer_calls_post")
        reference_discovery = lookup(
            summary, background, "nova_memory_reference", "correct_discovery_occurred"
        )
        reference_reuse = lookup(
            summary,
            background,
            "nova_memory_reference",
            "post_discovery_future_reuse_accuracy",
        )
        reference_reuse_rate = lookup(
            summary,
            background,
            "nova_memory_reference",
            "post_discovery_future_reuse_rate",
        )
        lines.extend(
            [
                f"- **{background}:** adding the shared novelty gate changes novel recall from "
                f"{percent(cnn_flag)} to {percent(cnn_gate)} for CNN, {percent(mem_flag)} to "
                f"{percent(mem_gate)} for the MemStream proxy, and {percent(at_flag)} to "
                f"{percent(at_gate)} for the Anomaly Transformer proxy.",
                f"- **{background}:** the memory reference uses "
                f"{('n/a' if reference_calls is None else f'{reference_calls:.1f}')} post queries versus "
                f"{('n/a' if no_reuse_calls is None else f'{no_reuse_calls:.1f}')} without reuse; correct discovery occurs in "
                f"{percent(reference_discovery)} of seeds, followed by {percent(reference_reuse)} conditional "
                f"future-reuse accuracy at {percent(reference_reuse_rate)} reuse coverage.",
            ]
        )

    lines.extend(
        [
            "",
            "These observations do not establish superiority over the named external methods: the score baselines are explicitly tiny repository proxies, and the naming benchmark is evidence-aligned.",
            "",
            "## Mechanism Definitions",
            "",
            "- `nova_memory_reference`: frozen common detector, exact evidence-component key, labeled prototype reuse within a shared radius, shared namer and vocabulary growth on a memory miss.",
            "- `nova_no_growth`: identical keyed reuse, but a newly named label is not admitted to memory or vocabulary.",
            "- `nova_no_reuse`: every detected anomaly calls the shared namer; labels may grow, but no prototype is retrieved.",
            "- `nearest_prototype`: global nearest labeled prototype with radius-gated naming and growth; no evidence-key partition.",
            "- `clustering_no_semantic_verify`: online radius clustering with opaque cluster IDs and no semantic namer. Purity is an offline diagnostic only.",
            "- This pilot contains no classifier replay or detector update in any NOVA arm. It is a memory-only mechanism experiment and does not claim a no-replay ablation.",
            "",
            "## Interpretation Limits",
            "",
            "1. This run is a pilot with compact training budgets; it is intended to validate fairness and mechanism direction, not establish an AAAI headline.",
            "2. Shared-namer arms isolate semantic access from detection. They should be compared with their detector-only counterpart for detection invariance and naming gain.",
            "3. `Typed incl. queries` includes direct current-window naming and is not a reuse metric. `Future reuse acc.` is conditional on a correct discovery having occurred and uses only later novel windows with `queried=false`; always read it with correct-discovery and reuse rates.",
            "4. Detector-only typed accuracy is zero for score-only methods by task definition, not evidence of inferior binary detection.",
            "5. The controlled benchmark is evidence-aligned. The deterministic rule namer can be unusually strong and does not demonstrate LLM necessity.",
            "6. Real-background injection is not a native typed real-fault benchmark.",
            "",
            "## Integrity Checks",
            "",
            *[f"- `{key}`: `{value}`" for key, value in payload["integrity"].items()],
            "",
            "## Artifacts",
            "",
            f"- Result JSON: `{payload['artifacts']['result_json']}`",
            f"- Runner: `{Path(__file__).resolve()}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs" / "fair_openvocab_ablation_2026-07-09")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--backgrounds", default="synthetic,SMD:1-1")
    parser.add_argument("--normal-stats-n", type=int, default=60)
    parser.add_argument("--cnn-train-per-class", type=int, default=40)
    parser.add_argument("--cnn-epochs", type=int, default=6)
    parser.add_argument("--normal-train-n", type=int, default=64)
    parser.add_argument("--normal-cal-n", type=int, default=48)
    parser.add_argument("--unsup-epochs", type=int, default=2)
    parser.add_argument("--memstream-emb-dim", type=int, default=16)
    parser.add_argument("--anomaly-transformer-d-model", type=int, default=16)
    parser.add_argument("--anomaly-transformer-heads", type=int, default=2)
    parser.add_argument("--anomaly-transformer-layers", type=int, default=1)
    parser.add_argument("--warm-n", type=int, default=60)
    parser.add_argument("--post-n", type=int, default=120)
    parser.add_argument("--score-quantile", type=float, default=0.95)
    parser.add_argument("--namer-mode", choices=("rule", "llm"), default="rule")
    parser.add_argument("--namer-threshold", type=float, default=2.0)
    parser.add_argument("--novelty-threshold", type=float, default=2.3)
    parser.add_argument("--prototype-radius-quantile", type=float, default=0.90)
    parser.add_argument("--prototype-radius-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        seeds=args.seeds,
        seed_start=args.seed_start,
        backgrounds=tuple(item.strip() for item in args.backgrounds.split(",") if item.strip()),
        normal_stats_n=args.normal_stats_n,
        cnn_train_per_class=args.cnn_train_per_class,
        cnn_epochs=args.cnn_epochs,
        normal_train_n=args.normal_train_n,
        normal_cal_n=args.normal_cal_n,
        unsup_epochs=args.unsup_epochs,
        memstream_emb_dim=args.memstream_emb_dim,
        anomaly_transformer_d_model=args.anomaly_transformer_d_model,
        anomaly_transformer_heads=args.anomaly_transformer_heads,
        anomaly_transformer_layers=args.anomaly_transformer_layers,
        warm_n=args.warm_n,
        post_n=args.post_n,
        score_quantile=args.score_quantile,
        namer_mode=args.namer_mode,
        namer_threshold=args.namer_threshold,
        novelty_threshold=args.novelty_threshold,
        prototype_radius_quantile=args.prototype_radius_quantile,
        prototype_radius_scale=args.prototype_radius_scale,
    )
    if config.seeds < 1 or config.warm_n < 1 or config.post_n < 1:
        raise ValueError("seeds, warm_n, and post_n must be positive")
    if not 0 < config.score_quantile < 1 or not 0 < config.prototype_radius_quantile < 1:
        raise ValueError("quantiles must be in (0, 1)")
    if config.anomaly_transformer_d_model % config.anomaly_transformer_heads:
        raise ValueError("Anomaly Transformer d_model must be divisible by heads")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    backgrounds: list[dict[str, Any]] = []
    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        for background_name in config.backgrounds:
            background = activate_background(background_name)
            backgrounds.append(background)
            print(f"[background] {background_name}: {background['kind']}", flush=True)
            for seed in range(config.seed_start, config.seed_start + config.seeds):
                print(f"  [seed {seed}] training and evaluating", flush=True)
                seed_rows, manifest = run_seed(background, seed, config, api_key)
                rows.extend(seed_rows)
                manifests.append(manifest)
                concise = " | ".join(
                    f"{row['method']} novTypedInclQ={row['novel_typed_accuracy_including_queries']:.0%} "
                    f"futureReuse={row['post_discovery_future_reuse_accuracy']} F1={row['binary_f1']:.2f}"
                    for row in seed_rows
                    if row["method"]
                    in {
                        "cnn_detector_only",
                        "cnn_shared_namer_flag_or_novelty_gate",
                        "nova_memory_reference",
                        "nova_no_reuse",
                    }
                )
                print(f"    {concise}", flush=True)
    finally:
        RB.deactivate()

    summary = summarize(rows)
    integrity = integrity_checks(rows, config)
    result_json = args.output_dir / "fair_openvocab_ablation_result.json"
    report_md = args.output_dir / "fair_openvocab_ablation_report.md"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": asdict(config),
        "protocol": {
            "known_vocab": DT.BASE_VOCAB,
            "known_anomalies": DT.KNOWN_ANOM,
            "novel": DT.NOVEL,
            "normal_fraction": 0.5,
            "stream_shared_across_arms": True,
            "real_background_is_controlled_injection": True,
            "native_typed_real_faults": False,
        },
        "backgrounds": backgrounds,
        "methods": METHODS,
        "provenance": provenance(),
        "integrity": integrity,
        "manifests": manifests,
        "rows": rows,
        "summary": summary,
        "warnings": [
            "MemStream is the repository compact AE-memory proxy, not the official external implementation.",
            "Anomaly Transformer is the repository compact reimplementation, not the official external implementation.",
            "The default rule namer is not an LLM and cannot establish LLM necessity.",
            "NOVA arms use a frozen detector and contain no replay; this is a memory-only pilot, not a full-NOVA result or no-replay ablation.",
            "SMD results use controlled injections on normal background, not native typed faults.",
            "This small run is a mechanism pilot, not a final statistical comparison.",
        ],
        "artifacts": {"result_json": str(result_json), "report_markdown": str(report_md)},
    }
    result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_report(payload, report_md)
    print(f"saved result -> {result_json}")
    print(f"saved report -> {report_md}")


if __name__ == "__main__":
    main()
