#!/usr/bin/env python3
"""OpenMax-style EVT rejection with the fair pilot's shared namer.

This is an auditable time-series adaptation, not the official OpenMax code and
not a layer-for-layer reproduction of the CVPR 2016 visual model.  It uses the
closed-set CNN logit vector as an activation vector, estimates a mean activation
vector (MAV) for each known class, fits a SciPy Weibull distribution to each
class's largest MAV distances, and rejects a prediction when its predicted-
class Weibull negative log-survival exceeds a known-only pre-stream calibration
quantile. Negative log-survival is used instead of the equivalent CDF ordering
because large distances otherwise round to CDF=1 in float64.

All data construction, structured evidence, known/novel taxonomy, CNN training,
and deterministic namer are imported from run_fair_openvocab_ablation.py.  The
event ledger enforces predict -> record -> query -> update.  A queried current
window can receive a post-query name but is never counted as autonomous reuse.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable

import numpy as np
from scipy.stats import weibull_min
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import scripts.exp_detection_tie as DT  # noqa: E402
import sigla_exp.ovbench as CB  # noqa: E402
from sota_compare import run_fair_openvocab_ablation as FAIR  # noqa: E402
import sota_compare.realbench as RB  # noqa: E402


UNKNOWN = "unknown"


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
    warm_n: int
    post_n: int
    namer_threshold: float
    novelty_threshold: float
    prototype_radius_quantile: float
    prototype_radius_scale: float
    evt_tail_size: int
    evt_cal_per_class: int
    evt_cal_quantile: float
    evt_min_correct_fit: int


METHODS: dict[str, dict[str, Any]] = {
    "cnn_closed_set": {
        "family": "closed_set_reference",
        "description": "The same trained CNN with no rejection and no namer.",
        "shared_namer": False,
        "memory": False,
    },
    "openmax_style_evt_reject_only": {
        "family": "open_set_rejection",
        "description": "Predicted-class MAV distance, Weibull negative-log-survival rejection, unknown sentinel.",
        "shared_namer": False,
        "memory": False,
    },
    "openmax_style_evt_shared_namer_no_memory": {
        "family": "open_set_rejection_plus_shared_namer",
        "description": "Every EVT rejection calls the same deterministic evidence namer; no reuse memory.",
        "shared_namer": True,
        "memory": False,
    },
    "openmax_style_evt_shared_namer_memory": {
        "family": "open_set_rejection_plus_shared_namer_and_memory",
        "description": "EVT rejection first consults prior keyed evidence prototypes, then calls the same namer on a miss.",
        "shared_namer": True,
        "memory": True,
    },
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance() -> dict[str, Any]:
    dependencies = [
        Path(__file__).resolve(),
        Path(FAIR.__file__).resolve(),
        ROOT / "scripts" / "exp_detection_tie.py",
        ROOT / "sigla_exp" / "ovbench.py",
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
        "scipy": __import__("scipy").__version__,
        "torch": torch.__version__,
        "device": str(DT.device),
        "command": [sys.executable, *sys.argv],
    }


def fair_config(config: Config) -> FAIR.Config:
    """Construct the exact compact-CNN/evidence configuration used by the fair pilot."""
    return FAIR.Config(
        seeds=config.seeds,
        seed_start=config.seed_start,
        backgrounds=config.backgrounds,
        normal_stats_n=config.normal_stats_n,
        cnn_train_per_class=config.cnn_train_per_class,
        cnn_epochs=config.cnn_epochs,
        normal_train_n=config.normal_train_n,
        normal_cal_n=config.normal_cal_n,
        unsup_epochs=2,
        memstream_emb_dim=16,
        anomaly_transformer_d_model=16,
        anomaly_transformer_heads=2,
        anomaly_transformer_layers=1,
        warm_n=config.warm_n,
        post_n=config.post_n,
        score_quantile=0.95,
        namer_mode="rule",
        namer_threshold=config.namer_threshold,
        novelty_threshold=config.novelty_threshold,
        prototype_radius_quantile=config.prototype_radius_quantile,
        prototype_radius_scale=config.prototype_radius_scale,
    )


@torch.no_grad()
def activation_vectors(model: Any, windows: list[np.ndarray]) -> np.ndarray:
    model.eval()
    batch = torch.tensor(np.stack(windows)).to(DT.device)
    return model(batch).detach().cpu().numpy().astype(np.float64)


def fit_evt_models(
    train_activations: np.ndarray,
    train_labels: list[str],
    config: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    closed_indices = np.argmax(train_activations, axis=1)
    models: dict[str, dict[str, Any]] = {}
    fallbacks: list[str] = []
    for class_index, label in enumerate(DT.BASE_VOCAB):
        labeled_indices = np.asarray(
            [index for index, value in enumerate(train_labels) if value == label], dtype=int
        )
        correct_indices = labeled_indices[closed_indices[labeled_indices] == class_index]
        use_correct = len(correct_indices) >= config.evt_min_correct_fit
        fit_indices = correct_indices if use_correct else labeled_indices
        if not use_correct:
            fallbacks.append(label)
        mav = train_activations[fit_indices].mean(axis=0)
        distances = np.linalg.norm(train_activations[fit_indices] - mav, axis=1)
        tail_n = min(config.evt_tail_size, len(distances))
        if tail_n < 3:
            raise RuntimeError(f"too few activation distances to fit EVT for {label}")
        tail = np.sort(np.maximum(distances, 1e-8))[-tail_n:]
        shape, location, scale = weibull_min.fit(tail, floc=0.0)
        if not (np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0):
            raise RuntimeError(f"invalid Weibull parameters for {label}")
        models[label] = {
            "class_index": class_index,
            "mav": mav,
            "shape": float(shape),
            "location": float(location),
            "scale": float(scale),
            "tail_n": int(tail_n),
            "fit_n": int(len(fit_indices)),
            "correct_fit_n": int(len(correct_indices)),
            "used_correct_only": bool(use_correct),
            "tail_min": float(tail[0]),
            "tail_max": float(tail[-1]),
        }
    return models, {
        "classes_using_all_labeled_fallback": fallbacks,
        "fallback_count": len(fallbacks),
    }


def evt_scores(
    activations: np.ndarray,
    closed_predictions: list[str],
    evt_models: dict[str, dict[str, Any]],
) -> np.ndarray:
    scores: list[float] = []
    for activation, label in zip(activations, closed_predictions):
        model = evt_models[label]
        distance = float(np.linalg.norm(activation - model["mav"]))
        # This is monotone-equivalent to the Weibull CDF and remains finite in
        # the regime where scipy's CDF has rounded to exactly one.
        score = float(
            -weibull_min.logsf(
                distance,
                model["shape"],
                loc=model["location"],
                scale=model["scale"],
            )
        )
        scores.append(max(0.0, score))
    return np.asarray(scores, dtype=np.float64)


def make_known_calibration(
    rng: np.random.Generator, config: Config
) -> tuple[list[np.ndarray], list[str]]:
    windows: list[np.ndarray] = []
    labels: list[str] = []
    for label in DT.BASE_VOCAB:
        for _ in range(config.evt_cal_per_class):
            windows.append(FAIR.make_window(label, rng))
            labels.append(label)
    return windows, labels


def clone_prototypes(prototypes: list[FAIR.Prototype]) -> list[FAIR.Prototype]:
    return [
        FAIR.Prototype(item.vector.copy(), item.label, item.key, item.support)
        for item in prototypes
    ]


def run_arm(
    method: str,
    closed_predictions: list[str],
    scores: np.ndarray,
    reject_threshold: float,
    truths: list[str],
    evidence: list[dict[str, float]],
    features: np.ndarray,
    mu: dict[str, float],
    sd: dict[str, float],
    namer: Callable[..., str | None],
    initial_prototypes: list[FAIR.Prototype],
    prototype_radius: float,
    config: Config,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(method)
    uses_evt = method != "cnn_closed_set"
    uses_namer = bool(METHODS[method]["shared_namer"])
    uses_memory = bool(METHODS[method]["memory"])
    prototypes = clone_prototypes(initial_prototypes) if uses_memory else []
    predictions: list[str] = []
    events: list[dict[str, Any]] = []
    state_version = 0
    query_calls = 0
    memory_reuses = 0

    for index, (closed, score, truth, row, feature) in enumerate(
        zip(closed_predictions, scores, truths, evidence, features)
    ):
        rejected = bool(uses_evt and score > reject_threshold)
        state_before = state_version
        queried = False
        query_result: str | None = None
        autonomous_reuse = False
        route = "closed_accept"
        update_action = "none"

        if not rejected:
            pre_query_prediction = closed
            final_prediction = closed
        elif method == "openmax_style_evt_reject_only":
            pre_query_prediction = UNKNOWN
            final_prediction = UNKNOWN
            route = "evt_reject_unknown"
        elif uses_memory:
            key = FAIR.evidence_key(feature, config.namer_threshold)
            candidates = [item for item in prototypes if item.key == key]
            match, distance = FAIR.nearest(feature, candidates)
            if match is not None and distance <= prototype_radius:
                pre_query_prediction = match.label
                final_prediction = match.label
                autonomous_reuse = True
                memory_reuses += 1
                route = "evt_reject_prior_memory_reuse"
                # Prediction and scoring use the old prototype. Updating occurs only now.
                match.update(feature)
                state_version += 1
                update_action = "prototype_update_after_record"
            else:
                pre_query_prediction = UNKNOWN
                queried = True
                query_calls += 1
                query_result = namer(row, feature, mu, sd)
                final_prediction = query_result if query_result is not None else UNKNOWN
                route = "evt_reject_memory_miss_query"
                if query_result is not None:
                    prototypes.append(FAIR.Prototype(feature.copy(), query_result, key))
                    state_version += 1
                    update_action = "prototype_create_after_query"
        elif uses_namer:
            pre_query_prediction = UNKNOWN
            queried = True
            query_calls += 1
            query_result = namer(row, feature, mu, sd)
            final_prediction = query_result if query_result is not None else UNKNOWN
            route = "evt_reject_query_no_memory"
        else:
            raise AssertionError(f"unhandled method {method}")

        event = {
            "stream_index": index,
            "truth": truth,
            "phase_order": ["predict", "record", "query", "update"],
            "closed_prediction": closed,
            "evt_score": float(score),
            "evt_rejected_before_query": rejected,
            "prediction_before_query": pre_query_prediction,
            "recorded_before_query": True,
            "queried": queried,
            "query_result": query_result,
            "discovery_name": query_result,
            "prediction_after_query": final_prediction,
            "autonomous_reuse": autonomous_reuse,
            "query_current_window_counted_as_reuse": False,
            "route": route,
            "state_version_before": state_before,
            "state_version_after": state_version,
            "update_action": update_action,
        }
        if queried and autonomous_reuse:
            raise AssertionError("a current-window query cannot also be memory reuse")
        if queried and state_before != event["state_version_before"]:
            raise AssertionError("state changed before the query was recorded")
        predictions.append(final_prediction)
        events.append(event)

    return predictions, events, {
        "namer_calls_total_ledger": query_calls,
        "memory_reuses_total_ledger": memory_reuses,
        "final_state_version": state_version,
        "active_prototypes": len(prototypes),
        "prototype_radius": prototype_radius if uses_memory else None,
    }


def arm_metrics(
    predictions: list[str],
    events: list[dict[str, Any]],
    closed_predictions: list[str],
    truths: list[str],
    onset: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    common_events = [
        {
            "prediction_before_query": event["prediction_before_query"],
            "queried": event["queried"],
            "discovery_name": event["discovery_name"],
            "autonomous_reuse": event["autonomous_reuse"],
            "route": event["route"],
        }
        for event in events
    ]
    base = FAIR.metrics(predictions, common_events, truths, onset, extra={})
    post_predictions = predictions[onset:]
    post_events = events[onset:]
    post_closed = closed_predictions[onset:]
    post_truths = truths[onset:]
    novel_indices = [index for index, truth in enumerate(post_truths) if truth == DT.NOVEL]
    known_anomaly_indices = [
        index for index, truth in enumerate(post_truths) if truth in DT.KNOWN_ANOM
    ]
    known_vocab_indices = [
        index for index, truth in enumerate(post_truths) if truth in DT.BASE_VOCAB
    ]
    normal_indices = [index for index, truth in enumerate(post_truths) if truth == DT.NORMAL]
    first_query_indices = [
        index for index in novel_indices if bool(post_events[index]["queried"])
    ]
    first_query_index = first_query_indices[0] if first_query_indices else None
    correct_query_indices = [
        index
        for index in novel_indices
        if post_events[index]["queried"]
        and post_events[index]["query_result"] == DT.NOVEL
    ]
    rejected = [bool(event["evt_rejected_before_query"]) for event in post_events]
    base.update(
        {
            "closed_known_vocab_accuracy": float(
                np.mean([post_closed[index] == post_truths[index] for index in known_vocab_indices])
            ),
            "closed_known_anomaly_accuracy": float(
                np.mean(
                    [post_closed[index] == post_truths[index] for index in known_anomaly_indices]
                )
            ),
            "novel_evt_reject_recall": float(
                np.mean([rejected[index] for index in novel_indices])
            ),
            "known_vocab_evt_reject_rate": float(
                np.mean([rejected[index] for index in known_vocab_indices])
            ),
            "known_anomaly_evt_reject_rate": float(
                np.mean([rejected[index] for index in known_anomaly_indices])
            ),
            "normal_evt_reject_rate": float(
                np.mean([rejected[index] for index in normal_indices])
            ),
            "first_novel_evt_rejected": int(rejected[novel_indices[0]]),
            "first_novel_pre_query_typed_correct": int(
                post_events[novel_indices[0]]["prediction_before_query"] == DT.NOVEL
            ),
            "novel_pre_query_typed_accuracy": float(
                np.mean(
                    [
                        post_events[index]["prediction_before_query"] == DT.NOVEL
                        for index in novel_indices
                    ]
                )
            ),
            "first_novel_query_attempted": int(first_query_index is not None),
            "first_novel_query_result": (
                post_events[first_query_index]["query_result"]
                if first_query_index is not None
                else None
            ),
            "first_novel_query_name_correct": (
                int(post_events[first_query_index]["query_result"] == DT.NOVEL)
                if first_query_index is not None
                else None
            ),
            "eventual_correct_novel_discovery": int(bool(correct_query_indices)),
            "evt_rejections_warm": int(
                sum(bool(event["evt_rejected_before_query"]) for event in events[:onset])
            ),
            "evt_rejections_post": int(sum(rejected)),
            "ledger_sha256": FAIR.stable_hash(events),
        }
    )
    base.update(extra)
    if base["namer_calls_total"] != extra["namer_calls_total_ledger"]:
        raise AssertionError("query ledger and recomputed query count disagree")
    if base["post_discovery_future_reuse_count"] > 0 and not METHODS[extra["method"]]["memory"]:
        raise AssertionError("non-memory method reported future autonomous reuse")
    return base


def reference_manifest(
    background: dict[str, Any],
    seed: int,
    windows: list[np.ndarray],
    truths: list[str],
    onset: int,
    config: Config,
) -> dict[str, Any]:
    # Regenerate the fair runner's unused score-detector normal splits solely to
    # prove the stream/config identity against its stored manifest hash.
    rng_unsup = np.random.default_rng(300_000 + seed)
    normal_train = [CB.make_window(None, rng_unsup) for _ in range(config.normal_train_n)]
    normal_cal = [CB.make_window(None, rng_unsup) for _ in range(config.normal_cal_n)]
    fair_payload = {
        "background": background,
        "seed": seed,
        "onset": onset,
        "labels": truths,
        "window_digest": hashlib.sha256(np.stack(windows).tobytes()).hexdigest(),
        "normal_train_digest": hashlib.sha256(np.stack(normal_train).tobytes()).hexdigest(),
        "normal_calibration_digest": hashlib.sha256(np.stack(normal_cal).tobytes()).hexdigest(),
        "score_detector_calibration_split_shared": True,
    }
    return {
        "background": background["name"],
        "seed": seed,
        "fair_manifest_sha256_reproduced": FAIR.stable_hash(fair_payload),
        "window_sha256": fair_payload["window_digest"],
        "onset": onset,
    }


def run_seed(
    background: dict[str, Any], seed: int, config: Config
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shared_config = fair_config(config)
    rng_stats = np.random.default_rng(100_000 + seed)
    rng_train = np.random.default_rng(200_000 + seed)
    rng_evt_cal = np.random.default_rng(250_000 + seed)
    rng_stream = np.random.default_rng(400_000 + seed)
    mu, sd = CB.normal_stats(rng_stats, n=config.normal_stats_n)
    namer, _ = FAIR.make_namer(shared_config, api_key="")

    cnn, train_windows, train_labels = FAIR.train_cnn(seed, shared_config, rng_train)
    train_activations = activation_vectors(cnn, train_windows)
    evt_models, fit_audit = fit_evt_models(train_activations, train_labels, config)

    cal_windows, cal_truths = make_known_calibration(rng_evt_cal, config)
    cal_activations = activation_vectors(cnn, cal_windows)
    cal_predictions = [
        DT.BASE_VOCAB[int(index)] for index in np.argmax(cal_activations, axis=1)
    ]
    cal_scores = evt_scores(cal_activations, cal_predictions, evt_models)
    reject_threshold = float(np.quantile(cal_scores, config.evt_cal_quantile))
    cal_reject_rate = float(np.mean(cal_scores > reject_threshold))

    windows, truths, onset = FAIR.build_stream(rng_stream, shared_config)
    evidence, features = FAIR.evidence_and_z(windows, mu, sd)
    activations = activation_vectors(cnn, windows)
    closed_predictions = [
        DT.BASE_VOCAB[int(index)] for index in np.argmax(activations, axis=1)
    ]
    scores = evt_scores(activations, closed_predictions, evt_models)
    train_evidence, train_features = FAIR.evidence_and_z(train_windows, mu, sd)
    del train_evidence
    initial_prototypes, prototype_radius = FAIR.prototype_setup(
        train_features, train_labels, shared_config
    )

    rows: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    reject_flags_reference: list[bool] | None = None
    for method in METHODS:
        predictions, events, extra = run_arm(
            method,
            closed_predictions,
            scores,
            reject_threshold,
            truths,
            evidence,
            features,
            mu,
            sd,
            namer,
            initial_prototypes,
            prototype_radius,
            config,
        )
        extra.update(
            {
                "method": method,
                "evt_reject_threshold": reject_threshold,
                "evt_cal_reject_rate": cal_reject_rate,
            }
        )
        row = arm_metrics(
            predictions, events, closed_predictions, truths, onset, extra
        )
        row.update(
            {
                "background": background["name"],
                "background_kind": background["kind"],
                "seed": seed,
                "method": method,
                "implementation_status": "auditable_openmax_style_scipy_weibull_adaptation",
                "official_openmax_implementation": False,
            }
        )
        rows.append(row)
        ledgers.append(
            {
                "background": background["name"],
                "seed": seed,
                "method": method,
                "onset": onset,
                "events": events,
            }
        )
        current_flags = [bool(item["evt_rejected_before_query"]) for item in events]
        if method != "cnn_closed_set":
            if reject_flags_reference is None:
                reject_flags_reference = current_flags
            elif current_flags != reject_flags_reference:
                raise AssertionError("EVT rejection decisions changed across downstream interfaces")

    manifest = reference_manifest(background, seed, windows, truths, onset, config)
    manifest.update(
        {
            "evt_fit": {
                label: {
                    key: value
                    for key, value in model.items()
                    if key != "mav"
                }
                for label, model in evt_models.items()
            },
            "evt_fit_audit": fit_audit,
            "evt_calibration": {
                "known_only": True,
                "pre_stream": True,
                "per_class_n": config.evt_cal_per_class,
                "truth_counts": {
                    label: int(sum(item == label for item in cal_truths))
                    for label in DT.BASE_VOCAB
                },
                "quantile": config.evt_cal_quantile,
                "threshold": reject_threshold,
                "strict_greater_than_threshold": True,
                "empirical_reject_rate": cal_reject_rate,
                "same_background_entity_as_stream": background["name"] != "synthetic",
            },
            "closed_prediction_sha256": FAIR.stable_hash(closed_predictions),
            "evt_score_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
        }
    )
    return rows, ledgers, manifest


SUMMARY_METRICS = [
    "closed_known_vocab_accuracy",
    "closed_known_anomaly_accuracy",
    "novel_evt_reject_recall",
    "known_vocab_evt_reject_rate",
    "known_anomaly_evt_reject_rate",
    "normal_evt_reject_rate",
    "first_novel_evt_rejected",
    "first_novel_pre_query_typed_correct",
    "novel_pre_query_typed_accuracy",
    "first_novel_query_attempted",
    "first_novel_query_name_correct",
    "eventual_correct_novel_discovery",
    "novel_typed_accuracy_including_queries",
    "post_discovery_future_reuse_accuracy",
    "post_discovery_future_reuse_rate",
    "post_discovery_future_reuse_count",
    "namer_calls_warm",
    "namer_calls_post",
    "namer_calls_total",
    "evt_rejections_warm",
    "evt_rejections_post",
    "binary_f1",
    "known_typed_accuracy",
    "normal_false_alarm_rate",
    "evt_reject_threshold",
    "evt_cal_reject_rate",
    "active_prototypes",
]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["background"], row["method"])].append(row)
    summary: list[dict[str, Any]] = []
    for (background, method), group in sorted(groups.items()):
        for metric in SUMMARY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ]
            if values:
                summary.append(
                    {
                        "background": background,
                        "method": method,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }
                )
    return summary


def load_fair_reference() -> dict[tuple[str, int], str]:
    path = REPO / "docs" / "fair_openvocab_ablation_2026-07-09" / "fair_openvocab_ablation_result.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (item["background"], int(item["seed"])): item["sha256"]
        for item in payload.get("manifests", [])
    }


def integrity_checks(
    rows: list[dict[str, Any]],
    ledgers: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    expected = len(config.backgrounds) * config.seeds * len(METHODS)
    reference = load_fair_reference()
    reference_matches = [
        manifest["fair_manifest_sha256_reproduced"]
        == reference.get((manifest["background"], int(manifest["seed"])))
        for manifest in manifests
        if (manifest["background"], int(manifest["seed"])) in reference
    ]
    phase_order_ok = True
    query_reuse_exclusive = True
    recorded_before_query = True
    state_monotone = True
    query_counts_exact = True
    no_memory_zero_reuse = True
    for ledger in ledgers:
        events = ledger["events"]
        phase_order_ok &= all(
            item["phase_order"] == ["predict", "record", "query", "update"]
            for item in events
        )
        query_reuse_exclusive &= all(
            not (item["queried"] and item["autonomous_reuse"]) for item in events
        )
        recorded_before_query &= all(item["recorded_before_query"] for item in events)
        state_monotone &= all(
            item["state_version_after"] >= item["state_version_before"]
            for item in events
        )
        if not METHODS[ledger["method"]]["memory"]:
            no_memory_zero_reuse &= all(not item["autonomous_reuse"] for item in events)
        matching = next(
            row
            for row in rows
            if row["background"] == ledger["background"]
            and row["seed"] == ledger["seed"]
            and row["method"] == ledger["method"]
        )
        query_counts_exact &= matching["namer_calls_total"] == sum(
            bool(item["queried"]) for item in events
        )
    checks = {
        "expected_aggregate_rows": len(rows) == expected,
        "expected_ledgers": len(ledgers) == expected,
        "strict_phase_order_recorded": bool(phase_order_ok),
        "query_and_reuse_mutually_exclusive": bool(query_reuse_exclusive),
        "every_prediction_recorded_before_query": bool(recorded_before_query),
        "memory_state_versions_monotone": bool(state_monotone),
        "query_counts_recomputed_from_ledgers": bool(query_counts_exact),
        "non_memory_arms_have_zero_autonomous_reuse": bool(no_memory_zero_reuse),
        "fair_reference_manifests_available": len(reference_matches),
        "fair_reference_streams_exactly_match": bool(reference_matches) and all(reference_matches),
        "aggregate_rows": len(rows),
        "event_count": int(sum(len(item["events"]) for item in ledgers)),
    }
    boolean_values = [value for value in checks.values() if isinstance(value, bool)]
    if not all(boolean_values):
        raise AssertionError(f"integrity checks failed: {checks}")
    return checks


def lookup(
    summary: list[dict[str, Any]], background: str, method: str, metric: str
) -> float | None:
    for item in summary:
        if (
            item["background"] == background
            and item["method"] == method
            and item["metric"] == metric
        ):
            return float(item["mean"])
    return None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def build_report(payload: dict[str, Any], output: Path) -> None:
    config = payload["config"]
    summary = payload["summary"]
    lines = [
        "# OpenMax-Style EVT Rejection + Shared Deterministic Namer",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Status and Scope",
        "",
        "This is a compact, auditable time-series adaptation inspired by OpenMax. It is **not** the official OpenMax implementation, not a reproduction of the CVPR visual architecture, and not an external SOTA number. The adaptation retains the core class-MAV plus Weibull-tail meta-recognition idea but uses the repository CNN's four logits and a predicted-class rejection rule.",
        "",
        "It deliberately omits OpenMax's top-alpha multi-class activation revision and K+1 softmax reconstruction. Therefore every paper table must label it `OpenMax-style EVT`, never simply `OpenMax`.",
        "",
        "Reference: Bendale and Boult, [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), CVPR 2016.",
        "",
        "## Shared Interface",
        "",
        "- The stream, known classes, held-out type, CNN training budget, structured evidence, z-scoring, and deterministic namer are imported unchanged from `run_fair_openvocab_ablation.py`.",
        f"- Known vocabulary: `{DT.BASE_VOCAB}`; held-out novel type: `{DT.NOVEL}`.",
        f"- Backgrounds: `{config['backgrounds']}`; seeds {config['seed_start']} through {config['seed_start'] + config['seeds'] - 1}.",
        f"- Each class MAV uses correctly classified training activations when at least {config['evt_min_correct_fit']} exist; otherwise the report records an all-labeled fallback.",
        f"- Weibull tail size: {config['evt_tail_size']}. Rejection threshold: q{100 * config['evt_cal_quantile']:.0f} of Weibull `-log survival` scores on {config['evt_cal_per_class']} pre-stream, known-only windows per class. This is monotone-equivalent to CDF thresholding but avoids CDF saturation at 1.",
        "- In the SMD condition, this calibration uses disjoint windows drawn from the same SMD:1-1 normal-background entity. It uses no stream window and no novel label, but it is not cross-entity calibration.",
        "- The real-background result remains controlled injection on normal SMD background, not a native typed-fault benchmark.",
        "",
        "## Strict Online Timeline",
        "",
        "Every stored event has the immutable phase order `predict -> record -> query -> update`. EVT models and the rejection threshold are frozen before the stream. A memory hit is computed from state available before the current window; its prototype update occurs after recording. A memory miss may call the shared namer and create a prototype only after the pre-query prediction is recorded. A queried current window is never counted as autonomous reuse.",
        "",
        "`Novel typed incl. query` reports the optional post-query response on that current window. `Novel pre-query typed` and `First novel pre-query typed` are the leakage-safe predictions available before external naming. `Future reuse` contains only later windows with `queried=false` after a correct discovery.",
        "",
        "## Method Arms",
        "",
        "| Method | EVT rejection | Shared namer | Memory |",
        "|---|---:|---:|---:|",
    ]
    for method, metadata in METHODS.items():
        lines.append(
            f"| `{method}` | {'no' if method == 'cnn_closed_set' else 'yes'} | "
            f"{'yes' if metadata['shared_namer'] else 'no'} | "
            f"{'yes' if metadata['memory'] else 'no'} |"
        )

    lines.extend(["", "## Results"])
    for background in config["backgrounds"]:
        lines.extend(
            [
                "",
                f"### {background}",
                "",
                "| Method | Closed known-vocab acc. | Novel EVT reject recall | Known-vocab reject rate | First novel rejected | Any novel query | First actual novel-query correct (conditional) | Eventual discovery | Novel pre-query typed | Novel typed incl. query | Future reuse acc. | Future reuse rate | Queries post |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            lines.append(
                f"| `{method}` | {pct(lookup(summary, background, method, 'closed_known_vocab_accuracy'))} | "
                f"{pct(lookup(summary, background, method, 'novel_evt_reject_recall'))} | "
                f"{pct(lookup(summary, background, method, 'known_vocab_evt_reject_rate'))} | "
                f"{pct(lookup(summary, background, method, 'first_novel_evt_rejected'))} | "
                f"{pct(lookup(summary, background, method, 'first_novel_query_attempted'))} | "
                f"{pct(lookup(summary, background, method, 'first_novel_query_name_correct'))} | "
                f"{pct(lookup(summary, background, method, 'eventual_correct_novel_discovery'))} | "
                f"{pct(lookup(summary, background, method, 'novel_pre_query_typed_accuracy'))} | "
                f"{pct(lookup(summary, background, method, 'novel_typed_accuracy_including_queries'))} | "
                f"{pct(lookup(summary, background, method, 'post_discovery_future_reuse_accuracy'))} | "
                f"{pct(lookup(summary, background, method, 'post_discovery_future_reuse_rate'))} | "
                f"{num(lookup(summary, background, method, 'namer_calls_post'))} |"
            )
        threshold = lookup(
            summary, background, "openmax_style_evt_reject_only", "evt_reject_threshold"
        )
        cal_rate = lookup(
            summary, background, "openmax_style_evt_reject_only", "evt_cal_reject_rate"
        )
        known_anomaly_rate = lookup(
            summary,
            background,
            "openmax_style_evt_reject_only",
            "known_anomaly_evt_reject_rate",
        )
        normal_rate = lookup(
            summary,
            background,
            "openmax_style_evt_reject_only",
            "normal_evt_reject_rate",
        )
        lines.extend(
            [
                "",
                f"Mean EVT threshold {num(threshold)}; known-only calibration rejection {pct(cal_rate)}; post-stream known-anomaly rejection {pct(known_anomaly_rate)}; normal rejection {pct(normal_rate)}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. The reject-only arm measures whether class-conditional activation tails identify the held-out mechanism. It receives no typed credit for the `unknown` sentinel.",
            "2. The no-memory shared-namer arm isolates semantic access after exactly the same EVT decisions. Its future autonomous reuse is zero by construction; repeated rejected windows incur repeated queries.",
            "3. The memory arm is reported separately because it changes the interface after rejection. Its keyed evidence prototypes are the same compact memory mechanism used in the fair pilot, not part of OpenMax.",
            "4. First-query naming uses the evidence-aligned deterministic rule, not an LLM. A strong value demonstrates the value of the shared structured interface, not free-form semantic induction.",
            "5. The threshold is calibrated on known-only pre-stream data. For SMD it is same-entity, so these values cannot support a cross-entity generalization claim.",
            "6. Conditional percentages can have fewer than five contributing seeds. In particular, SMD memory future-reuse accuracy is 100% in only one seed with any autonomous post-discovery reuse; its mean reuse-rate denominator includes the three seeds with a correct discovery and later novel windows.",
            "7. Five seeds and compact CNN training are adequate for a diagnostic baseline but not for an AAAI headline or an official-method comparison.",
            "",
            "## Integrity Checks",
            "",
            *[f"- `{key}`: `{value}`" for key, value in payload["integrity"].items()],
            "",
            "## Artifacts",
            "",
            f"- Result JSON: `{payload['artifacts']['result_json']}`",
            f"- Markdown report: `{payload['artifacts']['report_markdown']}`",
            f"- Runner: `{Path(__file__).resolve()}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "openmax_style_shared_namer_2026-07-09",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--backgrounds", default="synthetic,SMD:1-1")
    parser.add_argument("--normal-stats-n", type=int, default=60)
    parser.add_argument("--cnn-train-per-class", type=int, default=40)
    parser.add_argument("--cnn-epochs", type=int, default=6)
    parser.add_argument("--normal-train-n", type=int, default=64)
    parser.add_argument("--normal-cal-n", type=int, default=48)
    parser.add_argument("--warm-n", type=int, default=60)
    parser.add_argument("--post-n", type=int, default=120)
    parser.add_argument("--namer-threshold", type=float, default=2.0)
    parser.add_argument("--novelty-threshold", type=float, default=2.3)
    parser.add_argument("--prototype-radius-quantile", type=float, default=0.90)
    parser.add_argument("--prototype-radius-scale", type=float, default=1.0)
    parser.add_argument("--evt-tail-size", type=int, default=10)
    parser.add_argument("--evt-cal-per-class", type=int, default=30)
    parser.add_argument("--evt-cal-quantile", type=float, default=0.95)
    parser.add_argument("--evt-min-correct-fit", type=int, default=10)
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
        cnn_train_per_class=args.cnn_train_per_class,
        cnn_epochs=args.cnn_epochs,
        normal_train_n=args.normal_train_n,
        normal_cal_n=args.normal_cal_n,
        warm_n=args.warm_n,
        post_n=args.post_n,
        namer_threshold=args.namer_threshold,
        novelty_threshold=args.novelty_threshold,
        prototype_radius_quantile=args.prototype_radius_quantile,
        prototype_radius_scale=args.prototype_radius_scale,
        evt_tail_size=args.evt_tail_size,
        evt_cal_per_class=args.evt_cal_per_class,
        evt_cal_quantile=args.evt_cal_quantile,
        evt_min_correct_fit=args.evt_min_correct_fit,
    )
    if config.seeds < 1 or config.warm_n < 1 or config.post_n < 1:
        raise ValueError("seeds, warm_n, and post_n must be positive")
    if config.evt_tail_size < 3 or config.evt_cal_per_class < 3:
        raise ValueError("EVT tail and calibration counts must each be at least three")
    if not 0.0 < config.evt_cal_quantile < 1.0:
        raise ValueError("EVT calibration quantile must be in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    backgrounds: list[dict[str, Any]] = []
    try:
        for background_name in config.backgrounds:
            background = FAIR.activate_background(background_name)
            backgrounds.append(background)
            print(f"[background] {background_name}: {background['kind']}", flush=True)
            for seed in range(config.seed_start, config.seed_start + config.seeds):
                print(f"  [seed {seed}] fit CNN/MAV/Weibull and evaluate", flush=True)
                seed_rows, seed_ledgers, manifest = run_seed(background, seed, config)
                rows.extend(seed_rows)
                ledgers.extend(seed_ledgers)
                manifests.append(manifest)
                concise = " | ".join(
                    f"{row['method']} reject={row['novel_evt_reject_recall']:.0%} "
                    f"discover={row['eventual_correct_novel_discovery']:.0%} "
                    f"reuse={row['post_discovery_future_reuse_accuracy']} "
                    f"q={row['namer_calls_post']}"
                    for row in seed_rows
                    if row["method"] != "cnn_closed_set"
                )
                print(f"    {concise}", flush=True)
    finally:
        RB.deactivate()

    summary = summarize(rows)
    integrity = integrity_checks(rows, ledgers, manifests, config)
    result_json = args.output_dir / "openmax_style_shared_namer_result.json"
    report_md = args.output_dir / "openmax_style_shared_namer_report.md"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": asdict(config),
        "protocol": {
            "known_vocab": DT.BASE_VOCAB,
            "known_anomalies": DT.KNOWN_ANOM,
            "novel": DT.NOVEL,
            "openmax_status": "auditable_simplified_adaptation_not_official_reproduction",
            "activation_vector": "four closed-set CNN logits",
            "evt": "per-class SciPy Weibull fit to largest MAV distances",
            "rejection": "predicted-class Weibull negative log-survival greater than known-only pre-stream calibration quantile",
            "namer": "same deterministic evidence argmax rule as fair pilot",
            "timeline": ["predict", "record", "query", "update"],
            "current_query_is_reuse": False,
            "evt_frozen_during_stream": True,
            "stream_shared_across_arms": True,
            "real_background_is_controlled_injection": True,
            "native_typed_real_faults": False,
        },
        "methods": METHODS,
        "backgrounds": backgrounds,
        "provenance": provenance(),
        "integrity": integrity,
        "manifests": manifests,
        "rows": rows,
        "summary": summary,
        "event_ledgers": ledgers,
        "warnings": [
            "This is OpenMax-style EVT rejection, not the official OpenMax implementation or an official SOTA result.",
            "The adaptation omits top-alpha multi-class activation revision and K+1 OpenMax probability reconstruction.",
            "The deterministic namer is evidence-aligned and is not an LLM or free-form naming evaluation.",
            "The memory arm adds the fair pilot's keyed evidence prototype mechanism; memory is not part of OpenMax.",
            "SMD is a controlled injection condition and uses known-only, same-entity pre-stream threshold calibration.",
            "Five compact-pilot seeds are not a submission-level statistical comparison.",
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
