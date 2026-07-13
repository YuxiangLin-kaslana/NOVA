#!/usr/bin/env python3
"""Representation and gate follow-up for the many-type memory benchmark.

This runner is deliberately separate from ``run_longtail_prequential.py``.  It
keeps the corrected locked-reuse protocol, but tests the bottlenecks identified
by the previous audit:

1. The current 14D representation loses subtype information, especially scope.
2. The anomaly/known gate absorbs many novel windows before memory sees them.
3. Memory radius should be calibrated separately from the known-class radius.

The script reports oracle representation ceilings, split known-gate sweeps, and
a small locked memory rerun for flat / strict-key / key+fallback memories.
Discovery labels are still controlled oracle labels and must not be described
as LLM outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
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
from sigla_exp.prequential_memory import MemoryConfig, OnlinePrototypeMemory  # noqa: E402


FEATURES = ("current14_scopefix", "observable72")
GATE_SCORE_QUANTILES = (0.90, 0.95, 0.98)
GATE_KNOWN_QUANTILES = (0.50, 0.75, 0.90, 0.95)
MEMORY_METHODS = ("flat", "hier", "hier_fallback")


@dataclass(frozen=True)
class FollowupConfig:
    dev_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    ks: tuple[int, ...]
    discovery_values: tuple[int, ...]
    reuse_per_type: int
    normal_stats_n: int
    normal_train_n: int
    normal_cal_n: int
    train_per_known_type: int
    cal_per_known_type: int
    bootstrap_samples: int
    memory_discovery: int
    memory_radius_quantile: float
    memory_component_threshold: float


@dataclass
class FeatureContext:
    ev_mu: dict[str, float]
    ev_sd: dict[str, float]
    mechanism_q99: dict[str, float]
    scope_peak_q95: float


@dataclass
class SeedDataset:
    seed: int
    novel_n: int
    known_specs: list[dict[str, Any]]
    novel_specs: list[dict[str, Any]]
    normal_stats: list[np.ndarray]
    normal_train: list[np.ndarray]
    normal_cal: list[np.ndarray]
    known_train: list[tuple[str, np.ndarray]]
    known_cal: list[tuple[str, np.ndarray]]
    discovery: dict[str, list[np.ndarray]]
    reuse: list[tuple[str, np.ndarray]]
    component_by_label: dict[str, tuple[str, ...]]


@dataclass
class FeatureBundle:
    dataset: SeedDataset
    name: str
    context: FeatureContext
    normal_train_raw: list[np.ndarray]
    normal_cal_raw: list[np.ndarray]
    known_train_raw: list[tuple[str, np.ndarray]]
    known_cal_raw: list[tuple[str, np.ndarray]]
    discovery_raw: dict[str, list[np.ndarray]]
    reuse_raw: list[tuple[str, np.ndarray]]
    scores: dict[str, Any]
    old_raw: dict[str, Any]


@dataclass
class ScaledBundle:
    feature: FeatureBundle
    discovery_n: int
    scaler_scope: str
    scale_mu: np.ndarray
    scale_sd: np.ndarray
    normal_train: list[np.ndarray]
    normal_cal: list[np.ndarray]
    known_train: list[tuple[str, np.ndarray]]
    known_cal: list[tuple[str, np.ndarray]]
    discovery: dict[str, list[np.ndarray]]
    reuse: list[tuple[str, np.ndarray]]


@dataclass(frozen=True)
class GateConfig:
    score_quantile: float
    known_quantile: float
    split_known: bool

    @property
    def name(self) -> str:
        mode = "split" if self.split_known else "pooled"
        return f"{mode}_score{self.score_quantile:.2f}_known{self.known_quantile:.2f}"


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(sanitize(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fast_evidence(x: np.ndarray) -> dict[str, float]:
    mean = x.mean(axis=0)
    variance = x.var(axis=0) + 1e-9
    kurtosis = float(np.max(((x - mean) ** 4).mean(axis=0) / variance**2 - 3.0))
    values = {
        "kurtosis": kurtosis,
        "local_step": float(np.max(local_step_scores(x)[0])),
        "spectral_peak": OV._spectral_peak(x),
        "var_localiz": float(np.max(variance_ratio_scores(x))),
        "lin_r2": float(np.max(slope_scores(x)[1])),
        "decorr": float(1.0 - min(avg_abs_corr(x[start : start + 33]) for start in range(0, x.shape[0] - 33 + 1, 16))),
    }
    return {key: float(value) if np.isfinite(value) else 0.0 for key, value in values.items()}


def fast_features(x: np.ndarray, mu: dict[str, float], sd: dict[str, float]) -> np.ndarray:
    evidence = fast_evidence(x)
    z = np.asarray([(evidence[name] - mu[name]) / (sd[name] + 1e-9) for name in LT.STATS], dtype=np.float32)
    z = np.clip(z, -2.0, 10.0)
    step_scores, step_signs, step_locations = local_step_scores(x)
    best_step = int(np.argmax(step_scores)) if len(step_scores) else 0
    extra = np.asarray(
        [
            LT._spike_location(x),
            float(step_locations[best_step]) if len(step_locations) else 0.5,
            LT._variance_location(x),
            LT._corr_location(x),
            LT._spectral_freq(x),
            LT._scope_estimate(x),
            LT._slope_sign(x),
            float(step_signs[best_step]) if len(step_signs) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([z, extra]).astype(np.float32)


def component_key(raw_feature: np.ndarray, threshold: float) -> tuple[str, ...]:
    components = LT.component_signature(raw_feature, top=2, threshold=threshold)
    return tuple(sorted(components)) if components else ("unknown",)


def observed_component_key(x: np.ndarray, context: FeatureContext, threshold: float) -> tuple[str, ...]:
    z = evidence_z(x, context)
    idx = np.argsort(-z)
    stat_to_family = {OV.STAT_OF[concept]: concept for concept in LT.FAMILIES}
    components: list[str] = []
    for item in idx[:2]:
        if float(z[int(item)]) < threshold:
            continue
        components.append(stat_to_family[LT.STATS[int(item)]])
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


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance() -> dict[str, Any]:
    dirty = git_value("status", "--short", "--untracked-files=all")
    return {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_sha256": {
            str(Path(__file__).relative_to(REPO)): file_hash(Path(__file__)),
            "code/sota_compare/run_longtail_prequential.py": file_hash(
                ROOT / "sota_compare" / "run_longtail_prequential.py"
            ),
            "code/sigla_exp/prequential_memory.py": file_hash(
                ROOT / "sigla_exp" / "prequential_memory.py"
            ),
            "code/sigla_exp/longtail_bench.py": file_hash(
                ROOT / "sigla_exp" / "longtail_bench.py"
            ),
        },
    }


def qstats(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        [
            np.quantile(array, 0.50),
            np.quantile(array, 0.75),
            np.quantile(array, 0.90),
            np.max(array),
        ],
        dtype=np.float32,
    )


def local_step_scores(x: np.ndarray, window: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[0] < 2 * window + 1:
        return np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])
    views = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
    medians = np.median(views, axis=-1)
    differences = medians[window:] - medians[:-window]
    abs_diff = np.abs(differences)
    best_index = np.argmax(abs_diff, axis=0)
    dims = np.arange(x.shape[1])
    best = differences[best_index, dims]
    location = (window + best_index) / x.shape[0]
    return np.abs(best), np.sign(best), location


def variance_ratio_scores(x: np.ndarray, kernel: int = 7, segments: int = 5) -> np.ndarray:
    if x.shape[0] < kernel + segments:
        return np.ones(x.shape[1], dtype=np.float32)
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
    return np.max(scale_array, axis=0) / (np.median(scale_array, axis=0) + 1e-6)


def slope_scores(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_index = np.arange(x.shape[0], dtype=np.float64)
    centered_time = time_index - time_index.mean()
    centered_x = x - x.mean(axis=0, keepdims=True)
    denom = float(np.square(centered_time).sum()) + 1e-12
    slopes = (centered_time[:, None] * centered_x).sum(axis=0) / denom
    fitted = x.mean(axis=0, keepdims=True) + centered_time[:, None] * slopes
    residual = np.square(x - fitted).sum(axis=0)
    total = np.square(centered_x).sum(axis=0) + 1e-6
    r2 = np.clip(1.0 - residual / total, 0.0, 1.0)
    return slopes, r2


def spectral_scores(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    det = x - x.mean(axis=0, keepdims=True)
    mag = np.abs(np.fft.rfft(det, axis=0))[1:]
    if mag.size == 0:
        zeros = np.zeros(x.shape[1], dtype=np.float32)
        return zeros, zeros, zeros
    high = mag[14:] if mag.shape[0] > 14 else mag
    peak = high.max(axis=0) / (mag.sum(axis=0) + 1e-6)
    full_peak_index = np.argmax(mag, axis=0) + 1
    frequency = full_peak_index / max(1.0, x.shape[0] / 2.0)
    probability = mag / (mag.sum(axis=0, keepdims=True) + 1e-6)
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=0)
    entropy = entropy / max(1.0, math.log(mag.shape[0]))
    return peak, frequency, entropy


def avg_abs_corr(segment: np.ndarray) -> float:
    mask = segment.std(axis=0) > 1e-6
    if int(mask.sum()) < 2:
        return 1.0
    centered = segment[:, mask] - segment[:, mask].mean(axis=0, keepdims=True)
    gram = centered.T @ centered
    norms = np.sqrt(np.diag(gram))
    correlation = gram / (np.outer(norms, norms) + 1e-12)
    count = correlation.shape[0]
    return float((np.abs(correlation).sum() - count) / (count * (count - 1)))


def corr_features(x: np.ndarray) -> np.ndarray:
    mask = x.std(axis=0) > 1e-6
    if int(mask.sum()) < 2:
        return np.zeros(6, dtype=np.float32)
    centered = x[:, mask] - x[:, mask].mean(axis=0, keepdims=True)
    corr = np.corrcoef(centered, rowvar=False)
    upper = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
    eigvals = np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0))
    eigvals = np.clip(eigvals, 1e-9, None)
    top_share = float(eigvals[-1] / eigvals.sum())
    probability = eigvals / eigvals.sum()
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-12))) / len(eigvals))
    segment_corrs = []
    matrices = []
    for segment in np.array_split(x, 5, axis=0):
        segment_corrs.append(avg_abs_corr(segment))
        if segment.shape[0] >= 4:
            sub = segment[:, mask] - segment[:, mask].mean(axis=0, keepdims=True)
            matrices.append(np.nan_to_num(np.corrcoef(sub, rowvar=False), nan=0.0))
    changes = [
        float(np.linalg.norm(matrices[index + 1] - matrices[index], ord="fro") / matrices[index].shape[0])
        for index in range(len(matrices) - 1)
    ]
    return np.asarray(
        [
            np.quantile(upper, 0.10),
            np.quantile(upper, 0.50),
            np.quantile(upper, 0.90),
            top_share,
            effective_rank,
            max(changes) if changes else 0.0,
        ],
        dtype=np.float32,
    )


def mechanism_scores(x: np.ndarray) -> dict[str, np.ndarray]:
    median = np.median(x, axis=0, keepdims=True)
    spike = np.max(np.abs(x - median), axis=0)
    step, step_sign, step_location = local_step_scores(x)
    variance = variance_ratio_scores(x)
    slope, r2 = slope_scores(x)
    spectral_peak, spectral_frequency, spectral_entropy = spectral_scores(x)
    return {
        "spike": spike.astype(np.float32),
        "step": step.astype(np.float32),
        "step_sign": step_sign.astype(np.float32),
        "step_location": step_location.astype(np.float32),
        "variance": variance.astype(np.float32),
        "slope_abs": np.abs(slope).astype(np.float32),
        "slope_sign": np.sign(slope).astype(np.float32),
        "trend_r2": r2.astype(np.float32),
        "spectral_peak": spectral_peak.astype(np.float32),
        "spectral_frequency": spectral_frequency.astype(np.float32),
        "spectral_entropy": spectral_entropy.astype(np.float32),
    }


def fit_feature_context(normal_windows: list[np.ndarray]) -> FeatureContext:
    evidence = [fast_evidence(window) for window in normal_windows]
    ev_mu = {name: float(np.mean([row[name] for row in evidence])) for name in LT.STATS}
    ev_sd = {name: float(np.std([row[name] for row in evidence]) + 1e-6) for name in LT.STATS}
    buckets: dict[str, list[float]] = defaultdict(list)
    peaks = []
    for window in normal_windows:
        scores = mechanism_scores(window)
        for name in ("spike", "step", "variance", "slope_abs", "spectral_peak"):
            buckets[name].extend(float(value) for value in scores[name])
        peaks.extend(float(value) for value in scores["spike"])
    return FeatureContext(
        ev_mu=ev_mu,
        ev_sd=ev_sd,
        mechanism_q99={
            name: float(np.quantile(values, 0.99)) if values else 0.0
            for name, values in buckets.items()
        },
        scope_peak_q95=float(np.quantile(peaks, 0.95)) if peaks else 0.0,
    )


def evidence_z(x: np.ndarray, context: FeatureContext) -> np.ndarray:
    evidence = fast_evidence(x)
    return np.asarray(
        [
            (evidence[name] - context.ev_mu[name]) / (context.ev_sd[name] + 1e-9)
            for name in LT.STATS
        ],
        dtype=np.float32,
    )


def current14_scopefix(x: np.ndarray, context: FeatureContext) -> np.ndarray:
    z = np.arcsinh(evidence_z(x, context) / 3.0)
    scores = mechanism_scores(x)
    step = scores["step"]
    best_step = int(np.argmax(step)) if len(step) else 0
    extra = np.asarray(
        [
            LT._spike_location(x),
            float(scores["step_location"][best_step]) if len(step) else 0.5,
            LT._variance_location(x),
            LT._corr_location(x),
            LT._spectral_freq(x),
            float(np.mean(scores["spike"] > context.scope_peak_q95)),
            float(scores["slope_sign"][int(np.argmax(scores["slope_abs"]))]) if len(scores["slope_abs"]) else 0.0,
            float(scores["step_sign"][best_step]) if len(step) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([z, extra]).astype(np.float32)


def window_profile(x: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for segment in np.array_split(x, 5, axis=0):
        scores = mechanism_scores(segment)
        values.extend(
            [
                float(np.max(scores["spike"])),
                float(np.max(scores["step"])),
                float(np.max(scores["spectral_peak"])),
                float(np.max(scores["variance"])),
                float(np.max(scores["trend_r2"])),
                float(1.0 - avg_abs_corr(segment)),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def observable72(x: np.ndarray, context: FeatureContext) -> np.ndarray:
    z = np.arcsinh(evidence_z(x, context) / 3.0)
    scores = mechanism_scores(x)
    values: list[float] = []
    for name in ("spike", "step", "spectral_peak", "variance", "slope_abs"):
        raw = scores[name]
        values.extend(float(item) for item in qstats(raw))
        threshold = context.mechanism_q99.get(name, float("inf"))
        values.append(float(np.mean(raw > threshold)))

    step_weights = scores["step"] + 1e-9
    slope_weights = scores["slope_abs"] + 1e-9
    spectral_weights = scores["spectral_peak"] + 1e-9
    sign_and_spectrum = np.asarray(
        [
            float(np.sum(scores["step_sign"] * step_weights) / np.sum(step_weights)),
            float(np.sum(scores["slope_sign"] * slope_weights) / np.sum(slope_weights)),
            float(np.quantile(scores["spectral_frequency"], 0.50)),
            float(np.quantile(scores["spectral_frequency"], 0.90)),
            float(np.sum(scores["spectral_entropy"] * spectral_weights) / np.sum(spectral_weights)),
        ],
        dtype=np.float32,
    )
    out = np.concatenate(
        [
            z.astype(np.float32),
            np.asarray(values, dtype=np.float32),
            window_profile(x),
            sign_and_spectrum,
            corr_features(x),
        ]
    )
    if out.shape[0] != 72:
        raise AssertionError(f"observable72 produced {out.shape[0]} dimensions")
    return out.astype(np.float32)


def feature_vector(name: str, x: np.ndarray, context: FeatureContext) -> np.ndarray:
    if name == "current14":
        return fast_features(x, context.ev_mu, context.ev_sd)
    if name == "current14_scopefix":
        return current14_scopefix(x, context)
    if name == "observable72":
        return observable72(x, context)
    raise KeyError(name)


def anomaly_score(x: np.ndarray, context: FeatureContext, feature_name: str) -> float:
    if feature_name == "current14":
        return float(np.max(fast_features(x, context.ev_mu, context.ev_sd)[: len(LT.STATS)]))
    return float(np.max(evidence_z(x, context)))


def fit_scaler(vectors: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    array = np.stack(list(vectors)).astype(np.float64)
    return array.mean(axis=0), array.std(axis=0) + 1e-6


def apply_scaler(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (vector - mean) / std


def centroids(rows: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for label, vector in rows:
        grouped[label].append(vector)
    return {label: np.mean(vectors, axis=0) for label, vectors in grouped.items()}


def nearest_label(vector: np.ndarray, centers: dict[str, np.ndarray], allowed: set[str] | None = None) -> tuple[str, float]:
    names = list(allowed) if allowed is not None else list(centers)
    label = min(names, key=lambda name: float(np.linalg.norm(vector - centers[name])))
    return label, float(np.linalg.norm(vector - centers[label]))


def build_seed_dataset(seed: int, novel_n: int, config: FollowupConfig) -> SeedDataset:
    catalog = LT.generate_taxonomy(216)
    known_specs = [spec for spec in catalog if len(spec["components"]) == 1][:6]
    composite_specs = [spec for spec in catalog if len(spec["components"]) == 2]
    novel_specs = composite_order(seed, composite_specs)[:novel_n]
    rng = np.random.default_rng(710_000 + seed * 10_000 + novel_n)

    normal_stats = [LT.make_window(None, rng) for _ in range(config.normal_stats_n)]
    normal_train = [LT.make_window(None, rng) for _ in range(config.normal_train_n)]
    normal_cal = [LT.make_window(None, rng) for _ in range(config.normal_cal_n)]

    known_train: list[tuple[str, np.ndarray]] = []
    known_cal: list[tuple[str, np.ndarray]] = []
    for spec in known_specs:
        label = str(spec["name"])
        for _ in range(config.train_per_known_type):
            known_train.append((label, LT.make_window(spec, rng)))
        for _ in range(config.cal_per_known_type):
            known_cal.append((label, LT.make_window(spec, rng)))

    max_discovery = max(config.discovery_values)
    discovery: dict[str, list[np.ndarray]] = defaultdict(list)
    reuse: list[tuple[str, np.ndarray]] = []
    for spec in novel_specs:
        label = str(spec["name"])
        for _ in range(max_discovery):
            discovery[label].append(LT.make_window(spec, rng))
        for _ in range(config.reuse_per_type):
            reuse.append((label, LT.make_window(spec, rng)))
    rng.shuffle(reuse)
    return SeedDataset(
        seed=seed,
        novel_n=novel_n,
        known_specs=known_specs,
        novel_specs=novel_specs,
        normal_stats=normal_stats,
        normal_train=normal_train,
        normal_cal=normal_cal,
        known_train=known_train,
        known_cal=known_cal,
        discovery=dict(discovery),
        reuse=reuse,
        component_by_label={
            str(spec["name"]): tuple(sorted(str(component) for component in spec["components"]))
            for spec in novel_specs
        },
    )


def build_old_raw(dataset: SeedDataset, context: FeatureContext) -> dict[str, Any]:
    return {
        "discovery": {
            label: [fast_features(window, context.ev_mu, context.ev_sd) for window in windows]
            for label, windows in dataset.discovery.items()
        },
        "reuse": [
            (label, fast_features(window, context.ev_mu, context.ev_sd))
            for label, window in dataset.reuse
        ],
    }


def build_feature_bundle(dataset: SeedDataset, name: str, context: FeatureContext) -> FeatureBundle:

    def vectorize(windows: Iterable[np.ndarray]) -> list[np.ndarray]:
        return [feature_vector(name, window, context) for window in windows]

    return FeatureBundle(
        dataset=dataset,
        name=name,
        context=context,
        normal_train_raw=vectorize(dataset.normal_train),
        normal_cal_raw=vectorize(dataset.normal_cal),
        known_train_raw=[(label, feature_vector(name, window, context)) for label, window in dataset.known_train],
        known_cal_raw=[(label, feature_vector(name, window, context)) for label, window in dataset.known_cal],
        discovery_raw={
            label: [feature_vector(name, window, context) for window in windows]
            for label, windows in dataset.discovery.items()
        },
        reuse_raw=[(label, feature_vector(name, window, context)) for label, window in dataset.reuse],
        scores={
            "normal_cal": [anomaly_score(window, context, name) for window in dataset.normal_cal],
            "known_cal": [(label, anomaly_score(window, context, name)) for label, window in dataset.known_cal],
            "reuse": [(label, anomaly_score(window, context, name)) for label, window in dataset.reuse],
            "discovery": {
                label: [anomaly_score(window, context, name) for window in windows]
                for label, windows in dataset.discovery.items()
            },
        },
        old_raw={},
    )


def scale_bundle(bundle: FeatureBundle, discovery_n: int, scaler_scope: str = "base_discovery") -> ScaledBundle:
    base_vectors = [*bundle.normal_train_raw, *[vector for _, vector in bundle.known_train_raw]]
    if scaler_scope == "base_discovery":
        scale_source = [
            *base_vectors,
            *[
                vector
                for vectors in bundle.discovery_raw.values()
                for vector in vectors[:discovery_n]
            ],
        ]
    elif scaler_scope == "base":
        scale_source = base_vectors
    else:
        raise KeyError(scaler_scope)
    mean, std = fit_scaler(scale_source)
    return ScaledBundle(
        feature=bundle,
        discovery_n=discovery_n,
        scaler_scope=scaler_scope,
        scale_mu=mean,
        scale_sd=std,
        normal_train=[apply_scaler(vector, mean, std) for vector in bundle.normal_train_raw],
        normal_cal=[apply_scaler(vector, mean, std) for vector in bundle.normal_cal_raw],
        known_train=[(label, apply_scaler(vector, mean, std)) for label, vector in bundle.known_train_raw],
        known_cal=[(label, apply_scaler(vector, mean, std)) for label, vector in bundle.known_cal_raw],
        discovery={
            label: [apply_scaler(vector, mean, std) for vector in vectors[:discovery_n]]
            for label, vectors in bundle.discovery_raw.items()
        },
        reuse=[(label, apply_scaler(vector, mean, std)) for label, vector in bundle.reuse_raw],
    )


def topk_correct(vector: np.ndarray, centers: dict[str, np.ndarray], label: str, k: int = 5) -> bool:
    ranked = sorted(centers, key=lambda name: float(np.linalg.norm(vector - centers[name])))
    return label in ranked[:k]


def ceiling_rows(scaled: ScaledBundle, split: str) -> list[dict[str, Any]]:
    labels = sorted(scaled.discovery)
    prototypes = {
        label: np.mean(scaled.discovery[label], axis=0) for label in labels
    }
    pair_labels: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for label in labels:
        pair_labels[scaled.feature.dataset.component_by_label[label]].add(label)

    global_correct = []
    component_correct = []
    top5 = []
    margin_success = []
    margins = []
    for label, vector in scaled.reuse:
        pred, own_or_pred_dist = nearest_label(vector, prototypes)
        del own_or_pred_dist
        allowed = pair_labels[scaled.feature.dataset.component_by_label[label]]
        component_pred, _ = nearest_label(vector, prototypes, allowed=allowed)
        own_distance = float(np.linalg.norm(vector - prototypes[label]))
        other_distance = min(
            float(np.linalg.norm(vector - prototypes[other]))
            for other in labels
            if other != label
        )
        global_correct.append(pred == label)
        component_correct.append(component_pred == label)
        top5.append(topk_correct(vector, prototypes, label, 5))
        margin_success.append(own_distance < other_distance)
        margins.append(other_distance - own_distance)

    return [
        {
            "split": split,
            "seed": scaled.feature.dataset.seed,
            "novel_n": scaled.feature.dataset.novel_n,
            "feature": scaled.feature.name,
            "discovery_per_type": scaled.discovery_n,
            "scaler_scope": scaled.scaler_scope,
            "global_centroid_accuracy": float(np.mean(global_correct)),
            "oracle_component_accuracy": float(np.mean(component_correct)),
            "top5_accuracy": float(np.mean(top5)),
            "retrieval_margin_success": float(np.mean(margin_success)),
            "retrieval_margin_mean": float(np.mean(margins)),
        }
    ]


def gate_thresholds(scaled: ScaledBundle, gate: GateConfig) -> dict[str, Any]:
    bundle = scaled.feature
    normal_scores = np.asarray(bundle.scores["normal_cal"], dtype=float)
    score_threshold = float(np.quantile(normal_scores, gate.score_quantile))
    known_centers = centroids(scaled.known_train)
    known_labels = {str(spec["name"]) for spec in bundle.dataset.known_specs}
    if gate.split_known:
        distances: dict[str, list[float]] = defaultdict(list)
        for label, vector in scaled.known_cal:
            distances[label].append(float(np.linalg.norm(vector - known_centers[label])))
        radii = {
            label: float(np.quantile(values, gate.known_quantile))
            for label, values in distances.items()
        }
        pooled_radius = None
    else:
        train_distances = [
            float(np.linalg.norm(vector - known_centers[label]))
            for label, vector in scaled.known_train
        ]
        pooled_radius = float(np.quantile(train_distances, gate.known_quantile))
        radii = {label: pooled_radius for label in known_labels}
    return {
        "known_centers": known_centers,
        "known_labels": known_labels,
        "score_threshold": score_threshold,
        "known_radii": radii,
        "pooled_radius": pooled_radius,
    }


def route_with_gate(vector: np.ndarray, score: float, thresholds: dict[str, Any]) -> dict[str, Any]:
    if score <= thresholds["score_threshold"]:
        return {"route": "normal", "candidate": False, "anomaly": False, "pred_label": "normal", "known_distance": None}
    known_label, known_distance = nearest_label(vector, thresholds["known_centers"])
    if known_label in thresholds["known_labels"] and known_distance <= thresholds["known_radii"][known_label]:
        return {
            "route": "known",
            "candidate": False,
            "anomaly": True,
            "pred_label": known_label,
            "known_distance": known_distance,
        }
    return {
        "route": "candidate",
        "candidate": True,
        "anomaly": True,
        "pred_label": "unknown",
        "known_distance": known_distance,
    }


def gate_rows(scaled: ScaledBundle, gate: GateConfig, split: str) -> list[dict[str, Any]]:
    thresholds = gate_thresholds(scaled, gate)
    bundle = scaled.feature
    novel_candidate = []
    novel_anomaly = []
    known_to_candidate = []
    normal_to_candidate = []
    known_absorbed = []
    # Reuse windows are all novel.
    for (label, vector), (_, score) in zip(scaled.reuse, bundle.scores["reuse"]):
        route = route_with_gate(vector, float(score), thresholds)
        novel_candidate.append(route["candidate"])
        novel_anomaly.append(route["anomaly"])
        known_absorbed.append(route["anomaly"] and not route["candidate"])
    for (label, vector), (_, score) in zip(scaled.known_cal, bundle.scores["known_cal"]):
        del label
        route = route_with_gate(vector, float(score), thresholds)
        known_to_candidate.append(route["candidate"])
    for vector, score in zip(scaled.normal_cal, bundle.scores["normal_cal"]):
        route = route_with_gate(vector, float(score), thresholds)
        normal_to_candidate.append(route["candidate"])

    return [
        {
            "split": split,
            "seed": bundle.dataset.seed,
            "novel_n": bundle.dataset.novel_n,
            "feature": bundle.name,
            "discovery_per_type": scaled.discovery_n,
            "scaler_scope": scaled.scaler_scope,
            "gate": gate.name,
            "score_quantile": gate.score_quantile,
            "known_quantile": gate.known_quantile,
            "split_known": gate.split_known,
            "candidate_recall": float(np.mean(novel_candidate)),
            "detector_recall": float(np.mean(novel_anomaly)),
            "known_absorption_rate": float(np.mean(known_absorbed)),
            "known_to_candidate_rate": float(np.mean(known_to_candidate)),
            "normal_to_candidate_rate": float(np.mean(normal_to_candidate)),
        }
    ]


def select_gate(gate_summary: list[dict[str, Any]], split: str, novel_n: int, discovery_n: int, feature: str) -> GateConfig:
    candidates = [
        row
        for row in gate_summary
        if row["split"] == split
        and row["novel_n"] == novel_n
        and row["discovery_per_type"] == discovery_n
        and row["feature"] == feature
        and row["metric"] == "candidate_recall"
    ]
    by_key = {(row["gate"], row["score_quantile"], row["known_quantile"], row["split_known"]): row for row in candidates}
    scored = []
    for key, recall_row in by_key.items():
        gate_name, score_q, known_q, split_known = key
        known_far = next(
            row["mean"]
            for row in gate_summary
            if row["split"] == split
            and row["novel_n"] == novel_n
            and row["discovery_per_type"] == discovery_n
            and row["feature"] == feature
            and row["gate"] == gate_name
            and row["metric"] == "known_to_candidate_rate"
        )
        normal_far = next(
            row["mean"]
            for row in gate_summary
            if row["split"] == split
            and row["novel_n"] == novel_n
            and row["discovery_per_type"] == discovery_n
            and row["feature"] == feature
            and row["gate"] == gate_name
            and row["metric"] == "normal_to_candidate_rate"
        )
        feasible = known_far <= 0.20 and normal_far <= 0.10
        objective = recall_row["mean"] - 0.50 * known_far - 0.25 * normal_far
        scored.append((feasible, objective, recall_row["mean"], -known_far, -normal_far, score_q, known_q, split_known))
    if not scored:
        return GateConfig(score_quantile=0.95, known_quantile=0.75, split_known=True)
    scored.sort(reverse=True)
    _, _, _, _, _, score_q, known_q, split_known = scored[0]
    return GateConfig(score_quantile=float(score_q), known_quantile=float(known_q), split_known=bool(split_known))


def make_memory(method: str, radius: float) -> OnlinePrototypeMemory:
    return OnlinePrototypeMemory(
        MemoryConfig(
            name=method,
            hierarchical=method != "flat",
            radius=radius,
            fallback_global=method == "hier_fallback",
        )
    )


def memory_radius(scaled: ScaledBundle, quantile: float) -> float:
    distances = []
    for vectors in scaled.discovery.values():
        for i, first in enumerate(vectors):
            for second in vectors[i + 1 :]:
                distances.append(float(np.linalg.norm(first - second)))
    if not distances:
        return 1.0
    return max(1e-6, float(np.quantile(distances, quantile)))


def state_digest(memory: OnlinePrototypeMemory) -> str:
    return stable_hash(memory.state())


def run_memory(scaled: ScaledBundle, gate: GateConfig, method: str, split: str, radius: float) -> dict[str, Any]:
    thresholds = gate_thresholds(scaled, gate)
    bundle = scaled.feature
    memory = make_memory(method, radius)
    novel_names = set(scaled.discovery)
    events: list[dict[str, Any]] = []
    step = 0
    for label in sorted(scaled.discovery):
        for index, vector in enumerate(scaled.discovery[label]):
            score = float(bundle.scores["discovery"][label][index])
            route = route_with_gate(vector, score, thresholds)
            if route["candidate"]:
                key = observed_component_key(
                    bundle.dataset.discovery[label][index],
                    bundle.context,
                    threshold=1.6,
                )
                decision = memory.process(vector, key, label, True, step)
                events.append(
                    {
                        "phase": "discovery",
                        "true_label": label,
                        "candidate": True,
                        "pred_label": decision.pred_label,
                        "queried": decision.queried,
                        "created": decision.created,
                        "action": decision.action,
                    }
                )
            else:
                events.append(
                    {
                        "phase": "discovery",
                        "true_label": label,
                        "candidate": False,
                        "pred_label": route["pred_label"],
                        "queried": False,
                        "created": False,
                        "action": route["route"],
                    }
                )
            step += 1

    locked_hash_before = state_digest(memory)
    locked_rows = []
    for index, ((label, vector), (_, score)) in enumerate(zip(scaled.reuse, bundle.scores["reuse"])):
        route = route_with_gate(vector, float(score), thresholds)
        pred_label = route["pred_label"]
        cluster_id = None
        autonomous = False
        action = route["route"]
        if route["candidate"]:
            key = observed_component_key(
                bundle.dataset.reuse[index][1],
                bundle.context,
                threshold=1.6,
            )
            decision = memory.predict_locked(vector, key)
            pred_label = decision.pred_label
            cluster_id = decision.cluster_id
            autonomous = decision.autonomous_reuse
            action = decision.action
        row = {
            "phase": "locked_reuse",
            "true_label": label,
            "candidate": route["candidate"],
            "pred_label": pred_label,
            "cluster_id": cluster_id,
            "autonomous_reuse": autonomous,
            "action": action,
            "correct": pred_label == label,
        }
        locked_rows.append(row)
        events.append(row)
    locked_hash_after = state_digest(memory)
    if locked_hash_before != locked_hash_after:
        raise AssertionError("memory changed during locked reuse")

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in locked_rows:
        by_type[row["true_label"]].append(row)
    per_type = {
        label: float(np.mean([item["correct"] for item in rows]))
        for label, rows in by_type.items()
    }
    correct_types = {label for label, rows in by_type.items() if any(item["correct"] for item in rows)}
    autonomous_rows = [row for row in locked_rows if row["autonomous_reuse"]]
    return {
        "split": split,
        "seed": bundle.dataset.seed,
        "novel_n": bundle.dataset.novel_n,
        "feature": bundle.name,
        "discovery_per_type": scaled.discovery_n,
        "scaler_scope": scaled.scaler_scope,
        "gate": gate.name,
        "method": method,
        "memory_radius": radius,
        "memory_radius_quantile": None,
        "locked_macro_accuracy": float(np.mean(list(per_type.values()))),
        "locked_micro_accuracy": float(np.mean([row["correct"] for row in locked_rows])),
        "locked_type_coverage": len(correct_types) / len(novel_names),
        "locked_unknown_rate": float(np.mean([row["pred_label"] == "unknown" for row in locked_rows])),
        "locked_candidate_recall": float(np.mean([row["candidate"] for row in locked_rows])),
        "locked_autonomous_reuse_rate": len(autonomous_rows) / len(locked_rows) if locked_rows else 0.0,
        "locked_conditional_reuse_accuracy": float(np.mean([row["correct"] for row in autonomous_rows])) if autonomous_rows else 0.0,
        "annotation_queries": int(sum(row["queried"] for row in events if row["phase"] == "discovery")),
        "query_created_clusters": int(sum(row["created"] for row in events if row["phase"] == "discovery")),
        "active_vocab": memory.active_count,
        "committed_vocab": memory.committed_count,
        "historical_clusters": memory.historical_clusters,
        "locked_state_unchanged": True,
        "locked_state_sha256": locked_hash_after,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...], config: FollowupConfig) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    excluded = set(group_keys) | {"seed"}
    for key, group in sorted(groups.items()):
        metrics = sorted(
            name
            for name, value in group[0].items()
            if name not in excluded and isinstance(value, (int, float, np.integer, np.floating))
        )
        for metric in metrics:
            values = [float(row[metric]) for row in group if np.isfinite(float(row[metric]))]
            if not values:
                continue
            mean, std = mean_std(values)
            low, high = bootstrap_ci(values, config.bootstrap_samples, seed=int(stable_hash([key, metric])[:8], 16))
            summary_row = {name: item for name, item in zip(group_keys, key)}
            summary_row.update(
                {
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": len(values),
                }
            )
            output.append(summary_row)
    return output


def best_feature(ceiling_summary: list[dict[str, Any]], split: str, novel_n: int, discovery_n: int) -> str:
    rows = [
        row
        for row in ceiling_summary
        if row["split"] == split
        and row["novel_n"] == novel_n
        and row["discovery_per_type"] == discovery_n
        and row["scaler_scope"] == "base_discovery"
        and row["metric"] == "global_centroid_accuracy"
    ]
    if not rows:
        return "current14_scopefix"
    return max(rows, key=lambda row: row["mean"])["feature"]


def summary_lookup(summary: list[dict[str, Any]], **query: Any) -> dict[str, Any] | None:
    for row in summary:
        if all(row.get(key) == value for key, value in query.items()):
            return row
    return None


def plot_ceiling(summary: list[dict[str, Any]], config: FollowupConfig, output: Path) -> None:
    max_k = max(config.ks)
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    colors = {"current14": "#4c566a", "current14_scopefix": "#2f6b9a", "observable72": "#2a8c67"}
    labels = {
        "current14": "Current 14D",
        "current14_scopefix": "14D no-clip + scope fix",
        "observable72": "Observable 72D",
    }
    for feature in FEATURES:
        ys, errs = [], []
        for discovery in config.discovery_values:
            row = summary_lookup(
                summary,
                split="holdout",
                novel_n=max_k,
                feature=feature,
                discovery_per_type=discovery,
                scaler_scope="base_discovery",
                metric="global_centroid_accuracy",
            )
            ys.append(float("nan") if row is None else row["mean"])
            errs.append(0.0 if row is None else row["std"])
        axis.errorbar(
            config.discovery_values,
            ys,
            yerr=errs,
            marker="o",
            capsize=3,
            color=colors[feature],
            label=labels[feature],
        )
    axis.set_title(f"Representation ceiling on holdout K={max_k}")
    axis.set_xlabel("Discovery labels per novel type")
    axis.set_ylabel("Locked exact centroid accuracy")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def build_report(
    output: Path,
    result_path: Path,
    figure_path: Path,
    config: FollowupConfig,
    ceiling_summary: list[dict[str, Any]],
    gate_summary: list[dict[str, Any]],
    memory_summary: list[dict[str, Any]],
    selection: dict[str, Any],
) -> None:
    max_k = max(config.ks)
    lines = [
        "# Representation/Gate Follow-up for Many-Type Memory",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Protocol",
        "",
        "- Dev seeds are used only to select the follow-up feature/gate configuration.",
        "- Holdout seeds provide the reported conclusion.",
        "- Features use only the observed window; no generator `loc/scope/severity/name` fields are used at inference.",
        "- Discovery labels are controlled oracle labels. Locked reuse has no query and no memory update.",
        "",
        "## Holdout Representation Ceiling",
        "",
        "| K | D/type | Feature | Global centroid | Oracle component | Top-5 | Margin success |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for discovery in config.discovery_values:
        for feature in FEATURES:
            values = {
                metric: summary_lookup(
                    ceiling_summary,
                    split="holdout",
                    novel_n=max_k,
                    feature=feature,
                    discovery_per_type=discovery,
                    scaler_scope="base_discovery",
                    metric=metric,
                )
                for metric in (
                    "global_centroid_accuracy",
                    "oracle_component_accuracy",
                    "top5_accuracy",
                    "retrieval_margin_success",
                )
            }
            if values["global_centroid_accuracy"] is None:
                continue
            lines.append(
                f"| {max_k} | {discovery} | {feature} | "
                f"{pct(values['global_centroid_accuracy']['mean'])} | "
                f"{pct(values['oracle_component_accuracy']['mean'])} | "
                f"{pct(values['top5_accuracy']['mean'])} | "
                f"{pct(values['retrieval_margin_success']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Selected Gate on Holdout",
            "",
            f"Selected from dev: `{selection['feature']}` with gate `{selection['gate']}`.",
            "",
            "| Metric | Mean |",
            "|---|---:|",
        ]
    )
    for metric in (
        "candidate_recall",
        "detector_recall",
        "known_absorption_rate",
        "known_to_candidate_rate",
        "normal_to_candidate_rate",
    ):
        row = summary_lookup(
            gate_summary,
            split="holdout",
            novel_n=max_k,
            feature=selection["feature"],
            discovery_per_type=config.memory_discovery,
            scaler_scope="base_discovery",
            gate=selection["gate"],
            metric=metric,
        )
        if row is not None:
            lines.append(f"| {metric} | {pct(row['mean'])} |")
    lines.extend(
        [
            "",
            "## Holdout Locked Memory Rerun",
            "",
            "| Method | Locked macro acc. | Type coverage | Unknown rate | Candidate recall | Queries | Active vocab |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in MEMORY_METHODS:
        values = {
            metric: summary_lookup(
                memory_summary,
                split="holdout",
                novel_n=max_k,
                feature=selection["feature"],
                discovery_per_type=config.memory_discovery,
                scaler_scope="base_discovery",
                gate=selection["gate"],
                method=method,
                metric=metric,
            )
            for metric in (
                "locked_macro_accuracy",
                "locked_type_coverage",
                "locked_unknown_rate",
                "locked_candidate_recall",
                "annotation_queries",
                "active_vocab",
            )
        }
        if values["locked_macro_accuracy"] is None:
            continue
        lines.append(
            f"| {method} | {pct(values['locked_macro_accuracy']['mean'])} | "
            f"{pct(values['locked_type_coverage']['mean'])} | "
            f"{pct(values['locked_unknown_rate']['mean'])} | "
            f"{pct(values['locked_candidate_recall']['mean'])} | "
            f"{values['annotation_queries']['mean']:.1f} | "
            f"{values['active_vocab']['mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `current14_scopefix` directly tests the audit finding that the old scope coordinate was effectively constant.",
            "- `observable72` is still synthetic-benchmark engineering, not a final deployable representation claim.",
            "- If holdout ceiling improves but locked memory stays low, the remaining bottleneck is online assignment/radius/query policy.",
            "- If ceiling remains low, the AAAI claim should be narrowed before spending more effort on memory variants.",
            "",
            f"Result JSON: `{result_path}`",
            f"Figure: `{figure_path}`",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: FollowupConfig, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    ceiling: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    all_seeds = {"dev": config.dev_seeds, "holdout": config.holdout_seeds}

    bundles: dict[tuple[str, int, int, str], FeatureBundle] = {}
    for split, seeds in all_seeds.items():
        for seed in seeds:
            for novel_n in config.ks:
                dataset = build_seed_dataset(seed, novel_n, config)
                print(f"[dataset] split={split} seed={seed} K={novel_n}", flush=True)
                context = fit_feature_context(dataset.normal_stats)
                for feature in FEATURES:
                    bundle = build_feature_bundle(dataset, feature, context)
                    bundles[(split, seed, novel_n, feature)] = bundle
                    for discovery in config.discovery_values:
                        scaled = scale_bundle(bundle, discovery, scaler_scope="base_discovery")
                        ceiling.extend(ceiling_rows(scaled, split))
                        for score_q in GATE_SCORE_QUANTILES:
                            for known_q in GATE_KNOWN_QUANTILES:
                                for split_known in (False, True):
                                    gates.extend(
                                        gate_rows(
                                            scaled,
                                            GateConfig(score_q, known_q, split_known),
                                            split,
                                        )
                                    )

    ceiling_summary = summarize(
        ceiling,
        ("split", "novel_n", "feature", "discovery_per_type", "scaler_scope"),
        config,
    )
    gate_summary = summarize(
        gates,
        (
            "split",
            "novel_n",
            "feature",
            "discovery_per_type",
            "scaler_scope",
            "gate",
            "score_quantile",
            "known_quantile",
            "split_known",
        ),
        config,
    )
    max_k = max(config.ks)
    chosen_feature = best_feature(ceiling_summary, "dev", max_k, config.memory_discovery)
    chosen_gate = select_gate(gate_summary, "dev", max_k, config.memory_discovery, chosen_feature)
    print(f"[selection] feature={chosen_feature} gate={chosen_gate.name}", flush=True)

    for split, seeds in all_seeds.items():
        for seed in seeds:
            bundle = bundles[(split, seed, max_k, chosen_feature)]
            scaled = scale_bundle(bundle, config.memory_discovery, scaler_scope="base_discovery")
            radius = memory_radius(scaled, config.memory_radius_quantile)
            for method in MEMORY_METHODS:
                row = run_memory(scaled, chosen_gate, method, split, radius)
                row["memory_radius_quantile"] = config.memory_radius_quantile
                memory_rows.append(row)
            print(f"[memory] split={split} seed={seed} radius={radius:.3f}", flush=True)

    memory_summary = summarize(
        memory_rows,
        (
            "split",
            "novel_n",
            "feature",
            "discovery_per_type",
            "scaler_scope",
            "gate",
            "method",
        ),
        config,
    )
    selection = {
        "feature": chosen_feature,
        "gate": chosen_gate.name,
        "gate_config": asdict(chosen_gate),
        "selected_on": {"split": "dev", "novel_n": max_k, "discovery_per_type": config.memory_discovery},
    }
    result_path = output_dir / "representation_gate_followup_result.json"
    figure_path = output_dir / "representation_gate_ceiling.png"
    report_path = output_dir / "representation_gate_followup_report.md"
    plot_ceiling(ceiling_summary, config, figure_path)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.time() - started,
        "config": asdict(config),
        "selection": selection,
        "provenance": provenance(),
        "ceiling_rows": ceiling,
        "ceiling_summary": ceiling_summary,
        "gate_rows": gates,
        "gate_summary": gate_summary,
        "memory_rows": memory_rows,
        "memory_summary": memory_summary,
    }
    result_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    build_report(
        report_path,
        result_path,
        figure_path,
        config,
        ceiling_summary,
        gate_summary,
        memory_summary,
        selection,
    )
    print(f"saved -> {result_path}", flush=True)
    print(f"saved -> {figure_path}", flush=True)
    print(f"saved -> {report_path}", flush=True)
    return payload


def build_config(args: argparse.Namespace) -> FollowupConfig:
    if args.smoke:
        return FollowupConfig(
            dev_seeds=(0,),
            holdout_seeds=(20,),
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
        dev_seeds=parse_csv_ints(args.dev_seeds),
        holdout_seeds=parse_csv_ints(args.holdout_seeds),
        ks=parse_csv_ints(args.ks),
        discovery_values=parse_csv_ints(args.discovery_values),
        reuse_per_type=args.reuse_per_type,
        normal_stats_n=args.normal_stats_n,
        normal_train_n=args.normal_train_n,
        normal_cal_n=args.normal_cal_n,
        train_per_known_type=args.train_per_known_type,
        cal_per_known_type=args.cal_per_known_type,
        bootstrap_samples=args.bootstrap_samples,
        memory_discovery=args.memory_discovery,
        memory_radius_quantile=args.memory_radius_quantile,
        memory_component_threshold=args.memory_component_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-seeds", default="0,1,2")
    parser.add_argument("--holdout-seeds", default="30,31,32,33,34")
    parser.add_argument("--ks", default="20,100")
    parser.add_argument("--discovery-values", default="2,5,10")
    parser.add_argument("--reuse-per-type", type=int, default=5)
    parser.add_argument("--normal-stats-n", type=int, default=80)
    parser.add_argument("--normal-train-n", type=int, default=64)
    parser.add_argument("--normal-cal-n", type=int, default=80)
    parser.add_argument("--train-per-known-type", type=int, default=8)
    parser.add_argument("--cal-per-known-type", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--memory-discovery", type=int, default=10)
    parser.add_argument("--memory-radius-quantile", type=float, default=0.90)
    parser.add_argument("--memory-component-threshold", type=float, default=1.6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "docs" / "representation_gate_followup_2026-07-09",
    )
    args = parser.parse_args()
    config = build_config(args)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
