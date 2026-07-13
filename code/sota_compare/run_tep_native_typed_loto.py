#!/usr/bin/env python3
"""Minimal native-typed open-world bridge on the Tennessee Eastman Process.

The experiment holds out each native fault id in turn. Known-fault prototypes,
rejection calibration, discovery, and locked reuse use disjoint simulationRun
ranges. A single queried native fault id names the discovered prototype; this
tests discovery and future reuse, not free-form or zero-shot naming.

The deployable feature path never reads faultNumber. Fault ids are consumed
only when building the experimental split, fitting supervised known-class
prototypes, answering the explicitly counted naming query, and scoring output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "TEP" / "raw" / "TEP_Faulty_Training.RData"
DEFAULT_CACHE = ROOT / "data" / "TEP" / "processed" / "tep_run_evidence_post21_blocks4.npz"
DEFAULT_OUTPUT = ROOT / "docs" / "tep_native_typed_loto_2026-07-09"
OFFICIAL_DATASET = "https://doi.org/10.7910/DVN/6C3JR1"
RIETH_PAPER = "https://doi.org/10.1007/978-3-319-60384-1_6"
DOWNS_VOGEL_PAPER = "https://doi.org/10.1016/0098-1354(93)80018-I"
OPENMAX_PAPER = "https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html"
UNO_PAPER = "https://openaccess.thecvf.com/content/ICCV2021/html/Fini_A_Unified_Objective_for_Novel_Class_Discovery_ICCV_2021_paper.html"
TELEMANOM_SOURCE = "https://github.com/khundman/telemanom"
ITRUST_DATASETS = "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/"
EXPECTED_MD5 = "c5f594d54c47e620ff877feb58407fda"

# Downs and Vogel (1993), Table 8. IDs 16--20 are intentionally retained as
# "unknown" because that is how the original benchmark defines them.
FAULT_DESCRIPTIONS = {
    1: "A/C feed ratio step; B composition constant (stream 4)",
    2: "B composition step; A/C ratio constant (stream 4)",
    3: "D feed temperature step (stream 2)",
    4: "reactor cooling-water inlet temperature step",
    5: "condenser cooling-water inlet temperature step",
    6: "A feed loss step (stream 1)",
    7: "C header pressure loss / reduced availability step (stream 4)",
    8: "A, B, C feed composition random variation (stream 4)",
    9: "D feed temperature random variation (stream 2)",
    10: "C feed temperature random variation (stream 4)",
    11: "reactor cooling-water inlet temperature random variation",
    12: "condenser cooling-water inlet temperature random variation",
    13: "reaction kinetics slow drift",
    14: "reactor cooling-water valve sticking",
    15: "condenser cooling-water valve sticking",
    16: "unknown process disturbance 16",
    17: "unknown process disturbance 17",
    18: "unknown process disturbance 18",
    19: "unknown process disturbance 19",
    20: "unknown process disturbance 20",
}


@dataclass(frozen=True)
class Config:
    train_start: int = 1
    train_count: int = 50
    calibration_start: int = 51
    calibration_count: int = 50
    discovery_start: int = 101
    discovery_count: int = 50
    batch_discovery_count: int = 20
    reuse_start: int = 201
    reuse_count: int = 100
    known_replay_per_fault: int = 5
    post_fault_start: int = 21
    temporal_blocks: int = 4
    pca_dim: int = 32
    known_acceptance_quantile: float = 0.95
    seeds: int = 10


@dataclass(frozen=True)
class EvidenceCache:
    faults: np.ndarray
    runs: np.ndarray
    pre_mean: np.ndarray
    pre_second_moment: np.ndarray
    block_mean: np.ndarray
    post_std: np.ndarray
    post_slope: np.ndarray
    variable_names: tuple[str, ...]


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def run_ids(start: int, count: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.int64)


def assert_disjoint(config: Config) -> None:
    groups = {
        "train": set(run_ids(config.train_start, config.train_count)),
        "calibration": set(run_ids(config.calibration_start, config.calibration_count)),
        "discovery": set(run_ids(config.discovery_start, config.discovery_count)),
        "reuse": set(run_ids(config.reuse_start, config.reuse_count)),
    }
    for left, left_values in groups.items():
        for right, right_values in groups.items():
            if left >= right:
                continue
            overlap = left_values & right_values
            if overlap:
                raise ValueError(f"simulationRun leakage between {left} and {right}: {sorted(overlap)}")
    if config.batch_discovery_count > config.discovery_count:
        raise ValueError("batch_discovery_count exceeds the discovery pool")
    if config.known_replay_per_fault > config.reuse_count:
        raise ValueError("known_replay_per_fault exceeds the reuse pool")


def _ordered_tensor(frame: Any, variable_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faults = frame["faultNumber"].to_numpy(dtype=np.int16)
    runs = frame["simulationRun"].to_numpy(dtype=np.int16)
    samples = frame["sample"].to_numpy(dtype=np.int16)
    values = frame[variable_names].to_numpy(dtype=np.float32)

    expected_rows = 20 * 500 * 500
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(frame)}")
    expected_faults = np.repeat(np.arange(1, 21, dtype=np.int16), 500 * 500)
    expected_runs = np.tile(np.repeat(np.arange(1, 501, dtype=np.int16), 500), 20)
    expected_samples = np.tile(np.arange(1, 501, dtype=np.int16), 20 * 500)
    if not (
        np.array_equal(faults, expected_faults)
        and np.array_equal(runs, expected_runs)
        and np.array_equal(samples, expected_samples)
    ):
        order = np.lexsort((samples, runs, faults))
        faults = faults[order]
        runs = runs[order]
        samples = samples[order]
        values = values[order]
        if not (
            np.array_equal(faults, expected_faults)
            and np.array_equal(runs, expected_runs)
            and np.array_equal(samples, expected_samples)
        ):
            raise ValueError("RData rows do not form the documented 20 x 500 x 500 grid")
    return values.reshape(20, 500, 500, len(variable_names)), faults, runs


def load_evidence(path: Path, config: Config, cache_path: Path, source_md5: str) -> EvidenceCache:
    if cache_path.exists():
        packed = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(packed["metadata"].item()))
        expected = {
            "source_md5": source_md5,
            "post_fault_start": config.post_fault_start,
            "temporal_blocks": config.temporal_blocks,
        }
        if metadata == expected:
            print(f"loading evidence cache {cache_path}", flush=True)
            return EvidenceCache(
                faults=packed["faults"],
                runs=packed["runs"],
                pre_mean=packed["pre_mean"],
                pre_second_moment=packed["pre_second_moment"],
                block_mean=packed["block_mean"],
                post_std=packed["post_std"],
                post_slope=packed["post_slope"],
                variable_names=tuple(str(item) for item in packed["variable_names"]),
            )
        print(f"ignoring stale evidence cache {cache_path}: {metadata}", flush=True)

    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("pyreadr is required: python -m pip install pyreadr") from exc

    result = pyreadr.read_r(str(path))
    if "faulty_training" not in result:
        raise KeyError(f"faulty_training not found; objects={sorted(result)}")
    frame = result["faulty_training"]
    variable_names = [name for name in frame.columns if name.startswith(("xmeas_", "xmv_"))]
    if len(variable_names) != 52:
        raise ValueError(f"expected 52 process variables, found {len(variable_names)}")
    tensor, _, _ = _ordered_tensor(frame, variable_names)

    pre = tensor[:, :, : config.post_fault_start - 1]
    post = tensor[:, :, config.post_fault_start - 1 :]
    if post.shape[2] % config.temporal_blocks:
        raise ValueError("post-fault length must be divisible by temporal_blocks")
    block_size = post.shape[2] // config.temporal_blocks
    block_mean = post.reshape(20, 500, config.temporal_blocks, block_size, 52).mean(axis=3)
    post_std = post.std(axis=2)
    time_index = np.arange(post.shape[2], dtype=np.float32)
    centered_time = time_index - time_index.mean()
    denom = float(np.square(centered_time).sum())
    post_slope = np.einsum("fntv,t->fnv", post, centered_time, optimize=True) / denom

    cache = EvidenceCache(
        faults=np.arange(1, 21, dtype=np.int64),
        runs=np.arange(1, 501, dtype=np.int64),
        pre_mean=pre.mean(axis=2),
        pre_second_moment=np.square(pre).mean(axis=2),
        block_mean=block_mean.astype(np.float32),
        post_std=post_std.astype(np.float32),
        post_slope=post_slope.astype(np.float32),
        variable_names=tuple(variable_names),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_md5": source_md5,
        "post_fault_start": config.post_fault_start,
        "temporal_blocks": config.temporal_blocks,
    }
    np.savez_compressed(
        cache_path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        faults=cache.faults,
        runs=cache.runs,
        pre_mean=cache.pre_mean,
        pre_second_moment=cache.pre_second_moment,
        block_mean=cache.block_mean,
        post_std=cache.post_std,
        post_slope=cache.post_slope,
        variable_names=np.asarray(cache.variable_names),
    )
    print(f"saved evidence cache {cache_path}", flush=True)
    return cache


def baseline_stats(cache: EvidenceCache, known_faults: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = cache.pre_mean[np.ix_(known_faults - 1, train_indices)]
    seconds = cache.pre_second_moment[np.ix_(known_faults - 1, train_indices)]
    mean = means.mean(axis=(0, 1))
    variance = np.maximum(seconds.mean(axis=(0, 1)) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def observable_features(cache: EvidenceCache, baseline_mean: np.ndarray, baseline_sd: np.ndarray) -> np.ndarray:
    block_z = (cache.block_mean - baseline_mean) / baseline_sd
    log_scale = np.log((cache.post_std + 1e-6) / (baseline_sd + 1e-6))
    post_length = (500 - 21 + 1)
    slope_z = cache.post_slope * post_length / baseline_sd
    features = np.concatenate(
        [block_z.reshape(20, 500, -1), log_scale, slope_z], axis=2
    )
    if features.shape != (20, 500, 312):
        raise AssertionError(f"unexpected evidence shape: {features.shape}")
    return np.nan_to_num(features, nan=0.0, posinf=30.0, neginf=-30.0).astype(np.float32)


def fit_embedding(
    features: np.ndarray,
    known_faults: np.ndarray,
    train_indices: np.ndarray,
    pca_dim: int,
) -> tuple[StandardScaler, PCA, np.ndarray, dict[int, np.ndarray]]:
    x_train = features[np.ix_(known_faults - 1, train_indices)].reshape(-1, features.shape[-1])
    y_train = np.repeat(known_faults, len(train_indices))
    scaler = StandardScaler().fit(x_train)
    scaled = scaler.transform(x_train)
    n_components = min(pca_dim, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, whiten=True, svd_solver="full").fit(scaled)
    embedded = pca.transform(scaled).astype(np.float32)
    centers = {int(fault): embedded[y_train == fault].mean(axis=0) for fault in known_faults}
    return scaler, pca, embedded, centers


def transform_subset(
    features: np.ndarray,
    fault_ids: Iterable[int],
    indices: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray]:
    faults = np.asarray(list(fault_ids), dtype=np.int64)
    raw = features[np.ix_(faults - 1, indices)].reshape(-1, features.shape[-1])
    labels = np.repeat(faults, len(indices))
    return pca.transform(scaler.transform(raw)).astype(np.float32), labels


def distances_to_centers(x: np.ndarray, centers: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sorted(centers), dtype=np.int64)
    matrix = np.stack([centers[int(label)] for label in labels])
    distances = np.linalg.norm(x[:, None, :] - matrix[None, :, :], axis=2)
    nearest_index = np.argmin(distances, axis=1)
    return labels[nearest_index], distances[np.arange(len(x)), nearest_index]


def class_balanced_accuracy(y: np.ndarray, pred: np.ndarray, labels: Iterable[int]) -> float:
    scores = [float(np.mean(pred[y == label] == label)) for label in labels if np.any(y == label)]
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_fold(
    holdout: int,
    features: np.ndarray,
    config: Config,
) -> list[dict[str, Any]]:
    known_faults = np.asarray([fault for fault in range(1, 21) if fault != holdout], dtype=np.int64)
    train_indices = run_ids(config.train_start, config.train_count) - 1
    calibration_indices = run_ids(config.calibration_start, config.calibration_count) - 1
    discovery_indices = run_ids(config.discovery_start, config.discovery_count) - 1
    reuse_indices = run_ids(config.reuse_start, config.reuse_count) - 1

    scaler, pca, train_embedding, centers = fit_embedding(
        features, known_faults, train_indices, config.pca_dim
    )

    calibration_x, calibration_y = transform_subset(
        features, known_faults, calibration_indices, scaler, pca
    )
    own_distances = np.asarray(
        [np.linalg.norm(row - centers[int(label)]) for row, label in zip(calibration_x, calibration_y)]
    )
    acceptance_radius = float(np.quantile(own_distances, config.known_acceptance_quantile))

    novel_x, novel_y = transform_subset(features, [holdout], reuse_indices, scaler, pca)
    replay_indices = reuse_indices[: config.known_replay_per_fault]
    known_x, known_y = transform_subset(features, known_faults, replay_indices, scaler, pca)

    closed_novel_pred, closed_novel_distance = distances_to_centers(novel_x, centers)
    closed_known_pred, closed_known_distance = distances_to_centers(known_x, centers)
    open_novel_pred = np.where(closed_novel_distance <= acceptance_radius, closed_novel_pred, 0)
    open_known_pred = np.where(closed_known_distance <= acceptance_radius, closed_known_pred, 0)

    batch_indices = discovery_indices[: config.batch_discovery_count]
    batch_novel_x, batch_novel_y = transform_subset(
        features, [holdout], batch_indices, scaler, pca
    )
    batch_known_x, batch_known_y = transform_subset(
        features, known_faults, batch_indices[:1], scaler, pca
    )
    batch_x = np.concatenate([batch_novel_x, batch_known_x])
    batch_y = np.concatenate([batch_novel_y, batch_known_y])
    _, batch_distance = distances_to_centers(batch_x, centers)
    candidate_mask = batch_distance > acceptance_radius
    candidates = batch_x[candidate_mask]
    candidate_labels = batch_y[candidate_mask]
    batch_query_label = 0
    batch_center = None
    if len(candidates):
        provisional_center = candidates.mean(axis=0)
        medoid_index = int(np.argmin(np.linalg.norm(candidates - provisional_center, axis=1)))
        batch_query_label = int(candidate_labels[medoid_index])
        if batch_query_label == holdout:
            batch_center = provisional_center
    batch_centers = dict(centers)
    if batch_center is not None:
        batch_centers[holdout] = batch_center
    batch_novel_pred, _ = distances_to_centers(novel_x, batch_centers)
    batch_known_pred, _ = distances_to_centers(known_x, batch_centers)

    rows = []
    for seed in range(config.seeds):
        rng = np.random.default_rng(1_000_000 + 10_000 * holdout + seed)
        query_index = int(rng.choice(discovery_indices))
        discovery_x, _ = transform_subset(
            features, [holdout], np.asarray([query_index]), scaler, pca
        )
        queried_label = holdout
        one_query_centers = {**centers, queried_label: discovery_x[0]}
        one_novel_pred, _ = distances_to_centers(novel_x, one_query_centers)
        one_known_pred, _ = distances_to_centers(known_x, one_query_centers)

        rows.append(
            {
                "holdout_fault": holdout,
                "holdout_description": FAULT_DESCRIPTIONS[holdout],
                "seed": seed,
                "feature_dim": int(features.shape[-1]),
                "embedding_dim": int(pca.n_components_),
                "acceptance_radius": acceptance_radius,
                "closed_known_accuracy": class_balanced_accuracy(
                    known_y, closed_known_pred, known_faults
                ),
                "closed_novel_exact_accuracy": float(np.mean(closed_novel_pred == novel_y)),
                "open_novel_rejection_recall": float(np.mean(open_novel_pred == 0)),
                "open_known_rejection_rate": float(np.mean(open_known_pred == 0)),
                "open_known_typed_accuracy": class_balanced_accuracy(
                    known_y, open_known_pred, known_faults
                ),
                "one_query_run": query_index + 1,
                "one_query_annotation_count": 1,
                "one_query_returned_description": FAULT_DESCRIPTIONS[queried_label],
                "one_query_novel_reuse_accuracy": float(np.mean(one_novel_pred == novel_y)),
                "one_query_known_accuracy": class_balanced_accuracy(
                    known_y, one_known_pred, known_faults
                ),
                "one_query_known_to_novel_rate": float(np.mean(one_known_pred == holdout)),
                "batch_candidate_count": int(candidate_mask.sum()),
                "batch_candidate_novel_fraction": (
                    float(np.mean(candidate_labels == holdout)) if len(candidates) else 0.0
                ),
                "batch_query_label": batch_query_label,
                "batch_query_returned_description": (
                    FAULT_DESCRIPTIONS[batch_query_label] if batch_query_label else None
                ),
                "batch_query_naming_success": bool(batch_query_label == holdout),
                "batch_annotation_count": 1 if len(candidates) else 0,
                "batch_novel_reuse_accuracy": float(np.mean(batch_novel_pred == novel_y)),
                "batch_known_accuracy": class_balanced_accuracy(
                    known_y, batch_known_pred, known_faults
                ),
                "batch_known_to_novel_rate": float(np.mean(batch_known_pred == holdout)),
                "train_embedding_rows": int(len(train_embedding)),
            }
        )
    return rows


METRICS = (
    "closed_known_accuracy",
    "open_novel_rejection_recall",
    "open_known_rejection_rate",
    "open_known_typed_accuracy",
    "one_query_novel_reuse_accuracy",
    "one_query_known_accuracy",
    "one_query_known_to_novel_rate",
    "batch_query_naming_success",
    "batch_novel_reuse_accuracy",
    "batch_known_accuracy",
    "batch_known_to_novel_rate",
)


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    per_fault = []
    for fault in sorted({int(row["holdout_fault"]) for row in rows}):
        subset = [row for row in rows if row["holdout_fault"] == fault]
        record: dict[str, Any] = {
            "holdout_fault": fault,
            "holdout_description": FAULT_DESCRIPTIONS[fault],
            "n_seeds": len(subset),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            record[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        per_fault.append(record)

    macro: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        fault_means = np.asarray([record[metric]["mean"] for record in per_fault], dtype=float)
        macro[metric] = {
            "mean": float(fault_means.mean()),
            "std_across_faults": float(fault_means.std(ddof=1)),
            "n_faults": int(len(fault_means)),
        }
    return per_fault, macro


def plot(per_fault: list[dict[str, Any]], output: Path) -> None:
    faults = [row["holdout_fault"] for row in per_fault]
    reject = [row["open_novel_rejection_recall"]["mean"] for row in per_fault]
    one_query = [row["one_query_novel_reuse_accuracy"]["mean"] for row in per_fault]
    batch = [row["batch_novel_reuse_accuracy"]["mean"] for row in per_fault]
    x = np.arange(len(faults))
    width = 0.27
    fig, axis = plt.subplots(figsize=(11.5, 4.2))
    axis.bar(x - width, reject, width, label="Distance rejection (UNKNOWN)", color="#6b7280")
    axis.bar(x, one_query, width, label="One-query prototype", color="#2a8c67")
    axis.bar(x + width, batch, width, label="Batch candidate centroid", color="#b06c2f")
    axis.set_xticks(x, [str(fault) for fault in faults])
    axis.set_xlabel("Held-out native TEP fault id")
    axis.set_ylabel("Locked novel recall / exact reuse")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.set_title("TEP leave-one-fault-type-out: detection is not naming")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{value:.1%}"


def build_report(
    output: Path,
    config: Config,
    per_fault: list[dict[str, Any]],
    macro: dict[str, dict[str, float]],
    raw_path: Path,
    raw_md5: str,
    elapsed: float,
) -> None:
    lines = [
        "# TEP Native-Typed Leave-One-Fault-Out Bridge",
        "",
        "## Scope",
        "",
        "This is a minimal native-label bridge on the public Tennessee Eastman Process simulation data. It tests whether one explicit native fault-description query can create a prototype that is reused on independent future simulation runs. It does **not** test free-form or zero-shot semantic naming, and it is not an end-to-end anomaly detector.",
        "",
        f"- Official dataset: [{OFFICIAL_DATASET}]({OFFICIAL_DATASET})",
        f"- Dataset paper: [Rieth et al.]({RIETH_PAPER})",
        f"- Original process benchmark: [Downs and Vogel]({DOWNS_VOGEL_PAPER})",
        f"- Local source: `{raw_path}` (`{raw_path.stat().st_size}` bytes; MD5 `{raw_md5}`)",
        "- Official release used here: fault ids `1..20` (20 fault types), 500 independent simulation runs/type, 52 process variables.",
        "",
        "## Local Data Audit",
        "",
        "| Local source | Native type labels? | Decision |",
        "|---|---|---|",
        "| SMD | No. Binary point labels; `interpretation_label` lists affected dimension indices, not semantic fault classes. | Not a typed benchmark. |",
        "| PSM and UCR | No. Available labels are binary anomaly indicators. | Not a typed benchmark. |",
        f"| [NASA SMAP/MSL Telemanom]({TELEMANOM_SOURCE}) | Two author-provided sequence classes: `point` and `contextual` (105 events). | Valid backup bridge, but too coarse for multi-type discovery. |",
        f"| [SWaT/WADI]({ITRUST_DATASETS}) | Attack metadata exists in the controlled release. | Raw authorized files are absent locally; existing checkpoints/results are not typed ground truth and were not used. |",
        "| TEP | Yes: native fault ids 1--20 with process-disturbance descriptions. | Selected and run from the official Dataverse file. |",
        "",
        "## Protocol",
        "",
        f"- Known prototype training: simulationRun `{config.train_start}..{config.train_start + config.train_count - 1}`.",
        f"- Known rejection calibration: simulationRun `{config.calibration_start}..{config.calibration_start + config.calibration_count - 1}`; pooled `{config.known_acceptance_quantile:.0%}` within-known distance quantile.",
        f"- Discovery pool: simulationRun `{config.discovery_start}..{config.discovery_start + config.discovery_count - 1}`; one native fault-description query.",
        f"- Locked reuse: simulationRun `{config.reuse_start}..{config.reuse_start + config.reuse_count - 1}`; no query or state update.",
        f"- Observable evidence: `{4 * 52}` post-fault block means + `52` log scale ratios + `52` trajectory slopes = `312D`; known-only PCA to `{config.pca_dim}D`.",
        "- All arms share the same evidence vectors, known-class prototypes, native description vocabulary, and locked batch. Only discovery/query behavior changes.",
        "- `faultNumber` is not read by the feature extractor. It is used only for split construction, supervised known prototypes, the counted query response, and evaluation.",
        "",
        "The closed and open-rejection arms cannot output the held-out native name. The one-query arm is an oracle-triggered upper bound: it receives exactly one native fault description and stores that sample as a named prototype. Fault ID is retained only for array indexing and scoring. The batch arm first discovers candidates autonomously with the calibrated known-distance gate, forms one candidate centroid, and then spends one description query on its medoid before the locked batch.",
        "",
        "## Macro Results",
        "",
        "| Metric | Mean across held-out faults | Across-fault SD |",
        "|---|---:|---:|",
    ]
    display = [
        ("Known closed-set accuracy", "closed_known_accuracy"),
        ("Novel rejection recall (UNKNOWN only)", "open_novel_rejection_recall"),
        ("Known rejection rate", "open_known_rejection_rate"),
        ("One-query locked novel exact reuse", "one_query_novel_reuse_accuracy"),
        ("One-query known accuracy", "one_query_known_accuracy"),
        ("One-query known-to-novel error", "one_query_known_to_novel_rate"),
        ("Batch query names held-out fault", "batch_query_naming_success"),
        ("Batch locked novel exact reuse", "batch_novel_reuse_accuracy"),
        ("Batch known-to-novel error", "batch_known_to_novel_rate"),
    ]
    for label, metric in display:
        values = macro[metric]
        lines.append(f"| {label} | {percent(values['mean'])} | {percent(values['std_across_faults'])} |")

    lines.extend(
        [
            "",
            "The calibrated distance gate rejected only "
            f"`{percent(macro['open_novel_rejection_recall']['mean'])}` of held-out-fault runs. "
            "Even the oracle-triggered one-description upper bound reached only "
            f"`{percent(macro['one_query_novel_reuse_accuracy']['mean'])}` exact locked reuse, while "
            f"known-to-novel error stayed at `{percent(macro['one_query_known_to_novel_rate']['mean'])}`. "
            "The autonomous batch candidate bridge named the held-out type in only "
            f"`{percent(macro['batch_query_naming_success']['mean'])}` of folds. These are diagnostic negative results: "
            "they show that rejection, naming, and future reuse are separate bottlenecks.",
            "",
            "![Per-fault rejection and locked reuse](tep_native_typed_loto.png)",
        ]
    )

    lines.extend(
        [
            "",
            "## Per-Fault Results",
            "",
            "| Held-out fault | Native description | Reject as UNKNOWN | One-query exact reuse | One-query known→novel | Batch naming success | Batch exact reuse |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_fault:
        lines.append(
            f"| {row['holdout_fault']} | {row['holdout_description']} | "
            f"{percent(row['open_novel_rejection_recall']['mean'])} | "
            f"{percent(row['one_query_novel_reuse_accuracy']['mean'])} | "
            f"{percent(row['one_query_known_to_novel_rate']['mean'])} | "
            f"{percent(row['batch_query_naming_success']['mean'])} | "
            f"{percent(row['batch_novel_reuse_accuracy']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- Rejection recall is open-set detection, not native-type naming. An `UNKNOWN` output is scored separately from exact reuse.",
            "- The one-query arm is an oracle-triggered upper bound and memory-isolation control. It does not show that the system would choose the right query autonomously.",
            "- Only the batch arm autonomously generates novel candidates. Its clustering step is a deliberately small bridge, not an implementation of UNO, AutoNovel, or another deep NCD method.",
            "- Training-file simulation runs are independent RNG states but share the same simulator and operating regime. External plant generalization is not established.",
            "- The fault onset is supplied by the documented simulator protocol, so this experiment isolates post-fault typing rather than detection delay.",
            "",
            "## Adjacent Methods To Implement Next",
            "",
            f"1. [OpenMax]({OPENMAX_PAPER}): fit class-conditional Weibull tails on the same PCA evidence and report known accuracy versus unknown rejection. A distance quantile is used here, so these results must not be called OpenMax.",
            f"2. [UNO]({UNO_PAPER}): train a shared time-series encoder with supervised known-fault and multi-view pseudo-label objectives. This requires a multi-held-out protocol; leave-one-out has only one novel class and makes NCD clustering nearly trivial.",
            "",
            f"Runtime: `{elapsed:.1f}` seconds. Figure: `tep_native_typed_loto.png`. Machine-readable results: `tep_native_typed_loto_result.json`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_faults(value: str) -> tuple[int, ...]:
    faults = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    invalid = [fault for fault in faults if not 1 <= fault <= 20]
    if invalid:
        raise ValueError(f"fault ids outside 1..20: {invalid}")
    return faults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--evidence-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdouts", default=",".join(str(i) for i in range(1, 21)))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = Config(seeds=args.seeds)
    holdouts = parse_faults(args.holdouts)
    if args.smoke:
        config = Config(
            train_count=10,
            calibration_start=21,
            calibration_count=10,
            discovery_start=41,
            discovery_count=10,
            batch_discovery_count=5,
            reuse_start=61,
            reuse_count=20,
            known_replay_per_fault=2,
            pca_dim=12,
            seeds=min(args.seeds, 2),
        )
        holdouts = holdouts[:2]
    assert_disjoint(config)
    if not args.raw.exists():
        raise FileNotFoundError(
            f"{args.raw} not found; see data/TEP/README.md for the official download command"
        )
    raw_md5 = file_md5(args.raw)
    if raw_md5 != EXPECTED_MD5:
        raise ValueError(f"source MD5 mismatch: expected {EXPECTED_MD5}, found {raw_md5}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"loading {args.raw}", flush=True)
    cache = load_evidence(args.raw, config, args.evidence_cache, raw_md5)
    rows: list[dict[str, Any]] = []
    for holdout in holdouts:
        known_faults = np.asarray([fault for fault in range(1, 21) if fault != holdout], dtype=np.int64)
        train_indices = run_ids(config.train_start, config.train_count) - 1
        baseline_mean, baseline_sd = baseline_stats(cache, known_faults, train_indices)
        features = observable_features(cache, baseline_mean, baseline_sd)
        print(f"holdout={holdout:02d} seeds={config.seeds}", flush=True)
        rows.extend(evaluate_fold(holdout, features, config))

    per_fault, macro = summarize(rows)
    elapsed = time.time() - started
    payload = {
        "experiment": "tep_native_typed_loto",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": elapsed,
        "scope": "one native fault-description query followed by independent locked reuse; not zero-shot naming",
        "dataset": {
            "official_url": OFFICIAL_DATASET,
            "local_path": str(args.raw),
            "bytes": args.raw.stat().st_size,
            "md5": raw_md5,
            "evidence_cache": str(args.evidence_cache),
            "evidence_cache_md5": file_md5(args.evidence_cache),
            "native_fault_ids": list(range(1, 21)),
            "native_fault_descriptions": FAULT_DESCRIPTIONS,
            "normal_id": 0,
        },
        "config": asdict(config),
        "holdouts": list(holdouts),
        "integrity": {
            "run_ranges_disjoint": True,
            "feature_uses_fault_id": False,
            "locked_queries": 0,
            "source_checksum_matches_official": raw_md5 == EXPECTED_MD5,
        },
        "rows": rows,
        "per_fault": per_fault,
        "macro": macro,
        "references": {
            "dataset": OFFICIAL_DATASET,
            "dataset_paper": RIETH_PAPER,
            "process_paper": DOWNS_VOGEL_PAPER,
            "openmax": OPENMAX_PAPER,
            "uno": UNO_PAPER,
        },
    }
    result_path = args.output_dir / "tep_native_typed_loto_result.json"
    figure_path = args.output_dir / "tep_native_typed_loto.png"
    report_path = args.output_dir / "tep_native_typed_loto_report.md"
    result_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    plot(per_fault, figure_path)
    build_report(report_path, config, per_fault, macro, args.raw, raw_md5, elapsed)
    print(f"saved {result_path}", flush=True)
    print(f"saved {figure_path}", flush=True)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
