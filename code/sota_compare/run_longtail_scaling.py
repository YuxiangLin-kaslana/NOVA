#!/usr/bin/env python3
"""Run long-tail open-vocabulary anomaly scaling experiments.

The experiment compares four compact mechanisms:

1. Frozen closed-set centroids over known types.
2. Untyped normal-calibrated anomaly score.
3. NOVA/SigLA-style prototype memory with novelty-triggered vocabulary growth.
4. Hierarchical NOVA/SigLA memory with subtype keys and within-key merging.

This is intentionally a fast, local stress test.  It does not replace the P1
CNN/LLM experiments; it asks whether the open-vocabulary advantage becomes more
visible when the anomaly space is expanded to many long-tail composite types.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sigla_exp.longtail_bench as LT  # noqa: E402


KS = [int(x) for x in os.environ.get("LONGTAIL_KS", "6,20,50,100").split(",") if x.strip()]
NSEED = int(os.environ.get("LONGTAIL_NSEED", "3"))
SMOKE = os.environ.get("LONGTAIL_SMOKE", "0") == "1"
KNOWN_FRAC = float(os.environ.get("LONGTAIL_KNOWN_FRAC", "0.30"))
NORMAL_FRAC = float(os.environ.get("LONGTAIL_NORMAL_FRAC", "0.40"))
TRAIN_PER_TYPE = int(os.environ.get("LONGTAIL_TRAIN_PER_TYPE", "24" if SMOKE else "50"))
WARM_N = int(os.environ.get("LONGTAIL_WARM_N", "60" if SMOKE else "180"))
POST_N = int(os.environ.get("LONGTAIL_POST_N", "180" if SMOKE else "700"))
NORMAL_CAL_N = int(os.environ.get("LONGTAIL_NORMAL_CAL_N", "120" if SMOKE else "400"))
CLUSTER_SCALE = float(os.environ.get("LONGTAIL_CLUSTER_SCALE", "1.65"))
HMEM_CLUSTER_SCALE = float(os.environ.get("LONGTAIL_HMEM_CLUSTER_SCALE", "1.00"))
HMEM_MERGE_SCALE = float(os.environ.get("LONGTAIL_HMEM_MERGE_SCALE", "0.35"))
OUT_JSON = ROOT / "runs" / os.environ.get("LONGTAIL_OUTPUT_JSON", "longtail_scaling_result.json")
OUT_FIG = ROOT.parent / "docs" / "longtail_scaling_2026-07-09" / "longtail_scaling_results.png"


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    return mu, sd


def transform(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def centroids(x: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, label in zip(x, labels):
        buckets[label].append(row)
    return {label: np.mean(rows, axis=0) for label, rows in buckets.items()}


def nearest(row: np.ndarray, cents: dict[str, np.ndarray]) -> tuple[str, float]:
    best_label = ""
    best_dist = float("inf")
    for label, c in cents.items():
        d = float(np.linalg.norm(row - c))
        if d < best_dist:
            best_label = label
            best_dist = d
    return best_label, best_dist


def centroid_radius(x: np.ndarray, labels: list[str], cents: dict[str, np.ndarray]) -> float:
    dists = [float(np.linalg.norm(row - cents[label])) for row, label in zip(x, labels)]
    return float(np.quantile(dists, 0.90))


def make_training(
    specs_by_name: dict[str, dict[str, Any]],
    known_names: list[str],
    mu: dict[str, float],
    sd: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], list[float]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    normal_scores: list[float] = []
    for _ in range(max(TRAIN_PER_TYPE * 2, 80)):
        feat = LT.features(LT.make_window(None, rng), mu, sd)
        rows.append(feat)
        labels.append("normal")
        normal_scores.append(LT.anomaly_score(feat))
    for name in known_names:
        spec = specs_by_name[name]
        for _ in range(TRAIN_PER_TYPE):
            rows.append(LT.features(LT.make_window(spec, rng), mu, sd))
            labels.append(name)
    return np.stack(rows), labels, normal_scores


def zipf_weights(n: int, alpha: float = 1.15) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, alpha)
    return w / w.sum()


def make_stream(
    specs_by_name: dict[str, dict[str, Any]],
    known_names: list[str],
    novel_names: list[str],
    mu: dict[str, float],
    sd: dict[str, float],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[str], int]:
    xs: list[np.ndarray] = []
    ys: list[str] = []
    known_w = zipf_weights(len(known_names)) if known_names else np.asarray([])
    for _ in range(WARM_N):
        if rng.random() < NORMAL_FRAC:
            spec = None
            label = "normal"
        else:
            label = str(rng.choice(known_names, p=known_w))
            spec = specs_by_name[label]
        xs.append(LT.features(LT.make_window(spec, rng), mu, sd))
        ys.append(label)

    onset = len(xs)

    # Ensure every novel type appears at least once, then fill the rest with a
    # Zipf long-tail mixture of normal/known/novel windows.
    post_labels: list[str] = list(novel_names)
    all_anom = known_names + novel_names
    weights = zipf_weights(len(all_anom))
    while len(post_labels) < POST_N:
        if rng.random() < NORMAL_FRAC:
            post_labels.append("normal")
        else:
            post_labels.append(str(rng.choice(all_anom, p=weights)))
    rng.shuffle(post_labels)

    for label in post_labels:
        spec = None if label == "normal" else specs_by_name[label]
        xs.append(LT.features(LT.make_window(spec, rng), mu, sd))
        ys.append(label)
    return xs, ys, onset


def binary_metrics(is_anom_pred: list[bool], labels: list[str]) -> dict[str, float]:
    true = [y != "normal" for y in labels]
    tp = sum(p and t for p, t in zip(is_anom_pred, true))
    fp = sum(p and not t for p, t in zip(is_anom_pred, true))
    fn = sum((not p) and t for p, t in zip(is_anom_pred, true))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "prec": prec, "rec": rec}


def evaluate_frozen(preds: list[str], labels: list[str], novel: set[str]) -> dict[str, float]:
    post_novel = [i for i, y in enumerate(labels) if y in novel]
    is_anom = [p != "normal" for p in preds]
    out = binary_metrics(is_anom, labels)
    out["novel_recall"] = float(np.mean([is_anom[i] for i in post_novel])) if post_novel else float("nan")
    out["typed_utility"] = 0.0
    out["cluster_purity"] = 0.0
    out["type_coverage"] = 0.0
    out["reuse_accuracy"] = 0.0
    out["vocab_size"] = 0.0
    out["query_rate"] = 0.0
    return out


def evaluate_untyped(is_anom: list[bool], labels: list[str], novel: set[str]) -> dict[str, float]:
    post_novel = [i for i, y in enumerate(labels) if y in novel]
    out = binary_metrics(is_anom, labels)
    out["novel_recall"] = float(np.mean([is_anom[i] for i in post_novel])) if post_novel else float("nan")
    out["typed_utility"] = 0.0
    out["cluster_purity"] = 0.0
    out["type_coverage"] = 0.0
    out["reuse_accuracy"] = 0.0
    out["vocab_size"] = 0.0
    out["query_rate"] = 0.0
    return out


def evaluate_proto(
    preds: list[str],
    coarse_preds: list[tuple[str, ...]],
    labels: list[str],
    novel: set[str],
    specs_by_name: dict[str, dict[str, Any]],
    gate_calls: int,
) -> dict[str, float]:
    is_anom = [p != "normal" for p in preds]
    out = binary_metrics(is_anom, labels)
    post_novel = [i for i, y in enumerate(labels) if y in novel]
    out["novel_recall"] = float(np.mean([is_anom[i] for i in post_novel])) if post_novel else float("nan")

    cluster_to_labels: dict[str, list[str]] = defaultdict(list)
    for pred, true in zip(preds, labels):
        if pred.startswith("novel_cluster_"):
            cluster_to_labels[pred].append(true)

    cluster_majority: dict[str, str] = {}
    purities = []
    precise_clusters = 0
    for cluster, ys in cluster_to_labels.items():
        counts = Counter(ys)
        label, count = counts.most_common(1)[0]
        cluster_majority[cluster] = label
        purity = count / len(ys)
        purities.append(purity)
        if label in novel and purity >= 0.60:
            precise_clusters += 1

    covered = {label for label in cluster_majority.values() if label in novel}
    correct_reuse = 0
    correct_hier = 0
    for i in post_novel:
        pred = preds[i]
        if pred.startswith("novel_cluster_") and cluster_majority.get(pred) == labels[i]:
            correct_reuse += 1
        true_components = set(specs_by_name[labels[i]]["components"])
        pred_components = set(coarse_preds[i])
        if pred_components == true_components:
            correct_hier += 1

    n_clusters = len(cluster_to_labels)
    out["cluster_purity"] = float(np.mean(purities)) if purities else 0.0
    out["type_coverage"] = len(covered) / len(novel) if novel else 0.0
    out["reuse_accuracy"] = correct_reuse / len(post_novel) if post_novel else 0.0
    out["hierarchical_accuracy"] = correct_hier / len(post_novel) if post_novel else 0.0
    out["typed_utility"] = out["reuse_accuracy"]
    out["vocab_size"] = float(n_clusters)
    out["vocab_precision"] = precise_clusters / n_clusters if n_clusters else 0.0
    out["split_ratio"] = n_clusters / max(1, len(covered))
    out["query_rate"] = gate_calls / len(labels) if labels else 0.0
    return out


def run_proto(
    x: np.ndarray,
    raw_x: np.ndarray,
    labels: list[str],
    known_cents: dict[str, np.ndarray],
    known_radius: float,
    score_thresh: float,
) -> tuple[list[str], list[tuple[str, ...]], int]:
    known_labels = {k for k in known_cents if k != "normal"}
    cluster_radius = known_radius * CLUSTER_SCALE
    prototypes: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    preds: list[str] = []
    coarse_preds: list[tuple[str, ...]] = []
    gate_calls = 0
    next_id = 0

    for row, raw_row in zip(x, raw_x):
        score = LT.anomaly_score(raw_row)
        if score <= score_thresh:
            preds.append("normal")
            coarse_preds.append(())
            continue

        nearest_known, known_dist = nearest(row, known_cents)
        if nearest_known in known_labels and known_dist <= known_radius:
            preds.append(nearest_known)
            coarse_preds.append(LT.component_signature(raw_row, top=2, threshold=1.6))
            continue

        gate_calls += 1
        coarse = LT.component_signature(raw_row, top=2, threshold=1.6)
        if prototypes:
            nearest_cluster, cluster_dist = nearest(row, prototypes)
            if cluster_dist <= cluster_radius:
                preds.append(nearest_cluster)
                coarse_preds.append(coarse)
                counts[nearest_cluster] += 1
                eta = 1.0 / counts[nearest_cluster]
                prototypes[nearest_cluster] = (1 - eta) * prototypes[nearest_cluster] + eta * row
                continue

        name = f"novel_cluster_{next_id:03d}"
        next_id += 1
        prototypes[name] = row.copy()
        counts[name] = 1
        preds.append(name)
        coarse_preds.append(coarse)

    return preds, coarse_preds, gate_calls


def merge_memory(
    prototypes: dict[str, np.ndarray],
    counts: dict[str, int],
    radius: float,
) -> None:
    if len(prototypes) < 2:
        return
    names = list(prototypes)
    best_pair: tuple[str, str] | None = None
    best_dist = float("inf")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = float(np.linalg.norm(prototypes[a] - prototypes[b]))
            if d < best_dist:
                best_dist = d
                best_pair = (a, b)
    if best_pair is None or best_dist > radius:
        return
    a, b = best_pair
    if counts[b] > counts[a]:
        a, b = b, a
    total = counts[a] + counts[b]
    prototypes[a] = (counts[a] * prototypes[a] + counts[b] * prototypes[b]) / total
    counts[a] = total
    del prototypes[b]
    del counts[b]


def run_hier_memory(
    x: np.ndarray,
    raw_x: np.ndarray,
    labels: list[str],
    known_cents: dict[str, np.ndarray],
    known_sigs: set[tuple[str, str, str, str]],
    known_radius: float,
    score_thresh: float,
) -> tuple[list[str], list[tuple[str, ...]], int]:
    known_labels = {k for k in known_cents if k != "normal"}
    cluster_radius = known_radius * HMEM_CLUSTER_SCALE
    merge_radius = known_radius * HMEM_MERGE_SCALE
    prototypes: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    counts: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(dict)
    preds: list[str] = []
    coarse_preds: list[tuple[str, ...]] = []
    gate_calls = 0
    next_id = 0

    for row, raw_row in zip(x, raw_x):
        score = LT.anomaly_score(raw_row)
        if score <= score_thresh:
            preds.append("normal")
            coarse_preds.append(())
            continue

        sig = LT.memory_signature(raw_row, threshold=1.6)
        nearest_known, known_dist = nearest(row, known_cents)
        if nearest_known in known_labels and known_dist <= known_radius and sig in known_sigs:
            preds.append(nearest_known)
            coarse_preds.append(LT.component_signature(raw_row, top=2, threshold=1.6))
            continue

        gate_calls += 1
        coarse = tuple(sig[0].split("+")) if sig[0] not in {"normal", "unknown"} else ()
        bank = prototypes[sig]
        bank_counts = counts[sig]
        if bank:
            nearest_cluster, cluster_dist = nearest(row, bank)
            if cluster_dist <= cluster_radius:
                preds.append(nearest_cluster)
                coarse_preds.append(coarse)
                bank_counts[nearest_cluster] += 1
                eta = 1.0 / bank_counts[nearest_cluster]
                bank[nearest_cluster] = (1 - eta) * bank[nearest_cluster] + eta * row
                merge_memory(bank, bank_counts, merge_radius)
                continue

        sig_name = "_".join(sig).replace("+", "-")
        name = f"novel_cluster_h{next_id:03d}_{sig_name}"
        next_id += 1
        bank[name] = row.copy()
        bank_counts[name] = 1
        preds.append(name)
        coarse_preds.append(coarse)
        merge_memory(bank, bank_counts, merge_radius)

    return preds, coarse_preds, gate_calls


def run_seed(k: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    specs = LT.generate_taxonomy(k)
    specs_by_name = LT.spec_by_name(specs)
    names = [str(s["name"]) for s in specs]
    known_n = min(k - 1, max(3, int(round(k * KNOWN_FRAC)))) if k > 1 else 1
    known_names = names[:known_n]
    novel_names = names[known_n:]
    novel = set(novel_names)
    known_sigs = {LT.spec_signature(specs_by_name[name]) for name in known_names}

    ev_mu, ev_sd = LT.normal_stats(rng, n=NORMAL_CAL_N)
    train_raw, train_labels, normal_scores = make_training(specs_by_name, known_names, ev_mu, ev_sd, rng)
    scale_mu, scale_sd = fit_scaler(train_raw)
    train_x = transform(train_raw, scale_mu, scale_sd)
    cents = centroids(train_x, train_labels)
    radius = centroid_radius(train_x, train_labels, cents)
    score_thresh = float(np.quantile(normal_scores, 0.95))

    xs_raw, ys, onset = make_stream(specs_by_name, known_names, novel_names, ev_mu, ev_sd, rng)
    x = transform(np.stack(xs_raw), scale_mu, scale_sd)
    raw_arr = np.stack(xs_raw)
    post_raw_x = raw_arr[onset:]
    post_x = x[onset:]
    post_y = ys[onset:]

    frozen_preds = [nearest(row, cents)[0] for row in post_x]
    untyped_flags = [LT.anomaly_score(row) > score_thresh for row in post_raw_x]
    proto_preds, coarse_preds, gate_calls = run_proto(post_x, post_raw_x, post_y, cents, radius, score_thresh)
    hmem_preds, hmem_coarse_preds, hmem_gate_calls = run_hier_memory(
        post_x, post_raw_x, post_y, cents, known_sigs, radius, score_thresh
    )

    return {
        "k": k,
        "seed": seed,
        "known_n": known_n,
        "novel_n": len(novel_names),
        "post_n": len(post_y),
        "long_tail_unique_novel_seen": len(set(post_y) & novel),
        "frozen": evaluate_frozen(frozen_preds, post_y, novel),
        "untyped": evaluate_untyped(untyped_flags, post_y, novel),
        "sigla_proto": evaluate_proto(proto_preds, coarse_preds, post_y, novel, specs_by_name, gate_calls),
        "sigla_hmem": evaluate_proto(hmem_preds, hmem_coarse_preds, post_y, novel, specs_by_name, hmem_gate_calls),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    methods = ["frozen", "untyped", "sigla_proto", "sigla_hmem"]
    metrics = [
        "f1",
        "prec",
        "rec",
        "novel_recall",
        "typed_utility",
        "hierarchical_accuracy",
        "cluster_purity",
        "type_coverage",
        "reuse_accuracy",
        "vocab_size",
        "query_rate",
    ]
    for k in sorted({int(r["k"]) for r in rows}):
        subset = [r for r in rows if int(r["k"]) == k]
        for method in methods:
            for metric in metrics:
                vals = [float(r[method].get(metric, float("nan"))) for r in subset]
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                summary.append(
                    {
                        "k": k,
                        "method": method,
                        "metric": metric,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals),
                    }
                )
    return summary


def get(summary: list[dict[str, Any]], k: int, method: str, metric: str) -> float:
    for row in summary:
        if row["k"] == k and row["method"] == method and row["metric"] == metric:
            return float(row["mean"])
    return float("nan")


def plot_summary(summary: list[dict[str, Any]]) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted({int(r["k"]) for r in summary})
    methods = ["frozen", "untyped", "sigla_proto", "sigla_hmem"]
    labels = {
        "frozen": "Frozen closed-set",
        "untyped": "Untyped score",
        "sigla_proto": "SigLA prototype",
        "sigla_hmem": "SigLA h-memory",
    }
    colors = {"frozen": "#b9bdc9", "untyped": "#5a6b78", "sigla_proto": "#283593", "sigla_hmem": "#2e7d32"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        ("novel_recall", "Novel detection recall", (0, 1.05)),
        ("hierarchical_accuracy", "Hierarchical type accuracy", (0, 1.05)),
        ("type_coverage", "Novel type coverage", (0, 1.05)),
    ]
    for ax, (metric, title, ylim) in zip(axes, panels):
        for method in methods:
            vals = [get(summary, k, method, metric) for k in ks]
            ax.plot(ks, vals, marker="o", linewidth=2, label=labels[method], color=colors[method])
        ax.set_title(title)
        ax.set_xlabel("Number of anomaly types (K)")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Long-tail compositional anomaly scaling", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(
        f"longtail scaling KS={KS} NSEED={NSEED} SMOKE={SMOKE} known_frac={KNOWN_FRAC} "
        f"train_per_type={TRAIN_PER_TYPE} post_n={POST_N} cluster_scale={CLUSTER_SCALE} "
        f"hmem_cluster_scale={HMEM_CLUSTER_SCALE}"
    )
    rows = []
    for k in KS:
        for seed in range(NSEED):
            row = run_seed(k, seed)
            rows.append(row)
            sp = row["sigla_proto"]
            hm = row["sigla_hmem"]
            print(
                f"K={k:3d} seed={seed} known={row['known_n']:2d} novel={row['novel_n']:3d} "
                f"sigla novR={sp['novel_recall']:.2f} reuse={sp['reuse_accuracy']:.2f} "
                f"hier={sp['hierarchical_accuracy']:.2f} coverage={sp['type_coverage']:.2f} "
                f"| hmem reuse={hm['reuse_accuracy']:.2f} hier={hm['hierarchical_accuracy']:.2f} "
                f"coverage={hm['type_coverage']:.2f} vocab={hm['vocab_size']:.0f}"
            )
    summary = summarize(rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "ks": KS,
            "nseed": NSEED,
            "known_frac": KNOWN_FRAC,
            "normal_frac": NORMAL_FRAC,
            "train_per_type": TRAIN_PER_TYPE,
            "warm_n": WARM_N,
            "post_n": POST_N,
            "normal_cal_n": NORMAL_CAL_N,
            "cluster_scale": CLUSTER_SCALE,
            "hmem_cluster_scale": HMEM_CLUSTER_SCALE,
            "hmem_merge_scale": HMEM_MERGE_SCALE,
            "smoke": SMOKE,
        },
        "rows": rows,
        "summary": summary,
        "figure": str(OUT_FIG),
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot_summary(summary)
    print(f"saved -> {OUT_JSON}")
    print(f"figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
