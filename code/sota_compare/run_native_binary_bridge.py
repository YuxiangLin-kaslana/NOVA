#!/usr/bin/env python3
"""Native-label binary anomaly detection bridge.

This appendix runner evaluates official normal/anomaly labels directly, without
controlled novel-type injection. It is a sanity bridge for classic AD metrics:
window F1, point F1, event F1/recall, AUROC, AUPRC, FAR, and delay.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from sigla_exp.actions import find_events  # noqa: E402
from sota_compare.baselines import AnomalyTransformer, MemStream  # noqa: E402
import scripts.exp_detection_tie as DT  # noqa: E402


DATA = PROJECT / "data"
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "2"))
WIN = int(os.environ.get("NATIVE_WIN", "100"))
STEP = int(os.environ.get("NATIVE_STEP", "10"))
NVARS = int(os.environ.get("NATIVE_NVARS", "12"))
MAX_TRAIN_WINDOWS = int(os.environ.get("NATIVE_MAX_TRAIN_WINDOWS", "200" if SMOKE else "1500"))
MAX_CAL_WINDOWS = int(os.environ.get("NATIVE_MAX_CAL_WINDOWS", "100" if SMOKE else "400"))
UNSUP_EP = int(os.environ.get("NATIVE_UNSUP_EP", "3" if SMOKE else "15"))
PCA_DIM = int(os.environ.get("NATIVE_PCA_DIM", "16"))
METHODS = [x.strip() for x in os.environ.get("NATIVE_METHODS", "zscore,pca,anomaly_transformer,memstream").split(",") if x.strip()]
QS = [float(x) for x in os.environ.get("NATIVE_QS", "0.95,0.97").split(",")]
DATASETS = [x.strip().upper() for x in os.environ.get("NATIVE_DATASETS", "SMD,PSM,MSL,SMAP").split(",") if x.strip()]
DEFAULT_ENTITIES = {
    "SMD": "1-1,2-1,3-1,1-6,2-5",
    "PSM": "psm",
    "MSL": "M-1,F-7,T-4,C-1,P-10",
    "SMAP": "P-1,E-1,A-1,E-5,D-1",
}


def output_path(default_name: str) -> Path:
    explicit = os.environ.get("CMP_OUTPUT_JSON")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else ROOT / p
    tag = os.environ.get("CMP_RUN_TAG", "").strip()
    if tag:
        stem, suffix = Path(default_name).stem, Path(default_name).suffix
        return ROOT / "runs" / f"{stem}_{tag}{suffix}"
    return ROOT / "runs" / default_name


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def entity_paths(dataset: str, entity: str) -> tuple[Path, Path, Path]:
    if dataset == "SMD":
        ent = entity.removeprefix("machine-")
        pre = DATA / "ServerMachineDataset" / "preprocessed"
        return (
            pre / f"machine-{ent}_train.pkl",
            pre / f"machine-{ent}_test.pkl",
            pre / f"machine-{ent}_test_label.pkl",
        )
    if dataset == "PSM":
        pre = DATA / "PSM" / "preprocessed"
        return pre / "psm_train.pkl", pre / "psm_test.pkl", pre / "psm_test_label.pkl"
    pre = DATA / dataset / "preprocessed"
    return pre / f"{entity}_train.pkl", pre / f"{entity}_test.pkl", pre / f"{entity}_test_label.pkl"


def load_entity(dataset: str, entity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr, te, lab = entity_paths(dataset, entity)
    train = np.asarray(load_pickle(tr), dtype=np.float32)
    test = np.asarray(load_pickle(te), dtype=np.float32)
    labels = np.asarray(load_pickle(lab), dtype=np.int64).reshape(-1)
    if len(test) != len(labels):
        raise ValueError(f"{dataset}:{entity} test/label length mismatch: {len(test)} vs {len(labels)}")
    return train, test, labels


def select_standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if train.ndim == 1:
        train = train[:, None]
        test = test[:, None]
    keep = min(NVARS, train.shape[1])
    idx = np.argsort(np.nanvar(train, axis=0))[::-1][:keep]
    idx = np.sort(idx)
    tr = train[:, idx]
    te = test[:, idx]
    mu = np.nanmean(tr, axis=0, keepdims=True)
    sd = np.nanstd(tr, axis=0, keepdims=True) + 1e-6
    tr = np.nan_to_num((tr - mu) / sd).astype(np.float32)
    te = np.nan_to_num((te - mu) / sd).astype(np.float32)
    return tr, te


def windows(x: np.ndarray, labels: np.ndarray | None = None, step: int = STEP):
    starts = np.arange(0, max(0, len(x) - WIN + 1), step, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"series length {len(x)} shorter than WIN={WIN}")
    W = np.stack([x[s:s + WIN] for s in starts]).astype(np.float32)
    if labels is None:
        return W, starts, None
    y = np.asarray([int(np.any(labels[s:s + WIN] > 0)) for s in starts], dtype=np.int64)
    return W, starts, y


def sample_windows(W: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    if len(W) <= n:
        return [w for w in W]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(W), size=n, replace=False))
    return [W[i] for i in idx]


class ZScoreEnergy:
    def fit(self, normal_windows: list[np.ndarray]):
        return self

    def score_stream(self, windows_: list[np.ndarray] | np.ndarray, update: bool = False) -> np.ndarray:
        arr = np.asarray(windows_, dtype=np.float32)
        return np.sqrt(np.mean(arr * arr, axis=(1, 2))).astype(np.float32)


class PCARecon:
    def __init__(self, n_components: int = PCA_DIM):
        self.n_components = n_components

    def fit(self, normal_windows: list[np.ndarray]):
        X = np.asarray(normal_windows, dtype=np.float32).reshape(len(normal_windows), -1)
        self.mu = X.mean(axis=0, keepdims=True)
        Xc = X - self.mu
        k = max(1, min(self.n_components, Xc.shape[0] - 1, Xc.shape[1]))
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = vt[:k].astype(np.float32)
        return self

    def score_stream(self, windows_: list[np.ndarray] | np.ndarray, update: bool = False) -> np.ndarray:
        X = np.asarray(windows_, dtype=np.float32).reshape(len(windows_), -1)
        Xc = X - self.mu
        proj = Xc @ self.components.T @ self.components
        return np.mean((Xc - proj) ** 2, axis=1).astype(np.float32)


def make_model(method: str, seed: int, nvars: int):
    if method == "zscore":
        return ZScoreEnergy()
    if method == "pca":
        return PCARecon()
    if method == "anomaly_transformer":
        return AnomalyTransformer(WIN, nvars, DT.device, epochs=UNSUP_EP, seed=seed)
    if method == "memstream":
        return MemStream(WIN, nvars, DT.device, epochs=UNSUP_EP, seed=seed)
    raise ValueError(f"unknown native method {method!r}")


def safe_auc(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))
    except Exception:
        return float("nan"), float("nan")


def point_prediction(starts: np.ndarray, alarms: np.ndarray, length: int) -> np.ndarray:
    pred = np.zeros(length, dtype=np.int64)
    for s, a in zip(starts, alarms):
        if a:
            pred[int(s): min(length, int(s) + WIN)] = 1
    return pred


def binary_prf(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(bool)
    p = p.astype(bool)
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = float(np.mean(p[~y])) if np.any(~y) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "far": far}


def overlaps(a, b) -> bool:
    return a.onset <= b.end and b.onset <= a.end


def event_metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth = find_events(labels)
    alarms = find_events(pred)
    truth_hit = [any(overlaps(t, a) for a in alarms) for t in truth]
    alarm_hit = [any(overlaps(t, a) for t in truth) for a in alarms]
    er = float(np.mean(truth_hit)) if truth else float("nan")
    ep = float(np.mean(alarm_hit)) if alarms else 0.0
    ef1 = 2 * ep * er / (ep + er) if ep + er else 0.0
    delays = []
    for t, hit in zip(truth, truth_hit):
        if not hit:
            continue
        idx = np.flatnonzero(pred[t.onset:t.end + 1] > 0)
        if len(idx):
            delays.append(int(idx[0]))
    return {
        "event_precision": float(ep),
        "event_recall": float(er),
        "event_f1": float(ef1),
        "delay_mean": float(np.mean(delays)) if delays else float("nan"),
        "n_events": int(len(truth)),
        "n_alarm_events": int(len(alarms)),
    }


def eval_threshold(scores: np.ndarray, threshold: float, starts: np.ndarray, win_y: np.ndarray, point_y: np.ndarray) -> dict[str, float]:
    alarms = (scores > threshold).astype(np.int64)
    point_pred = point_prediction(starts, alarms, len(point_y))
    win = binary_prf(win_y, alarms)
    point = binary_prf(point_y, point_pred)
    ev = event_metrics(point_y, point_pred)
    out = {
        "window_precision": win["precision"],
        "window_recall": win["recall"],
        "window_f1": win["f1"],
        "window_far": win["far"],
        "point_precision": point["precision"],
        "point_recall": point["recall"],
        "point_f1": point["f1"],
        "point_far": point["far"],
        **ev,
    }
    return out


def ms(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def dataset_entities(dataset: str) -> list[str]:
    env = os.environ.get(f"NATIVE_ENTITIES_{dataset}")
    raw = env if env else DEFAULT_ENTITIES[dataset]
    return [x.strip() for x in raw.split(",") if x.strip()]


def run_entity(dataset: str, entity: str) -> dict[str, Any]:
    train_raw, test_raw, point_y = load_entity(dataset, entity)
    train, test = select_standardize(train_raw, test_raw)
    train_W, _, _ = windows(train, None)
    test_W, starts, win_y = windows(test, point_y)
    nvars = train.shape[1]
    rows = []
    for seed in range(NSEED):
        train_sample = sample_windows(train_W, MAX_TRAIN_WINDOWS, 10_000 + seed)
        cal_sample = sample_windows(train_W, MAX_CAL_WINDOWS, 20_000 + seed)
        for method in METHODS:
            model = make_model(method, seed, nvars)
            model.fit(train_sample)
            cal_scores = np.asarray(model.score_stream(cal_sample, update=False), dtype=float)
            scores = np.asarray(model.score_stream([w for w in test_W], update=True), dtype=float)
            auroc, auprc = safe_auc(win_y, scores)
            for q in QS:
                thr = float(np.quantile(cal_scores, q))
                rec = eval_threshold(scores, thr, starts, win_y, point_y)
                rec.update({
                    "seed": seed,
                    "method": method,
                    "q": q,
                    "threshold": thr,
                    "window_auroc": auroc,
                    "window_auprc": auprc,
                })
                rows.append(rec)
                print(f"{dataset}:{entity} seed={seed} {method:20s} q={q:.2f} "
                      f"winF1={rec['window_f1']:.2f} evtF1={rec['event_f1']:.2f} FAR={rec['point_far']:.2f}")
    summary = []
    for method in METHODS:
        for q in QS:
            subset = [r for r in rows if r["method"] == method and r["q"] == q]
            metrics = {}
            for metric in [
                "window_f1", "window_precision", "window_recall", "window_far",
                "point_f1", "point_precision", "point_recall", "point_far",
                "event_f1", "event_precision", "event_recall", "delay_mean",
                "window_auroc", "window_auprc",
            ]:
                metrics[metric] = dict(zip(("mean", "std"), ms([r[metric] for r in subset])))
            summary.append({"method": method, "q": q, "metrics": metrics})
    return {
        "dataset": dataset,
        "entity": entity,
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
        "anomaly_ratio": float(np.mean(point_y)),
        "n_events": int(len(find_events(point_y))),
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    print(f"device={DT.device} datasets={DATASETS} methods={METHODS} nseed={NSEED} q={QS} win={WIN} step={STEP} epochs={UNSUP_EP}")
    results = {}
    for dataset in DATASETS:
        entities = dataset_entities(dataset)
        for entity in entities:
            key = f"{dataset}:{entity}"
            results[key] = run_entity(dataset, entity)
    outp = output_path("native_binary_bridge.json")
    payload = {
        "experiment": "native_binary_bridge",
        "datasets": DATASETS,
        "methods": METHODS,
        "nseed": NSEED,
        "win": WIN,
        "step": STEP,
        "nvars": NVARS,
        "quantiles": QS,
        "max_train_windows": MAX_TRAIN_WINDOWS,
        "max_cal_windows": MAX_CAL_WINDOWS,
        "unsup_epochs": UNSUP_EP,
        "results": results,
    }
    json.dump(payload, open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
