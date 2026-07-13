#!/usr/bin/env python3
"""P2 appendix: q-threshold strategy sweep on a fixed stream.

Fits each scoring model once per seed, then applies q90/q95/q97/q99 thresholds
to the same calibration and test scores. This evaluates threshold choice without
confounding it with retraining noise.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.exp_detection_tie as DT  # noqa: E402
import sigla_exp.ovbench as CB  # noqa: E402
import sota_compare.realbench as RB  # noqa: E402
from sota_compare.baselines import AnomalyTransformer, MemStream  # noqa: E402
from sota_compare.run_hparam_sweep import local_recognize_top1  # noqa: E402


SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
N_PT = 300 if SMOKE else 3000
N_NORM_TR = 200 if SMOKE else 1500
UNSUP_EP = 3 if SMOKE else 40
DATASET = os.environ.get("P2_DATASET", "synthetic")
ENTITY = os.environ.get("P2_ENTITY", "")
NAMER_MODE = os.environ.get("P2_NAMER", "local").lower()
QS = [float(x) for x in os.environ.get("P2_QUANTILES", "0.90,0.95,0.97,0.99").split(",")]

if NAMER_MODE == "local":
    CB.gpt_recognize_top1 = local_recognize_top1

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


def make_labeled(concept_or_normal: str, rng: np.random.Generator) -> np.ndarray:
    c = None if concept_or_normal == DT.NORMAL else concept_or_normal
    return CB.make_window(c, rng)


def binary_metrics(alarms: np.ndarray, trues: list[str], onset: int) -> dict[str, float]:
    pred = alarms[onset:].astype(bool)
    yy = np.asarray([t != DT.NORMAL for t in trues[onset:]], dtype=bool)
    nov = np.asarray([t == DT.NOVEL for t in trues[onset:]], dtype=bool)
    normal = ~yy
    tp = int(np.sum(pred & yy))
    fp = int(np.sum(pred & normal))
    fn = int(np.sum((~pred) & yy))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "f1": float(f1),
        "prec": float(prec),
        "rec": float(rec),
        "far": float(np.mean(pred[normal])) if np.any(normal) else 0.0,
        "nov_recall": float(np.mean(pred[nov])) if np.any(nov) else float("nan"),
    }


def train_closed(seed: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)
    det = DT.make_detector(len(DT.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(N_PT):
        c = DT.BASE_VOCAB[rng.integers(len(DT.BASE_VOCAB))]
        Xpt.append(make_labeled(c, rng))
        Ypt.append(DT.onehot(DT.BASE_VOCAB.index(c), len(DT.BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))
    stream, onset = DT.build_stream(rng)
    return rng, mu, sd, det, pre_state, replay, stream, onset


def closed_scores(det, windows):
    return 1.0 - DT.proba(det, windows)[:, 0]


def ms(xs):
    arr = np.asarray(xs, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main() -> None:
    if DATASET.lower() != "synthetic":
        if not ENTITY:
            raise SystemExit("P2_ENTITY is required for real-background threshold sweep")
        RB.activate(ENTITY, dataset=DATASET)
        bg = f"{DATASET}:{ENTITY}"
    else:
        bg = "synthetic"

    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = (NAMER_MODE == "llm") and bool(key) and not SMOKE
    print(f"device={DT.device} bg={bg} nseed={NSEED} q={QS} namer={NAMER_MODE} net_ok={net_ok}")

    rows = []
    for seed in range(NSEED):
        rng, mu, sd, det, pre_state, replay, stream, onset = train_closed(seed)
        windows = [x for x, _ in stream]
        trues = [t for _, t in stream]
        normal_cal = [CB.make_window(None, rng) for _ in range(400)]

        frozen_cal = closed_scores(det, normal_cal)
        frozen_scores = closed_scores(det, windows)

        score_bank = {
            "closed_cnn": (frozen_cal, frozen_scores, {}),
        }

        rng2 = np.random.default_rng(10_000 + seed)
        normal_train = [CB.make_window(None, rng2) for _ in range(N_NORM_TR)]
        unsup_cal = [CB.make_window(None, rng2) for _ in range(400)]
        for cls, name in [(AnomalyTransformer, "anomaly_transformer"), (MemStream, "memstream")]:
            model = cls(CB.WIN, CB.NVARS, DT.device, epochs=UNSUP_EP, seed=seed)
            model.fit(normal_train)
            score_bank[name] = (
                model.score_stream(unsup_cal, update=False),
                model.score_stream(windows, update=True),
                {},
            )

        for method, (cal, scores, extra) in score_bank.items():
            for q in QS:
                thr = float(np.quantile(cal, q))
                rec = binary_metrics(np.asarray(scores) > thr, trues, onset)
                rec.update(extra)
                rec.update({"seed": seed, "method": method, "q": q, "threshold": thr})
                rows.append(rec)
                print(f"seed={seed} {method:20s} q={q:.2f} F1={rec['f1']:.2f} "
                      f"novR={rec['nov_recall']:.2f} FAR={rec['far']:.2f}")

    summary = []
    for method in sorted({r["method"] for r in rows}):
        for q in QS:
            subset = [r for r in rows if r["method"] == method and r["q"] == q]
            metrics = {}
            for metric in ("f1", "prec", "rec", "far", "nov_recall", "nov_classacc", "namer_call_rate"):
                vals = [r[metric] for r in subset if metric in r]
                if vals:
                    metrics[metric] = dict(zip(("mean", "std"), ms(vals)))
            summary.append({"method": method, "q": q, "metrics": metrics})

    outp = output_path("threshold_strategy_sweep.json")
    payload = {
        "experiment": "threshold_strategy_sweep",
        "background": bg,
        "nseed": NSEED,
        "namer_mode": NAMER_MODE,
        "quantiles": QS,
        "summary": summary,
        "rows": rows,
    }
    json.dump(payload, open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
