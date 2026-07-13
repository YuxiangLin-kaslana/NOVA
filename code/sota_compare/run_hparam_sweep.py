#!/usr/bin/env python3
"""P2 appendix: core open-vocabulary gate hyperparameter sweep.

This isolates online-gate behavior from SOTA baseline training. It reuses the
same detection/naming stream as the P1 synthetic and real-background injection
experiments, but sweeps NOVEL_Z, TAU, AUDIT, and RETRAIN_EVERY for Ours.
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


SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
N_PT = 300 if SMOKE else 3000
DATASET = os.environ.get("P2_DATASET", "synthetic")
ENTITY = os.environ.get("P2_ENTITY", "")
NAMER_MODE = os.environ.get("P2_NAMER", "local").lower()
MAX_CONFIGS = int(os.environ.get("P2_MAX_CONFIGS", "0") or "0")

DEFAULT = {
    "tau": 0.50,
    "audit": 0.08,
    "novel_z": 2.30,
    "retrain_every": 15,
}


def local_recognize_top1(ev, key, mu, sd=None, model="local-rule"):
    sd = sd or {k: 1.0 for k in mu}
    z = {k: abs((ev[k] - mu[k]) / (sd[k] + 1e-9)) for k in mu}
    dom = max(z, key=z.get)
    if z[dom] < float(os.environ.get("P2_LOCAL_NAMER_Z", "2.0")):
        return None
    stat_to_concept = {v: k for k, v in CB.STAT_OF.items()}
    return stat_to_concept.get(dom)


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


def config_key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return tuple((k, cfg[k]) for k in sorted(cfg))


def make_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def add(name: str, cfg: dict[str, Any]) -> None:
        row = dict(DEFAULT)
        row.update(cfg)
        row["name"] = name
        configs.append(row)

    add("default", {})
    for v in [1.7, 2.0, 2.3, 2.6, 3.0]:
        add(f"novel_z={v:g}", {"novel_z": v})
    for v in [0.40, 0.50, 0.60, 0.70]:
        add(f"tau={v:g}", {"tau": v})
    for v in [0.00, 0.04, 0.08, 0.12]:
        add(f"audit={v:g}", {"audit": v})
    for v in [8, 12, 15, 24]:
        add(f"retrain_every={v}", {"retrain_every": v})
    for z in [2.0, 2.3, 2.6, 3.0]:
        for tau in [0.40, 0.50, 0.60, 0.70]:
            add(f"grid_z={z:g}_tau={tau:g}", {"novel_z": z, "tau": tau})

    seen = set()
    deduped = []
    for cfg in configs:
        key = config_key({k: cfg[k] for k in DEFAULT})
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    names = [x.strip() for x in os.environ.get("P2_CONFIG_NAMES", "").split(",") if x.strip()]
    if names:
        by_name = {cfg["name"]: cfg for cfg in deduped}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Unknown P2_CONFIG_NAMES entries: {missing}; available={sorted(by_name)}")
        deduped = [by_name[name] for name in names]
    return deduped[:MAX_CONFIGS] if MAX_CONFIGS else deduped


def make_labeled(concept_or_normal: str, rng: np.random.Generator) -> np.ndarray:
    c = None if concept_or_normal == DT.NORMAL else concept_or_normal
    return CB.make_window(c, rng)


def suspect_novel(ev: dict[str, float], mu: dict[str, float], sd: dict[str, float], novel_z: float) -> bool:
    devz = {k: abs((ev[k] - mu[k]) / sd[k]) for k in mu}
    dom = max(devz, key=devz.get)
    return (dom not in DT.KNOWN_STATS) and (devz[dom] > novel_z)


def run_bootstrap(
    stream,
    onset: int,
    pre_state,
    replay,
    mu: dict[str, float],
    sd: dict[str, float],
    cfg: dict[str, Any],
    key: str,
    net_ok: bool,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(50_000 + seed)
    det = DT.make_detector(len(DT.BASE_VOCAB))
    det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(DT.BASE_VOCAB)
    buf = []
    boot_pred = []
    namer_flags = []
    pending = 0

    for x, _ in stream:
        ev = CB.evidence(x)
        p = DT.proba(det, [x])[0]
        mx = float(p.max())
        pred_idx = int(np.argmax(p))
        susp = suspect_novel(ev, mu, sd, float(cfg["novel_z"]))
        mislabel = (vocab[pred_idx] in (DT.KNOWN_ANOM + [DT.NORMAL])) and susp
        audit = rng.random() < float(cfg["audit"])
        if mx >= float(cfg["tau"]) and not mislabel and not audit:
            pred, called = vocab[pred_idx], 0
        else:
            c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            c = None if c == "__ERROR__" else c
            pred = c if c else vocab[pred_idx]
            called = 1
            if pred not in vocab:
                if susp:
                    vocab.append(pred)
                    DT.grow_head(det, len(vocab))
                    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                else:
                    pred = vocab[pred_idx]
            buf.append((x, vocab.index(pred)))
            pending += 1
            if pending >= int(cfg["retrain_every"]):
                per_class = {}
                for w, li in replay:
                    per_class.setdefault(li, []).append(w)
                for w, li in buf:
                    per_class.setdefault(li, []).append(w)
                Xb, Yb = [], []
                for li, ws in per_class.items():
                    for j in rng.integers(0, len(ws), 40):
                        Xb.append(ws[j])
                        Yb.append(DT.onehot(li, len(vocab)))
                DT.train_on(det, opt, Xb, Yb, epochs=2)
                pending = 0
        boot_pred.append(pred)
        namer_flags.append(called)

    trues = [t for _, t in stream]
    metrics = DT.detect_metrics(boot_pred[onset:], trues[onset:], vocab)
    metrics.update({
        "namer_call_rate": float(np.mean(namer_flags[onset:])),
        "grew": int(DT.NOVEL in vocab),
        "vocab_size": int(len(vocab)),
        "spurious_vocab": int(len([c for c in vocab if c not in DT.BASE_VOCAB + [DT.NOVEL]])),
    })
    return metrics


def prepare_seed(seed: int):
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
    return stream, onset, pre_state, replay, mu, sd


def ms(xs):
    arr = np.asarray(xs, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main() -> None:
    if DATASET.lower() != "synthetic":
        if not ENTITY:
            raise SystemExit("P2_ENTITY is required for real-background hparam sweep")
        RB.activate(ENTITY, dataset=DATASET)
        bg = f"{DATASET}:{ENTITY}"
    else:
        bg = "synthetic"

    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = (NAMER_MODE == "local") or ((NAMER_MODE == "llm") and bool(key) and not SMOKE)
    configs = make_configs()
    print(f"device={DT.device} bg={bg} nseed={NSEED} configs={len(configs)} namer={NAMER_MODE} net_ok={net_ok}")

    per_config = []
    seed_cache = [prepare_seed(seed) for seed in range(NSEED)]
    for cfg in configs:
        recs = []
        for seed, prepared in enumerate(seed_cache):
            stream, onset, pre_state, replay, mu, sd = prepared
            rec = run_bootstrap(stream, onset, pre_state, replay, mu, sd, cfg, key, net_ok, seed)
            rec["seed"] = seed
            recs.append(rec)
        summary = {}
        for metric in ("nov_recall", "nov_classacc", "f1", "prec", "rec", "namer_call_rate", "grew", "vocab_size", "spurious_vocab"):
            summary[metric] = dict(zip(("mean", "std"), ms([r[metric] for r in recs])))
        per_config.append({"config": cfg, "summary": summary, "per_seed": recs})
        print(f"{cfg['name']:22s} novR={summary['nov_recall']['mean']:.2f} "
              f"name={summary['nov_classacc']['mean']:.2f} call={summary['namer_call_rate']['mean']:.2f} "
              f"grew={summary['grew']['mean']:.2f}")

    outp = output_path("hparam_sweep.json")
    payload = {
        "experiment": "hparam_sweep",
        "background": bg,
        "nseed": NSEED,
        "namer_mode": NAMER_MODE,
        "defaults": DEFAULT,
        "per_config": per_config,
    }
    json.dump(payload, open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
