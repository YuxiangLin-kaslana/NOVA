#!/usr/bin/env python3
"""P2 appendix: guarded-update hyperparameter sweep."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sigla_exp.ovbench as CB  # noqa: E402
import sota_compare.realbench as RB  # noqa: E402
import sota_compare.exp_drift_vs_novel as EXP  # noqa: E402
from sota_compare.robust_loop import robust_ours_loop  # noqa: E402
from sota_compare.run_hparam_sweep import local_recognize_top1  # noqa: E402
from sota_compare.run_robust_multi import pretrain, drift_stream  # noqa: E402


SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "2"))
MACHINES = os.environ.get("P2_MACHINES", "1-1").split(",")
NAMER_MODE = os.environ.get("P2_NAMER", "local").lower()
MAX_CONFIGS = int(os.environ.get("P2_MAX_CONFIGS", "0") or "0")

if NAMER_MODE == "local":
    CB.gpt_recognize_top1 = local_recognize_top1

DEFAULT = {
    "K_CONFIRM": 3,
    "HORIZON": 60,
    "Z_MARGIN": 2.8,
    "Z_STRONG": 4.0,
    "FA_TARGET": 0.05,
    "GUARD_N": 200,
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


def make_configs() -> list[dict[str, Any]]:
    configs = []

    def add(name: str, cfg: dict[str, Any]) -> None:
        row = dict(DEFAULT)
        row.update(cfg)
        row["name"] = name
        configs.append(row)

    add("default", {})
    for v in [2, 3, 4]:
        add(f"K_CONFIRM={v}", {"K_CONFIRM": v})
    for v in [40, 60, 90]:
        add(f"HORIZON={v}", {"HORIZON": v})
    for v in [2.4, 2.8, 3.2]:
        add(f"Z_MARGIN={v:g}", {"Z_MARGIN": v})
    for v in [0.03, 0.05, 0.08]:
        add(f"FA_TARGET={v:g}", {"FA_TARGET": v})

    deduped, seen = [], set()
    for cfg in configs:
        key = tuple((k, cfg[k]) for k in sorted(DEFAULT))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped[:MAX_CONFIGS] if MAX_CONFIGS else deduped


def ms(xs):
    arr = np.asarray(xs, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def run_prepared(seed: int, cfg: dict[str, Any], prepared: tuple[Any, ...], key: str, net_ok: bool) -> dict[str, Any]:
    mu, sd, dvec, pre, replay, thr, W1, W2, is_nov, phase = prepared
    nspur = lambda v: len([c for c in v if c not in EXP.BASE_VOCAB])

    a1, _, v1, st1 = robust_ours_loop(
        pre, replay, W1, mu, sd, key, net_ok, thr,
        K_CONFIRM=int(cfg["K_CONFIRM"]),
        HORIZON=int(cfg["HORIZON"]),
        Z_MARGIN=float(cfg["Z_MARGIN"]),
        Z_STRONG=float(cfg["Z_STRONG"]),
        FA_TARGET=float(cfg["FA_TARGET"]),
        GUARD_N=int(cfg["GUARD_N"]),
    )
    a2, names, v2, st2 = robust_ours_loop(
        pre, replay, W2, mu, sd, key, net_ok, thr,
        K_CONFIRM=int(cfg["K_CONFIRM"]),
        HORIZON=int(cfg["HORIZON"]),
        Z_MARGIN=float(cfg["Z_MARGIN"]),
        Z_STRONG=float(cfg["Z_STRONG"]),
        FA_TARGET=float(cfg["FA_TARGET"]),
        GUARD_N=int(cfg["GUARD_N"]),
    )
    nov = is_nov == 1
    return {
        "seed": seed,
        "rob_fa": float(np.mean(a1)),
        "rob_fa_late": float(np.mean(a1[-200:])),
        "rob_spur": nspur(v1),
        "t2_drift_fa": float(a2[phase == "drift"].mean()),
        "t2_recall": float(a2[nov].mean()),
        "t2_name": float(np.mean([names[i] == EXP.NOVEL for i in range(len(W2)) if is_nov[i]])),
        "t2_grew": int(EXP.NOVEL in v2),
        "commit": int(st1["commit"] + st2["commit"]),
        "recalib": int(st1["recalib"] + st2["recalib"]),
    }


def prepare_seed(seed: int):
    rng = np.random.default_rng(seed)
    import torch
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)
    dvec = rng.normal(0, 1, CB.NVARS).astype(np.float32)
    dvec /= (np.linalg.norm(dvec) / np.sqrt(CB.NVARS) + 1e-6)
    pre, replay, thr = pretrain(rng)
    W1 = drift_stream(rng, dvec)
    rng2 = np.random.default_rng(1000 + seed)
    W2, lab, is_nov, phase, creep = EXP.build_stream(rng2, dvec)
    return mu, sd, dvec, pre, replay, thr, W1, W2, is_nov, phase


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = (NAMER_MODE == "local") or ((NAMER_MODE == "llm") and bool(key) and not SMOKE)
    configs = make_configs()
    print(f"device={EXP.device} machines={MACHINES} nseed={NSEED} configs={len(configs)} namer={NAMER_MODE} net_ok={net_ok}")

    results = {}
    for machine in [m.strip() for m in MACHINES if m.strip()]:
        RB.activate(machine)
        prepared = [prepare_seed(seed) for seed in range(NSEED)]
        cfg_rows = []
        for cfg in configs:
            recs = [run_prepared(seed, cfg, prepared[seed], key, net_ok) for seed in range(NSEED)]
            summary = {}
            for metric in ("rob_fa", "rob_fa_late", "rob_spur", "t2_drift_fa", "t2_recall", "t2_name", "t2_grew", "commit", "recalib"):
                summary[metric] = dict(zip(("mean", "std"), ms([r[metric] for r in recs])))
            cfg_rows.append({"config": cfg, "summary": summary, "per_seed": recs})
            print(f"{machine} {cfg['name']:18s} driftFA={summary['rob_fa']['mean']:.2f} "
                  f"rec={summary['t2_recall']['mean']:.2f} name={summary['t2_name']['mean']:.2f} "
                  f"spur={summary['rob_spur']['mean']:.2f}")
        results[machine] = cfg_rows

    outp = output_path("guard_sweep.json")
    payload = {
        "experiment": "guard_sweep",
        "machines": MACHINES,
        "nseed": NSEED,
        "namer_mode": NAMER_MODE,
        "defaults": DEFAULT,
        "per_machine": results,
    }
    json.dump(payload, open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
