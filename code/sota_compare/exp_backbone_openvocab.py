#!/usr/bin/env python3
"""【backbone 无关性:CNN/MLP/RF/HGB 的开放词表闭环】填补 NOVA 的 backbone 实验(RF/HGB 之前没跑)。

无 LLM(用规则命名器,因已证规则在闭集命名上追平 LLM)。证据 z-向量特征 + grow-vocab + 类平衡重训。
单 novel(trend/variance_burst/correlation_break 各 hold-out),报每 backbone 的 novel 检测召回 + 命名。
用法: sbatch sota_compare/run_backbone.sh
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import sigla_exp.ovbench as CB

STATS = CB.STATS; STAT2C = {v: k for k, v in CB.SIG.items()}
KNOWN = ["spike", "level_shift", "oscillation"]; NOVELS = ["trend", "variance_burst", "correlation_break"]
KNOWN_STATS = {CB.SIG[c] for c in KNOWN}
NOVEL_Z, TAU, KCONF, NSEED = 2.3, 0.6, 3, int(os.environ.get("CMP_NSEED", "3"))


def output_path(default_name):
    explicit = os.environ.get("CMP_OUTPUT_JSON")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else ROOT / p
    tag = os.environ.get("CMP_RUN_TAG", "").strip()
    if tag:
        stem, suffix = Path(default_name).stem, Path(default_name).suffix
        return ROOT / "runs" / f"{stem}_{tag}{suffix}"
    return ROOT / "runs" / default_name


def zfeat(x, mu, sd):
    ev = CB.evidence(x); return np.array([(ev[s] - mu[s]) / (sd[s] + 1e-9) for s in STATS], np.float32)


def mk(name):
    if name == "MLP": return MLPClassifier((64, 64), max_iter=300, early_stopping=False)
    if name == "RF": return RandomForestClassifier(150, n_jobs=4)
    return HistGradientBoostingClassifier(max_iter=200)


def rule_name(z):
    d = {STATS[i]: z[i] for i in range(len(STATS))}
    dom = max(d, key=d.get)
    return STAT2C[dom] if (d[dom] > NOVEL_Z and dom not in KNOWN_STATS) else None


def run(backbone, novel, seed):
    rng = np.random.default_rng(seed); mu, sd = CB.normal_stats(rng)
    base = ["normal"] + KNOWN
    buf = {c: [] for c in base}
    for _ in range(500):
        c = base[rng.integers(len(base))]
        x = CB.make_window(None, rng) if c == "normal" else CB.make_window_strength(c, rng, float(rng.uniform(.6, 1)))
        buf[c].append(zfeat(x, mu, sd))
    vocab = list(base)

    def fit():
        X, y = [], []
        K = 200
        for c in vocab:
            arr = buf[c]
            for j in rng.integers(0, len(arr), K):
                X.append(arr[j]); y.append(c)
        m = mk(backbone); m.fit(np.array(X), np.array(y)); return m
    clf = fit()

    # stream: warmup known, then novel emerges
    stream = []
    for _ in range(150):
        c = "normal" if rng.random() < .5 else KNOWN[rng.integers(3)]
        stream.append((c, CB.make_window(None, rng) if c == "normal" else CB.make_window_strength(c, rng, 1.)))
    pool = KNOWN + [novel]
    for _ in range(400):
        c = "normal" if rng.random() < .5 else pool[rng.integers(len(pool))]
        stream.append((c, CB.make_window(None, rng) if c == "normal" else CB.make_window_strength(c, rng, 1.)))
    onset = 150
    pending = []; preds = []
    for i, (tc, x) in enumerate(stream):
        z = zfeat(x, mu, sd)
        proba = clf.predict_proba([z])[0]; cls = clf.classes_
        pi = int(np.argmax(proba)); pred = cls[pi]; conf = proba[pi]
        rn = rule_name(z)
        if rn is not None and pred != rn:   # 证据驱动门控(不靠置信度→避开OOD过自信)
            if rn in vocab:
                pred = rn; buf[rn].append(z)
            else:
                pending.append((rn, z))
                if sum(1 for r, _ in pending if r == rn) >= KCONF:
                    vocab.append(rn); buf.setdefault(rn, [])
                    for r, zz in pending:
                        if r == rn: buf[rn].append(zz)
                    pending = [(r, zz) for r, zz in pending if r != rn]
                    clf = fit(); pred = rn
        preds.append(pred)
    tr = [c for c, _ in stream]
    nov = [(p, t) for p, t in zip(preds[onset:], tr[onset:]) if t == novel]
    det = np.mean([p != "normal" for p, _ in nov]) if nov else np.nan
    name = np.mean([p == novel for p, _ in nov]) if nov else np.nan
    return float(det), float(name), int(novel in vocab)


def main():
    print(f"Backbone 无关性(无LLM,规则命名器)  NSEED={NSEED}\n")
    print(f"{'backbone':10s}{'novel':18s}{'novel检测召回':>14s}{'命名准确率':>12s}{'词表长全':>10s}")
    print("-" * 64)
    res = {}
    for bb in ["MLP", "RF", "HGB"]:
        for nv in NOVELS:
            ds, ns, gr = [], [], []
            for s in range(NSEED):
                d, n, g = run(bb, nv, s); ds.append(d); ns.append(n); gr.append(g)
            res[f"{bb}/{nv}"] = dict(det=float(np.nanmean(ds)), name=float(np.nanmean(ns)), grew=float(np.mean(gr)))
            print(f"{bb:10s}{nv:18s}{np.nanmean(ds)*100:>11.0f}%{np.nanmean(ns)*100:>10.0f}%{np.mean(gr)*100:>8.0f}%")
        print()
    json.dump(res, open(output_path("backbone_openvocab.json"), "w"), indent=2)
    print("判读:若 RF/HGB/MLP 都拿到正的 novel 检测+命名 → 开放词表闭环 backbone 无关(NOVA 的 backbone 声称成立)。")


if __name__ == "__main__":
    main()
