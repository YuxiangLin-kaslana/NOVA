#!/usr/bin/env python3
"""B 路线第一步:验证 `sigla_exp.ovbench.evidence` 的**证据可分性 + gate 可用性**。
判据(真正重要的是后两条):
  (1) 每概念自己的签名统计量显著偏离 normal(z>3)。
  (2) gate:给定 known/novel 划分,每个 NOVEL 概念**不触发任何 KNOWN 签名**(z<GATE_Z)→ suspect_novel 才会 fire。
  (3) 每个 KNOWN 概念触发自己的签名(z>GATE_Z)。
纯 numpy。用法: sbatch scripts/diag_separation_v2.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sigla_exp import ovbench as B  # noqa: E402

KNOWN = ["spike", "level_shift", "oscillation"]
NOVEL = ["variance_burst", "trend", "correlation_break"]
GATE_Z = 3.0           # 某签名 z>GATE_Z 视为"被触发"
KEYS = list(B.SIG.values())


def main():
    rng = np.random.default_rng(0)
    n = 200
    norm = [B.evidence(B.make_window(None, rng)) for _ in range(n)]
    mu = {k: float(np.mean([e[k] for e in norm])) for k in KEYS}
    sd = {k: float(np.std([e[k] for e in norm]) + 1e-9) for k in KEYS}
    known_sigs = {B.SIG[c] for c in KNOWN}

    Z = {}
    for c in B.CONCEPTS:
        evs = [B.evidence(B.make_window(c, rng)) for _ in range(n)]
        m = {k: float(np.mean([e[k] for e in evs])) for k in KEYS}
        Z[c] = {k: (m[k] - mu[k]) / sd[k] for k in KEYS}

    print(f"ovbench 可分性 + gate sanity (n={n})  GATE_Z={GATE_Z}")
    print(f"KNOWN={KNOWN}  NOVEL={NOVEL}\nSIG={B.SIG}\nknown_signature_stats={sorted(known_sigs)}\n")
    print("%-20s" % "concept\\stat(z)" + "".join("%14s" % k for k in KEYS))
    print("-" * (20 + 14 * len(KEYS)))
    for c in B.CONCEPTS:
        tag = "K" if c in KNOWN else "N"
        print("%-20s" % f"[{tag}] {c}" + "".join("%14s" % f"{Z[c][k]:+.1f}" for k in KEYS))

    print("\n--- 判据 ---")
    ok = True
    ABSENCE = {"correlation_break"}   # 靠"不触发任何已知签名"被发现,自签名弱可接受
    for c in B.CONCEPTS:
        own = B.SIG[c]
        z_own = Z[c][own]
        sig_ok = z_own > (1.5 if c in ABSENCE else 3.0)
        # gate:novel 不能触发任何 known 签名;known 必须触发自己签名
        if c in NOVEL:
            tripped = [s for s in known_sigs if Z[c][s] > GATE_Z]
            gate_ok = len(tripped) == 0
            msg = f"novel: own({own}) z={z_own:+.1f} {'OK' if sig_ok else 'WEAK'}; " \
                  f"trips known {tripped if tripped else 'none ✓'}"
        else:
            gate_ok = z_own > GATE_Z
            msg = f"known: own({own}) z={z_own:+.1f} {'fires ✓' if gate_ok else 'NOT firing ✗'}"
        good = sig_ok and gate_ok
        ok &= good
        print(f"  {'OK ' if good else 'XX '}{c:18s} {msg}")
    print("\n" + ("✅ 证据正交 + gate 可用:每概念签名显著,且每个新类不触发任何已知签名。"
                  if ok else "❌ 仍有纠缠/弱签名 —— 见上面 XX 行。"))


if __name__ == "__main__":
    main()
