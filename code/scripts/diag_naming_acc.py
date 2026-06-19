#!/usr/bin/env python3
"""诊断多新类闭环的瓶颈:gpt-4o-mini 在**干净正交证据**上做 top-1 命名的精度(混淆矩阵)。
若新类命名精度低 → 伪标签噪声大 → 检测器学不动(解释 38–64% 天花板)。
用法: sbatch scripts/diag_naming_acc.sh"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sigla_exp import ovbench as B  # noqa: E402


def main():
    rng = np.random.default_rng(1)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("no key"); return
    mu, sd = B.normal_stats(rng)
    n = 25
    print(f"top-1 命名精度(每概念 n={n});行=真值,列=LLM命名\n")
    conf = {}
    for c in B.CONCEPTS:
        cnt = collections.Counter()
        for _ in range(n):
            ev = B.evidence(B.make_window(c, rng))
            got = B.gpt_recognize_top1(ev, key, mu, sd)
            cnt[got if got and got != "__ERROR__" else "<none>"] += 1
        conf[c] = cnt
    cols = B.CONCEPTS + ["<none>"]
    print("%-20s" % "true\\pred" + "".join("%16s" % p[:15] for p in cols) + "   acc")
    for c in B.CONCEPTS:
        row = "%-20s" % c
        for p in cols:
            row += "%16d" % conf[c].get(p, 0)
        row += "   %.0f%%" % (100 * conf[c].get(c, 0) / n)
        print(row)
    novacc = np.mean([conf[c].get(c, 0) / n for c in ["variance_burst", "trend", "correlation_break"]])
    print(f"\n新类平均命名精度: {novacc:.0%}  (若低,则伪标签噪声是 38–64% 天花板主因)")


if __name__ == "__main__":
    main()
