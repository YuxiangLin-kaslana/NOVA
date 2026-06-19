#!/usr/bin/env python3
"""诊断 exp_openvocab_multi 的两处异常:
  (1) warm-up(仅已知类)期间词表就增长 → 已知类窗被误判 suspect_novel。
  (2) variance_burst 进了词表却学不会(~1%) → 可能 LLM 命名混淆。
对每个概念各采样若干窗,报告:suspect_novel 触发率、主导偏离统计量分布、LLM 命名分布。
纯 numpy + 少量 LLM 探针。用法: sbatch scripts/diag_gate_naming.sh
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.exp_novel_concept as NC  # noqa: E402

KNOWN_STATS = {"max_abs_zscore", "max_step_change", "high_freq_energy_frac"}
ALL = ["spike", "level_shift", "oscillation", "variance_burst", "trend", "correlation_break"]


def main():
    rng = np.random.default_rng(0)
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    norm_evs = [NC.evidence(NC.make_window(None, rng)) for _ in range(300)]
    keys = list(norm_evs[0])
    mu = {k: float(np.mean([e[k] for e in norm_evs])) for k in keys}
    sd = {k: float(np.std([e[k] for e in norm_evs]) + 1e-6) for k in keys}

    def dom_stat(ev):
        devz = {k: abs((ev[k] - mu[k]) / sd[k]) for k in keys}
        return max(devz, key=devz.get), devz

    print(f"net_ok={net_ok}  KNOWN_STATS={sorted(KNOWN_STATS)}")
    print("normal mu:", {k: round(mu[k], 3) for k in keys})
    print("normal sd:", {k: round(sd[k], 3) for k in keys})
    print("\n%-18s %-8s %-26s %s" % ("concept", "susp率", "主导统计量分布", "LLM命名分布(n=20)"))
    print("-" * 110)
    for c in ALL:
        wins = [NC.make_window(c, rng) for _ in range(100)]
        doms = collections.Counter()
        susp = 0
        for x in wins:
            ev = NC.evidence(x)
            d, _ = dom_stat(ev)
            doms[d] += 1
            if d not in KNOWN_STATS:
                susp += 1
        # LLM 命名探针(20 窗)
        names = collections.Counter()
        if net_ok:
            for x in wins[:20]:
                got = NC.gpt_recognize(NC.evidence(x), key, mu)
                got = [g for g in got if g != "__ERROR__"]
                names[tuple(sorted(got)) if got else ("<empty>",)] += 1
        dom_str = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in doms.most_common())
        name_str = "; ".join(f"{'+'.join(k)}×{v}" for k, v in names.most_common(4))
        print("%-18s %-8s %-26s %s" % (c, f"{susp/100:.0%}", dom_str, name_str))


if __name__ == "__main__":
    main()
