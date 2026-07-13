#!/usr/bin/env python3
"""确证诊断:ours 长期漂移高误报 = 循环内"自我毒化"(良性窗触发门控→LLM命名→长类+重训污染 normal)?
在**纯正常+漂移(无新异常)**流上跑 EXP.ours_loop,对比 net_ok=False(关LLM,不会长类)vs True(开LLM)。
若关 LLM FA≈5%、开 LLM FA飙升且词表增长 → 坐实自我毒化。env REAL_MACHINE 选背景。用法 sbatch sota_compare/diag_drift2.sh
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.exp_detection_tie as DT          # noqa: E402
import sigla_exp.ovbench as CB                  # noqa: E402
import sota_compare.exp_drift_vs_novel as EXP   # noqa: E402

device = DT.device
REAL = os.environ.get("REAL_MACHINE", "")


def build_drift_only(rng, dvec, n_warm=200, n_drift=600):
    W = [CB.base_normal(rng) for _ in range(n_warm)]
    for i in range(n_drift):
        frac = (i + 1) / n_drift
        W.append(CB.base_normal(rng) + frac * EXP.DRIFT_D * dvec)
    return W, n_warm


def main():
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    else:
        bg = "synthetic"
    key = os.environ.get("OPENAI_API_KEY", "")
    rng = np.random.default_rng(0); torch.manual_seed(0)
    mu, sd = CB.normal_stats(rng)
    dvec = rng.normal(0, 1, CB.NVARS).astype(np.float32)
    dvec /= (np.linalg.norm(dvec) / np.sqrt(CB.NVARS) + 1e-6)

    det = DT.make_detector(len(EXP.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(2400):
        c = EXP.BASE_VOCAB[rng.integers(len(EXP.BASE_VOCAB))]
        base = CB.base_normal(rng)
        x = base if c == EXP.NORMAL else EXP.inject(c, base, rng, float(rng.uniform(0.5, 1.0)))
        Xpt.append(EXP.mc(x)); Ypt.append(DT.onehot(EXP.BASE_VOCAB.index(c), len(EXP.BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)
    import copy
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))
    cal_s = 1.0 - DT.proba(det, [EXP.mc(CB.base_normal(rng)) for _ in range(400)])[:, 0]
    thr = float(np.quantile(cal_s, 0.95))

    W, n_warm = build_drift_only(rng, dvec)
    print(f"bg={bg} thr_ours={thr:.4f}  流: {n_warm} warm + {len(W)-n_warm} drift(全正常,无新异常)\n")
    for net_ok, tag in [(False, "关LLM"), (bool(key), "开LLM")]:
        a, names, vocab = EXP.ours_loop(pre_state, replay, W, mu, sd, key, net_ok, thr)
        a = np.array(a)
        grew = [v for v in vocab if v not in EXP.BASE_VOCAB]
        # 误报随时间(前/后半)
        fa_all = a.mean(); fa_warm = a[:n_warm].mean(); fa_drift = a[n_warm:].mean()
        fa_late = a[-200:].mean()
        print(f"[{tag}] FA 全程={fa_all:.0%} warm={fa_warm:.0%} drift段={fa_drift:.0%} 末200窗={fa_late:.0%} "
              f"| 词表增长={grew if grew else '无'}")
    print("\n判读:关LLM≈5%(纯漂移无碍)而开LLM飙升+词表增长 → 良性窗触发门控被LLM命名→长类重训毒化 normal;")
    print("即'长期漂移高误报'的真因不是漂移,而是开放词表闭环在长流上的**门控假阳性自我毒化**(真实数据更甚)。")


if __name__ == "__main__":
    main()
