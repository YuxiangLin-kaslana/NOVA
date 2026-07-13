#!/usr/bin/env python3
"""验证鲁棒闭环修复:
  测试1(纯正常+漂移,无真异常):原版 vs 鲁棒版 → 鲁棒版应 FA≈5%、词表不长(毒化消失)。
  测试2(含缓慢爬升真新异常 oscillation):鲁棒版 → 应仍漂移FA低 + 新异常被检+命名(本事没丢)。
env REAL_MACHINE 选背景。用法 sbatch sota_compare/run_robust_diag.sh
"""
from __future__ import annotations
import copy, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.exp_detection_tie as DT          # noqa: E402
import sigla_exp.ovbench as CB                  # noqa: E402
import sota_compare.exp_drift_vs_novel as EXP   # noqa: E402
from sota_compare.robust_loop import robust_ours_loop  # noqa: E402

device = DT.device
REAL = os.environ.get("REAL_MACHINE", "")


def pretrain(rng):
    det = DT.make_detector(len(EXP.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(2400):
        c = EXP.BASE_VOCAB[rng.integers(len(EXP.BASE_VOCAB))]
        base = CB.base_normal(rng)
        x = base if c == EXP.NORMAL else EXP.inject(c, base, rng, float(rng.uniform(0.5, 1.0)))
        Xpt.append(EXP.mc(x)); Ypt.append(DT.onehot(EXP.BASE_VOCAB.index(c), len(EXP.BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)
    pre = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))
    thr = float(np.quantile(1.0 - DT.proba(det, [EXP.mc(CB.base_normal(rng)) for _ in range(400)])[:, 0], 0.95))
    return pre, replay, thr


def main():
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    else:
        bg = "synthetic"
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    mu, sd = CB.normal_stats(rng)
    dvec = rng.normal(0, 1, CB.NVARS).astype(np.float32); dvec /= (np.linalg.norm(dvec) / np.sqrt(CB.NVARS) + 1e-6)
    pre, replay, thr = pretrain(rng)
    print(f"bg={bg} net_ok={net_ok} thr_ours={thr:.4f}\n")

    # ---- 测试1:纯正常+漂移(无真异常) ---- #
    W = [CB.base_normal(rng) for _ in range(200)]
    for i in range(600):
        W.append(CB.base_normal(rng) + (i + 1) / 600 * EXP.DRIFT_D * dvec)
    print("【测试1:纯正常+漂移,无真异常 → 期望 FA≈5%、词表不长】")
    for tag, fn in [("原版", EXP.ours_loop), ("鲁棒版", robust_ours_loop)]:
        res = fn(pre, replay, W, mu, sd, key, net_ok, thr)
        a, names, vocab = res[0], res[1], res[2]                  # 鲁棒版返回 4 元组(末位 stats)
        a = np.array(a); grew = [v for v in vocab if v not in EXP.BASE_VOCAB]
        print(f"  [{tag}] FA 全程={a.mean():.0%} 末200窗={a[-200:].mean():.0%} | 伪类增长={grew if grew else '无'}")

    # ---- 测试2:含缓慢爬升真新异常 oscillation(鲁棒版应仍检+命名) ---- #
    rng2 = np.random.default_rng(1)
    W2, lab, is_nov, phase, creep = EXP.build_stream(rng2, dvec)
    a, names, vocab, _ = robust_ours_loop(pre, replay, W2, mu, sd, key, net_ok, thr)
    drift_mask = (phase == "drift"); nov = (is_nov == 1)
    fa = float(a[drift_mask].mean()); rec = float(a[nov].mean())
    nameacc = float(np.mean([names[i] == EXP.NOVEL for i in range(len(W2)) if is_nov[i]]))
    cv = []
    for q in range(4):
        m = nov & (creep >= q / 4) & (creep < (q + 1) / 4 if q < 3 else creep <= 1.01)
        cv.append(round(float(a[m].mean()), 2) if m.any() else float("nan"))
    print(f"\n【测试2:鲁棒版 on 含真新异常流】漂移FA={fa:.0%} 新异常召回={rec:.0%} 命名={nameacc:.0%} "
          f"词表长全={'是' if EXP.NOVEL in vocab else '否'} 召回随爬升={cv}")
    print("\n判读:测试1 鲁棒版 FA≈5%且不长伪类 → 毒化修复;测试2 鲁棒版仍检出+命名真新异常 → 本事没丢。")


if __name__ == "__main__":
    main()
