#!/usr/bin/env python3
"""诊断:ours 在长期漂移段误报为何高?对比预训练 CNN 的 anomaly_score=1-P(normal) 分布:
  A 无漂移正常(mc)          —— 标定基准,thr=q95(A)
  B 漂移 frac=0.5(mc)       —— 应≈A(mc 抵消恒定偏移)
  C 漂移 frac=1.0(mc)       —— 应≈A
  D 漂移 frac=1.0(不做 mc)  —— 对照:不归一化会怎样
报 mean/q50/q95 与 FA=mean(score>thr)。若 B/C≈A 且 FA≈5% → 高误报来自循环内 vocab 增长/重训污染(另查);
若 B/C≫A → mc 没抵消掉漂移(真 bug)。env REAL_MACHINE 选背景(默认合成)。用法 sbatch sota_compare/diag_drift.sh
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
import sota_compare.exp_drift_vs_novel as EXP   # 复用 mc/inject/BASE_VOCAB/常量  # noqa: E402

device = DT.device
REAL = os.environ.get("REAL_MACHINE", "")


def main():
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    else:
        bg = "synthetic"
    rng = np.random.default_rng(0); torch.manual_seed(0)
    dvec = rng.normal(0, 1, CB.NVARS).astype(np.float32)
    dvec /= (np.linalg.norm(dvec) / np.sqrt(CB.NVARS) + 1e-6)

    # 预训练 ours 的 CNN(与实验同口径:mc 后的 normal+3已知)
    det = DT.make_detector(len(EXP.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(2400):
        c = EXP.BASE_VOCAB[rng.integers(len(EXP.BASE_VOCAB))]
        base = CB.base_normal(rng)
        x = base if c == EXP.NORMAL else EXP.inject(c, base, rng, float(rng.uniform(0.5, 1.0)))
        Xpt.append(EXP.mc(x)); Ypt.append(DT.onehot(EXP.BASE_VOCAB.index(c), len(EXP.BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)

    def score(ws):                                  # 1 - P(normal)
        return 1.0 - DT.proba(det, ws)[:, 0]

    A = score([EXP.mc(CB.base_normal(rng)) for _ in range(400)])
    B = score([EXP.mc(CB.base_normal(rng) + 0.5 * EXP.DRIFT_D * dvec) for _ in range(400)])
    C = score([EXP.mc(CB.base_normal(rng) + 1.0 * EXP.DRIFT_D * dvec) for _ in range(400)])
    D = score([(CB.base_normal(rng) + 1.0 * EXP.DRIFT_D * dvec) for _ in range(400)])   # 不做 mc
    thr = float(np.quantile(A, 0.95))

    # 同时看 argmax 类别分布(漂移窗被判成什么)
    Cw = [EXP.mc(CB.base_normal(rng) + 1.0 * EXP.DRIFT_D * dvec) for _ in range(400)]
    pred_idx = np.argmax(DT.proba(det, Cw), 1)
    dist = {EXP.BASE_VOCAB[i]: int((pred_idx == i).sum()) for i in range(len(EXP.BASE_VOCAB))}

    print(f"bg={bg}  thr=q95(A)={thr:.4f}\n")
    print(f"{'set':28s}{'mean':>8s}{'q50':>8s}{'q95':>8s}{'FA(>thr)':>10s}")
    for nm, S in [("A 无漂移(mc)", A), ("B 漂移0.5(mc)", B), ("C 漂移1.0(mc)", C), ("D 漂移1.0(无mc)", D)]:
        print(f"{nm:28s}{S.mean():>8.3f}{np.quantile(S,.5):>8.3f}{np.quantile(S,.95):>8.3f}{np.mean(S>thr)*100:>9.0f}%")
    print(f"\n漂移1.0(mc) 的 argmax 类别分布: {dist}")
    print("\n判读:B/C 的 FA 若≈5% → mc 有效,实验里的高误报来自循环内增长/重训污染(需另查);")
    print("     B/C 的 FA 若高、且 mean≫A → mc 没抵消漂移(真 bug);看 argmax 分布知漂移被误判成哪个类。")


if __name__ == "__main__":
    main()
