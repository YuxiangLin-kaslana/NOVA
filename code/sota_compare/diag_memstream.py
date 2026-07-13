#!/usr/bin/env python3
"""MemStream 为何 F1≈0.05?诊断:比较两种异常分数在 normal / 各已知异常 / novel 上的分离度。
  A) 嵌入到记忆最近邻 L2 距离(当前实现)
  B) AE 重构误差(标准化输入空间)
对每个概念报:分数均值、以及在 5% FAR(normal q95)阈下的检出召回。看哪个分数能分开异常。
另查:在线更新是否把异常吸进记忆(对比 update on/off 的 normal 阈漂移)。
用法: sbatch sota_compare/diag.sh
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigla_exp.ovbench as CB                  # noqa: E402
from sota_compare.baselines import MemStream    # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    rng = np.random.default_rng(0)
    m = MemStream(CB.WIN, CB.NVARS, device, seed=0)
    m.fit([CB.make_window(None, rng) for _ in range(1500)])

    @torch.no_grad()
    def recon_err(ws):
        Xf = (m._flat(ws) - m.mu) / m.sd
        rec, _ = m.ae(Xf)
        return ((rec - Xf) ** 2).mean(1).cpu().numpy()

    def emb_dist(ws):
        return m.score_stream(ws, update=False)

    normal = [CB.make_window(None, rng) for _ in range(400)]
    nd_emb, nd_rec = emb_dist(normal), recon_err(normal)
    th_emb, th_rec = np.quantile(nd_emb, 0.95), np.quantile(nd_rec, 0.95)
    print(f"device={device}  emb β(memupd)={m.beta:.3f}")
    print(f"normal: emb_dist mean={nd_emb.mean():.3f} q95={th_emb:.3f} | recon mean={nd_rec.mean():.4f} q95={th_rec:.4f}\n")
    print(f"{'concept':18s}{'emb_dist mean':>14s}{'recall@5%FAR':>14s}{'recon mean':>14s}{'recall@5%FAR':>14s}")
    for c in CB.CONCEPTS:
        ws = [CB.make_window(c, rng) for _ in range(200)]
        e, r = emb_dist(ws), recon_err(ws)
        print(f"{c:18s}{e.mean():>14.3f}{np.mean(e > th_emb)*100:>12.0f}%{r.mean():>14.4f}{np.mean(r > th_rec)*100:>12.0f}%")
    print("\n判读:若 recon 列召回 >> emb_dist 列 → 分数应改用重构误差(嵌入距离对这些注入异常不敏感)。")


if __name__ == "__main__":
    main()
