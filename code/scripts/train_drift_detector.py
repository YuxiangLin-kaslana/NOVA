#!/usr/bin/env python3
"""在合成漂移流的 regime-0 正常窗上训练一个重建自编码器(抗漂移实验的共同起点)。

为什么:SMD 预训练 detector 不迁移到合成流(异常分≈正常分,无判别力)。抗漂移实验
需要一个在 regime-0 正常分布上专门训练的 detector —— 它在 regime-0 工作良好,信号
漂移出 regime-0 分布后重建误差升高(正是 FP 机制)。frozen 臂保持它,online 臂继续适应。

只用 regime-0(漂移前)且 label=0 的窗训练;标签仅用于剔除异常窗,不进重建目标。
输出 checkpoint 兼容 run_online.py 的 load_detector({"detector": state_dict})。

用法(轻量,登录节点 CPU 即可):
  python scripts/train_drift_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sigla_exp.model import MLPAnomalyDetector  # noqa: E402

# optional argv[1] = stream stem (e.g. drift_gradual / drift_abrupt)
STEM = sys.argv[1] if len(sys.argv) > 1 else "drift_gradual"
STREAM = ROOT.parent / "specific_data" / "Online_training" / "streams" / f"{STEM}.npz"
OUT = ROOT / "runs" / f"drift_detector_{STEM}_regime0" / "checkpoint_best.pt"
WIN, STEP_TRAIN = 100, 5
DRIFT0 = 3000  # regime 0 = points [0, 3000)


def windows(x, y, lo, hi, step, normal_only):
    starts = np.arange(lo, hi - WIN + 1, step, dtype=np.int64)
    W, keep = [], []
    for s in starts:
        is_anom = bool(np.any(y[s:s + WIN] == 1))
        if normal_only and is_anom:
            continue
        W.append(x[s:s + WIN]); keep.append(s)
    return np.asarray(W, dtype=np.float32), np.asarray(keep)


def main():
    torch.manual_seed(0); np.random.seed(0)
    d = np.load(STREAM)
    x, y, regime = d["x"].astype(np.float32), d["y"].astype(np.int64), d["regime"].astype(np.int64)
    n_vars = x.shape[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, _ = windows(x, y, 0, DRIFT0, STEP_TRAIN, normal_only=True)
    print(f"训练集: regime-0 正常窗 {Xtr.shape}  device={dev}")

    det = MLPAnomalyDetector(WIN, n_vars, latent_dim=128, hidden_dim=128).to(dev)
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(dev)
    det.train()
    n = len(Xt); bs = 64
    for epoch in range(120):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            batch = Xt[idx]
            recon = det(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch+1:3d}  recon_mse={tot/n:.6f}")

    # ---- 验证:在整条流上看判别力 + 各 regime 正常窗误差 ---- #
    det.eval()
    @torch.no_grad()
    def score(W):
        out = []
        for i in range(0, len(W), 256):
            b = torch.from_numpy(W[i:i+256]).to(dev)
            r = det(b)
            out.append(torch.mean((r - b) ** 2, dim=(1, 2)).cpu().numpy())
        return np.concatenate(out)
    allW, starts = windows(x, y, 0, len(x), STEP_TRAIN, normal_only=False)
    sc = score(allW)
    lab = np.array([int(np.any(y[s:s+WIN] == 1)) for s in starts])
    reg = np.array([int(regime[s+WIN-1]) for s in starts])
    # ROC-AUC(整体判别力)
    order = np.argsort(sc); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(sc))
    pos = lab == 1; npos, nneg = int(pos.sum()), int((~pos).sum())
    auc = (ranks[pos].sum() - npos*(npos-1)/2) / (npos*nneg)
    print(f"\n整条流 ROC-AUC = {auc:.3f}  (异常窗={npos})")
    print(f"  正常窗分: mean={sc[~pos].mean():.5f}  异常窗分: mean={sc[pos].mean():.5f}")
    print("  各 regime 正常窗平均重建误差(frozen 视角,漂移应使其升高):")
    for r in range(4):
        m = (reg == r) & (~pos)
        print(f"    regime {r}: {sc[m].mean():.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"detector": det.state_dict(),
                "args": {"win_size": WIN, "n_vars": n_vars, "latent_dim": 128, "hidden_dim": 128}}, OUT)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
