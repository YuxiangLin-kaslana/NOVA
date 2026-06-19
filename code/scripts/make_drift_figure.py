#!/usr/bin/env python3
"""抗漂移主图 + 数值证据:在线适应 vs 冻结。

证明:漂移后,冻结 detector 的重建误差(对正常窗)与 FP 率随区间上升 → precision 退化;
持续在线适应把重建误差压住 → FP 下降 → precision 回升。

读取 run_drift.sh 产出的两臂逐窗 CSV(含 detector_score、label、regime),输出:
  1. 每区间 P/R/F1 + 正常窗 FP 率 + 正常窗平均重建误差(两臂对比)。
  2. 主图 runs/online/drift/drift_main.png:滑窗 FP 率 与 正常窗重建误差 随时间,
     标注漂移点,online vs frozen。

需 matplotlib(用 ragenv2 环境的 python 运行):
  /u/ylin30/.conda/envs/ragenv2/bin/python scripts/make_drift_figure.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

ROOT = Path(__file__).resolve().parents[1]
# optional argv[1] = stream stem (drift_gradual / drift_abrupt); default to legacy dir
_STEM = sys.argv[1] if len(sys.argv) > 1 else None
DRIFT = ROOT / "runs" / "online" / (f"drift_{_STEM}" if _STEM else "drift")


def drift_window_indices(regime: np.ndarray) -> list[int]:
    """漂移点的窗索引 = regime 值发生变化处(从数据导出,与 step 无关)。"""
    return [int(i) for i in range(1, len(regime)) if regime[i] != regime[i - 1]]


FIXED_Q = 0.95  # 固定阈值的分位:在 regime-0 正常窗分数上取该分位,部署后冻结不变


def load(name: str):
    rows = list(csv.DictReader(open(DRIFT / f"pred_{name}.csv")))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "regime": g("regime").astype(int), "label": g("label").astype(int),
        "score": g("detector_score"),
    }


def fixed_threshold(arm: dict) -> float:
    """在 regime-0 正常窗的重建分数上取分位作为固定阈值(部署即冻结)。

    两臂在 regime 0 起点相同,用 frozen 臂的 regime-0 正常分数定阈,对两臂一致施加,
    隔离掉自适应校准器的干扰——纯看 detector 适应对漂移段 FP 的影响。
    """
    m = (arm["regime"] == 0) & (arm["label"] == 0)
    return float(np.quantile(arm["score"][m], FIXED_Q))


def apply_threshold(arms: dict):
    thr = fixed_threshold(arms["frozen"])
    for a in arms.values():
        a["pred"] = (a["score"] > thr).astype(int)
    return thr


def prf(label, pred):
    tp = int(((label == 1) & (pred == 1)).sum()); fp = int(((label == 0) & (pred == 1)).sum())
    tn = int(((label == 0) & (pred == 0)).sum()); fn = int(((label == 1) & (pred == 0)).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F = 2 * P * R / max(1e-12, P + R)
    fpr = fp / max(1, fp + tn)  # 正常窗里被误报的比例
    return P, R, F, fpr, tp, fp, fn, tn


def per_regime_table(arms: dict):
    print("=" * 92)
    print("每区间对比 (regime 0 = 训练区间,1/2/3 = 漂移后)。late=区间后 50% 窗(适应稳态)")
    print("=" * 92)
    print(f"{'regime':>6} | {'arm':>12} | {'P':>5} {'R':>5} {'F1':>5} | {'FP率':>6} | {'late FP率':>8} | {'late重建误差':>11}")
    for r in range(4):
        for name, a in arms.items():
            m = a["regime"] == r
            P, R, F, fpr, tp, fp, fn, tn = prf(a["label"][m], a["pred"][m])
            # late = 区间后半段(适应应已收敛)
            idx = np.where(m)[0]
            late = np.zeros_like(m); late[idx[len(idx) // 2:]] = True
            _, _, _, lfpr, *_ = prf(a["label"][late], a["pred"][late])
            late_err = float(a["score"][late & (a["label"] == 0)].mean())
            print(f"{r:>6} | {name:>12} | {P:5.3f} {R:5.3f} {F:5.3f} | {fpr:5.1%} | {lfpr:7.1%} | {late_err:11.5f}")
        print("-" * 92)


def overall(arms: dict):
    print("整体:")
    for name, a in arms.items():
        P, R, F, fpr, tp, fp, fn, tn = prf(a["label"], a["pred"])
        print(f"  {name:>7}: P/R/F1 = {P:.3f}/{R:.3f}/{F:.3f}  FP率={fpr:.1%}  (tp={tp} fp={fp} fn={fn})")


def sliding(arr_label, arr_pred, arr_score, k=40):
    """滑窗 FP 率与正常窗平均重建误差。"""
    n = len(arr_label); fpr = np.full(n, np.nan); err = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - k // 2); hi = min(n, i + k // 2)
        lab = arr_label[lo:hi]; pr = arr_pred[lo:hi]; sc = arr_score[lo:hi]
        normal = lab == 0
        if normal.sum() > 0:
            fpr[i] = (pr[normal] == 1).mean()
            err[i] = sc[normal].mean()
    return fpr, err


def make_figure(arms: dict):
    if not HAVE_MPL:
        print("[warn] 无 matplotlib,跳过出图(数值表已打印)。用 ragenv2 的 python 重跑出图。")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    colors = {"online_track": "#1f77b4", "online_naive": "#ff7f0e", "frozen": "#d62728",
              "online": "#1f77b4"}
    for name, a in arms.items():
        fpr, err = sliding(a["label"], a["pred"], a["score"])
        x = np.arange(len(fpr))
        ax1.plot(x, err, label=name, color=colors.get(name), lw=2)
        ax2.plot(x, fpr, label=name, color=colors.get(name), lw=2)
    dwin = drift_window_indices(next(iter(arms.values()))["regime"])
    for ax in (ax1, ax2):
        for dp in dwin:
            ax.axvline(dp, color="gray", ls="--", lw=1, alpha=0.7)
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax1.set_ylabel("正常窗平均重建误差")
    ax1.set_title("抗漂移:在线适应 vs 冻结(虚线=漂移点)")
    ax2.set_ylabel("滑窗 FP 率(正常窗误报率)"); ax2.set_xlabel("窗索引(时间)")
    fig.tight_layout()
    out = DRIFT / "drift_main.png"
    fig.savefig(out, dpi=140); print(f"\nsaved figure -> {out}")


def main():
    names = [n for n in ("frozen", "online_naive", "online_track", "online") if (DRIFT / f"pred_{n}.csv").exists()]
    arms = {n: load(n) for n in names}
    thr = apply_threshold(arms)
    print(f"固定阈值(regime-0 正常窗 q{FIXED_Q})= {thr:.5f}，部署后冻结，对两臂一致施加。\n")
    overall(arms); print()
    per_regime_table(arms)
    make_figure(arms)


if __name__ == "__main__":
    main()
