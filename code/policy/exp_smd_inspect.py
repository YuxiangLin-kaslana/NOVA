#!/usr/bin/env python3
"""【新数据集 SMD:成本感知 inspect 的原生形式 inspect=调更多通道】

SMD(28机×38通道真实工业传感器)。inspect 的 SigLA 原生形式:平时只看**便宜子集**(方差最高 6 通道,常监控),
风险模糊时才**调取全部 38 通道**(花 λ_insp:带宽/算力/采集)。任务=异常检测(SMD onset少、前兆稀,故用检测;
同一 inspect 机制)。比 阈值(便宜) / 全量(always-full) / 成本感知(模糊才调全量),跨机器。
判据:成本感知在小λ下 utility 最高 + 只对一部分窗调全量 → inspect 框架在新数据集/原生通道形式上泛化。LLM 无关。
用法: python policy/exp_smd_inspect.py
"""
from __future__ import annotations
import pickle, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

D = Path("/u/ylin30/sigLA/data/ServerMachineDataset/preprocessed")
MACHINES = ["1-1", "1-6", "1-7", "2-1", "3-1", "3-7"]
K = 20                      # 窗长
N_CHEAP = 6                 # 便宜通道数(方差最高)
MAXPTS = 2500               # 每机器决策点(子采样)
FP_W, FN_W, LAM = 0.8, 1.0, 0.05
N_SPLIT = 6


def load(ent):
    tr = np.asarray(pickle.load(open(D / f"machine-{ent}_train.pkl", "rb")), dtype=np.float32)
    te = np.asarray(pickle.load(open(D / f"machine-{ent}_test.pkl", "rb")), dtype=np.float32)
    lb = np.asarray(pickle.load(open(D / f"machine-{ent}_test_label.pkl", "rb"))).reshape(-1).astype(int)
    cheap = np.argsort(tr.var(0))[::-1][:N_CHEAP]                 # 便宜=训练方差最高通道
    return te, lb, np.sort(cheap)


def feats(win, chans):
    seg = win[:, chans]
    return np.concatenate([seg.mean(0), seg.std(0), seg.max(0) - seg.min(0)])


def utility(alarm, n_ins, y, lam):
    tp = np.sum(alarm & (y == 1)); fp = np.sum(alarm & (y == 0)); fn = np.sum((~alarm) & (y == 1))
    return (tp - FP_W * fp - FN_W * fn - lam * n_ins) / len(y)


def costaware(pc, pf, lo, hi, tauf):
    al = np.zeros(len(pc), bool); n = 0
    for i in range(len(pc)):
        if pc[i] >= hi:
            al[i] = True
        elif pc[i] > lo:
            n += 1; al[i] = pf[i] > tauf
    return al, n


def run(ent, rng):
    te, lb, cheap = load(ent)
    allc = np.arange(te.shape[1])
    idx = [t for t in range(K, len(te)) if True]
    if len(idx) > MAXPTS:
        idx = sorted(rng.choice(idx, MAXPTS, replace=False))
    y = np.array([lb[t] for t in idx])
    if y.sum() < 20 or (1 - y).sum() < 20:
        return None
    Xc = np.array([feats(te[t - K + 1:t + 1], cheap) for t in idx])
    Xf = np.array([feats(te[t - K + 1:t + 1], allc) for t in idx])
    A = {k: [] for k in ("aucc", "aucf", "thr", "full", "cost", "rr")}
    for tr, ev in StratifiedShuffleSplit(N_SPLIT, test_size=0.4, random_state=0).split(Xc, y):
        scc = StandardScaler().fit(Xc[tr]); lrc = LogisticRegression(max_iter=1000).fit(scc.transform(Xc[tr]), y[tr])
        pc = lrc.predict_proba(scc.transform(Xc))[:, 1]
        scf = StandardScaler().fit(Xf[tr]); lrf = LogisticRegression(max_iter=1000).fit(scf.transform(Xf[tr]), y[tr])
        pf = lrf.predict_proba(scf.transform(Xf))[:, 1]
        tau = float(np.quantile(pc[tr][y[tr] == 0], 0.85)); tauf = float(np.quantile(pf[tr][y[tr] == 0], 0.85))
        best = max(((utility(*costaware(pc[tr], pf[tr], lo, hi, tauf), y[tr], LAM), lo, hi)
                    for lo in np.linspace(0.1, 0.4, 7) for hi in np.linspace(0.5, 0.9, 9)), key=lambda z: z[0])
        lo, hi = best[1], best[2]
        al_c, ins = costaware(pc[ev], pf[ev], lo, hi, tauf)
        A["aucc"].append(roc_auc_score(y[ev], pc[ev])); A["aucf"].append(roc_auc_score(y[ev], pf[ev]))
        A["thr"].append(utility(pc[ev] > tau, 0, y[ev], LAM))
        A["full"].append(utility(pf[ev] > tauf, len(ev), y[ev], LAM))
        A["cost"].append(utility(al_c, ins, y[ev], LAM)); A["rr"].append(ins / len(ev))
    return {k: float(np.mean(v)) for k, v in A.items()} | {"n": len(idx), "pos": float(y.mean())}


def main():
    rng = np.random.default_rng(0)
    print(f"SMD 成本感知 inspect=调更多通道(便宜{N_CHEAP}通道 vs 全38)  λ_insp={LAM}  LLM无关\n")
    print(f"{'machine':9s}{'n':>6s}{'异常%':>6s}{'AUC便宜':>9s}{'AUC全量':>9s}{'thr':>8s}{'full':>8s}{'cost':>8s}{'调全量率':>9s}{'赢家':>10s}")
    print("-" * 86)
    res = {}
    for m in MACHINES:
        r = run(m, rng)
        if r is None:
            print(f"{m:9s} (正例不足跳过)"); continue
        res[m] = r
        win = max([("thr", r["thr"]), ("full", r["full"]), ("cost", r["cost"])], key=lambda z: z[1])[0]
        print(f"{m:9s}{r['n']:>6d}{r['pos']*100:>5.0f}%{r['aucc']:>9.3f}{r['aucf']:>9.3f}{r['thr']:>8.3f}"
              f"{r['full']:>8.3f}{r['cost']:>8.3f}{r['rr']*100:>7.0f}%{win:>10s}")
    print("\n判据:成本感知 utility ≥ thr/full 且只对部分窗调全量 → inspect 框架在 SMD(新数据集,原生通道inspect)泛化成立。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/smd_inspect.json", "w"), indent=2)


if __name__ == "__main__":
    main()
