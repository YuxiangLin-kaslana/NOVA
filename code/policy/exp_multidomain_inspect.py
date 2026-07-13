#!/usr/bin/env python3
"""【成本感知 inspect 的自适应性:跨领域 graceful degradation】

论点:成本感知策略**自己判断文本值不值得读**——文本正交的领域(Environment)选择性读文本并获胜;
文本冗余的领域(ILI 等)读取率自动→低、退化为 thr-num(不亏)。统一"文本何时有用"+"成本感知动作",LLM 无关(TF-IDF)。

每领域:OT 事件前早预警点(复用 exp_mmd_multidomain),数值特征 + TF-IDF 文本。报:
  AUC_num / AUC_fuse(文本正交度)、thr/fuse/cost utility、文本读取率、赢家(λ_insp=0.05,8 次划分)。
用法: sbatch policy/run_mdinspect.sh  (纯 sklearn,无 LLM)
"""
from __future__ import annotations
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/u/ylin30/sigLA/code")
import sota_compare.exp_mmd_multidomain as MD

DOMAINS = ["Environment", "Health_US", "Climate", "Energy", "Economy", "Traffic"]
FP_W, FN_W = 0.8, 1.0
N_SPLIT, LAM = 8, 0.05


def feats(ot, i, K=8):
    seg = ot[i - K + 1:i + 1]
    return [ot[i], seg.mean(), ot[i] - ot[i - 3], (ot[i] - ot[i - 1]) - (ot[i - 3] - ot[i - 4]), seg.max(), seg.std()]


def utility(alarm, n_ins, y, lam):
    tp = np.sum(alarm & (y == 1)); fp = np.sum(alarm & (y == 0)); fn = np.sum((~alarm) & (y == 1))
    return (tp - FP_W * fp - FN_W * fn - lam * n_ins) / len(y)


def costaware(pn, pf, lo, hi, tauf):
    alarm = np.zeros(len(pn), bool); n = 0
    for i in range(len(pn)):
        if pn[i] >= hi:
            alarm[i] = True
        elif pn[i] > lo:
            n += 1; alarm[i] = pf[i] > tauf
    return alarm, n


def run_domain(dom):
    ot, dates, rep = MD.load_domain(dom)
    pts, thr = MD.build_points(ot, dates, rep)
    if len(pts) < 60 or sum(p[1] for p in pts) < 15:
        return None
    num = np.array([feats(ot, i) for i, _, _ in pts], float)
    texts = [t for _, _, t in pts]; y = np.array([p[1] for p in pts])
    A = {k: [] for k in ("aucn", "aucf", "thr", "fuse", "cost", "rr")}
    for tr, te in StratifiedShuffleSplit(N_SPLIT, test_size=0.4, random_state=0).split(num, y):
        scn = StandardScaler().fit(num[tr]); lrn = LogisticRegression(max_iter=1000).fit(scn.transform(num[tr]), y[tr])
        pn = lrn.predict_proba(scn.transform(num))[:, 1]
        vec = TfidfVectorizer(max_features=150, stop_words="english", ngram_range=(1, 2)).fit([texts[j] for j in tr])
        Xtf = vec.transform(texts).toarray(); fused = np.column_stack([num, Xtf])
        scf = StandardScaler().fit(fused[tr]); lrf = LogisticRegression(max_iter=2000, C=0.5).fit(scf.transform(fused[tr]), y[tr])
        pf = lrf.predict_proba(scf.transform(fused))[:, 1]
        tau = float(np.quantile(pn[tr][y[tr] == 0], 0.80)); tauf = float(np.quantile(pf[tr][y[tr] == 0], 0.80))
        best = max(((utility(*costaware(pn[tr], pf[tr], lo, hi, tauf), y[tr], LAM), lo, hi)
                    for lo in np.linspace(0.1, 0.4, 7) for hi in np.linspace(0.5, 0.85, 8)), key=lambda z: z[0])
        lo, hi = best[1], best[2]
        al_c, ins_c = costaware(pn[te], pf[te], lo, hi, tauf)
        A["aucn"].append(roc_auc_score(y[te], pn[te])); A["aucf"].append(roc_auc_score(y[te], pf[te]))
        A["thr"].append(utility(pn[te] > tau, 0, y[te], LAM))
        A["fuse"].append(utility(pf[te] > tauf, len(te), y[te], LAM))
        A["cost"].append(utility(al_c, ins_c, y[te], LAM)); A["rr"].append(ins_c / len(te))
    return {k: float(np.mean(v)) for k, v in A.items()} | {"n": len(pts), "pos": float(y.mean())}


def main():
    print(f"成本感知 inspect 自适应(跨领域,λ_insp={LAM},LLM无关)\n")
    print(f"{'domain':12s}{'n':>5s}{'AUCnum':>8s}{'AUCfuse':>9s}{'Δauc':>7s}{'thr':>8s}{'fuse':>8s}{'cost':>8s}{'文本读取':>9s}{'赢家':>11s}")
    print("-" * 86)
    res = {}
    for d in DOMAINS:
        r = run_domain(d)
        if r is None:
            print(f"{d:12s} (点不足跳过)"); continue
        res[d] = r; dauc = r["aucf"] - r["aucn"]
        win = max([("thr", r["thr"]), ("fuse", r["fuse"]), ("cost", r["cost"])], key=lambda z: z[1])[0]
        tag = "  文本正交" if dauc > 0.02 else ("  文本冗余" if dauc < 0.01 else "")
        print(f"{d:12s}{r['n']:>5d}{r['aucn']:>8.3f}{r['aucf']:>9.3f}{dauc:>+7.3f}{r['thr']:>8.3f}{r['fuse']:>8.3f}"
              f"{r['cost']:>8.3f}{r['rr']*100:>7.0f}%{win:>11s}{tag}")
    print("\n判据(graceful degradation):文本正交领域 cost 赢且读取率高;文本冗余领域 cost≈thr、读取率自动→低")
    print("→ 策略自适应判断'文本值不值得读',只在有用时付 inspect 成本。LLM 无关。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/multidomain_inspect.json", "w"), indent=2)


if __name__ == "__main__":
    main()
