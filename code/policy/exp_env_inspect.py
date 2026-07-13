#!/usr/bin/env python3
"""【真实数据:成本感知 inspect 策略】Time-MMD Environment(AQI 早预警),统一"文本正交价值"+"成本感知动作"。

便宜信号=AQI 数值(real surge 与 benign 波动在数值上混);inspect 动作=读报告文本(花 λ_insp,但 TF-IDF 文本
正交、AUC 0.78→0.92)。三策略(train 调参,test 评估,多次划分误差棒):
  threshold-num   只用数值:p_num>τ 报警(从不读文本)
  always-fuse     每点都读文本:p_fused>τ 报警(精度高但文本成本=100%)
  cost-aware      数值低→wait;数值高→直接 alarm;仅**数值模糊**时 inspect(读文本)再用 p_fused 决策
效用 = +tp − fp_w·fp − fn_w·fn − λ_insp·读文本次数。判据:成本感知在小λ_insp下 utility 最高,且只读一小部分文本即近 always-fuse 精度。
LLM 无关(文本=TF-IDF)。用法: python policy/exp_env_inspect.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, "/u/ylin30/sigLA/code")
import sota_compare.exp_env_deepdive as ED

FP_W, FN_W = 0.8, 1.0
N_SPLIT = 8


def utility(pred_alarm, n_inspect, y, lam):
    tp = np.sum(pred_alarm & (y == 1)); fp = np.sum(pred_alarm & (y == 0)); fn = np.sum((~pred_alarm) & (y == 1))
    return (tp * 1.0 - fp * FP_W - fn * FN_W - n_inspect * lam) / len(y)


def eval_split(num, texts, y, tr, te, lam_list):
    scn = StandardScaler().fit(num[tr])
    lrn = LogisticRegression(max_iter=1000).fit(scn.transform(num[tr]), y[tr])
    pnum = lrn.predict_proba(scn.transform(num))[:, 1]
    vec = TfidfVectorizer(max_features=150, stop_words="english", ngram_range=(1, 2)).fit([texts[j] for j in tr])
    Xtf = vec.transform(texts).toarray()
    fused = np.column_stack([num, Xtf])
    scf = StandardScaler().fit(fused[tr])
    lrf = LogisticRegression(max_iter=2000, C=0.5).fit(scf.transform(fused[tr]), y[tr])
    pfused = lrf.predict_proba(scf.transform(fused))[:, 1]

    # 在 train 上定操作阈值 τ(目标 FP≈20%);成本感知的模糊带 (lo,hi) 取 p_num 的分位
    tau = float(np.quantile(pnum[tr][y[tr] == 0], 0.80))
    tauf = float(np.quantile(pfused[tr][y[tr] == 0], 0.80))
    out = {}
    for lam in lam_list:
        # cost-aware:网格选 (lo,hi) 最大化 train utility
        best = None
        for lo in np.linspace(0.1, 0.4, 7):
            for hi in np.linspace(0.5, 0.85, 8):
                al, ins = costaware(pnum[tr], pfused[tr], lo, hi, tauf)
                u = utility(al, ins, y[tr], lam)
                if best is None or u > best[0]:
                    best = (u, lo, hi)
        _, lo, hi = best
        al_t = (pnum[te] > tau); u_t = utility(al_t, 0, y[te], lam)
        al_f = (pfused[te] > tauf); u_f = utility(al_f, len(te), y[te], lam)
        al_c, ins_c = costaware(pnum[te], pfused[te], lo, hi, tauf)
        u_c = utility(al_c, ins_c, y[te], lam)
        out[lam] = dict(thr=u_t, fuse=u_f, cost=u_c, readrate=ins_c / len(te))
    return out


def costaware(pn, pf, lo, hi, tauf):
    """数值低→wait;高→alarm;模糊带→inspect(读文本)用 pf 决策。返回 (alarm bool[], 读文本次数)。"""
    alarm = np.zeros(len(pn), bool); n_ins = 0
    for i in range(len(pn)):
        if pn[i] >= hi:
            alarm[i] = True
        elif pn[i] > lo:                      # 模糊 → inspect
            n_ins += 1
            alarm[i] = pf[i] > tauf
    return alarm, n_ins


def main():
    ot, dates, rep = ED.load_env()
    rows, thr = ED.build(ot, dates, rep)
    num = np.array([r[0] for r in rows], float)
    texts = [r[1] for r in rows]; y = np.array([r[3] for r in rows])
    print(f"Environment 成本感知 inspect(真实数据,LLM无关)  n={len(y)} 正例率={y.mean():.0%}\n")
    lam_list = [0.02, 0.05, 0.10, 0.20, 0.40]
    sss = StratifiedShuffleSplit(N_SPLIT, test_size=0.4, random_state=0)
    agg = {lam: {"thr": [], "fuse": [], "cost": [], "rr": []} for lam in lam_list}
    for tr, te in sss.split(num, y):
        o = eval_split(num, texts, y, tr, te, lam_list)
        for lam in lam_list:
            for k in ("thr", "fuse", "cost"):
                agg[lam][k].append(o[lam][k])
            agg[lam]["rr"].append(o[lam]["readrate"])
    print(f"{'λ_insp':>8s}{'thr-num':>11s}{'always-fuse':>14s}{'cost-aware':>13s}{'文本读取率':>12s}{'赢家':>12s}")
    print("-" * 70)
    res = {}
    for lam in lam_list:
        ut = np.mean(agg[lam]["thr"]); uf = np.mean(agg[lam]["fuse"]); uc = np.mean(agg[lam]["cost"])
        rr = np.mean(agg[lam]["rr"]); win = max([("thr-num", ut), ("always-fuse", uf), ("cost-aware", uc)], key=lambda z: z[1])[0]
        res[str(lam)] = dict(thr=ut, fuse=uf, cost=uc, readrate=rr)
        print(f"{lam:>8.2f}{ut:>11.3f}{uf:>14.3f}{uc:>13.3f}{rr*100:>10.0f}%{win:>13s}")
    print("\n判据:成本感知在小λ_insp下 utility ≥ always-fuse(只读一小部分文本→省成本)且 ≥ thr-num(借文本消歧)→")
    print("真实数据上验证'成本感知 inspect 动作'框架:选择性读文本兼得精度与低成本。LLM 无关(TF-IDF)。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/env_inspect.json", "w"), indent=2)


if __name__ == "__main__":
    main()
