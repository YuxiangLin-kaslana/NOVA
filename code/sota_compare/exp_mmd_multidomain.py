#!/usr/bin/env python3
"""【Time-MMD 多领域:文本"正交增量"是否领域特异?】

Health/ILI 上文本冗余(CDC 报告复述 ILI)→ 无增量。但这可能领域特异——别的领域(野火→空气、新闻→经济/能源/交通)
文本也许带数值外的正交信息。跨多领域做同一"事件前早预警"任务 + 学习式融合,看 [num,text] 是否在某些领域 > [num]。

每领域:OT 按日期聚合 → 事件=未来H期超阈(q80)且当前未超 + 有对齐报告(≤t,无泄漏)。
通道:numeric(level/mean/slope/accel) / text(LLM 只看报告→风险)。5 折 CV 逻辑回归比 [num] vs [num,text] 留出 AUC。
判据:某领域 Δ(num+text − num) 显著 >0 → 该领域文本带正交增量(找到 LLM 加值的真实场景)。
用法: sbatch sota_compare/run_mmd_multi.sh
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = Path("/u/ylin30/sigLA/data/data/Time-MMD")
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
DOMAINS = os.environ.get("MMD_DOMAINS", "Climate,Environment,Energy,Economy,Traffic,Health_US").split(",")
K, H = 8, 6
LOOKBACK = 90                       # 报告对齐(天),跨周/月 cadence
MAXPTS = 40 if SMOKE else 220
META = {"Climate": "regional climate/weather index", "Environment": "air quality index",
        "Energy": "retail gasoline price", "Economy": "macroeconomic indicator",
        "Traffic": "traffic volume/congestion", "Health_US": "influenza-like illness %",
        "Health_AFR": "disease indicator", "SocialGood": "social indicator", "Agriculture": "agriculture index"}


def load_domain(dom):
    num = pd.read_csv(DATA / f"numerical/{dom}/{dom}.csv")
    num["dt"] = pd.to_datetime(num["end_date"], errors="coerce")
    g = num.dropna(subset=["dt"]).groupby("dt")["OT"].mean().sort_index()
    ot, dates = g.values.astype(float), g.index.values
    rep = pd.read_csv(DATA / f"textual/{dom}/{dom}_report.csv")
    rep["e"] = pd.to_datetime(rep["end_date"], errors="coerce")
    rep = rep[rep["fact"].notna() & (rep["fact"].astype(str).str.len() > 30)].dropna(subset=["e"]).sort_values("e")
    return ot, dates, rep


def recent_report(rep, day):
    win = rep[(rep["e"] <= day) & (rep["e"] >= day - np.timedelta64(LOOKBACK, "D"))]
    return str(win.iloc[-1]["fact"])[:700] if len(win) else None


def build_points(ot, dates, rep):
    thr = float(np.quantile(ot, 0.80)); pts = []
    for i in range(K, len(ot) - H):
        if ot[i] > thr:
            continue
        rtxt = recent_report(rep, dates[i])
        if rtxt is None:
            continue
        label = int(np.any(ot[i + 1:i + 1 + H] > thr))
        pts.append((i, label, rtxt))
    return pts, thr


def gpt_text_risk(text, dom, key, model="gpt-4o-mini"):
    metric = META.get(dom, "the monitored indicator")
    instr = (f"You assess early-warning risk for time-series of {metric} ({dom} domain). Based ONLY on this report "
             f"(no numbers given), estimate the RISK (0-100) that {metric} will SURGE abnormally in the next {H} "
             "periods, judging from emerging conditions/events the report implies. Use the full 0-100 range. "
             'Respond ONLY JSON {"risk": <0-100>}. No markdown.')
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": f"report: \"{text}\""}], "max_output_tokens": 40}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            return float(json.loads(txt[s:e + 1]).get("risk"))
        except Exception:
            continue
    return None


def oof_auc(X, y, n_rep=15):
    aucs = []
    for rs in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=rs); oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


def run_domain(dom, key, net_ok, rng):
    ot, dates, rep = load_domain(dom)
    pts, thr = build_points(ot, dates, rep)
    if len(pts) < 40 or sum(p[1] for p in pts) < 8:
        return None
    if len(pts) > MAXPTS:
        sel = sorted(rng.choice(len(pts), MAXPTS, replace=False)); pts = [pts[j] for j in sel]
    y = np.array([p[1] for p in pts])
    if y.sum() < 8 or (1 - y).sum() < 8:                  # 子采样后两类都需够,防 CV 退化
        return None
    feat = np.array([[ot[i], np.mean(ot[i - K + 1:i + 1]), ot[i] - ot[i - 3],
                      (ot[i] - ot[i - 1]) - (ot[i - 3] - ot[i - 4])] for i, _, _ in pts])
    text = np.array([gpt_text_risk(t, dom, key) if net_ok else np.nan for _, _, t in pts], float)
    text = np.nan_to_num(text, nan=np.nanmean(text)) if net_ok and np.isfinite(np.nanmean(text)) else np.zeros(len(pts))
    a_num, s_num = oof_auc(feat, y)
    a_nt, s_nt = (oof_auc(np.column_stack([feat, text]), y) if net_ok else (float("nan"), 0))
    a_t, _ = (oof_auc(text.reshape(-1, 1), y) if net_ok else (float("nan"), 0))
    return dict(n=len(pts), pos=float(y.mean()), num=(a_num, s_num), numtext=(a_nt, s_nt), textonly=a_t)


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    rng = np.random.default_rng(0)
    print(f"Time-MMD 多领域融合早预警  net_ok={net_ok} 领域={DOMAINS} H={H}\n")
    print(f"{'domain':12s}{'n':>5s}{'正例':>6s}{'[num]AUC':>12s}{'[num,text]AUC':>16s}{'Δ':>9s}{'text-only':>11s}")
    print("-" * 72)
    out = {}
    for d in DOMAINS:
        r = run_domain(d.strip(), key, net_ok, rng)
        if r is None:
            print(f"{d:12s} (点/正例不足,跳过)"); continue
        out[d] = r
        an, sn = r["num"]; ant, snt = r["numtext"]; delta = ant - an
        flag = "  ←文本加值!" if (net_ok and delta > 2 * max(sn, snt, 1e-6)) else ""
        print(f"{d:12s}{r['n']:>5d}{r['pos']*100:>5.0f}%{an:>9.3f}±{sn:<3.3f}"
              f"{(str(round(ant,3)) if net_ok else 'NA'):>13s}±{snt:<3.3f}{delta:>+8.3f}{(round(r['textonly'],3) if net_ok else 0):>11}{flag}")
    print("\n判据:某领域 Δ=[num,text]−[num] 显著>0(超 ~2×std)→ 该领域真实文本带正交增量(LLM 加值的真实场景);")
    print("     全部 Δ≈0 → Time-MMD 文本普遍与数值冗余(更强的一般性负结果)。")
    json.dump(out, open(Path("/u/ylin30/sigLA/code/runs/mmd_multidomain.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
