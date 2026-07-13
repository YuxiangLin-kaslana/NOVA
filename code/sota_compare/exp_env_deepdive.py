#!/usr/bin/env python3
"""【深挖 Environment 正例:文本正交增量是否真实、是不是野火机制、能否做大】

多领域跑出唯一正例:Environment(空气质量)Δ(num+text−num)=+0.021,text-only AUC 0.72,疑因野火新闻=AQI 前瞻外生信号。
本实验严格验证 4 问:
  Q1 稳健性:全部对齐点 + 更丰富数值特征 + 50 次重复 CV → Δ 的紧误差棒。
  Q2 机制:仅"野火关键词 flag"(wildfire/smoke/fire/haze…,无 LLM)是否也加值?→ 证实野火机制。
  Q3 时间:正例是否集中 2023 野火期?去掉 2023 后文本还加值吗?
  Q4 文本通道:TF-IDF(折内拟合防泄漏)vs LLM 标量,谁更强、能否把 Δ 做大。
用法: sbatch sota_compare/run_envdd.sh
"""
from __future__ import annotations
import json, os, sys, urllib.request, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("/u/ylin30/sigLA/data/data/Time-MMD")
K, H, LOOKBACK = 8, 6, 90
KW = ["wildfire", "smoke", "fire", "haze", "burning", "air quality alert", "advisory", "particulate", "pm2.5", "ozone"]


def load_env():
    num = pd.read_csv(DATA / "numerical/Environment/Environment.csv")
    num["dt"] = pd.to_datetime(num["end_date"], errors="coerce")
    g = num.dropna(subset=["dt"]).groupby("dt")["OT"].mean().sort_index()
    rep = pd.read_csv(DATA / "textual/Environment/Environment_report.csv")
    rep["e"] = pd.to_datetime(rep["end_date"], errors="coerce")
    rep = rep[rep["fact"].notna() & (rep["fact"].astype(str).str.len() > 30)].dropna(subset=["e"]).sort_values("e")
    return g.values.astype(float), g.index.values, rep


def recent_report(rep, day):
    w = rep[(rep["e"] <= day) & (rep["e"] >= day - np.timedelta64(LOOKBACK, "D"))]
    return str(w.iloc[-1]["fact"])[:800] if len(w) else None


def gpt_text_risk(text, key, model="gpt-4o-mini"):
    instr = ("You assess early-warning risk for air-quality (AQI). Based ONLY on this report (no numbers), estimate "
             f"the RISK (0-100) that AQI will SURGE to unhealthy levels in the next {H} weeks, judging from events "
             "the report implies (e.g., wildfires, smoke transport, pollution episodes). Use the full 0-100 range. "
             'Respond ONLY JSON {"risk": <0-100>}. No markdown.')
    payload = {"model": model, "instructions": instr, "input": [{"role": "user", "content": f"report: \"{text}\""}],
               "max_output_tokens": 40}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode())
            t = data.get("output_text")
            if not isinstance(t, str):
                t = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = t.find("{"), t.rfind("}")
            return float(json.loads(t[s:e + 1]).get("risk"))
        except Exception:
            continue
    return None


def build(ot, dates, rep):
    thr = float(np.quantile(ot, 0.80)); rows = []
    for i in range(K, len(ot) - H):
        if ot[i] > thr:
            continue
        rt = recent_report(rep, dates[i])
        if rt is None:
            continue
        seg = ot[i - K + 1:i + 1]
        feat = [ot[i], seg.mean(), ot[i] - ot[i - 3], (ot[i] - ot[i - 1]) - (ot[i - 3] - ot[i - 4]),
                seg.max(), seg.std()]
        label = int(np.any(ot[i + 1:i + 1 + H] > thr))
        kw = float(sum(k in rt.lower() for k in KW))
        rows.append((feat, rt, kw, label, pd.Timestamp(dates[i]).year))
    return rows, thr


def cv_auc(num, txtscore, kwarr, texts, y, use, n_rep=50, tfidf=False):
    """折内拟合 TF-IDF(防泄漏)。use=要拼的列名集合;返回 OOF AUC mean,std。"""
    aucs = []
    for rs in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=rs); oof = np.zeros(len(y))
        for tr, te in skf.split(num, y):
            cols_tr, cols_te = [num[tr]], [num[te]]
            if "llm" in use:
                cols_tr.append(txtscore[tr, None]); cols_te.append(txtscore[te, None])
            if "kw" in use:
                cols_tr.append(kwarr[tr, None]); cols_te.append(kwarr[te, None])
            if "tfidf" in use:
                vec = TfidfVectorizer(max_features=150, stop_words="english", ngram_range=(1, 2))
                Xt = vec.fit_transform([texts[j] for j in tr]).toarray()
                Xe = vec.transform([texts[j] for j in te]).toarray()
                cols_tr.append(Xt); cols_te.append(Xe)
            Xtr = np.column_stack(cols_tr); Xte = np.column_stack(cols_te)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), y[tr])
            oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key)
    ot, dates, rep = load_env()
    rows, thr = build(ot, dates, rep)
    num = np.array([r[0] for r in rows], float); texts = [r[1] for r in rows]
    kwarr = np.array([r[2] for r in rows], float); y = np.array([r[3] for r in rows]); yr = np.array([r[4] for r in rows])
    txt = np.array([gpt_text_risk(t, key) if net_ok else np.nan for t in texts], float)
    txt = np.nan_to_num(txt, nan=np.nanmean(txt)) if net_ok else np.zeros(len(rows))
    print(f"Environment 深挖  n={len(rows)} 正例率={y.mean():.0%}  野火关键词点占比={np.mean(kwarr>0):.0%}  "
          f"2023点占比={np.mean(yr==2023):.0%}\n")

    def show(tag, use):
        m, s = cv_auc(num, txt, kwarr, texts, y, use)
        base = main._base
        print(f"{tag:22s}AUC={m:.3f}±{s:.3f}" + ("" if use == set() or tag.startswith("[num]") and use == {"num"} else f"  Δvs[num]={m-base:+.3f}"))
        return m
    print("Q1/Q4 通道对比(50×5折CV,TF-IDF折内拟合):")
    main._base = cv_auc(num, txt, kwarr, texts, y, {"num"})[0]
    print(f"{'[num]':22s}AUC={main._base:.3f}  ← baseline")
    show("[num,llm]", {"num", "llm"})
    show("[num,kw]", {"num", "kw"})           # Q2 机制:纯野火关键词
    show("[num,tfidf]", {"num", "tfidf"})     # Q4 文本通道
    show("[num,llm,kw,tfidf]", {"num", "llm", "kw", "tfidf"})

    # Q3 时间:去掉 2023 野火期
    m = yr != 2023
    if m.sum() > 40 and y[m].sum() > 8:
        b2 = cv_auc(num[m], txt[m], kwarr[m], [texts[j] for j in np.where(m)[0]], y[m], {"num"})[0]
        t2 = cv_auc(num[m], txt[m], kwarr[m], [texts[j] for j in np.where(m)[0]], y[m], {"num", "llm", "kw", "tfidf"})[0]
        print(f"\nQ3 去掉2023野火期(n={m.sum()}):[num]={b2:.3f}  [num+全文本]={t2:.3f}  Δ={t2-b2:+.3f}")
    # 例子:文本分高且真爆发
    print("\n文本分最高的几个点(看是否野火→AQI):")
    for j in np.argsort(-txt)[:4]:
        print(f"  txt={txt[j]:.0f} kw={kwarr[j]:.0f} label={y[j]} yr={yr[j]} | {texts[j][:150]}")
    json.dump({"n": len(rows), "pos": float(y.mean())}, open("/u/ylin30/sigLA/code/runs/env_deepdive.json", "w"))


if __name__ == "__main__":
    main()
