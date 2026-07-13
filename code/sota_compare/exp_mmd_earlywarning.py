#!/usr/bin/env python3
"""【真实多模态早预警 pilot:Time-MMD / Health_US(流感 ILI)】

回答"真实文本上下文是否在数值之外还加信息、帮早预警"——这是命名/合成 EW 之后,唯一可能让 LLM 发光的设置。
任务:在**事件发生前**,用近 K 周 ILI 轨迹 [+ ≤当周的真实报告文本] 预测未来 [t+1,t+H] 周内是否流感爆发(ILI 超阈)。
三臂同口径(matched FA):
  numeric    近 K 周 ILI 的"水平+上升"信号分(纯数值早预警基线)
  LLM−text   LLM 看 K 周 ILI 数值 → 风险 0-100
  LLM+text   LLM 看 K 周 ILI 数值 + 最近的真实报告 fact(≤当周,无泄漏)→ 风险 0-100
关键消融:LLM+text vs LLM−text 隔离"真实文本增量";若 ≫ 且 LLM−text≈numeric → 真实文本带来信息优势(支持 SigLA);
若 ≈ → 真实文本只是复述数值,LLM 无增量(诚实负结果)。无泄漏:只用 ≤决策周 的数值与报告。
用法: sbatch sota_compare/run_mmd.sh  (env CMP_NSEED 子采样次数, CMP_SMOKE)
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/u/ylin30/sigLA/data/data/Time-MMD")
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
K = 8                       # 历史窗(周)
H = 6                       # 预警horizon(未来H周内是否爆发)
REP_LOOKBACK = 28          # 报告对齐:取决策周前 REP_LOOKBACK 天内最近的非空 fact
TARGET_FA = 0.20
MAXPTS = 60 if SMOKE else 250   # 决策点数(控 LLM 调用量)


def load():
    num = pd.read_csv(DATA / "numerical/Health_US/Health_US.csv")
    num["d"] = pd.to_datetime(num["end_date"]); num = num.sort_values("d").reset_index(drop=True)
    ot = num["OT"].astype(float).values
    dates = num["d"].values
    rep = pd.read_csv(DATA / "textual/Health_US/Health_US_report.csv")
    rep["e"] = pd.to_datetime(rep["end_date"])
    rep = rep[rep["fact"].notna() & (rep["fact"].astype(str).str.len() > 30)].sort_values("e").reset_index(drop=True)
    return ot, dates, rep


def recent_report(rep, day):
    """决策周 day 之前 REP_LOOKBACK 天内最近的非空报告 fact(无泄漏)。"""
    win = rep[(rep["e"] <= day) & (rep["e"] >= day - np.timedelta64(REP_LOOKBACK, "D"))]
    if len(win) == 0:
        return None
    return str(win.iloc[-1]["fact"])[:700]


def gpt_risk(traj, ctx, key, model="gpt-4o-mini"):
    tr = [round(float(x), 2) for x in traj]
    base = (
        "You forecast influenza early warning. Given recent WEEKLY influenza-like-illness (ILI) percentages, "
        f"estimate the RISK (0-100) that ILI will SURGE (rise sharply above its recent baseline) within the next "
        f"{H} weeks. A sustained upward trend signals an emerging outbreak. Use the full 0-100 range. "
    )
    ctx_part = (f"Recent public-health report: \"{ctx}\". Weigh this together with the numbers. " if ctx else "")
    instr = base + ctx_part + 'Respond ONLY JSON {"risk": <0-100>}. No markdown.'
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": f"recent weekly ILI %: {json.dumps(tr)}"}],
               "max_output_tokens": 40}
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


def build_points(ot, dates, rep):
    thr = float(np.quantile(ot, 0.75))
    pts = []                                                    # (i, label, has_report_day)
    for i in range(K, len(ot) - H):
        if ot[i] > thr:                                        # 已在爆发中 → 跳过(只做"事件前"早预警)
            continue
        day = dates[i]
        rtxt = recent_report(rep, day)
        if rtxt is None:                                       # 只保留有对齐报告的多模态点
            continue
        label = int(np.any(ot[i + 1:i + 1 + H] > thr))         # 未来 H 周是否爆发
        pts.append((i, label, rtxt))
    return pts, thr


def run(seed, key, net_ok):
    ot, dates, rep = load()
    pts, thr = build_points(ot, dates, rep)
    rng = np.random.default_rng(seed)
    if len(pts) > MAXPTS:
        sel = rng.choice(len(pts), MAXPTS, replace=False); pts = [pts[j] for j in sorted(sel)]
    labels = np.array([p[1] for p in pts])
    # numeric 信号分:近 K 周水平 + 上升幅度
    def numsig(i):
        seg = ot[i - K + 1:i + 1]
        return (ot[i] - thr) + (ot[i] - ot[i - 3])             # 当前相对阈 + 近3周上升
    num_s = np.array([numsig(i) for i, _, _ in pts])
    out = {"n": len(pts), "pos_rate": float(labels.mean())}
    out["numeric"] = vew_at_fa(num_s[labels == 1], num_s[labels == 0], TARGET_FA)
    if net_ok:
        for tag, use in [("llm_notext", False), ("llm_text", True)]:
            sc = []
            for i, _, rtxt in pts:
                traj = ot[i - K + 1:i + 1]
                sc.append(gpt_risk(traj, rtxt if use else None, key))
            sc = np.array([np.nan if x is None else x for x in sc])
            ok = ~np.isnan(sc)
            out[tag] = vew_at_fa(sc[(labels == 1) & ok], sc[(labels == 0) & ok], TARGET_FA)
    return out


def vew_at_fa(s_pos, s_neg, fa):
    if len(s_pos) == 0 or len(s_neg) == 0:
        return float("nan")
    thr = float(np.quantile(s_neg, 1.0 - fa))
    return float(np.mean(np.asarray(s_pos) > thr))


def ms(xs):
    a = np.array(xs, float); return np.nanmean(a), np.nanstd(a)


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "4"))
    print(f"Time-MMD Health_US(ILI) 早预警  net_ok={net_ok} K={K} H={H} targetFA={TARGET_FA} MAXPTS={MAXPTS}\n")
    res = [run(s, key, net_ok) for s in range(NSEED)]
    print(f"决策点 n≈{res[0]['n']}  正例率(未来{H}周爆发)≈{res[0]['pos_rate']:.0%}\n")
    print(f"{'方法':16s}{'早预警召回 @FA=%d%%' % int(TARGET_FA*100):>20s}")
    print("-" * 38)
    rows = [("numeric", "numeric(纯数值)")]
    if net_ok:
        rows += [("llm_notext", "LLM−text"), ("llm_text", "LLM+text(真实报告)")]
    for tag, nm in rows:
        v = ms([r[tag] for r in res])
        print(f"{nm:16s}{v[0]*100:>14.0f}±{v[1]*100:<3.0f}")
    print("\n判读:LLM+text ≫ LLM−text≈numeric → 真实文本在数值之外带来信息优势(支持 SigLA 语义轴);")
    print("     LLM+text ≈ LLM−text → 真实报告只是复述数值,LLM 无增量(诚实负结果)。")
    json.dump(dict(nseed=NSEED, per_seed=res), open(ROOT / "runs" / "mmd_earlywarning.json", "w"), indent=2)


if __name__ == "__main__":
    main()
