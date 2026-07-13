#!/usr/bin/env python3
"""【真实多模态早预警:融合(不让 LLM 单挑,而是补全数值)】

用户洞察:LLM 单独做不如数值;正确用法是**晚期融合**——数值信号 + LLM/文本通道,只要 fused > numeric 就算加值。
任务/数据同 exp_mmd_earlywarning(Time-MMD Health_US ILI,事件前 H 周早预警,无泄漏)。

通道(对全部多模态点各算一次分):
  numeric    ILI 轨迹"水平+上升"信号分
  text       LLM **只看真实报告文本** → 风险 0-100(隔离文本独立贡献,与数值正交)
  llm_full   LLM 看数值+文本 → 风险(对照)
融合(等权 z-sum,无调参→不偷看):
  fuse_nt    z(numeric) + z(text)
  fuse_nf    z(numeric) + z(llm_full)
指标:VEW@FA 与 AUC,bootstrap 子采样出误差棒。**判据:fuse_* > numeric → LLM/文本带来互补信息(支持多模态)。**
用法: sbatch sota_compare/run_mmdfuse.sh
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sota_compare.exp_mmd_earlywarning as M   # 复用 load/build_points/recent_report/常量  # noqa: E402

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NBOOT = 3 if SMOKE else 200
TARGET_FA = M.TARGET_FA


def gpt_text_only(ctx, key, model="gpt-4o-mini"):
    """LLM 只看报告文本(不给数值)→ 未来是否流感爆发的风险 0-100。隔离文本通道。"""
    instr = ("You assess influenza early-warning risk from a public-health report ALONE (no numbers given). "
             f"Estimate the RISK (0-100) that influenza-like illness will SURGE in the next {M.H} weeks based "
             "only on what this report implies about emerging outbreak conditions. Use the full 0-100 range. "
             'Respond ONLY JSON {"risk": <0-100>}. No markdown.')
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": f"report: \"{ctx}\""}], "max_output_tokens": 40}
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


def zscore(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / (s + 1e-9)


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    ot, dates, rep = M.load()
    pts, thr = M.build_points(ot, dates, rep)
    if SMOKE:
        pts = pts[:40]
    labels = np.array([p[1] for p in pts])
    print(f"Time-MMD ILI 融合早预警  net_ok={net_ok} n={len(pts)} 正例率={labels.mean():.0%} H={M.H} FA={TARGET_FA}\n")

    # 对全部点各算一次分(避免按 seed 重复调 LLM)
    num = np.array([(ot[i] - thr) + (ot[i] - ot[i - 3]) for i, _, _ in pts])
    text = np.full(len(pts), np.nan); full = np.full(len(pts), np.nan)
    if net_ok:
        for j, (i, _, rtxt) in enumerate(pts):
            t = gpt_text_only(rtxt, key)
            f = M.gpt_risk(ot[i - M.K + 1:i + 1], rtxt, key)
            text[j] = np.nan if t is None else t
            full[j] = np.nan if f is None else f

    chans = {"numeric": num}
    if net_ok:
        chans["text"] = text
        chans["llm_full"] = full
        chans["fuse_nt(num+text)"] = zscore(num) + zscore(np.nan_to_num(text, nan=np.nanmean(text)))
        chans["fuse_nf(num+llm_full)"] = zscore(num) + zscore(np.nan_to_num(full, nan=np.nanmean(full)))

    def vew(score, idx):
        s, l = score[idx], labels[idx]
        if l.sum() == 0 or (l == 0).sum() == 0:
            return np.nan
        th = np.quantile(s[l == 0], 1 - TARGET_FA)
        return float(np.mean(s[l == 1] > th))

    def auc(score, idx):
        s, l = score[idx], labels[idx]
        return roc_auc_score(l, s) if l.sum() and (l == 0).sum() else np.nan

    rng = np.random.default_rng(0); N = len(pts)
    print(f"{'通道':22s}{'VEW@%d%%' % int(TARGET_FA*100):>14s}{'AUC':>14s}")
    print("-" * 50)
    base_v = []
    for name, sc in chans.items():
        vs = [vew(sc, rng.integers(0, N, N)) for _ in range(NBOOT)]
        au = [auc(sc, rng.integers(0, N, N)) for _ in range(NBOOT)]
        vm, vsd = np.nanmean(vs), np.nanstd(vs); am, asd = np.nanmean(au), np.nanstd(au)
        tag = "  ← baseline" if name == "numeric" else ""
        print(f"{name:22s}{vm*100:>9.0f}±{vsd*100:<3.0f}{am:>10.2f}±{asd:<4.2f}{tag}")
    print("\n判据:fuse_nt / fuse_nf 的 VEW 或 AUC > numeric → 加 LLM/文本带来互补信息(多模态有用,即便 LLM 单独更弱);")
    print("     若融合 ≤ numeric → 文本与数值冗余,加 LLM 不改善。")
    json.dump({"n": len(pts), "pos_rate": float(labels.mean())}, open(ROOT / "runs" / "mmd_fusion.json", "w"))


if __name__ == "__main__":
    main()
