#!/usr/bin/env python3
"""【一锤定音:LLM 没优势是"信号表示瓶颈"还是"信息不对称/LLM不擅数值"?】

命名任务(有明确正确答案=注入的概念),三臂:
  rule_feat  6 个证据 z-score 上 argmax→STAT_OF反查→概念(追平 LLM 的硬规则)
  llm_feat   LLM 看 6 个 z-score(我们之前的设置)
  llm_raw    LLM 看**原始窗口**(降采样多变量数值,无手工特征)→ 自己做信号识别

判读:
  llm_raw > rule_feat → 信号表示是瓶颈(给 LLM 原始信号它能超过手工特征规则)→ 出路=学习式编码器+LLM。
  llm_raw ≤ rule_feat(尤其远低)→ 印证 LLM 不擅原始数值(Tan et al.),根因=信息不对称,出路=真实语义上下文。
env REAL_MACHINE 选背景,CMP_NSEED 默认2。用法 sbatch sota_compare/run_bottleneck.sh
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigla_exp.ovbench as CB                  # noqa: E402

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "2"))
REAL = os.environ.get("REAL_MACHINE", "")
N_PER = 6 if SMOKE else 15
T_DS = 20                                        # 原始窗口时间降采样点数
STAT_TO_CONCEPT = {v: k for k, v in CB.STAT_OF.items()}


def rule_feat(ev, mu, sd, thresh=2.0):
    z = {k: (ev[k] - mu[k]) / (sd[k] + 1e-9) for k in mu}
    dom = max(z, key=z.get)
    return STAT_TO_CONCEPT.get(dom) if z[dom] >= thresh else None


def _gpt(instr, user, key, model="gpt-4o-mini"):
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": user}], "max_output_tokens": 60}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses",
                                         data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            c = json.loads(txt[s:e + 1]).get("concept")
            return c if c in CB.CONCEPTS else None
        except Exception:
            continue
    return "__ERROR__"


def llm_raw(x, key):
    """LLM 看降采样原始窗口(无手工特征),自己识别概念。"""
    idx = np.linspace(0, len(x) - 1, T_DS).astype(int)
    arr = np.round(x[idx], 1).tolist()                       # [T_DS, NVARS] 原始数值
    instr = (
        "You are given a multivariate time-series window: a list of timesteps, each a vector of channel values "
        "(downsampled, normalized). Identify which anomaly concept it exhibits by reading the raw shapes "
        "(isolated spikes, an abrupt persistent level shift, a high-frequency oscillation, a localized variance "
        "burst, a gradual linear trend, or a cross-channel correlation breakdown). Concepts:\n" +
        "\n".join(f"- {k}: {v}" for k, v in CB.DEFS.items()) +
        '\nRespond ONLY JSON {"concept":"<name|null>"}. No markdown.'
    )
    return _gpt(instr, "window (time x channels): " + json.dumps(arr), key)


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    mu, sd = CB.normal_stats(rng)
    out = {}
    for c in CB.CONCEPTS:
        rf = lf = lr = 0
        for _ in range(N_PER):
            x = CB.make_window(c, rng); ev = CB.evidence(x)
            rf += int(rule_feat(ev, mu, sd) == c)
            if net_ok:
                lf += int(CB.gpt_recognize_top1(ev, key, mu, sd) == c)
                lr += int(llm_raw(x, key) == c)
        out[c] = (rf / N_PER, lf / N_PER if net_ok else float("nan"), lr / N_PER if net_ok else float("nan"))
    return out


def ms(xs):
    a = np.array(xs, float); return np.nanmean(a), np.nanstd(a)


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    bg = "synthetic"
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    print(f"net_ok={net_ok} NSEED={NSEED} bg={bg} N_PER={N_PER} T_DS={T_DS}\n")
    res = [run_seed(s, key, net_ok) for s in range(NSEED)]
    print(f"{'concept':20s}{'rule_feat':>12s}{'llm_feat':>12s}{'llm_raw':>12s}")
    print("-" * 56)
    for c in CB.CONCEPTS:
        rf = ms([r[c][0] for r in res])[0]; lf = ms([r[c][1] for r in res])[0]; lr = ms([r[c][2] for r in res])[0]
        print(f"{c:20s}{rf*100:>10.0f}%{lf*100:>11.0f}%{lr*100:>11.0f}%")
    rfo = ms([[r[c][0] for c in CB.CONCEPTS] for r in res])[0]
    lfo = ms([[r[c][1] for c in CB.CONCEPTS] for r in res])[0]
    lro = ms([[r[c][2] for c in CB.CONCEPTS] for r in res])[0]
    print("-" * 56)
    print(f"{'总体':20s}{rfo*100:>10.0f}%{lfo*100:>11.0f}%{lro*100:>11.0f}%")
    print("\n判读:llm_raw > rule_feat → 信号表示瓶颈(LLM 给原始信号能超手工特征规则);")
    print("     llm_raw ≤ rule_feat(尤其远低)→ LLM 不擅原始数值,根因=信息不对称,出路=真实语义上下文。")
    json.dump(dict(bg=bg, nseed=NSEED, per_seed=res), open(ROOT / "runs" / f"signal_bottleneck{'_'+REAL if REAL else ''}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
