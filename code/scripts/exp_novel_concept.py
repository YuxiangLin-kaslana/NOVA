#!/usr/bin/env python3
"""路线B 决定性实验:新异常类型识别 —— LLM zero-shot vs 参数化(+在线重训)。

硬 claim:概念漂移(新异常类型涌现)在无标签流里,参数化检测器**结构性**无法识别
(没有新类型的标签 → 输出维永远是死的,在线重训也救不回);而 LLM 用语义先验 +
通用统计证据,zero-shot 命名新类型。这正是"为什么必须用 LLM"。

设定:已知 5 类进概念检测器训练;correlation_break **留出**(训练从不出现),仅测试出现。
同一批新类型窗上比"识别率"(是否正确命名 correlation_break):参数化 vs LLM。
公平性:① 已知类型上参数化应当很好(非菜,是对新类型失明);② LLM 拿的是 6 类**通用**
统计证据(不是"答案"),要自己推理哪类匹配。

用法: sbatch scripts/exp_novel_concept.sh
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sigla_exp.model import CNNConceptDetector  # noqa: E402

CONCEPTS = ("spike", "level_shift", "oscillation", "variance_burst", "trend", "correlation_break")
KNOWN = ("spike", "level_shift", "oscillation", "variance_burst", "trend")  # 进训练
NOVEL = "correlation_break"                                                 # 留出
WIN, NVARS = 100, 12
DEFS = {  # 给 LLM 的 6 类一句话定义(通用,非答案)
    "spike": "a brief sharp excursion (a few extreme points) in one or more channels",
    "level_shift": "an abrupt step change in the mean level partway through the window",
    "oscillation": "an injected high-frequency oscillatory pattern",
    "variance_burst": "a localized burst of increased variance/volatility",
    "trend": "a gradual directional drift (ramp) within the window",
    "correlation_break": "the cross-channel correlation structure is disrupted (channels desynchronize)",
}


# --------------------------------------------------------------------------- #
#  可控幅度注入器:相关基底 + 6 类各有清晰、可分离的统计签名                    #
# --------------------------------------------------------------------------- #
def base_normal(rng):
    """3 个共享潜因子 → 通道间强相关(correlation_break 才检测得到)。"""
    t = np.arange(WIN)
    nf = 3
    fac = np.stack([np.sin(2 * np.pi * (0.03 + 0.04 * rng.random()) * t + 2 * np.pi * rng.random())
                    for _ in range(nf)], 1).astype(np.float32)
    w = rng.normal(0, 1, (nf, NVARS)).astype(np.float32)
    x = fac @ w + rng.normal(0, 0.1, (WIN, NVARS)).astype(np.float32)
    return ((x - x.mean(0)) / (x.std(0) + 1e-6)).astype(np.float32)


def _dims(rng, k):
    return rng.choice(NVARS, size=min(k, NVARS), replace=False)


def _spike(x, rng):
    for d in _dims(rng, 3):
        for tt in rng.integers(0, WIN, size=2):
            x[tt, d] += rng.choice([-1.0, 1.0]) * rng.uniform(6, 10)


def _level_shift(x, rng):
    s = int(rng.integers(WIN // 4, WIN // 2))
    for d in _dims(rng, 4):
        x[s:, d] += rng.choice([-1.0, 1.0]) * rng.uniform(2.5, 4.0)


def _oscillation(x, rng):
    t = np.arange(WIN); f = rng.uniform(0.30, 0.45)
    for d in _dims(rng, 3):
        x[:, d] += rng.uniform(1.5, 2.5) * np.sin(2 * np.pi * f * t + 2 * np.pi * rng.random())


def _variance_burst(x, rng):
    s = int(rng.integers(WIN // 2, 3 * WIN // 4)); e = min(WIN, s + int(rng.integers(15, 30)))
    for d in _dims(rng, 4):
        x[s:e, d] += rng.normal(0, rng.uniform(1.6, 2.2), e - s).astype(np.float32)


def _trend(x, rng):
    ramp = np.linspace(0, 1, WIN).astype(np.float32)
    for d in _dims(rng, 4):
        x[:, d] += rng.choice([-1.0, 1.0]) * rng.uniform(3.5, 5.0) * ramp


def _corr_break(x, rng):
    """把选中通道的一段**平滑地**换成独立低频信号(同均值/方差)→ 破坏跨通道同步,
    但不引入高频跳变(否则会被误认成 oscillation)。每通道边际分布几乎不变。"""
    s = int(rng.integers(0, WIN // 3)); e = s + int(rng.integers(WIN // 3, WIN // 2))
    tt = np.arange(e - s)
    for d in _dims(rng, max(3, NVARS // 2)):
        mu, sd = x[s:e, d].mean(), x[s:e, d].std() + 1e-6
        f = rng.uniform(0.03, 0.07)
        x[s:e, d] = (np.sin(2 * np.pi * f * tt + 2 * np.pi * rng.random()) * sd + mu).astype(np.float32)


INJ = {"spike": _spike, "level_shift": _level_shift, "oscillation": _oscillation,
       "variance_burst": _variance_burst, "trend": _trend, "correlation_break": _corr_break}


def make_window(concept, rng):
    x = base_normal(rng)
    if concept is not None:
        INJ[concept](x, rng)
    return x.astype(np.float32)


def evidence(x):
    """6 个通用、可解释的统计证据(不是答案);每个概念在其中一个上偏离正常基线。"""
    z = (x - np.median(x, 0)) / (1.4826 * np.median(np.abs(x - np.median(x, 0)), 0) + 1e-6)
    mid = WIN // 2; L, R = x[:mid], x[mid:]
    det = x - x.mean(0, keepdims=True); mag = np.abs(np.fft.rfft(det, axis=0))[1:]
    hf = float(np.mean(mag[mag.shape[0] // 2:].sum(0) / (mag.sum(0) + 1e-6)))
    tt = np.arange(WIN); slope = float(np.max(np.abs(np.polyfit(tt, x, 1)[0])))
    vb = float(np.max(R.std(0) / (L.std(0) + 1e-6)))
    s = x.std(0) > 1e-6                                # 丢常数通道,避免 corrcoef 出 NaN
    if int(s.sum()) >= 2:
        C = np.corrcoef(x[:, s], rowvar=False); n = C.shape[0]
        meancorr = float((np.abs(C).sum() - n) / (n * (n - 1)))
    else:
        meancorr = 1.0
    out = {
        "max_abs_zscore": round(float(np.max(np.abs(z))), 2),
        "max_step_change": round(float(np.max(np.abs(np.median(R, 0) - np.median(L, 0)))), 2),
        "high_freq_energy_frac": round(hf, 3),
        "right_left_std_ratio": round(vb, 2),
        "max_linear_slope": round(slope, 4),
        "mean_cross_channel_corr": round(meancorr, 3),
    }
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}  # 保证有限,可 JSON


def gpt_recognize(ev, key, base, model="gpt-4o-mini"):
    instr = (
        "You identify which time-series anomaly concept(s) are present in a window, given generic "
        "statistical evidence and the typical NORMAL value of each statistic. A concept is present "
        "when its signature statistic clearly deviates from normal. Taxonomy:\n" +
        "\n".join(f"- {k}: {v}" for k, v in DEFS.items()) +
        "\nTypical NORMAL statistic values: " + json.dumps({k: round(v, 3) for k, v in base.items()}) +
        "\nReason over the evidence vs these normals; a concept may be unseen by any trained detector "
        "but you can still name it from its definition. "
        "Respond with ONLY a JSON object {\"concepts\":[...]} (subset of the taxonomy; empty if none "
        "clearly deviates). No explanation, no markdown, no other text."
    )
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": json.dumps(ev)}], "max_output_tokens": 400}
    for attempt in range(3):                          # 重试,避免一次超时崩掉整个 job
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses",
                                         data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            return [c for c in json.loads(txt[s:e + 1]).get("concepts", []) if c in CONCEPTS]
        except Exception:
            continue
    return ["__ERROR__"]                              # 标记失败(不计为命中,也不污染分母)


def gpt_recognize_top1(ev, key, base, model="gpt-4o-mini"):
    """单标签版:强制 LLM 返回**唯一最显著**的概念(或 none)。解决 gpt-4o-mini 过度列举、取首项随机
    的问题——既是公平的 LLM-only 分类基线,也是多新类闭环的前置修复。返回概念字符串 / None / "__ERROR__"。"""
    instr = (
        "You classify a time-series window into the SINGLE most salient anomaly concept, given generic "
        "statistical evidence and the typical NORMAL value of each statistic. Pick the ONE concept whose "
        "signature statistic deviates most strongly from normal. Taxonomy:\n" +
        "\n".join(f"- {k}: {v}" for k, v in DEFS.items()) +
        "\nTypical NORMAL statistic values: " + json.dumps({k: round(v, 3) for k, v in base.items()}) +
        "\nReason over the evidence vs these normals; a concept may be unseen by any trained detector "
        "but you can still name it from its definition. Choose exactly one, the most dominant. "
        "Respond with ONLY a JSON object {\"concept\":\"<name>\"} where <name> is one taxonomy key, "
        "or \"none\" if nothing clearly deviates. No explanation, no markdown, no other text."
    )
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": json.dumps(ev)}], "max_output_tokens": 60}
    for attempt in range(3):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses",
                                         data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            c = json.loads(txt[s:e + 1]).get("concept", "none")
            return c if c in CONCEPTS else None           # "none"/未知 → None(视为正常/无概念)
        except Exception:
            continue
    return "__ERROR__"


def sanity_separation(rng):
    import collections
    keys = list(evidence(make_window(None, rng)).keys())
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in [None, *CONCEPTS]:
        for _ in range(80):
            ev = evidence(make_window(c, rng))
            for k in keys:
                data[c or "normal"][k].append(ev[k])
    print("=== 证据分离 sanity(每概念应在某个统计上偏离 normal)===")
    print("%-18s" % "concept" + "".join("%24s" % k for k in keys))
    base = {}
    for c in ["normal", *CONCEPTS]:
        print("%-18s" % c + "".join("%24s" % f"{np.mean(data[c][k]):.3f}" for k in keys))
        if c == "normal":
            base = {k: float(np.mean(data[c][k])) for k in keys}
    print()
    return base


def main():
    rng = np.random.default_rng(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  known={KNOWN}  novel(held-out)={NOVEL}")
    base = sanity_separation(np.random.default_rng(7))

    # ---- 训练集:只含已知 5 类(+正常),correlation_break 从不出现 ---- #
    Xtr, Ytr = [], []
    for _ in range(4000):
        c = KNOWN[rng.integers(len(KNOWN))] if rng.random() < 0.7 else None
        Xtr.append(make_window(c, rng))
        y = np.zeros(len(CONCEPTS), np.float32)
        if c is not None:
            y[CONCEPTS.index(c)] = 1.0
        Ytr.append(y)
    Xtr = torch.tensor(np.stack(Xtr)).to(dev); Ytr = torch.tensor(np.stack(Ytr)).to(dev)

    det = CNNConceptDetector(WIN, NVARS, n_concepts=len(CONCEPTS), kernel_size=7).to(dev)
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    det.train()
    for ep in range(40):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            loss = F.binary_cross_entropy_with_logits(det(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    det.eval()

    # ---- 测试集:已知类型(每类若干) + 新类型 correlation_break ---- #
    def build_test(concept, n):
        return [make_window(concept, rng) for _ in range(n)]
    test = {c: build_test(c, 60) for c in KNOWN}
    test[NOVEL] = build_test(NOVEL, 60)

    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    print(f"OPENAI key set: {net_ok}")

    @torch.no_grad()
    def param_probs(W):
        return det.predict_proba(torch.tensor(np.stack(W)).to(dev)).cpu().numpy()

    print("\n%-18s | %-22s | %-22s" % ("true concept", "参数化识别率", "LLM 识别率"))
    print("-" * 70)
    rows = {}
    for c, W in test.items():
        probs = param_probs(W)
        ci = CONCEPTS.index(c)
        param_rec = float(np.mean(probs[:, ci] > 0.5))  # 参数化是否命中该类
        # LLM: 仅对一个子样本调用(省成本),已知类抽样12,新类全量60
        n_llm = 60 if c == NOVEL else 12
        llm_hits = n_ok = 0
        if net_ok:
            for x in W[:n_llm]:
                got = gpt_recognize(evidence(x), key, base)
                if got == ["__ERROR__"]:
                    continue
                n_ok += 1
                llm_hits += int(c in got)
        llm_rec = (llm_hits / n_ok) if (net_ok and n_ok) else float("nan")
        rows[c] = (param_rec, llm_rec, probs[:, ci].mean())
        tag = "  <<< 新类型" if c == NOVEL else ""
        print("%-18s | %-22s | %-22s%s" % (
            c, f"{param_rec:.1%} (avg p={probs[:,ci].mean():.2f})",
            (f"{llm_rec:.1%}" if net_ok else "n/a"), tag))

    pn, ln, _ = rows[NOVEL]
    known_param = np.mean([rows[c][0] for c in KNOWN])
    print("\n" + "=" * 70)
    print(f"已知类型 参数化平均识别率 = {known_param:.1%}  (证明参数化对已知类型有效)")
    print(f"新类型 correlation_break: 参数化 = {pn:.1%}   LLM = {ln:.1%}" if net_ok else
          f"新类型 correlation_break: 参数化 = {pn:.1%}   LLM = (无网络,跳过)")
    if net_ok and ln > pn + 0.3:
        print("结论:✅ LLM zero-shot 识别新类型,参数化结构性失明 —— 'LLM 是概念漂移必需品' 成立。")
    print("=" * 70)
    json.dump({k: {"param_rec": v[0], "llm_rec": v[1]} for k, v in rows.items()},
              open(ROOT / "runs" / "novel_concept_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
