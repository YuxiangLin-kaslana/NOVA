#!/usr/bin/env python3
"""干净可分的 6 概念合成基准:每个概念**只**激活一个通用统计量(签名),互不遮蔽。

动机:旧 evidence(exp_novel_concept)里 spike↔variance_burst(都抬 max-z)、level_shift↔trend
(都抬 slope/step)天然纠缠,只有 correlation_break 正交 → 多留出类的新颖门控崩。
本模块重设计注入器与统计量,使 6 概念在 6 个统计量上一一对应、z 分离清晰:

  spike            -> peakedness   = max|z| / p95|z|        (孤立极值:比值大;方差爆发:≈1)
  level_shift      -> step_jump    = 最大局部中位数跳变      (去趋势台阶:无斜率,锐跳变)
  oscillation      -> hf_frac      = 高频能量占比
  variance_burst   -> var_ratio    = 滑窗最大局部std / 中位局部std
  trend            -> lin_r2       = 线性拟合 R²(去趋势台阶≈0,纯斜坡≈1)
  correlation_break-> neg_corr     = -(跨通道平均|相关|)     (相关性下降 → 该统计量大幅变小)

诊断(main):打印每概念在 6 统计量上的 z(对正常基线),验证 argmax = 自身签名且 z 分离够大。
用法: sbatch scripts/clean_bench.sh
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.exp_novel_concept as NC  # noqa: E402  复用相关基底 base_normal

WIN, NVARS = NC.WIN, NC.NVARS
CONCEPTS = ["spike", "level_shift", "oscillation", "variance_burst", "trend", "correlation_break"]
STAT_OF = {"spike": "peakedness", "level_shift": "step_jump", "oscillation": "hf_frac",
           "variance_burst": "var_ratio", "trend": "lin_r2", "correlation_break": "neg_corr"}
STATS = ["peakedness", "step_jump", "hf_frac", "var_ratio", "lin_r2", "neg_corr"]


# --------------------------------------------------------------------------- #
#  注入器:每个只激活自身签名                                                   #
# --------------------------------------------------------------------------- #
def _dims(rng, k):
    return rng.choice(NVARS, size=min(k, NVARS), replace=False)


def _spike(x, rng):
    """少量孤立极值点 → 高 peakedness,不抬持续方差。"""
    for d in _dims(rng, 3):
        for tt in rng.integers(0, WIN, size=2):
            x[tt, d] += rng.choice([-1.0, 1.0]) * rng.uniform(8, 12)


def _level_shift(x, rng):
    """锐台阶,但**去除线性分量** → 跳变保留(step_jump 高),整体斜率≈0(lin_r2 低)。"""
    s = int(rng.integers(WIN // 3, 2 * WIN // 3))
    t = np.arange(WIN)
    for d in _dims(rng, 4):
        col = x[:, d].astype(np.float64).copy()
        col[s:] += rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 5.0)
        a, b = np.polyfit(t, col, 1)            # 去趋势:减掉最佳拟合直线 → 无净斜率
        x[:, d] = (col - (a * t + b) + x[:, d].mean()).astype(np.float32)


def _oscillation(x, rng):
    t = np.arange(WIN); f = rng.uniform(0.30, 0.45)
    for d in _dims(rng, 3):
        x[:, d] += rng.uniform(1.5, 2.5) * np.sin(2 * np.pi * f * t + 2 * np.pi * rng.random())


def _variance_burst(x, rng):
    """一段内**平滑放大已有基底波动**(gain,绕均值) → 该段局部 std 升高 → var_ratio 高。
    不加白噪声 → 不产生孤立极值(peakedness 不被触发);基底平滑 → 不引入高频/台阶。"""
    s = int(rng.integers(WIN // 4, WIN // 2)); e = min(WIN, s + int(rng.integers(30, 45)))
    L = e - s
    env = 1.0 + (rng.uniform(3.5, 4.5) - 1.0) * np.sin(np.pi * np.arange(L) / L) ** 2  # 边缘=1,中心=G
    for d in _dims(rng, 6):
        m = x[s:e, d].mean()
        x[s:e, d] = (m + (x[s:e, d] - m) * env).astype(np.float32)                     # 平滑放大,无边界突变


def _trend(x, rng):
    ramp = np.linspace(0, 1, WIN).astype(np.float32)
    for d in _dims(rng, 4):
        x[:, d] += rng.choice([-1.0, 1.0]) * rng.uniform(3.5, 5.0) * ramp


def _corr_break(x, rng):
    """选中通道一段平滑换成独立低频 → 破坏跨通道同步(neg_corr 大),不引入高频/极值。"""
    s = int(rng.integers(0, WIN // 4)); e = s + int(rng.integers(WIN // 2, 3 * WIN // 4))
    tt = np.arange(e - s)
    for d in _dims(rng, NVARS - 2):                        # 去同步几乎所有通道 → 相关性压到最低
        mu, sd = x[s:e, d].mean(), x[s:e, d].std() + 1e-6
        f = rng.uniform(0.03, 0.07)
        x[s:e, d] = (np.sin(2 * np.pi * f * tt + 2 * np.pi * rng.random()) * sd + mu).astype(np.float32)


INJ = {"spike": _spike, "level_shift": _level_shift, "oscillation": _oscillation,
       "variance_burst": _variance_burst, "trend": _trend, "correlation_break": _corr_break}


def make_window(concept, rng):
    x = NC.base_normal(rng)
    if concept is not None:
        INJ[concept](x, rng)
    return x.astype(np.float32)


# --------------------------------------------------------------------------- #
#  干净统计量:6 个,每个对应一个概念签名                                         #
# --------------------------------------------------------------------------- #
def evidence(x):
    WIN_, NV = x.shape
    med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0) + 1e-6
    absz = np.abs((x - med) / mad)
    top1 = float(absz.max()); p95 = float(np.percentile(absz, 95))
    peakedness = top1 / (p95 + 1e-6)                       # 孤立极值 vs 广泛抬高

    # 滑窗局部 std:variance_burst 一段显著高于其余
    w = 20; locs = [x[i:i + w].std(0).mean() for i in range(0, WIN_ - w, 4)]
    locs = np.array(locs)
    var_ratio = float(locs.max() / (np.median(locs) + 1e-6))

    # 持久台阶跳变:用较长中位块(持久 level_shift 保留;瞬态高方差摆动被平均掉)
    h = 20; meds = np.array([np.median(x[i:i + h], 0) for i in range(0, WIN_ - h, 2)])
    step_jump = float(np.max(np.abs(np.diff(meds, axis=0))))

    det = x - x.mean(0, keepdims=True); mag = np.abs(np.fft.rfft(det, axis=0))[1:]
    hf_frac = float(np.mean(mag[mag.shape[0] // 2:].sum(0) / (mag.sum(0) + 1e-6)))

    t = np.arange(WIN_); A = np.vstack([t, np.ones(WIN_)]).T   # 线性拟合 R²:trend 高
    coef, *_ = np.linalg.lstsq(A, x, rcond=None); fit = A @ coef
    ss_res = ((x - fit) ** 2).sum(0); ss_tot = ((x - x.mean(0)) ** 2).sum(0) + 1e-6
    lin_r2 = float(np.mean(np.clip(1 - ss_res / ss_tot, 0, 1)))

    s = x.std(0) > 1e-6
    if int(s.sum()) >= 2:
        C = np.corrcoef(x[:, s], rowvar=False); n = C.shape[0]
        meancorr = float((np.abs(C).sum() - n) / (n * (n - 1)))
    else:
        meancorr = 1.0
    neg_corr = -meancorr                                   # 相关性下降 → neg_corr 升高

    out = {"peakedness": peakedness, "step_jump": step_jump, "hf_frac": hf_frac,
           "var_ratio": var_ratio, "lin_r2": lin_r2, "neg_corr": neg_corr}
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}


DEFS = {  # 给 LLM 的语义定义(通用,非答案)
    "spike": "a few brief isolated extreme points in one or more channels",
    "level_shift": "an abrupt, persistent step change in the level (no overall slope)",
    "oscillation": "an injected high-frequency oscillatory pattern",
    "variance_burst": "a localized burst of increased variance/volatility (no isolated outliers)",
    "trend": "a gradual linear ramp/drift across the whole window",
    "correlation_break": "the cross-channel correlation structure is disrupted (channels desynchronize)",
}


def gpt_recognize_top1(ev, key, mu, model="gpt-4o-mini"):
    """返回**单个**最可能的概念名(或 None)。强制 top-1 → 杜绝过度列举(旧版多留出崩溃的元凶)。"""
    instr = (
        "You identify the SINGLE most likely time-series anomaly concept present in a window, given "
        "generic statistical evidence and the typical NORMAL value of each statistic. Pick the ONE "
        "concept whose signature statistic deviates most from normal. Taxonomy:\n" +
        "\n".join(f"- {k}: {v}" for k, v in DEFS.items()) +
        "\nTypical NORMAL statistic values: " + json.dumps({k: round(v, 3) for k, v in mu.items()}) +
        "\nStatistic meanings: peakedness=isolated-outlier-ness; step_jump=persistent level jump; "
        "hf_frac=high-frequency energy; var_ratio=localized variance burst; lin_r2=linearity (ramp); "
        "neg_corr=loss of cross-channel correlation.\n"
        "Respond with ONLY a JSON object {\"concept\":\"<name>\"} using exactly one taxonomy name, "
        "or {\"concept\":null} if nothing clearly deviates. No explanation, no markdown."
    )
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": json.dumps({k: round(v, 3) for k, v in ev.items()})}],
               "max_output_tokens": 200}
    for _ in range(3):
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
            c = json.loads(txt[s:e + 1]).get("concept")
            return c if c in CONCEPTS else None
        except Exception:
            continue
    return "__ERROR__"


def normal_stats(rng, n=400):
    evs = [evidence(make_window(None, rng)) for _ in range(n)]
    mu = {k: float(np.mean([e[k] for e in evs])) for k in STATS}
    sd = {k: float(np.std([e[k] for e in evs]) + 1e-6) for k in STATS}
    return mu, sd


def main():
    rng = np.random.default_rng(0)
    mu, sd = normal_stats(rng)
    print("=== 干净基准:每概念在 6 统计量上的 z(对正常基线)===")
    print("期望:每行 argmax 落在自身签名(标 *),且远大于其它\n")
    print("%-18s" % "concept" + "".join("%13s" % s for s in STATS))
    ok = True
    for c in CONCEPTS:
        evs = [evidence(make_window(c, rng)) for _ in range(120)]
        zc = {s: float(np.mean([(e[s] - mu[s]) / sd[s] for e in evs])) for s in STATS}
        dom = max(zc, key=zc.get)
        row = "%-18s" % c
        for s in STATS:
            star = "*" if s == STAT_OF[c] else " "
            row += "%12.1f%s" % (zc[s], star)
        hit = (dom == STAT_OF[c])
        ok = ok and hit
        print(row + ("   <-OK" if hit else f"   <-BAD dom={dom}"))
    # 任取一个 3/3 划分,验证门控:novel 应判 suspect(argmax z 不在已知签名集)
    print("\n=== 门控验证(已知=前3,新=后3)===")
    known = CONCEPTS[:3]; novel = CONCEPTS[3:]
    known_stats = {STAT_OF[c] for c in known}
    for c in CONCEPTS:
        evs = [evidence(make_window(c, rng)) for _ in range(120)]
        susp = np.mean([max({s: (e[s] - mu[s]) / sd[s] for s in STATS},
                            key=lambda k: (e[k] - mu[k]) / sd[k]) not in known_stats for e in evs])
        tag = "novel→应高" if c in novel else "known→应低"
        print("  %-18s suspect=%.0f%%  (%s)" % (c, susp * 100, tag))
    print("\n总判定:", "✅ 6 概念干净可分" if ok else "❌ 仍有遮蔽,需调注入器/统计量")


if __name__ == "__main__":
    main()
