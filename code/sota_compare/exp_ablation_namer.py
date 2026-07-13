#!/usr/bin/env python3
"""【存亡级 ablation:语义轴的 LLM 是否不可替代?】

`gpt_recognize_top1` 的 prompt 实际指示 LLM 做的是:"找 z 最大的统计量 → 映射到它对应的概念"。
这等于一个 argmax+查表 的硬规则。本实验直接对比**同一命名任务**上:
  - LLM 命名器   `CB.gpt_recognize_top1`
  - 硬规则命名器 `rule_namer` = argmax(z) → STAT_OF 反查 → 概念名(z<阈 → None)
对每个概念各生成 N 窗,看两者命名准确率。

判读:若硬规则 ≈ LLM → 当前(证据正交+把映射喂给LLM的)benchmark **证明不了 LLM 的必要性**,
语义轴需换到 LLM 真正加值的设置(开放词表/真实模糊证据)。若 LLM 显著 > 规则 → 找到加值点。
env REAL_MACHINE 选背景,CMP_NSEED 默认3。用法 sbatch sota_compare/run_ablation.sh
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigla_exp.ovbench as CB                  # noqa: E402

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
REAL = os.environ.get("REAL_MACHINE", "")
N_PER = 10 if SMOKE else 40
STAT_TO_CONCEPT = {v: k for k, v in CB.STAT_OF.items()}   # 反查:签名统计量 → 概念


def output_path(default_name):
    explicit = os.environ.get("CMP_OUTPUT_JSON")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else ROOT / p
    tag = os.environ.get("CMP_RUN_TAG", "").strip()
    if tag:
        stem, suffix = Path(default_name).stem, Path(default_name).suffix
        return ROOT / "runs" / f"{stem}_{tag}{suffix}"
    return ROOT / "runs" / default_name


def rule_namer(ev, mu, sd, thresh=2.0):
    """硬规则:取偏离正常最大的统计量,反查它是哪个概念的签名(z<阈→None)。正是 prompt 让 LLM 做的事。"""
    z = {k: (ev[k] - mu[k]) / (sd[k] + 1e-9) for k in mu}
    dom = max(z, key=z.get)
    if z[dom] < thresh:
        return None
    return STAT_TO_CONCEPT.get(dom)


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    mu, sd = CB.normal_stats(rng)
    out = {}
    for c in CB.CONCEPTS:
        ln = rn = 0
        for _ in range(N_PER):
            ev = CB.evidence(CB.make_window(c, rng))
            r = rule_namer(ev, mu, sd)
            l = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            rn += int(r == c); ln += int(l == c)
        out[c] = (ln / N_PER, rn / N_PER)
    # normal 窗的"误命名率"(被判成某概念而非 None)
    lf = rf = 0
    for _ in range(N_PER):
        ev = CB.evidence(CB.make_window(None, rng))
        r = rule_namer(ev, mu, sd)
        l = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
        rf += int(r is not None); lf += int(l not in (None, "__ERROR__"))
    out["__normal_misname__"] = (lf / N_PER, rf / N_PER)
    return out


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    bg = "synthetic"
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    print(f"net_ok={net_ok} SMOKE={SMOKE} NSEED={NSEED} bg={bg} N_PER={N_PER}\n")
    res = [run_seed(s, key, net_ok) for s in range(NSEED)]

    keys = list(CB.CONCEPTS) + ["__normal_misname__"]
    def agg(c, i):
        a = np.array([r[c][i] for r in res], float); return a.mean(), a.std()
    print(f"{'concept':22s}{'LLM 命名准确率':>16s}{'硬规则准确率':>14s}{'差(LLM-规则)':>14s}")
    print("-" * 70)
    L = R = []
    for c in CB.CONCEPTS:
        lm, ls = agg(c, 0); rm, rs = agg(c, 1)
        print(f"{c:22s}{lm*100:>10.0f}±{ls*100:<3.0f}{rm*100:>10.0f}±{rs*100:<3.0f}{(lm-rm)*100:>+12.0f}")
    lo = np.mean([[r[c][0] for c in CB.CONCEPTS] for r in res])
    ro = np.mean([[r[c][1] for c in CB.CONCEPTS] for r in res])
    nm_l, _ = agg("__normal_misname__", 0); nm_r, _ = agg("__normal_misname__", 1)
    print("-" * 70)
    print(f"{'总体(6概念均值)':22s}{lo*100:>10.0f}    {ro*100:>10.0f}    {(lo-ro)*100:>+12.0f}")
    print(f"{'normal 误命名率↓':22s}{nm_l*100:>10.0f}    {nm_r*100:>10.0f}")
    print("\n判读:LLM≈硬规则 → 当前 benchmark 证明不了 LLM 必要性(语义轴=手工特征+查表);")
    print("     LLM≫规则(尤其真实背景/模糊证据)→ LLM 在歧义证据上加值,记下该设置作为语义轴卖点。")
    json.dump(dict(bg=bg, nseed=NSEED, per_seed=res),
              open(output_path(f"ablation_namer{'_'+REAL if REAL else ''}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
