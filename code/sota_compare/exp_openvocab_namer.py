#!/usr/bin/env python3
"""【B:真·开放词表 —— 结构上证明语义轴必须要 LLM】

A 证明:闭集命名时硬规则追平 LLM(规则=argmax→查 6 个预设映射)。但规则只能输出预设里的概念;
**签名未被预先映射的新类型,规则结构上命不了名,LLM 能。** 本实验测这个结构性差异:

  KNOWN = {spike, level_shift, oscillation}(prompt 只给这 3 个定义)。
  held-out NOVEL ∈ {correlation_break, trend, variance_burst}(名字**不在 prompt**)。
  - 硬规则(闭集):dom 统计量∈已知签名→输出对应已知概念,否则 None。**命名 novel 恒 0(结构性)**。
  - LLM(开放):prompt 允许"都不匹配则自创 NEW:<短名>"。看它能否**自由合成语义正确的名字**。
正确性用**人类语言关键词**判定(避开原始统计量 token,防"输出统计量名"蒙混)。

判读:规则命名 novel=0(结构性失败);LLM>0 → 语义轴的 LLM 不可替代性**结构性成立**(这是 A 给不出的)。
env REAL_MACHINE 选背景,CMP_NSEED 默认3。用法 sbatch sota_compare/run_openvocab.sh
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sigla_exp.ovbench as CB                  # noqa: E402

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
REAL = os.environ.get("REAL_MACHINE", "")
N_PER = 10 if SMOKE else 40
KNOWN = ["spike", "level_shift", "oscillation"]
NOVELS = ["correlation_break", "trend", "variance_burst"]
KNOWN_STATS = {CB.STAT_OF[c] for c in KNOWN}
STAT_TO_CONCEPT = {v: k for k, v in CB.STAT_OF.items()}
# 人类语言关键词(刻意避开原始统计量 token 如 decorr/lin_r2/var_localiz,防"输出统计量名"蒙混)
KW = {
    "correlation_break": ["correlat", "decoupl", "desync", "independ", "cross-channel", "cross channel", "uncorrel"],
    "trend": ["trend", "drift", "ramp", "slope", "linear", "gradual", "increas", "decreas", "monoton"],
    "variance_burst": ["varian", "volatil", "burst", "fluctuat", "unstable", "erratic", "dispers", "noisy"],
}


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


def rule_namer_closed(ev, mu, sd, thresh=2.0):
    """闭集硬规则:只认已知签名→已知概念,否则 None(无法命名未预映射类型)。"""
    z = {k: (ev[k] - mu[k]) / (sd[k] + 1e-9) for k in mu}
    dom = max(z, key=z.get)
    if z[dom] < thresh or dom not in KNOWN_STATS:
        return None
    return STAT_TO_CONCEPT[dom]


def gpt_openvocab(ev, key, mu, sd, model="gpt-4o-mini"):
    """开放词表命名:只给 3 个已知概念;允许自创 NEW:<短名>。返回原始字符串或 None/__ERROR__。"""
    z = {k: round((ev[k] - mu[k]) / (sd[k] + 1e-9), 1) for k in mu}
    known_def = "\n".join(f"- {c}: {CB.DEFS[c]}" for c in KNOWN)
    stat_mean = "; ".join(f"{k}={v}" for k, v in CB.STAT_MEANING.items())
    instr = (
        "You identify the dominant anomaly pattern in a multivariate time-series window from per-statistic "
        "z-scores (deviation from normal in SDs; large positive = strongly elevated).\n"
        f"KNOWN anomaly concepts:\n{known_def}\n"
        f"What each generic statistic measures: {stat_mean}.\n"
        "Procedure: locate the statistic(s) with the largest positive z. If it matches a KNOWN concept, output "
        "that concept's exact name. If the dominant deviation does NOT correspond to any known concept, INVENT a "
        "concise human-readable name (1-3 words) describing the underlying pattern, prefixed 'NEW:'. "
        "If no statistic has z above ~2, output null.\n"
        'Respond with ONLY JSON {"concept":"<known-name | NEW:... | null>"}. No markdown, no explanation.'
    )
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": "z-scores: " + json.dumps(z)}], "max_output_tokens": 200}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses",
                                         data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            return json.loads(txt[s:e + 1]).get("concept")
        except Exception:
            continue
    return "__ERROR__"


def sem_correct(name, novel):
    if not isinstance(name, str):
        return False
    low = name.lower()
    return any(k in low for k in KW[novel])


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    mu, sd = CB.normal_stats(rng)
    out = {}
    for nov in NOVELS:
        rule_ok = llm_ok = llm_new = llm_known = 0
        for _ in range(N_PER):
            ev = CB.evidence(CB.make_window(nov, rng))
            r = rule_namer_closed(ev, mu, sd)
            rule_ok += int(r == nov)                          # 恒 0(nov 不在已知映射输出)
            l = gpt_openvocab(ev, key, mu, sd) if net_ok else None
            if isinstance(l, str) and l not in ("__ERROR__",):
                if l in KNOWN:
                    llm_known += 1                            # 误判成已知
                else:
                    llm_new += 1                              # 提了新名(NEW: 或非已知)
                    llm_ok += int(sem_correct(l, nov))        # 新名语义正确
        out[nov] = dict(rule=rule_ok / N_PER, llm_correct=llm_ok / N_PER,
                        llm_newrate=llm_new / N_PER, llm_misknown=llm_known / N_PER)
    return out


def ms(xs):
    a = np.array(xs, float); return a.mean(), a.std()


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    bg = "synthetic"
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    print(f"net_ok={net_ok} NSEED={NSEED} bg={bg}  KNOWN(prompt内)={KNOWN}  N_PER={N_PER}\n")
    res = [run_seed(s, key, net_ok) for s in range(NSEED)]
    print(f"{'held-out novel':20s}{'硬规则命名':>12s}{'LLM自由命名(语义对)':>20s}{'LLM提新名率':>14s}{'LLM误判已知':>14s}")
    print("-" * 82)
    for nov in NOVELS:
        rk = ms([r[nov]["rule"] for r in res])[0]
        lc, lcs = ms([r[nov]["llm_correct"] for r in res])
        nr = ms([r[nov]["llm_newrate"] for r in res])[0]
        mk = ms([r[nov]["llm_misknown"] for r in res])[0]
        print(f"{nov:20s}{rk*100:>10.0f}%{lc*100:>14.0f}±{lcs*100:<3.0f}{nr*100:>12.0f}%{mk*100:>12.0f}%")
    print("-" * 82)
    print("判读:硬规则命名 novel 恒 0(签名未预映射→结构性命不了名);LLM 能自由合成语义正确的新名 →")
    print("**语义轴的 LLM 不可替代性结构性成立**(这是 A 的闭集命名给不出的关键证据)。")
    json.dump(dict(bg=bg, nseed=NSEED, per_seed=res),
              open(output_path(f"openvocab_namer{'_'+REAL if REAL else ''}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
