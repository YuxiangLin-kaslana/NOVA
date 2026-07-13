#!/usr/bin/env python3
"""【早预警判别:LLM 整合弱前兆信号 + 语言上下文,是否优于纯信号方法?】

回答"LLM 拿到早信号并判别是否有优势",并像命名 ablation 一样**早去风险**——关键是隔离"上下文整合"的价值。

设定(对齐 AAAI SigLA 的前兆窗 + 风险累积 ρ_t):每个事件 = 前兆窗内 K 个窗的轨迹 + 一条语言上下文。
  - 真实前兆(should-warn=1):弱但**持续**的签名(每窗弱信号),之后会 onset。
  - 良性扰动(should-warn=0):**瞬时**强 blip(1 窗) 然后回正常,无 onset → 信号上像异常(骗峰值检测器)。
  → 信号部分可分(持续 vs 瞬时),但有重叠;**只有语言上下文能彻底消歧**(informative log/rule/case)。

四臂(都在前兆窗内做"报警/等待"决策,同口径 matched FA):
  peak-detector  轨迹峰值异常分(单窗最大 z)→ 被良性 blip 骗,FA 高。
  rho_t          风险累积 ρ=αρ+(1-α)score → 持续才高,抗瞬时 blip(强信号基线)。
  LLM-context    LLM 只看 z 轨迹决策(≈聪明的 ρ_t,无语言)。
  LLM+context    LLM 看 z 轨迹 + 语言上下文决策。
关键消融:LLM+context vs LLM-context 隔离"上下文整合"价值;若 ≫ 且 LLM-context≈检测器 → 价值在上下文(可辩护)。
env REAL_MACHINE 选信号背景,CMP_NSEED 默认3。用法 sbatch sota_compare/run_ewdisc.sh
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
N_EV = 20 if SMOKE else 60                      # 每类事件数(real / benign 各)
K = 6                                            # 前兆窗轨迹长度
NOVEL = "oscillation"                            # 渐发振动(可作前兆)
ALPHA = 0.6                                      # ρ_t 累积衰减
P_INFORM = 0.7                                   # 上下文为"信息型"(可消歧)的比例,否则干扰型
TARGET_FA = 0.10                                 # 匹配的误报率

V = "sensor_07"
CTX_REAL = [
    f"Operator log: {V} shows sustained creeping deviation since last shift; engineers flagged a possible developing fault.",
    f"Maintenance record: no scheduled work on {V}; the slow trend is abnormal for this operating phase.",
    f"Historical case: a similar slow build-up on {V} preceded a confirmed failure two weeks ago.",
]
CTX_BENIGN = [
    f"Maintenance log: scheduled recalibration of {V} is in progress this window; transient readings are expected.",
    f"Operator note: {V} fluctuation is normal during the current startup/ramp phase.",
    f"Historical case: brief blips on {V} during load changes have never led to faults.",
]
CTX_DISTRACT = [
    "System status nominal. Routine telemetry; no operator annotations for this interval.",
    "Shift handover note: general monitoring, nothing specific reported on this subsystem.",
]


def maxz(ev, mu, sd):
    return max((ev[k] - mu[k]) / (sd[k] + 1e-9) for k in mu)


def make_event(is_real, rng, mu, sd):
    """2×2 设计:信号只有一半信息量,留出上下文消歧空间。返回 (z轨迹[K], 上下文, signal_clear)。
    - 真前兆:60% 'clear'(持续中等信号→ρ_t 高),40% 'ambiguous'(几乎无信号→像正常,ρ_t 漏)。
    - 良性:  40% 'misleading'(持续信号→像前兆,骗 ρ_t),60% 'quiet'(正常)。
    → 信号(ρ_t)对 clear-real/quiet-benign 有效,但对 ambiguous-real/misleading-benign 失效;
      仅语言上下文(P_INFORM 概率为信息型)能救这部分重叠。"""
    traj = []
    if is_real:
        clear = rng.random() < 0.6
        for _ in range(K):
            s = float(rng.uniform(0.30, 0.50)) if clear else float(rng.uniform(0.0, 0.12))
            traj.append(maxz(CB.evidence(CB.make_window_strength(NOVEL, rng, s)), mu, sd))
        ctx = (rng.choice(CTX_REAL) if rng.random() < P_INFORM else rng.choice(CTX_DISTRACT))
    else:
        misleading = rng.random() < 0.4
        for _ in range(K):
            s = float(rng.uniform(0.30, 0.50)) if misleading else float(rng.uniform(0.0, 0.12))
            traj.append(maxz(CB.evidence(CB.make_window_strength(NOVEL, rng, s)), mu, sd))
        ctx = (rng.choice(CTX_BENIGN) if rng.random() < P_INFORM else rng.choice(CTX_DISTRACT))
    return np.array(traj, float), ctx


def gpt_confidence(traj, ctx, key, model="gpt-4o-mini"):
    """LLM 输出 0–100 的"真前兆"风险置信度(非二值,避免报警偏置)。ctx=None 不给语言上下文。返回 float/None。
    与信号方法同口径:置信度再卡阈到同一 FA 比 VEW。"""
    tr = [round(float(x), 1) for x in traj]
    base = (
        "You are an early-warning monitor for a multivariate sensor. You see a trajectory of per-window anomaly "
        "z-scores (deviation from normal in SDs) inside the pre-onset interval of a possible event. Estimate the "
        "RISK that a REAL precursor is developing (vs a benign transient). A real precursor tends to be a sustained "
        "build-up; a benign blip is brief. Output a calibrated risk score 0-100 (0=clearly benign, 100=clearly a "
        "developing precursor); use the full range, do not default to extremes. "
    )
    ctx_part = (f"Operational context: \"{ctx}\". Weigh this context together with the evidence. " if ctx else "")
    instr = base + ctx_part + 'Respond ONLY JSON {"risk": <0-100>}. No markdown.'
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": "z-score trajectory: " + json.dumps(tr)}],
               "max_output_tokens": 50}
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
            return float(json.loads(txt[s:e + 1]).get("risk"))
        except Exception:
            continue
    return None


def vew_at_fa(score_real, score_benign, target_fa):
    """信号方法:阈值设到 benign 误报=target_fa,读 real 的有效预警召回。"""
    thr = float(np.quantile(score_benign, 1.0 - target_fa))
    return float(np.mean(np.asarray(score_real) > thr))


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    mu, sd = CB.normal_stats(rng)
    reals = [make_event(True, rng, mu, sd) for _ in range(N_EV)]
    benigns = [make_event(False, rng, mu, sd) for _ in range(N_EV)]

    def peak(tr): return float(np.max(tr))
    def rho(tr):
        r = 0.0
        for x in tr:
            r = ALPHA * r + (1 - ALPHA) * x
        return r

    pr_r = [peak(t) for t, _ in reals]; pr_b = [peak(t) for t, _ in benigns]
    rh_r = [rho(t) for t, _ in reals]; rh_b = [rho(t) for t, _ in benigns]

    out = {"peak": vew_at_fa(pr_r, pr_b, TARGET_FA), "rho": vew_at_fa(rh_r, rh_b, TARGET_FA)}
    if net_ok:
        for tag, use_ctx in [("llm_noctx", False), ("llm_ctx", True)]:
            cr = [gpt_confidence(t, c if use_ctx else None, key) for t, c in reals]
            cb = [gpt_confidence(t, c if use_ctx else None, key) for t, c in benigns]
            cr = [x for x in cr if x is not None]; cb = [x for x in cb if x is not None]
            out[tag] = vew_at_fa(cr, cb, TARGET_FA) if cr and cb else float("nan")  # 同口径 @10%FA
    return out


def ms(xs):
    a = np.array(xs, float); return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    bg = "synthetic"
    if REAL:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL); bg = f"real-{REAL}(T={len(Z)})"
    print(f"net_ok={net_ok} NSEED={NSEED} bg={bg} novel={NOVEL} K={K} P_INFORM={P_INFORM} targetFA={TARGET_FA}\n")
    res = [run_seed(s, key, net_ok) for s in range(NSEED)]
    print(f"全部 @ 同口径 FA={TARGET_FA*100:.0f}%(置信度卡阈)\n{'方法':18s}{'有效预警召回(VEW)':>18s}")
    print("-" * 40)
    rows = [("peak", "peak-detector"), ("rho", "rho_t 风险累积")]
    if net_ok:
        rows += [("llm_noctx", "LLM−上下文"), ("llm_ctx", "LLM+上下文")]
    for tag, nm in rows:
        v = ms([r[tag] for r in res])
        print(f"{nm:18s}{v[0]*100:>12.0f}±{v[1]*100:<3.0f}")
    print("\n判读:若 LLM+上下文 的 VEW ≫ peak/rho/LLM−上下文(同 FA)→ LLM 整合语言上下文在弱前兆下做更好早预警决策;")
    print("     且 LLM−上下文 ≈ 信号方法 → 价值确在**上下文整合**(非 LLM 读信号更强)。反之则是伪优势。")
    json.dump(dict(bg=bg, nseed=NSEED, per_seed=res), open(ROOT / "runs" / f"ew_discriminate{'_'+REAL if REAL else ''}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
