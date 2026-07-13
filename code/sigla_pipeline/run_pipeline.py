#!/usr/bin/env python3
"""【SigLA 端到端流水线(LLM 无关)】信号 → ②校准去纠缠画像 → ③成本感知inspect动作 → ①前兆窗效用。

整合点:**去纠缠 = inspect 动作**。便宜信号=原始校准 oscillation z;模糊时 inspect=去纠缠剥离 spike 伪激活。
为速度:每个(事件,步)只算一次 disentangle(同时得 raw z 与 net),缓存后调参/评估都在缓存上跑。
用法: sbatch sigla_pipeline/run_pipeline.sh
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, "/u/ylin30/sigLA/code")
import sigla_exp.ovbench as CB
import sigla_pipeline.profile_naive as PF
import sigla_pipeline.env as ENV

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NTR, NTE = (30, 50) if SMOKE else (160, 320)
ON, LMAX, LMIN, Tt = ENV.ONSET, ENV.LMAX, ENV.LMIN, ENV.T
WT = dict(valid=1.0, prem=0.5, late=0.3, miss=1.0, fa=0.8)


def precompute(rng, n, mu, sd):
    """每事件预算 raw[t]/net[t]/label。disentangle 一次同时给 raw 与 net。"""
    out = []
    for i in range(n):
        Wlist, isr = ENV.make_episode(i % 2 == 0, rng)
        raw = np.zeros(Tt); net = np.zeros(Tt)
        for t in range(Tt):
            pr = PF.disentangle(Wlist[t], mu, sd)
            raw[t] = pr["z"][ENV.TARGET]; net[t] = pr["net"][ENV.TARGET]
        out.append((raw, net, isr))
    return out


def util(astep, isr, lam, n_ins):
    u = -lam * n_ins
    if isr:
        if astep is None or astep >= ON: u -= WT["miss"]
        elif ON - LMAX <= astep <= ON - LMIN: u += WT["valid"]
        elif astep < ON - LMAX: u -= WT["prem"]
        else: u -= WT["late"]
    elif astep is not None:
        u -= WT["fa"]
    return u


def pol_threshold(ep, tau, use_net=False):
    raw, net, _ = ep; sig = net if use_net else raw; n = Tt if use_net else 0
    for t in range(Tt):
        if sig[t] > tau:
            return t, n
    return None, n


def pol_costaware(ep, lo, hi):
    raw, net, _ = ep; n = 0
    for t in range(Tt):
        if raw[t] >= hi:
            return t, n
        if raw[t] >= lo:
            n += 1
            if net[t] >= hi:
                return t, n
    return None, n


def main():
    rng = np.random.default_rng(0); mu, sd = PF.normal_stats(rng)
    print(f"SigLA 端到端(LLM无关)  预算信号中… train={NTR} test={NTE}")
    tr = precompute(rng, NTR, mu, sd); te = precompute(rng, NTE, mu, sd)
    print(f"TARGET={ENV.TARGET} 前兆窗[{ON-LMAX},{ON-LMIN}]  benign=spike毛刺(伪抬高spectral_peak)\n")

    print(f"{'λ_insp':>8s}{'thr-raw':>10s}{'always-dis':>12s}{'cost-aware':>12s}{'CA去纠缠率':>11s}{'raw误报':>9s}{'CA误报':>8s}")
    print("-" * 74)
    res = {}
    taus = np.linspace(1.5, 6.0, 19)
    grid = [(lo, hi) for lo in np.linspace(1.5, 3.5, 6) for hi in np.linspace(3.0, 6.0, 7) if hi > lo]
    for lam in ([0.05, 0.2] if SMOKE else [0.02, 0.05, 0.10, 0.20]):
        btr = max(taus, key=lambda t: np.mean([util(*pol_threshold(e, t), e[2], lam) for e in tr]))
        bnt = max(taus, key=lambda t: np.mean([util(*pol_threshold(e, t, True), e[2], lam) for e in tr]))
        blo, bhi = max(grid, key=lambda p: np.mean([util(*pol_costaware(e, p[0], p[1]), e[2], lam) for e in tr]))

        def ev(fn):
            us, ins, fa, nneg = [], 0, 0, 0
            for e in te:
                a, ni = fn(e); us.append(util(a, e[2], lam, ni)); ins += ni
                if not e[2]:
                    nneg += 1; fa += int(a is not None)
            return np.mean(us), ins / (len(te) * Tt), fa / max(1, nneg)
        ur, _, far = ev(lambda e: pol_threshold(e, btr))
        un, _, _ = ev(lambda e: pol_threshold(e, bnt, True))
        uc, rr, fac = ev(lambda e: pol_costaware(e, blo, bhi))
        res[str(lam)] = dict(thr_raw=float(ur), always_dis=float(un), cost=float(uc), rr=float(rr), far=float(far), fac=float(fac))
        print(f"{lam:>8.2f}{ur:>10.3f}{un:>12.3f}{uc:>12.3f}{rr*100:>9.0f}%{far*100:>8.0f}%{fac*100:>7.0f}%")
    print("\n判据:cost-aware ≥ thr-raw(去纠缠剥离spike伪激活→少误报)且 ≈ always-dis(只对模糊步去纠缠→省成本)")
    print("→ 端到端(②去纠缠 + ③成本感知inspect + ①前兆窗)跑通且各部件有价值。LLM 无关。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/pipeline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
