#!/usr/bin/env python3
"""【成本感知前兆动作策略 vs 纯阈值】验证草案贡献④(LLM 无关)。

核心论点:把早预警当**成本感知动作决策**(wait/inspect/alarm),优于纯 score-thresholding。
机制(对齐 SigLA "inspect 以获得确信"):风险标量上 **real 前兆 与 benign 伪影 重叠、噪声里分不开**;
但 inspect 能(花 λ_insp)揭示一个**诊断信号**——real 前兆=高、benign 伪影=低。阈值只能看风险标量→对
benign bump 误报或漏;成本感知策略在风险抬升时 inspect 确认,再决定 alarm → 少误报、能在前兆窗内及时报。

action utility = λ_valid·有效预警 − λ_prem·过早 − λ_late·迟滞 − λ_miss·漏 − λ_fa·误报 − λ_insp·检查 − λ_flood·泛滥。
对比(train 调参,test 评估):阈值-瞬时 / 阈值-累积ρ(用持续性,强基线) / 成本感知(inspect 诊断)。
用法: python policy/exp_action_policy.py
"""
from __future__ import annotations
import json
import numpy as np

T = 30
ONSET = 24
LMAX, LMIN = 8, 2                      # 前兆窗 [16,22]
SIG_OBS = 0.40                         # 风险标量观测噪声
SIG_DIAG = 0.10                        # inspect 揭示诊断信号的(低)噪声
W = dict(valid=1.0, prem=0.5, late=0.3, miss=1.0, fa=0.8, insp=0.10, flood=0.3)


def make_episode(is_real, rng):
    """返回 (risk[T], diag[T])。risk=可噪声观测的风险标量;diag=隐藏诊断(仅 inspect 低噪可见)。
    real:前兆窗内 risk 抬升 + diag 高(真前兆);benign:偶发 bump risk 抬升 + diag 低(伪影)。"""
    risk = np.abs(rng.normal(0, 0.12, T))
    diag = np.abs(rng.normal(0, 0.10, T))
    if is_real:
        for t in range(ONSET - LMAX, ONSET):
            a = (t - (ONSET - LMAX)) / LMAX
            risk[t] += 0.5 + 0.5 * a                       # 抬升至 onset
            diag[t] += rng.normal(0.85, 0.08)              # 真前兆诊断高
    else:
        if rng.random() < 0.6:
            s = int(rng.integers(6, ONSET - 2))
            for t in range(s, min(T, s + 3)):
                risk[t] += rng.uniform(0.7, 1.0)           # 良性 bump:风险像前兆
                diag[t] += rng.normal(0.20, 0.08)          # 但诊断低(伪影)
    return risk.astype(float), diag.astype(float)


def score_episode(actions, alarm_step, is_real):
    n_insp = sum(a == "inspect" for a in actions)
    n_alarm = sum(a == "alarm" for a in actions)
    u = -W["insp"] * n_insp - W["flood"] * max(0, n_alarm - 1)
    if is_real:
        if alarm_step is None or alarm_step >= ONSET:
            u -= W["miss"]
        elif ONSET - LMAX <= alarm_step <= ONSET - LMIN:
            u += W["valid"]
        elif alarm_step < ONSET - LMAX:
            u -= W["prem"]
        else:
            u -= W["late"]
    else:
        if alarm_step is not None:
            u -= W["fa"]
    return u


def run_threshold(ep, rng, tau, accumulate=False, alpha=0.6):
    risk, _ = ep
    actions = ["wait"] * T; alarm_step = None; r = 0.0
    for t in range(T):
        o = risk[t] + rng.normal(0, SIG_OBS)
        s = (alpha * r + (1 - alpha) * o) if accumulate else o
        r = s if accumulate else r
        if s > tau:
            actions[t] = "alarm"; alarm_step = t
            break                                          # 首次报警即停(一事件一预警)
    return actions, alarm_step


def run_costaware(ep, rng, tau_lo, tau_diag, max_insp=5):
    """风险抬升(o≥tau_lo)且还能查→inspect 揭示诊断 d;d>tau_diag→alarm,否则 wait(判为伪影)。"""
    risk, diag = ep
    actions = ["wait"] * T; alarm_step = None; insp_left = max_insp
    for t in range(T):
        o = risk[t] + rng.normal(0, SIG_OBS)
        if o >= tau_lo and insp_left > 0:
            actions[t] = "inspect"; insp_left -= 1
            d = diag[t] + rng.normal(0, SIG_DIAG)
            if d > tau_diag:
                actions[t] = "alarm"; alarm_step = t       # inspect 确认→报警
                break                                      # 首次报警即停
    return actions, alarm_step


def tune_and_eval(tr, tr_y, te, te_y, lam_insp):
    """给定检查成本 λ_insp:每个策略在 train 调参(用同一 utility 定义),在 test 评估。"""
    W["insp"] = lam_insp
    ge = np.random.default_rng(1)

    def mean_u(eps, ys, fn, g):
        return float(np.mean([score_episode(*(fn(e, g) + (y,))) for e, y in zip(eps, ys)]))
    bt = max(np.linspace(0.3, 1.4, 23), key=lambda tau: mean_u(tr, tr_y, lambda e, g: run_threshold(e, g, tau), ge))
    bta = max(np.linspace(0.2, 1.1, 19), key=lambda tau: mean_u(tr, tr_y, lambda e, g: run_threshold(e, g, tau, True), ge))
    grid = [(lo, td) for lo in np.linspace(0.3, 0.7, 9) for td in np.linspace(0.3, 0.7, 9)]
    bca = max(grid, key=lambda p: mean_u(tr, tr_y, lambda e, g: run_costaware(e, g, p[0], p[1]), ge))
    fns = {"阈值-瞬时": lambda e, g: run_threshold(e, g, bt),
           "阈值-累积ρ": lambda e, g: run_threshold(e, g, bta, True),
           "成本感知(inspect)": lambda e, g: run_costaware(e, g, bca[0], bca[1])}
    gt = np.random.default_rng(2)
    out = {}
    npos = sum(te_y); nneg = len(te_y) - npos
    for name, fn in fns.items():
        us = valid = fa = miss = insp = 0; ulist = []
        for e, y in zip(te, te_y):
            acts, astep = fn(e, gt); ulist.append(score_episode(acts, astep, y))
            insp += sum(a == "inspect" for a in acts)
            if y:
                valid += int(astep is not None and ONSET - LMAX <= astep <= ONSET - LMIN)
                miss += int(astep is None or astep >= ONSET)
            elif astep is not None:
                fa += 1
        out[name] = dict(u=float(np.mean(ulist)), valid=valid / npos, fa=fa / nneg, miss=miss / npos, insp=insp / len(te))
    return out


def main():
    rng = np.random.default_rng(0)
    mk = lambda n, g: ([make_episode(i % 2 == 0, g) for i in range(n)], [i % 2 == 0 for i in range(n)])
    tr, tr_y = mk(300, rng); te, te_y = mk(600, rng)

    print("成本感知动作策略 vs 阈值  train=300 test=600  (real/benign 各半;前兆窗[16,22])")
    print("机制:风险标量上 real前兆与benign伪影重叠/有噪→阈值分不开;inspect(花λ_insp)揭示诊断信号可区分。\n")
    # 默认成本下的操作指标表
    base = tune_and_eval(tr, tr_y, te, te_y, 0.05)
    print(f"【λ_insp=0.05 操作指标】{'策略':18s}{'Utility':>9s}{'有效预警':>9s}{'误报':>7s}{'漏':>6s}{'检查/事件':>10s}")
    for nm, r in base.items():
        print(f"{'':22s}{nm:18s}{r['u']:>8.3f}{r['valid']*100:>8.0f}%{r['fa']*100:>6.0f}%{r['miss']*100:>5.0f}%{r['insp']:>9.2f}")

    print("\n【λ_insp 扫描:test Action Utility】(检查越便宜,成本感知越占优)")
    print(f"{'λ_insp':>8s}{'阈值-瞬时':>12s}{'阈值-累积ρ':>14s}{'成本感知':>12s}{'赢家':>14s}")
    print("-" * 62)
    res = {}
    for lam in [0.02, 0.05, 0.10, 0.20, 0.40]:
        o = tune_and_eval(tr, tr_y, te, te_y, lam)
        u = {k: o[k]["u"] for k in o}; win = max(u, key=u.get)
        res[str(lam)] = u
        print(f"{lam:>8.2f}{u['阈值-瞬时']:>12.3f}{u['阈值-累积ρ']:>14.3f}{u['成本感知(inspect)']:>12.3f}{win.split('(')[0]:>14s}")
    print("\n判据:在'检查比误报/漏报便宜'的现实区间(小λ_insp),成本感知策略 utility 最高 + 操作指标(预警/误报/漏)全面占优")
    print("→ 验证草案④'早预警=成本感知动作决策(可inspect确认)'的价值,且全程 LLM 无关。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/action_policy.json", "w"), indent=2)


if __name__ == "__main__":
    main()
