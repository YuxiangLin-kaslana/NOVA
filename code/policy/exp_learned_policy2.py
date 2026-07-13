#!/usr/bin/env python3
"""【学习式成本感知动作策略:BC(草案④阶段1)+ 可选 RL 微调】

REINFORCE 从零崩(稀疏延迟奖励)。按草案④:先用**专家轨迹行为克隆**,再 RL 微调。
专家=基于观测的成本感知策略(2步:风险模糊→inspect;下一步看诊断 d 决定 alarm/wait)。
state=[o, t/T, last_d, has_d];action={wait,inspect,alarm}。验证:BC 学到的策略≈手调专家(自己学会 inspect 行为),
且换 λ_insp 时专家(进而 BC)自动改读取率 → 学习式动作策略可行(LLM 无关)。
用法: sbatch policy/run_learned.sh (指向本文件)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/u/ylin30/sigLA/code")
import policy.exp_action_policy as AP

T, ONSET, LMAX, LMIN, SIG_OBS, SIG_DIAG = AP.T, AP.ONSET, AP.LMAX, AP.LMIN, AP.SIG_OBS, AP.SIG_DIAG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"


def terminal_reward(astep, isr, lam, n_ins):
    W = AP.W; u = -lam * n_ins
    if isr:
        if astep is None or astep >= ONSET: u -= W["miss"]
        elif ONSET - LMAX <= astep <= ONSET - LMIN: u += W["valid"]
        elif astep < ONSET - LMAX: u -= W["prem"]
        else: u -= W["late"]
    elif astep is not None: u -= W["fa"]
    return u


def expert_act(s, lo, hi, tauf):
    o, _, last_d, has_d = s
    if has_d > 0.5:                       # 上一步刚 inspect → 看诊断决定
        return 2 if last_d > tauf else 0
    if o >= hi: return 2                  # 极高 → 直接 alarm
    if o > lo: return 1                   # 模糊 → inspect
    return 0                              # wait


def rollout(ep, act_fn, rng):
    """2步 inspect 动力学。act_fn(state)->action。返回 [(state,action)], astep, n_ins, isr。"""
    risk, diag, isr = ep
    last_d, has_d = -1.0, 0.0; n_ins = 0; astep = None; traj = []
    for t in range(T):
        o = risk[t] + rng.normal(0, SIG_OBS)
        s = [float(o), t / T, float(last_d), float(has_d)]
        a = act_fn(s); traj.append((s, a))
        if a == 2:
            astep = t; break
        elif a == 1:
            n_ins += 1; last_d = float(diag[t] + rng.normal(0, SIG_DIAG)); has_d = 1.0
        else:
            last_d, has_d = -1.0, 0.0
    return traj, astep, n_ins, isr


def tune_expert(lam, rng):
    """train 上网格调专家 (lo,hi,tauf) 最大化 utility。"""
    eps = [AP.make_episode(i % 2 == 0, rng) + (i % 2 == 0,) for i in range(400)]
    ge = np.random.default_rng(3)
    best = None
    for lo in np.linspace(0.3, 0.7, 5):
        for hi in np.linspace(0.9, 1.5, 5):
            for tf in np.linspace(0.4, 0.7, 4):
                us = []
                for ep in eps:
                    _, a, ni, isr = rollout(ep, lambda s: expert_act(s, lo, hi, tf), ge)
                    us.append(terminal_reward(a, isr, lam, ni))
                m = float(np.mean(us))
                if best is None or m > best[0]: best = (m, lo, hi, tf)
    return best[1], best[2], best[3]


def bc_train(lam, lo, hi, tauf, rng, epochs=40):
    net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    ge = np.random.default_rng(5)
    # 收集专家轨迹
    S, A = [], []
    for i in range(1500 if not SMOKE else 200):
        ep = AP.make_episode(i % 2 == 0, rng) + (i % 2 == 0,)
        traj, *_ = rollout(ep, lambda s: expert_act(s, lo, hi, tauf), ge)
        for s, a in traj: S.append(s); A.append(a)
    S = torch.tensor(S, dtype=torch.float32, device=device); A = torch.tensor(A, device=device)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(S))
        for i in range(0, len(S), 256):
            idx = perm[i:i + 256]
            loss = lossf(net(S[idx]), A[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def net_act(net):
    def f(s):
        return int(torch.argmax(net(torch.tensor(s, dtype=torch.float32, device=device))))
    return f


def evaluate(act_fn, lam, n=2000):
    rng = np.random.default_rng(99); us = []; ins = valid = fa = miss = 0; npos = 0
    for i in range(n):
        isr = i % 2 == 0; npos += isr
        ep = AP.make_episode(isr, rng) + (isr,)
        _, a, ni, _ = rollout(ep, act_fn, rng)
        us.append(terminal_reward(a, isr, lam, ni)); ins += ni
        if isr:
            valid += int(a is not None and ONSET - LMAX <= a <= ONSET - LMIN); miss += int(a is None or a >= ONSET)
        elif a is not None: fa += 1
    return dict(u=float(np.mean(us)), valid=valid / npos, fa=fa / (n - npos), read=ins / n)


def main():
    print(f"学习式动作策略 BC(草案④阶段1)  device={device}\n")
    print(f"{'λ_insp':>8s}{'专家手调':>10s}{'BC学习式':>10s}{'BC有效预警':>11s}{'BC误报':>8s}{'BC读取':>8s}{'专家(lo,hi,tf)':>16s}")
    print("-" * 74)
    res = {}
    for lam in ([0.05, 0.20] if SMOKE else [0.02, 0.05, 0.10, 0.20, 0.40]):
        rng = np.random.default_rng(0)
        lo, hi, tf = tune_expert(lam, rng)
        exp_u = evaluate(lambda s: expert_act(s, lo, hi, tf), lam)
        net = bc_train(lam, lo, hi, tf, rng)
        bc_u = evaluate(net_act(net), lam)
        res[str(lam)] = dict(expert=exp_u["u"], bc=bc_u["u"], valid=bc_u["valid"], fa=bc_u["fa"], read=bc_u["read"])
        print(f"{lam:>8.2f}{exp_u['u']:>10.3f}{bc_u['u']:>10.3f}{bc_u['valid']*100:>9.0f}%{bc_u['fa']*100:>7.0f}%"
              f"{bc_u['read']:>7.2f}   ({lo:.2f},{hi:.2f},{tf:.2f})")
    print("\n判据:BC 学习式 ≈ 专家手调(从轨迹学会 inspect 行为),且换 λ_insp 自动改读取率 → 草案④'学习动作策略'可行(LLM无关)。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/learned_policy_bc.json", "w"), indent=2)


if __name__ == "__main__":
    main()
