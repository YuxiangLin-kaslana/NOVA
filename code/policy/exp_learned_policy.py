#!/usr/bin/env python3
"""【学习式成本感知动作策略(REINFORCE)】对上草案④"学习动作策略",取代手调阈值。

在动作模拟 MDP 上(state=观测风险+时序+最近诊断;action={wait,inspect,alarm};reward=成本感知 utility)
用策略梯度直接学策略。验证:学习策略**自己发现**"风险模糊→inspect→靠诊断决定 alarm"的行为,
匹配/超过手调成本感知策略,并**随 λ_insp 自动改变读取率**(无需人工调阈值)。
对比:阈值-累积ρ(手调强基线)/ 成本感知手调 / **学习式(REINFORCE)**。
用法: sbatch policy/run_learned.sh
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/u/ylin30/sigLA/code")
import policy.exp_action_policy as AP   # make_episode, ONSET, LMAX, LMIN, W, SIG_OBS, SIG_DIAG, run_threshold, run_costaware

T, ONSET, LMAX, LMIN = AP.T, AP.ONSET, AP.LMAX, AP.LMIN
SIG_OBS, SIG_DIAG = AP.SIG_OBS, AP.SIG_DIAG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
EPOCHS = 60 if not SMOKE else 8
BATCH = 256


def terminal_reward(alarm_step, is_real, lam, n_ins):
    W = AP.W
    u = -lam * n_ins
    if is_real:
        if alarm_step is None or alarm_step >= ONSET:
            u -= W["miss"]
        elif ONSET - LMAX <= alarm_step <= ONSET - LMIN:
            u += W["valid"]
        elif alarm_step < ONSET - LMAX:
            u -= W["prem"]
        else:
            u -= W["late"]
    elif alarm_step is not None:
        u -= W["fa"]
    return u


def rollout(net, ep, rng, greedy=False):
    """跑一个 episode。state=[o, t/T, last_d, has_d]。返回 logps, 终局 reward 占位(外部按 lam 算), 动作统计。"""
    risk, diag, is_real = ep
    last_d, has_d = -1.0, 0.0
    logps = []; n_ins = 0; alarm_step = None
    for t in range(T):
        o = risk[t] + rng.normal(0, SIG_OBS)
        s = torch.tensor([o, t / T, last_d, has_d], dtype=torch.float32, device=device)
        logits = net(s)
        if greedy:
            a = int(torch.argmax(logits))
            logps.append(torch.tensor(0.0, device=device))
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample()); logps.append(dist.log_prob(torch.tensor(a, device=device)))
        if a == 0:                                  # wait
            last_d, has_d = -1.0, 0.0
        elif a == 1:                                # inspect
            n_ins += 1; last_d = float(diag[t] + rng.normal(0, SIG_DIAG)); has_d = 1.0
        else:                                       # alarm
            alarm_step = t; break
    return logps, alarm_step, n_ins, is_real


def train(lam, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 3)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    baseline = 0.0
    for ep_i in range(EPOCHS):
        eps = [AP.make_episode(i % 2 == 0, rng) + (i % 2 == 0,) for i in range(BATCH)]
        losses = []; rets = []
        for risk, diag, isr in eps:
            logps, astep, n_ins, _ = rollout(net, (risk, diag, isr), rng)
            R = terminal_reward(astep, isr, lam, n_ins)
            rets.append(R)
            adv = R - baseline
            losses.append(-torch.stack(logps).sum() * adv)
        baseline = 0.9 * baseline + 0.1 * float(np.mean(rets))
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def evaluate(net, lam, n=2000, seed=99):
    rng = np.random.default_rng(seed)
    us, ins_tot, valid, fa, miss = [], 0, 0, 0, 0; npos = 0
    for i in range(n):
        isr = i % 2 == 0; npos += isr
        risk, diag = AP.make_episode(isr, rng)
        _, astep, n_ins, _ = rollout(net, (risk, diag, isr), rng, greedy=True)
        us.append(terminal_reward(astep, isr, lam, n_ins)); ins_tot += n_ins
        if isr:
            valid += int(astep is not None and ONSET - LMAX <= astep <= ONSET - LMIN)
            miss += int(astep is None or astep >= ONSET)
        elif astep is not None:
            fa += 1
    nneg = n - npos
    return dict(u=float(np.mean(us)), valid=valid / npos, fa=fa / nneg, miss=miss / npos, read=ins_tot / n)


def handtuned(lam):
    """手调基线(复用 AP):阈值-累积ρ 与 成本感知,train 调参 test 评估。"""
    AP.W["insp"] = lam
    rng = np.random.default_rng(0)
    tr = [AP.make_episode(i % 2 == 0, rng) for i in range(300)]; tr_y = [i % 2 == 0 for i in range(300)]
    ge = np.random.default_rng(1)
    mu = lambda fn: float(np.mean([AP.score_episode(*(fn(e, ge) + (y,))) for e, y in zip(tr, tr_y)]))
    bta = max(np.linspace(0.2, 1.1, 19), key=lambda tau: mu(lambda e, g: AP.run_threshold(e, g, tau, True)))
    grid = [(lo, td) for lo in np.linspace(0.3, 0.7, 9) for td in np.linspace(0.3, 0.7, 9)]
    bca = max(grid, key=lambda p: mu(lambda e, g: AP.run_costaware(e, g, p[0], p[1])))
    gt = np.random.default_rng(99); te = [AP.make_episode(i % 2 == 0, gt) for i in range(2000)]; te_y = [i % 2 == 0 for i in range(2000)]
    ge2 = np.random.default_rng(7)
    ur = float(np.mean([AP.score_episode(*(AP.run_threshold(e, ge2, bta, True) + (y,))) for e, y in zip(te, te_y)]))
    uc = float(np.mean([AP.score_episode(*(AP.run_costaware(e, ge2, bca[0], bca[1]) + (y,))) for e, y in zip(te, te_y)]))
    return ur, uc


def main():
    print(f"学习式成本感知动作策略(REINFORCE)  device={device} EPOCHS={EPOCHS}\n")
    print(f"{'λ_insp':>8s}{'阈值累积ρ':>11s}{'成本感知手调':>13s}{'学习式(RL)':>12s}{'RL有效预警':>11s}{'RL误报':>8s}{'RL读取':>8s}")
    print("-" * 74)
    res = {}
    for lam in ([0.05, 0.20] if SMOKE else [0.02, 0.05, 0.10, 0.20, 0.40]):
        ur, uc = handtuned(lam)
        net = train(lam)
        ev = evaluate(net, lam)
        res[str(lam)] = dict(thr_rho=ur, cost_hand=uc, rl=ev["u"], read=ev["read"], valid=ev["valid"], fa=ev["fa"])
        print(f"{lam:>8.2f}{ur:>11.3f}{uc:>13.3f}{ev['u']:>12.3f}{ev['valid']*100:>9.0f}%{ev['fa']*100:>7.0f}%{ev['read']:>7.2f}")
    print("\n判据:学习式 RL ≥ 成本感知手调(自己发现 inspect 策略)且随 λ_insp 自动改读取率 → 验证草案④'学习动作策略'。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/learned_policy.json", "w"), indent=2)


if __name__ == "__main__":
    main()
