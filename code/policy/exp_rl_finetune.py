#!/usr/bin/env python3
"""【草案④阶段2:BC-init 的 RL 微调】从 BC 策略出发,REINFORCE(熵正则+baseline)微调,超越手调专家。

从零 REINFORCE 会崩;BC 初始化后微调稳。看 RL 能否在专家弱处改进(小λ的~10%误报、λ=0.40 专家崩溃处)。
对比:专家手调 / BC / **BC+RL微调**。用法: sbatch policy/run_rl.sh
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/u/ylin30/sigLA/code")
import policy.exp_action_policy as AP
import policy.exp_learned_policy2 as LP

T, ONSET, LMAX, LMIN, SIG_OBS, SIG_DIAG = LP.T, LP.ONSET, LP.LMAX, LP.LMIN, LP.SIG_OBS, LP.SIG_DIAG
device = LP.device
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"


def rl_rollout(net, ep, rng):
    risk, diag, isr = ep; last_d, has_d = -1.0, 0.0; n_ins = 0; astep = None; logps, ents = [], []
    for t in range(T):
        o = risk[t] + rng.normal(0, SIG_OBS)
        s = torch.tensor([float(o), t / T, float(last_d), float(has_d)], dtype=torch.float32, device=device)
        dist = torch.distributions.Categorical(logits=net(s))
        a = dist.sample()
        logps.append(dist.log_prob(a)); ents.append(dist.entropy())
        ai = int(a)
        if ai == 2:
            astep = t; break
        elif ai == 1:
            n_ins += 1; last_d = float(diag[t] + rng.normal(0, SIG_DIAG)); has_d = 1.0
        else:
            last_d, has_d = -1.0, 0.0
    return logps, ents, astep, n_ins, isr


def rl_finetune(net, lam, epochs=80, lr=5e-4, beta=0.03, batch=256):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(11); base = 0.0
    for ep_i in range(epochs if not SMOKE else 8):
        eps = [AP.make_episode(i % 2 == 0, rng) + (i % 2 == 0,) for i in range(batch)]
        losses, rets = [], []
        for e in eps:
            logps, ents, astep, n_ins, isr = rl_rollout(net, e, rng)
            R = LP.terminal_reward(astep, isr, lam, n_ins); rets.append(R)
            adv = R - base
            pg = -torch.stack(logps).sum() * adv - beta * torch.stack(ents).sum()
            losses.append(pg)
        base = 0.9 * base + 0.1 * float(np.mean(rets))
        opt.zero_grad(); torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    return net


def main():
    print(f"BC-init RL 微调(草案④阶段2)  device={device}\n")
    print(f"{'λ_insp':>8s}{'专家手调':>10s}{'BC':>9s}{'BC+RL微调':>11s}{'RL有效预警':>11s}{'RL误报':>8s}{'RL读取':>8s}")
    print("-" * 70)
    res = {}
    for lam in ([0.05, 0.40] if SMOKE else [0.02, 0.05, 0.10, 0.20, 0.40]):
        rng = np.random.default_rng(0)
        lo, hi, tf = LP.tune_expert(lam, rng)
        exp_u = LP.evaluate(lambda s: LP.expert_act(s, lo, hi, tf), lam)["u"]
        net = LP.bc_train(lam, lo, hi, tf, rng)
        bc_u = LP.evaluate(LP.net_act(net), lam)["u"]
        net = rl_finetune(net, lam)
        rl = LP.evaluate(LP.net_act(net), lam)
        res[str(lam)] = dict(expert=exp_u, bc=bc_u, rl=rl["u"], valid=rl["valid"], fa=rl["fa"], read=rl["read"])
        print(f"{lam:>8.2f}{exp_u:>10.3f}{bc_u:>9.3f}{rl['u']:>11.3f}{rl['valid']*100:>9.0f}%{rl['fa']*100:>7.0f}%{rl['read']:>7.2f}")
    print("\n判据:BC+RL ≥ BC ≥ 专家,尤其在专家弱处(小λ误报、λ=0.40崩溃)RL 改进 → 草案④阶段2(成本感知离线/在线RL)成立。")
    json.dump(res, open("/u/ylin30/sigLA/code/runs/rl_finetune.json", "w"), indent=2)


if __name__ == "__main__":
    main()
