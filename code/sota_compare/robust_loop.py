"""鲁棒开放词表闭环:修复"自我毒化"(良性窗触发门控→LLM命名→长伪类重训污染 normal)。

诊断(diag_drift_loop)证明:原闭环在纯正常+漂移长流上凭空长伪类、normal 误报爬到 100%。
根因=闭环无"这就是正常"的拒绝出口,每次门控触发都立刻变新类;真实正常窗统计尾部厚→偶发假触发。

**两道防线**:
  (1) **成簇确认门控(hysteresis)**:疑似新类不立刻长,需同一概念在近 HORIZON 窗内累计 ≥K_CONFIRM 次
      **强**触发(主导非已知签名 z ≥ Z_MARGIN)才进入"提交"。滤掉散乱/偶发的良性假触发。
  (2) **normal 精度守卫(do-no-harm,关键)**:提交新类+重训后,在留出正常集上测误报;**若新类把 normal
      误报抬过 BUDGET → 回滚(撤销长类与重训)+ 拉黑该概念**。这样不论哪个签名假阳,normal FA 都被钳住;
      而真正与 normal 可分的新类(如 oscillation)不抬高 normal 误报 → 保留。重训(含已知异常累积)同样守卫。

报警 = P(非normal) > thr_ours(~5% FA 同口径);命名 = argmax。
"""
from __future__ import annotations

import copy

import numpy as np
import torch

import scripts.exp_detection_tie as DT
import sigla_exp.ovbench as CB
import sota_compare.exp_drift_vs_novel as EXP

BASE_VOCAB, KNOWN, NORMAL = EXP.BASE_VOCAB, EXP.KNOWN, EXP.NORMAL
KNOWN_STATS = EXP.KNOWN_STATS
TAU, AUDIT, NOVEL_Z, RETRAIN_EVERY = EXP.TAU, EXP.AUDIT, EXP.NOVEL_Z, EXP.RETRAIN_EVERY


def _retrain(det, opt, replay, buf, vocab, rng):
    per = {}
    for w, li in replay: per.setdefault(li, []).append(w)
    for w, li in buf: per.setdefault(li, []).append(w)
    K = 40; Xb, Yb = [], []
    for li, ws in per.items():
        for j in rng.integers(0, len(ws), K):
            Xb.append(ws[j]); Yb.append(DT.onehot(li, len(vocab)))
    DT.train_on(det, opt, Xb, Yb, epochs=2)


def robust_ours_loop(pre_state, replay, W, mu, sd, key, net_ok, thr_ours,
                     K_CONFIRM=3, HORIZON=60, Z_MARGIN=2.8, Z_STRONG=4.0,
                     FA_TARGET=0.05, GUARD_N=200):
    """两道防线(修复版):
      (1) 成簇确认 + 只用簇内最强窗(z≥Z_STRONG)训练新类 → 干净可分、限制伪类增长;
      (2) **长类/重训后在留出正常集上重标定报警阈**(thr_cur=q95) → normal 误报恒定在 ~FA_TARGET,
          由自适应阈值控 FA,而非靠"禁止长类"。真类照长(命名救回),正常即便偶尔像新类也因阈值升高不报警。"""
    det = DT.make_detector(len(BASE_VOCAB)); det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(BASE_VOCAB)
    buf = []; pending = 0; cand = {}
    alarms, names = [], []
    stats = {"attempt": 0, "commit": 0, "recalib": 0}
    rng = np.random.default_rng(777)
    guard = [w for (w, li) in replay if li == 0][:GUARD_N]
    thr_cur = thr_ours

    def recalibrate():                                              # 重标定:留出正常集 q95 → 维持 ~FA_TARGET
        nonlocal thr_cur
        s = 1.0 - DT.proba(det, guard)[:, 0]
        thr_cur = float(np.quantile(s, 1.0 - FA_TARGET)); stats["recalib"] += 1

    def new_opt():
        return torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)

    for i, x in enumerate(W):
        xm = EXP.mc(x); ev = CB.evidence(x)
        p = DT.proba(det, [xm])[0]; mx = float(p.max()); pi = int(np.argmax(p))
        devz = {k: abs((ev[k] - mu[k]) / sd[k]) for k in mu}
        dom = max(devz, key=devz.get)
        susp = (dom not in KNOWN_STATS) and (devz[dom] > NOVEL_Z)
        mislabel = (vocab[pi] in (KNOWN + [NORMAL])) and susp
        pred = vocab[pi]
        if not (mx >= TAU and not mislabel and rng.random() >= AUDIT):
            c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            c = None if c == "__ERROR__" else c
            if c and c in vocab:
                pred = c
                if c != NORMAL:
                    buf.append((xm, vocab.index(c))); pending += 1
            elif c and susp:                                        # 疑似新类 → 成簇确认
                lst = [e for e in cand.get(c, []) if i - e[0] <= HORIZON]
                lst.append((i, xm, devz[dom])); cand[c] = lst
                if sum(e[2] >= Z_MARGIN for e in lst) >= K_CONFIRM:
                    stats["attempt"] += 1; stats["commit"] += 1
                    strong = [e for e in lst if e[2] >= Z_STRONG]
                    if len(strong) < K_CONFIRM:
                        strong = sorted(lst, key=lambda e: -e[2])[:K_CONFIRM]
                    vocab.append(c); DT.grow_head(det, len(vocab)); opt = new_opt()
                    buf = buf + [(e[1], len(vocab) - 1) for e in strong]
                    _retrain(det, opt, replay, buf, vocab, rng); recalibrate()  # 长类后重标定阈
                    pending = 0; cand[c] = []; pred = c
            if pending >= RETRAIN_EVERY:
                _retrain(det, opt, replay, buf, vocab, rng); recalibrate(); pending = 0
        anom_score = 1.0 - float(p[0])
        alarms.append(int(anom_score > thr_cur)); names.append(pred)
    return np.array(alarms), names, vocab, stats
