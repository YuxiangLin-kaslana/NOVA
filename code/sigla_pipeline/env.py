"""SigLA 前兆事件流环境(LLM 无关)。目标概念 TARGET=oscillation 的**类型化早预警**。

real 事件:前兆窗内 TARGET(oscillation)逐步增强 → onset。应在前兆窗内报警(报对类型)。
benign 干扰:spike 毛刺(宽频谱**伪抬高** spectral_peak=oscillation 的签名)→ 无 onset。
  → 不去纠缠会把它误判成 oscillation 前兆而误报;去纠缠(压制 spike→spectral_peak 掉)应剥离,避免误报。
这正好让 ②校准去纠缠 在 ③动作策略 上产生可测价值,并由 ①前兆窗 评判。
"""
from __future__ import annotations
import numpy as np
import sigla_exp.ovbench as CB

WIN, NVARS = CB.WIN, CB.NVARS
T = 24
ONSET = 20
LMAX, LMIN = 8, 2                 # 前兆窗 [12,18]
TARGET = "variance_burst"
PREC_MIN = 0.25                  # 前兆起始强度


def _spike_blip(rng):
    x = CB.base_normal(rng)
    if rng.random() < 0.7:
        for _ in range(int(rng.integers(6, 12))):
            CB.INJ["spike"](x, rng)        # spike 串:抬高窗方差(伪激活 variance_burst)
    return x.astype(np.float32)


def make_episode(is_real, rng):
    """返回 windows[T](逐步窗)与 is_real。"""
    W = []
    for t in range(T):
        if is_real and ONSET - LMAX <= t < ONSET:
            a = (t - (ONSET - LMAX)) / LMAX
            s = PREC_MIN + (1 - PREC_MIN) * a
            W.append(CB.make_window_strength(TARGET, rng, s))       # 渐增 oscillation 前兆
        elif is_real and t >= ONSET:
            W.append(CB.make_window_strength(TARGET, rng, 1.0))     # 事件
        else:
            W.append(_spike_blip(rng))                              # 正常/毛刺干扰
    return W, is_real
