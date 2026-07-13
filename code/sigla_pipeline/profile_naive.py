"""朴素(会纠缠)证据画像 + 去纠缠,**百分位校准**(尺度无关,不爆炸)。

概念→朴素签名:spike→max|z|;variance_burst→窗方差;level_shift→半窗均值差;trend→线性斜率;
oscillation→高频能量占比;correlation_break→1−平均|corr|。
纠缠:spike 串会抬高 方差(伪激活 variance_burst);压制 spike(削极值)→ 方差掉 → 去纠缠剥离。
真 variance_burst=持续中等噪声(无极值)→ 削极值不掉 → 净分仍高。
"""
from __future__ import annotations
import numpy as np
import sigla_exp.ovbench as CB
from sigla_pipeline.profile import SUPPRESS, CONCEPTS

WIN, NVARS = CB.WIN, CB.NVARS
SIG = {"spike": "maxz", "variance_burst": "var", "level_shift": "step",
       "trend": "slope", "oscillation": "hf", "correlation_break": "decorr"}


def ev_naive(x):
    mu = x.mean(0); sdv = x.std(0) + 1e-6; xc = x - mu
    maxz = float(np.percentile(np.abs(xc / sdv), 99.5))
    var = float(x.var(0).mean())
    mid = WIN // 2
    step = float(np.max(np.abs(x[mid:].mean(0) - x[:mid].mean(0))))
    t = np.arange(WIN); slope = float(np.max(np.abs(np.polyfit(t, x, 1)[0])))
    F = np.abs(np.fft.rfft(xc, axis=0))[1:]
    hf = float(np.max(F[14:].sum(0) / (F.sum(0) + 1e-6)))
    m = x.std(0) > 1e-6
    if int(m.sum()) >= 2:
        C = np.corrcoef(x[:, m], rowvar=False); n = C.shape[0]
        decorr = float(1.0 - (np.abs(C).sum() - n) / (n * (n - 1)))
    else:
        decorr = 0.0
    return {"maxz": maxz, "var": var, "step": step, "slope": slope, "hf": hf, "decorr": decorr}


def normal_stats(rng, n=400):
    """返回 (normal_samples, None)。normal_samples[stat]=正常窗该统计量的排序样本(供百分位校准)。"""
    evs = [ev_naive(CB.base_normal(rng)) for _ in range(n)]
    samp = {k: np.sort([e[k] for e in evs]) for k in SIG.values()}
    return samp, None


def calibrated(ev, samp, _sd=None):
    """百分位校准:每概念签名统计量在正常样本中的上尾分位 → 有界异常分(尺度无关,可比)。"""
    out = {}
    for c in CONCEPTS:
        s = SIG[c]; arr = samp[s]
        pct = np.searchsorted(arr, ev[s]) / len(arr)
        out[c] = float(-np.log(1.0 - min(pct, 0.999) + 1e-3))      # 上尾→分越高越异常(~0..7)
    return out


def disentangle(x, samp, _sd=None, co_thresh=0.4):
    z = calibrated(ev_naive(x), samp)
    prim = max(z, key=z.get)
    zs = calibrated(ev_naive(SUPPRESS[prim](x)), samp)
    net = dict(z); co = {}
    for j in CONCEPTS:
        if j == prim:
            continue
        frac = (z[j] - zs[j]) / (abs(z[j]) + 1e-6)
        if z[j] > 1.5 and frac > co_thresh:
            co[j] = float(frac); net[j] = float(zs[j])
    return dict(z=z, primary=prim, prim_z=float(z[prim]), co_responses=co, net=net)
