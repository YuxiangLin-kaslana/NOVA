"""② 校准 + 去纠缠 异常画像(SigLA 流水线,LLM 无关)。

evidence(x) → 校准(对正常域 MAD z-score)→ **intervention 去纠缠**:
压制候选主导概念的模式,重算其它概念证据;若某概念证据大幅下降,则它是被主导概念带起来的**伴随响应**,
而非独立异常。输出结构化画像:各概念校准分、主导概念、伴随响应、支撑证据。
"""
from __future__ import annotations
import numpy as np
import sigla_exp.ovbench as CB

WIN, NVARS = CB.WIN, CB.NVARS
CONCEPTS = list(CB.CONCEPTS)
SIG = CB.SIG                                  # concept -> 签名统计量
STATS = list(SIG.values())


# --------------------------- 各概念的"压制"算子 ℐ_k --------------------------- #
def _sup_spike(x):
    y = x.copy()
    for d in range(NVARS):
        m = np.median(y[:, d]); mad = 1.4826 * np.median(np.abs(y[:, d] - m)) + 1e-6
        y[:, d] = np.clip(y[:, d], m - 3.5 * mad, m + 3.5 * mad)     # 削掉稀疏极值
    return y


def _roll_med(a, w):
    w = max(3, w | 1); pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + w]) for i in range(len(a))])


def _sup_levelshift(x):
    y = x.copy()
    for d in range(NVARS):
        rm = _roll_med(y[:, d], WIN // 4)
        y[:, d] = y[:, d] - rm + y[:, d].mean()                     # 去窗内中位数阶跃
    return y


def _sup_oscillation(x):
    y = x.copy(); t_mean = y.mean(0)
    det = y - t_mean
    F = np.fft.rfft(det, axis=0); mag = np.abs(F)
    for d in range(NVARS):
        hi = mag[15:, d]
        if len(hi):
            k = 15 + int(np.argmax(hi))
            for b in (k - 1, k, k + 1):
                if 0 <= b < F.shape[0]:
                    F[b, d] = 0.0                                    # 陷波主高频
    return (np.fft.irfft(F, n=WIN, axis=0) + t_mean).astype(np.float32)


def _sup_varburst(x):
    y = x.copy()
    for d in range(NVARS):
        sm = np.convolve(y[:, d], np.ones(7) / 7, mode="same"); r = y[:, d] - sm
        s = r.std() + 1e-6
        y[:, d] = sm + np.clip(r, -2 * s, 2 * s)                    # 收缩局部方差爆发
    return y


def _sup_trend(x):
    y = x.copy(); t = np.arange(WIN)
    for d in range(NVARS):
        a, b = np.polyfit(t, y[:, d], 1)
        y[:, d] = y[:, d] - (a * t + b) + y[:, d].mean()            # 去线性趋势
    return y


def _sup_corrbreak(x):
    y = x.copy(); mu = y.mean(0); xc = y - mu
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    r = min(3, len(S)); recon = (U[:, :r] * S[:r]) @ Vt[:r]
    return (0.5 * xc + 0.5 * recon + mu).astype(np.float32)         # 向低秩(强相关)重构靠拢→恢复相关


SUPPRESS = {"spike": _sup_spike, "level_shift": _sup_levelshift, "oscillation": _sup_oscillation,
            "variance_burst": _sup_varburst, "trend": _sup_trend, "correlation_break": _sup_corrbreak}


# --------------------------- 校准 + 去纠缠 --------------------------- #
def calibrated(ev, mu, sd):
    """各概念签名统计量对正常域的 z-score(校准分,抑制假激活)。"""
    return {c: (ev[SIG[c]] - mu[SIG[c]]) / (sd[SIG[c]] + 1e-9) for c in CONCEPTS}


def disentangle(x, mu, sd, co_thresh=0.5):
    """返回结构化画像:校准分 z / 主导概念 / 伴随响应 / 去纠缠后净分。
    主导 = 校准分最大的概念;对它做压制 ℐ,重算各概念签名 z;Δ_j = z_j − z_j(压制后)。
    Δ_j 占 z_j 比例大(且 j≠主导)→ j 是主导的伴随响应(净分清零)。"""
    ev = CB.evidence(x); z = calibrated(ev, mu, sd)
    prim = max(z, key=z.get)
    xs = SUPPRESS[prim](x); zs = calibrated(CB.evidence(xs), mu, sd)
    net = dict(z); co = {}
    for j in CONCEPTS:
        if j == prim:
            continue
        drop = z[j] - zs[j]
        # 伴随响应:压制主导后该概念 z 大幅下降(相对其自身)且本身被激活
        frac = drop / (abs(z[j]) + 1e-6)
        if z[j] > 1.5 and frac > co_thresh:
            co[j] = float(frac); net[j] = float(zs[j])             # 净分=压制后(扣掉伴随成分)
    return dict(z=z, primary=prim, prim_z=float(z[prim]), co_responses=co, net=net)


def profile_vector(prof):
    """画像→定长特征向量(供策略状态):6 概念净分 + 主导分 + 伴随数。"""
    net = prof["net"]
    return np.array([net[c] for c in CONCEPTS] + [prof["prim_z"], float(len(prof["co_responses"]))], np.float32)
