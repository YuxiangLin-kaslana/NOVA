"""B 路线:**证据正交可分**的 6 概念开放词表基准(解决 multi-type blocker 的根因)。

旧 evidence 的纠缠(导致多新类失败):
  trend↔level_shift(半窗中位数阶跃:斜坡也有大阶跃)
  variance_burst↔spike(全窗峰度/max-z:局部方差混合=重尾)
  variance_burst↔oscillation(高频能量占比:噪声爆发是宽带,也抬高 hf)

本模块把 6 个签名统计量都换成**对纠缠维度鲁棒**的版本,使**每个概念恰好且仅触发自己的签名**:
  spike          -> kurtosis        稀疏极值(几点 ^4 主导);爆发幅度调低后近高斯,不触发
  level_shift    -> local_step      局部窗口中位数突变;斜坡逐窗只差 slope*w → 不触发
  oscillation    -> spectral_peak   单频占比(窄带);宽带噪声爆发谱平坦 → 不触发
  variance_burst -> var_localiz     方差局部化(高通残差分段 max/median);spike 稀疏、平滑结构 → ≈1
  trend          -> kendall         值-时间 Kendall τ(单调);台阶 τ≈0.5、正弦 τ≈0 → 不触发
  correlation_break -> decorr       1 - 滑窗最小平均|corr|(定位被破坏子段)

注入器复用 exp_novel_concept,但 variance_burst 调低幅度、correlation_break 增强(更多通道)以保证可分。
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np

import scripts.exp_novel_concept as NC

WIN, NVARS = NC.WIN, NC.NVARS
CONCEPTS = list(NC.CONCEPTS)
DEFS = NC.DEFS
base_normal = NC.base_normal

# 每概念的签名统计量 key(应只有它在该统计量上显著偏离 normal)
SIG = {"spike": "kurtosis", "level_shift": "local_step", "oscillation": "spectral_peak",
       "variance_burst": "var_localiz", "trend": "lin_r2", "correlation_break": "decorr"}
STAT_OF = SIG                                              # 别名(供实验脚本通用引用)
STATS = list(SIG.values())

# 给 LLM 的统计量语义提示(对应正交签名)
STAT_MEANING = {
    "kurtosis": "isolated extreme outliers (spike-like peakedness)",
    "local_step": "an abrupt, persistent level jump (no overall slope)",
    "spectral_peak": "a dominant high-frequency oscillatory tone",
    "var_localiz": "a localized burst of increased variance (no isolated outliers)",
    "lin_r2": "a gradual linear ramp/drift across the whole window",
    "decorr": "loss of cross-channel correlation (channels desynchronize)",
}


# --------------------------- 注入器(可分化微调) --------------------------- #
def _level_shift(x, rng):
    """锐台阶但**去除线性分量** → 跳变保留(local_step 高),整体净斜率≈0(lin_r2 不被触发,与 trend 分开)。"""
    s = int(rng.integers(WIN // 3, 2 * WIN // 3))
    t = np.arange(WIN)
    for d in NC._dims(rng, 4):
        col = x[:, d].astype(np.float64).copy()
        col[s:] += rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 5.0)
        a, b = np.polyfit(t, col, 1)                       # 减掉最佳拟合直线 → 无净斜率
        x[:, d] = (col - (a * t + b) + x[:, d].mean()).astype(np.float32)


def _variance_burst(x, rng):
    """调低幅度、加长区段:保持方差局部化可检测,但不让混合变重尾(不触发 kurtosis)。"""
    s = int(rng.integers(WIN // 2, 3 * WIN // 4)); e = min(WIN, s + int(rng.integers(25, 45)))
    for d in NC._dims(rng, 4):
        x[s:e, d] += rng.normal(0, rng.uniform(0.7, 1.0), e - s).astype(np.float32)


def _corr_break(x, rng):
    """纯相关破坏:对中段每个通道做**不同的循环移位**,精确保留每通道边际/局部纹理
    (var_localiz/kurtosis/local_step 等逐通道签名全不变),只打散跨通道对齐 → 仅 decorr 升高。
    边界交叉淡化,消除移位接缝处的不连续。"""
    s = int(rng.integers(WIN // 6, WIN // 3)); e = min(WIN, s + int(rng.integers(WIN // 2, 2 * WIN // 3)))
    seg_len = e - s
    tt = np.arange(seg_len)
    bw = 6
    ramp = np.clip(np.minimum(tt, (seg_len - 1) - tt) / bw, 0.0, 1.0).astype(np.float32)
    for d in NC._dims(rng, NVARS):
        lag = int(rng.integers(seg_len // 4, 3 * seg_len // 4))   # 各通道不同的大位移
        shifted = np.roll(x[s:e, d], lag)
        x[s:e, d] = (ramp * shifted + (1.0 - ramp) * x[s:e, d]).astype(np.float32)  # 边界淡回原值


INJ = {**NC.INJ, "level_shift": _level_shift, "variance_burst": _variance_burst,
       "correlation_break": _corr_break}


def make_window(concept, rng):
    x = base_normal(rng)
    if concept is not None:
        INJ[concept](x, rng)
    return x.astype(np.float32)


def make_window_strength(concept, rng, strength=1.0):
    """强度可调的注入(供前兆/早预警):前兆=同一概念签名的**弱化版**(strength<1)。
    在同一 base 上注入,取偏离 = 注入后−base,再按 strength 缩放回叠加 → 保留签名方向、幅度按比例。"""
    x = base_normal(rng)
    if concept is None or strength <= 0:
        return x.astype(np.float32)
    xf = x.copy()
    INJ[concept](xf, rng)
    return (x + strength * (xf - x)).astype(np.float32)


# --------------------------- 正交证据统计量 --------------------------- #
def _lin_r2(x):
    """线性拟合 R²(逐通道取最大)。纯斜坡→≈1;去趋势台阶→≈0(无净斜率);正弦/噪声/spike→低。"""
    t = np.arange(WIN); A = np.vstack([t, np.ones(WIN)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None); fit = A @ coef
    ss_res = ((x - fit) ** 2).sum(0); ss_tot = ((x - x.mean(0)) ** 2).sum(0) + 1e-6
    return float(np.max(np.clip(1 - ss_res / ss_tot, 0, 1)))


def _local_step(y, w=10):
    """逐点局部中位数突变 max|median(y[t:t+w]) - median(y[t-w:t])|。台阶突变高;斜坡逐窗仅 slope*w → 低。"""
    best = 0.0
    for t in range(w, WIN - w + 1):
        best = max(best, abs(np.median(y[t:t + w]) - np.median(y[t - w:t])))
    return float(best)


def _var_localiz(y, k=7, nseg=5):
    """方差局部化比:高通残差按段分块,max段稳健尺度 / 中位段尺度。爆发单段突出→高;均匀/稀疏→≈1。"""
    sm = np.convolve(y, np.ones(k) / k, mode="same")
    r = y - sm
    sc = np.array([1.4826 * np.median(np.abs(b - np.median(b))) + 1e-6 for b in np.array_split(r, nseg)])
    return float(sc.max() / (np.median(sc) + 1e-6))


def _spectral_peak(x):
    """**高频带**内单频占比:freq>0.15 的最大频点能量 / 总能量(逐通道取最大)。
    base 低频正弦不计;窄带高频振荡→高;宽带噪声爆发能量分散→低。"""
    det = x - x.mean(0, keepdims=True)
    mag = np.abs(np.fft.rfft(det, axis=0))[1:]            # 去 DC;index i -> freq (i+1)/WIN
    hi = mag[14:]                                         # freq > 15/WIN = 0.15
    frac = hi.max(0) / (mag.sum(0) + 1e-6)
    return float(np.max(frac))


def _decorr(x, w=33):
    """1 - 滑窗内平均跨通道|corr|的最小值(定位被破坏子段)。正常→低;break→高。"""
    best = 1.0
    for s in range(0, WIN - w + 1, w // 2):
        seg = x[s:s + w]
        m = seg.std(0) > 1e-6
        if int(m.sum()) < 2:
            continue
        C = np.corrcoef(seg[:, m], rowvar=False); n = C.shape[0]
        best = min(best, float((np.abs(C).sum() - n) / (n * (n - 1))))
    return float(1.0 - best)


def evidence(x):
    """6 个正交、可解释的证据统计量;每概念恰好在自己的签名上偏离 normal。"""
    mu = x.mean(0); var = x.var(0) + 1e-9
    kurt = float(np.max(((x - mu) ** 4).mean(0) / var ** 2 - 3.0))
    local_step = float(np.max([_local_step(x[:, d]) for d in range(NVARS)]))
    spectral_peak = _spectral_peak(x)
    var_localiz = float(np.max([_var_localiz(x[:, d]) for d in range(NVARS)]))
    lin_r2 = _lin_r2(x)
    decorr = _decorr(x)
    out = {"kurtosis": kurt, "local_step": local_step, "spectral_peak": spectral_peak,
           "var_localiz": var_localiz, "lin_r2": lin_r2, "decorr": decorr}
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}


def normal_stats(rng, n=400):
    """正常基线每统计量的 mu/sd(供 z-score 门控与 LLM 提示)。"""
    evs = [evidence(make_window(None, rng)) for _ in range(n)]
    mu = {k: float(np.mean([e[k] for e in evs])) for k in STATS}
    sd = {k: float(np.std([e[k] for e in evs]) + 1e-6) for k in STATS}
    return mu, sd


_GPT_CACHE = {}                                              # (model, z-tuple) -> 概念名/None;纯加速,跨调用复用


def gpt_recognize_top1(ev, key, mu, sd=None, model="gpt-4o-mini"):
    """返回**单个**最可能的概念名(或 None / "__ERROR__")。强制 top-1 → 杜绝过度列举。
    给 LLM 的是每个统计量**对正常基线的 z-score(偏离几个标准差)**而非原始值——gpt-4o-mini 不擅长原始数值
    比大小(会一律锚定到 level_shift);喂 z 后它只需找"偏离最大的统计量"再用语义映射到概念(开放词表语义 grounding)。"""
    if sd is None:
        sd = {k: 1.0 for k in mu}
    z = {k: round((ev[k] - mu[k]) / (sd[k] + 1e-9), 1) for k in mu}
    ckey = (model, tuple(sorted(z.items())))                  # 纯加速:z 已四舍五入到 1 位,良性窗高命中
    if ckey in _GPT_CACHE:                                    # 缓存命中 → 跳过网络往返(不改数值)
        return _GPT_CACHE[ckey]
    instr = (
        "You name the SINGLE most likely time-series anomaly concept in a window. You are given, for each "
        "generic statistic, its DEVIATION FROM NORMAL in standard deviations (z-score): large positive z "
        "means that statistic is strongly elevated vs normal. Taxonomy (concept: definition):\n" +
        "\n".join(f"- {k}: {v}" for k, v in DEFS.items()) +
        "\nEach statistic is the signature of one concept: "
        + "; ".join(f"{k}={v}" for k, v in STAT_MEANING.items()) + ".\n"
        "Procedure: find the statistic with the largest positive z-score; map it to the concept it is the "
        "signature of (using the meanings above). Pick that ONE concept. If no statistic has z above ~2, "
        "respond null.\n"
        "Respond with ONLY a JSON object {\"concept\":\"<name>\"} using exactly one taxonomy name, "
        "or {\"concept\":null}. No explanation, no markdown."
    )
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": "z-scores: " + json.dumps(z)}],
               "max_output_tokens": 200}
    for _ in range(2):                                        # 重试 3→2,timeout 30→8s(失败更快返回)
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses",
                                         data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            c = json.loads(txt[s:e + 1]).get("concept")
            res = c if c in CONCEPTS else None
            _GPT_CACHE[ckey] = res                            # 仅缓存成功结果(含 None);__ERROR__ 不缓存
            return res
        except Exception:
            continue
    return "__ERROR__"
