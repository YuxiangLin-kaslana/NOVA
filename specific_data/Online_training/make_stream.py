#!/usr/bin/env python3
"""生成「带概念漂移」的合成时间序列流，用于**无真值在线训练**实验。

为什么要这个数据集
------------------
在线适应(online adaptation)的卖点是：数据随时间**漂移**时，系统在线重训
轻量模型把性能跟住。要证明这一点，就需要一条**漂移可控、漂移点已知**的流。
真实数据(SMD)漂移不可控，所以这里程序化地造一条：

  * 协变量漂移 P(X)：基础正常信号的频率/幅度/均值**逐区间缓慢或突变漂移**。
  * 概念漂移：每个区间**异常的概念构成不同**(区间1多 spike/oscillation，
    区间2转向 level_shift/trend ……)，模拟"异常形态本身在变"。

标签只用于**离线评测**(precision/recall、漂移后恢复曲线)，**不进训练**——
训练全程 label-free，符合无真值在线适应的设定。

概念注入函数直接复用 concept_detector 的 concept_synth.py，保证与 concept
detector 训练时的 6 个概念定义完全一致：
    spike / level_shift / oscillation / variance_burst / trend / correlation_break

输出(.npz)
----------
    x            float32 [T, n_vars]   归一化基础 ~[0,1]，注入处可越界
    y            int64   [T]           逐点异常标签(仅评测用)
    regime       int64   [T]           每个点属于第几个漂移区间
    drift_points int64   [R-1]         区间边界(漂移发生点)
并附 .json 记录每个区间的概念调度与生成参数。

用法
----
    python make_stream.py --out streams/drift_gradual.npz --drift gradual
    python make_stream.py --out streams/drift_abrupt.npz  --drift abrupt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIGLA_ROOT = HERE.parent.parent  # .../sigLA


def _load_concept_synth():
    path = SIGLA_ROOT / "specific_data" / "Concept_detector" / "concept_synth.py"
    spec = importlib.util.spec_from_file_location("concept_synth", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["concept_synth"] = mod
    spec.loader.exec_module(mod)
    return mod


CS = _load_concept_synth()
CONCEPT_NAMES = CS.CONCEPT_NAMES                       # 6 个概念
INJECTORS = CS.INJECTORS


# 每个漂移区间允许出现的概念子集(概念漂移：异常形态构成随区间变化)
REGIME_CONCEPTS = [
    ("spike", "oscillation"),
    ("level_shift", "trend"),
    ("variance_burst", "correlation_break"),
    ("spike", "level_shift", "trend"),
]


def _base_normal(length: int, n_vars: int, regime_of: np.ndarray, n_regimes: int,
                 drift: str, rng: np.random.Generator) -> np.ndarray:
    """生成基础正常信号，并让其统计量随区间漂移(协变量漂移 P(X))。"""
    t = np.arange(length, dtype=np.float64)
    x = np.zeros((length, n_vars), dtype=np.float32)

    # 每个变量一组基础参数 + 每个区间一组漂移增量
    base_freq = 0.01 + 0.004 * rng.random(n_vars)
    base_amp = 0.6 + 0.2 * rng.random(n_vars)
    base_mean = 0.5 + 0.0 * rng.random(n_vars)
    phase = 2 * np.pi * rng.random(n_vars)

    # 每个区间的漂移因子
    freq_drift = np.linspace(1.0, 1.8, n_regimes)        # 频率逐区间升高
    mean_drift = np.linspace(0.0, 0.25, n_regimes)       # 均值逐区间抬升
    amp_drift = np.linspace(1.0, 0.7, n_regimes)         # 幅度逐区间下降

    for r in range(n_regimes):
        mask = regime_of == r
        if not np.any(mask):
            continue
        if drift == "abrupt":
            f, m, a = freq_drift[r], mean_drift[r], amp_drift[r]
            ti = t[mask]
        else:  # gradual：区间内线性过渡到下一区间
            r_next = min(r + 1, n_regimes - 1)
            seg = np.linspace(0.0, 1.0, int(np.sum(mask)))
            f = freq_drift[r] + (freq_drift[r_next] - freq_drift[r]) * seg
            m = mean_drift[r] + (mean_drift[r_next] - mean_drift[r]) * seg
            a = amp_drift[r] + (amp_drift[r_next] - amp_drift[r]) * seg
            ti = t[mask]
        for d in range(n_vars):
            wave = base_amp[d] * a * np.sin(2 * np.pi * base_freq[d] * f * ti + phase[d])
            x[mask, d] = (base_mean[d] + m + 0.5 * wave).astype(np.float32)

    x += rng.normal(0.0, 0.03, size=x.shape).astype(np.float32)
    return x


def make_stream(length: int, n_vars: int, win_size: int, n_regimes: int,
                anomaly_rate: float, drift: str, seed: int):
    rng = np.random.default_rng(seed)

    # 区间划分(等长)
    bounds = np.linspace(0, length, n_regimes + 1, dtype=np.int64)
    regime_of = np.zeros(length, dtype=np.int64)
    for r in range(n_regimes):
        regime_of[bounds[r]:bounds[r + 1]] = r
    drift_points = bounds[1:-1].copy()

    x = _base_normal(length, n_vars, regime_of, n_regimes, drift, rng)
    y = np.zeros(length, dtype=np.int64)

    # 按区间概念调度注入异常事件
    n_events = max(1, int(anomaly_rate * length / win_size))
    schedule = []
    for _ in range(n_events):
        start = int(rng.integers(0, length - win_size))
        r = int(regime_of[start])
        allowed = REGIME_CONCEPTS[r % len(REGIME_CONCEPTS)]
        name = allowed[int(rng.integers(0, len(allowed)))]
        seg = x[start:start + win_size]
        INJECTORS[name](seg, rng)               # 原地注入到窗口
        x[start:start + win_size] = seg
        y[start:start + win_size] = 1
        schedule.append({"start": start, "regime": r, "concept": name})

    meta = {
        "length": length, "n_vars": n_vars, "win_size": win_size,
        "n_regimes": n_regimes, "drift": drift, "seed": seed,
        "anomaly_rate": anomaly_rate,
        "concept_names": list(CONCEPT_NAMES),
        "regime_concepts": [list(c) for c in REGIME_CONCEPTS[:n_regimes]],
        "drift_points": drift_points.tolist(),
        "n_events": n_events,
        "positive_points": int(y.sum()),
    }
    return x, y, regime_of, drift_points, meta, schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成带概念漂移的合成在线训练数据流")
    p.add_argument("--out", type=Path, default=HERE / "streams" / "drift_gradual.npz")
    p.add_argument("--length", type=int, default=12000)
    p.add_argument("--n_vars", type=int, default=38, help="默认 38 与 SMD 对齐")
    p.add_argument("--win_size", type=int, default=100)
    p.add_argument("--n_regimes", type=int, default=4)
    p.add_argument("--anomaly_rate", type=float, default=0.08, help="异常点占比目标")
    p.add_argument("--drift", choices=("gradual", "abrupt"), default="gradual")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x, y, regime_of, drift_points, meta, schedule = make_stream(
        args.length, args.n_vars, args.win_size, args.n_regimes,
        args.anomaly_rate, args.drift, args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, y=y, regime=regime_of, drift_points=drift_points)
    meta_path = args.out.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "schedule": schedule}, f, indent=2)

    print(f"saved stream  -> {args.out}")
    print(f"saved meta    -> {meta_path}")
    print(f"shape x={x.shape} y={y.shape}  positive={int(y.sum())} ({y.mean():.2%})")
    print(f"drift={args.drift}  regimes={args.n_regimes}  drift_points={drift_points.tolist()}")
    print(f"regime concept schedule: {meta['regime_concepts']}")


if __name__ == "__main__":
    main()
