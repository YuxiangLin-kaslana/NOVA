"""从正常数据合成「带 concept 标签」的窗口，用于训练 concept detector。

思路：异常检测器（AE）只能说「不正常」，但说不出「为什么不正常」。
concept detector 要在窗口里识别**异常的形态**——是尖峰？水平突变？震荡？……
真实异常没有形态标注，但我们可以**往正常窗口里程序化注入**已知形态，
注入了什么就知道标签是什么，从而得到大量带标签的训练数据。

6 个 concept（依据对 SMD 真实异常的形态统计，见 docstring 末尾）：
    spike             局部短促尖峰
    level_shift       某点起均值持续偏移
    oscillation       叠加高频振荡
    variance_burst    局部噪声/方差暴增（均值不变）
    trend             叠加线性斜坡（缓慢漂移）
    correlation_break 打乱部分维度间的相关结构（多变量）

设计要点（对齐真实异常特性）：
    * 多标签：一个窗口可同时注入多个 concept（真实异常常混合）。
    * 多变量：只往**部分维度**注入（真实平均 ~7/38 维受影响）。
    * 变时长：注入落在窗口内随机子区间，覆盖短/长形态。
    * 含纯正常样本（全 0 标签），让 detector 学会「什么都没有」。

输入正常窗口约定为 [win, n_vars]，数值大致在 [0,1]（SMD min-max 归一化）。
注入后允许略微超出 [0,1]（真实异常本就会越界），不强行裁剪。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CONCEPT_NAMES = (
    "spike",
    "level_shift",
    "oscillation",
    "variance_burst",
    "trend",
    "correlation_break",
)
N_CONCEPTS = len(CONCEPT_NAMES)
CONCEPT_INDEX = {name: i for i, name in enumerate(CONCEPT_NAMES)}


# --------------------------------------------------------------------------- #
#  单个 concept 的注入函数                                                     #
#  约定：原地修改 x[:, dims] 的一个时间子区间，x 形状 [win, n_vars]            #
# --------------------------------------------------------------------------- #
def _pick_dims(rng: np.random.Generator, n_vars: int) -> np.ndarray:
    """随机选一部分维度（1 ~ ~1/3 维），贴近真实「部分维受影响」。"""
    k = int(rng.integers(1, max(2, n_vars // 3) + 1))
    return rng.choice(n_vars, size=k, replace=False)


def _pick_segment(rng: np.random.Generator, win: int, min_len: int) -> tuple[int, int]:
    """在窗口内随机取一个长度 >= min_len 的子区间 [s, e)。"""
    length = int(rng.integers(min_len, win + 1))
    start = int(rng.integers(0, win - length + 1))
    return start, start + length


def inject_spike(x, rng):
    dims = _pick_dims(rng, x.shape[1])
    n_spikes = int(rng.integers(1, 4))
    for d in dims:
        pos = rng.choice(x.shape[0], size=n_spikes, replace=False)
        amp = rng.uniform(0.3, 0.9, size=n_spikes) * rng.choice([-1.0, 1.0], size=n_spikes)
        x[pos, d] += amp
    return x


def inject_level_shift(x, rng):
    dims = _pick_dims(rng, x.shape[1])
    s, e = _pick_segment(rng, x.shape[0], min_len=max(5, x.shape[0] // 4))
    for d in dims:
        x[s:e, d] += rng.uniform(0.2, 0.6) * rng.choice([-1.0, 1.0])
    return x


def inject_oscillation(x, rng):
    dims = _pick_dims(rng, x.shape[1])
    s, e = _pick_segment(rng, x.shape[0], min_len=max(8, x.shape[0] // 4))
    t = np.arange(e - s)
    for d in dims:
        period = rng.uniform(2.0, 6.0)            # 高频：周期 2~6 步
        amp = rng.uniform(0.1, 0.4)
        phase = rng.uniform(0, 2 * np.pi)
        wave = amp * np.sin(2 * np.pi * t / period + phase)
        x[s:e, d] += wave
    return x


def inject_variance_burst(x, rng):
    dims = _pick_dims(rng, x.shape[1])
    s, e = _pick_segment(rng, x.shape[0], min_len=max(5, x.shape[0] // 4))
    for d in dims:
        sigma = rng.uniform(0.1, 0.3)
        x[s:e, d] += rng.normal(0.0, sigma, size=e - s)   # 均值不变、方差增大
    return x


def inject_trend(x, rng):
    dims = _pick_dims(rng, x.shape[1])
    s, e = _pick_segment(rng, x.shape[0], min_len=max(8, x.shape[0] // 3))
    ramp = np.linspace(0.0, 1.0, e - s)
    for d in dims:
        slope = rng.uniform(0.2, 0.6) * rng.choice([-1.0, 1.0])
        x[s:e, d] += slope * ramp
    return x


def inject_correlation_break(x, rng):
    """打乱部分维度间相关：把选中维度在子区间内的时间顺序各自独立打乱。

    这样每个维度自身的边缘分布几乎不变（不引入明显尖峰/偏移），
    但维度之间原有的同步/相关结构被破坏。需要 >=2 维才有意义。
    """
    n_vars = x.shape[1]
    if n_vars < 2:
        return x
    k = int(rng.integers(2, max(3, n_vars // 3) + 1))
    dims = rng.choice(n_vars, size=k, replace=False)
    s, e = _pick_segment(rng, x.shape[0], min_len=max(8, x.shape[0] // 4))
    for d in dims:
        perm = rng.permutation(e - s)
        x[s:e, d] = x[s:e, d][perm]
    return x


INJECTORS = {
    "spike": inject_spike,
    "level_shift": inject_level_shift,
    "oscillation": inject_oscillation,
    "variance_burst": inject_variance_burst,
    "trend": inject_trend,
    "correlation_break": inject_correlation_break,
}


def synthesize(
    normal_window: np.ndarray,
    rng: np.random.Generator,
    p_normal: float = 0.2,
    max_concepts: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """对一个正常窗口注入 0~max_concepts 个 concept，返回 (window, multi_label[6])。

    p_normal 概率不注入任何 concept（纯正常样本，标签全 0）。
    """
    x = np.array(normal_window, dtype=np.float32, copy=True)
    label = np.zeros(N_CONCEPTS, dtype=np.float32)
    if rng.random() < p_normal:
        return x, label
    n = int(rng.integers(1, max_concepts + 1))
    chosen = rng.choice(N_CONCEPTS, size=n, replace=False)
    for ci in chosen:
        name = CONCEPT_NAMES[ci]
        INJECTORS[name](x, rng)
        label[ci] = 1.0
    return x, label


# --------------------------------------------------------------------------- #
#  Dataset：包装正常窗口，实时合成带标签样本                                   #
# --------------------------------------------------------------------------- #
class SyntheticConceptDataset(Dataset):
    """把一批正常窗口实时合成为「带 concept 多标签」的训练样本。

    参数
    ----
    normal_windows : 正常窗口来源，形状 [N, win, n_vars] 的 ndarray，
                     或任何支持 len()/[i]->[win,n_vars] 的序列。
    seed           : 基础随机种子；每个样本用 (seed, idx) 派生独立 RNG，
                     保证可复现且每个 idx 的注入稳定。
    p_normal/max_concepts : 见 synthesize。
    epoch          : 改变它可在不同 epoch 得到不同的注入（数据增强）；
                     默认 0 表示固定。

    每个样本返回:
        signal: FloatTensor [win, n_vars]
        label : FloatTensor [6]   多标签 0/1
    """

    def __init__(
        self,
        normal_windows,
        seed: int = 0,
        p_normal: float = 0.2,
        max_concepts: int = 3,
        epoch: int = 0,
    ) -> None:
        self.normal = normal_windows
        self.seed = int(seed)
        self.p_normal = float(p_normal)
        self.max_concepts = int(max_concepts)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.normal)

    def _window(self, idx: int) -> np.ndarray:
        w = self.normal[idx]
        if isinstance(w, dict):       # 兼容 SMDWindowDataset 返回的 dict
            w = w["signal"]
        if isinstance(w, torch.Tensor):
            w = w.numpy()
        return np.asarray(w, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        x, label = synthesize(self._window(idx), rng,
                              p_normal=self.p_normal, max_concepts=self.max_concepts)
        return {
            "signal": torch.from_numpy(x),
            "label": torch.from_numpy(label),
        }
