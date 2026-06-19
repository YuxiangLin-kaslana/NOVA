"""SMD 异常检测数据准备模块。

数据来源：Server Machine Dataset (SMD)
    /u/ylin30/sigLA/data/ServerMachineDataset
    ├── train/<machine>.txt        每台机器的训练序列  [T, 38]，已归一化到 [0,1]，无标签（默认全部正常）
    ├── test/<machine>.txt         每台机器的测试序列  [T, 38]
    └── test_label/<machine>.txt   每个时间点 0/1 异常标签  [T]

任务设定：
    输入  = 一个时间窗口 [win_size, n_vars]
    输出  = 该窗口是否存在异常（窗口内任一时刻被标为异常 -> 1）
            异常检测器训练后，也可用「与正常形态的偏离程度」（如重构误差）作为连续异常分数。

本模块只负责「加载 + 组织 + 切窗」，不训练任何模型。

设计要点：
    * 28 台机器全部合并：train_dataset 含所有机器的训练窗口，test_dataset 含所有机器的测试窗口。
    * 滑窗不跨机器边界：每台机器内部独立切窗，避免在两台机器拼接处产生无意义窗口。
    * 实时切窗：不预先把窗口落盘，Dataset.__getitem__ 时按 (机器, 起点) 现切，省内存。
    * 窗口标签：窗口内任一时刻异常即记为 1（与 sigLA 现有 eval_autoencoder.py 的约定一致）。
    * train 序列默认全部视为正常（标签 0）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# SMD 根目录（可在 build_smd_datasets 里覆盖）
DEFAULT_SMD_ROOT = Path("/u/ylin30/sigLA/data/ServerMachineDataset")
N_VARS = 38  # SMD 固定 38 维特征


# --------------------------------------------------------------------------- #
#  原始序列加载                                                                #
# --------------------------------------------------------------------------- #
def list_machines(root: str | Path = DEFAULT_SMD_ROOT) -> list[str]:
    """返回所有机器名（如 'machine-1-1'），按 (组, 编号) 自然排序。"""
    root = Path(root)
    names = {f[:-4] for f in os.listdir(root / "train") if f.endswith(".txt")}

    def key(name: str) -> tuple[int, int]:
        _, group, idx = name.split("-")
        return int(group), int(idx)

    return sorted(names, key=key)


def _load_txt(path: Path, dtype) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=dtype)


@dataclass
class MachineSeries:
    """一台机器的原始时间序列（未切窗）。"""

    name: str
    train_x: np.ndarray  # [T_train, n_vars] float32
    test_x: np.ndarray   # [T_test,  n_vars] float32
    test_y: np.ndarray   # [T_test]          int64  (0/1)
    train_y: np.ndarray  # [T_train]         int64  (全 0)


def load_machine(name: str, root: str | Path = DEFAULT_SMD_ROOT) -> MachineSeries:
    """加载单台机器的 train / test / test_label 三个序列。"""
    root = Path(root)
    train_x = _load_txt(root / "train" / f"{name}.txt", np.float32)
    test_x = _load_txt(root / "test" / f"{name}.txt", np.float32)
    test_y = _load_txt(root / "test_label" / f"{name}.txt", np.int64).reshape(-1)

    # loadtxt 在单行文件时会降到 1 维，这里统一成 2 维 [T, n_vars]
    train_x = np.atleast_2d(train_x)
    test_x = np.atleast_2d(test_x)
    train_y = np.zeros(len(train_x), dtype=np.int64)
    return MachineSeries(name, train_x, test_x, test_y, train_y)


def load_all_machines(
    root: str | Path = DEFAULT_SMD_ROOT,
    machines: list[str] | None = None,
) -> list[MachineSeries]:
    """加载全部（或指定子集）机器的原始序列。"""
    names = machines if machines is not None else list_machines(root)
    return [load_machine(n, root) for n in names]


# --------------------------------------------------------------------------- #
#  实时切窗 Dataset                                                            #
# --------------------------------------------------------------------------- #
class SMDWindowDataset(Dataset):
    """把若干条时间序列按滑动窗口暴露为样本，切窗在 __getitem__ 时实时进行。

    每个样本：
        signal: FloatTensor [win_size, n_vars]
        label:  LongTensor  标量，窗口内是否存在异常 (0/1)
        series: LongTensor  标量，该窗口来自第几条序列（机器索引）
        start:  LongTensor  标量，窗口在该序列内的起始下标

    参数
    ----
    series_list : 每条序列的特征数组列表，每个形状 [T_i, n_vars]
    label_list  : 每条序列的逐点标签列表，每个形状 [T_i]（0/1）
    win_size    : 窗口长度
    stride      : 相邻窗口起点的间隔（步长）

    滑窗不跨序列边界：长度 < win_size 的序列直接跳过。
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        label_list: list[np.ndarray],
        win_size: int = 100,
        stride: int = 10,
    ) -> None:
        assert len(series_list) == len(label_list)
        self.win_size = int(win_size)
        self.stride = int(stride)
        self.series = [np.ascontiguousarray(s, dtype=np.float32) for s in series_list]
        self.labels = [np.ascontiguousarray(y, dtype=np.int64).reshape(-1) for y in label_list]
        self.n_vars = self.series[0].shape[1] if self.series else N_VARS

        # 预计算窗口索引：(序列下标, 起点)。只存索引，不存窗口本身。
        index: list[tuple[int, int]] = []
        for si, x in enumerate(self.series):
            last_start = len(x) - self.win_size
            if last_start < 0:
                continue  # 序列太短，跳过
            starts = range(0, last_start + 1, self.stride)
            index.extend((si, st) for st in starts)
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        si, start = self._index[idx]
        end = start + self.win_size
        signal = self.series[si][start:end]                 # [win, n_vars]
        win_label = int(np.any(self.labels[si][start:end] == 1))
        return {
            "signal": torch.from_numpy(signal.copy()),       # FloatTensor [win, n_vars]
            "label": torch.tensor(win_label, dtype=torch.long),
            "series": torch.tensor(si, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }

    # ---- 一些便捷统计，方便检查数据 ---- #
    def anomaly_window_count(self) -> int:
        """异常窗口数量（label==1）。"""
        n = 0
        for si, start in self._index:
            if np.any(self.labels[si][start : start + self.win_size] == 1):
                n += 1
        return n


# --------------------------------------------------------------------------- #
#  顶层入口：一行拿到 train / test 两个 Dataset                                #
# --------------------------------------------------------------------------- #
@dataclass
class SMDBundle:
    train: SMDWindowDataset
    val: SMDWindowDataset
    test: SMDWindowDataset
    n_vars: int
    win_size: int
    stride: int
    machines: list[str]


def build_smd_datasets(
    root: str | Path = DEFAULT_SMD_ROOT,
    win_size: int = 100,
    stride: int = 10,
    machines: list[str] | None = None,
    val_split: float = 0.5,
) -> SMDBundle:
    """合并全部机器，构造 train / val / test 三个实时切窗 Dataset（干净无监督设定）。

    数据来源与切分：
        train ← 每台机器的 train 文件（纯正常，无异常）。
                模型只在此上做无监督重构训练，绝不接触异常。
        val/test ← 每台机器的 test 文件（含异常标签）按【时间顺序】切：
                前 val_split 比例 → val（用于早停、选阈值）
                其余             → test（最终评估）
        默认 val_split=0.5，即把 test 文件按时间切成 val:test ≈ 1:1。
        所有机器对应段再合并（不跨机器、不跨段切窗）。

    用法::

        from data import build_smd_datasets
        from torch.utils.data import DataLoader

        bundle = build_smd_datasets(win_size=100, stride=10, val_split=0.5)
        train_loader = DataLoader(bundle.train, batch_size=256, shuffle=True)
        val_loader   = DataLoader(bundle.val,   batch_size=256, shuffle=False)
        test_loader  = DataLoader(bundle.test,  batch_size=256, shuffle=False)

        batch = next(iter(train_loader))
        batch["signal"]  # [B, win_size, n_vars]
        batch["label"]   # [B]  (val/test 有真实异常；train 恒为 0)
    """
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split 必须在 [0,1) 内，收到 {val_split}")
    series = load_all_machines(root, machines)
    names = [m.name for m in series]

    # 每台机器的 test 文件按时间顺序切：前 val_split → val，其余 → test
    val_x, val_y, te_x, te_y = [], [], [], []
    for m in series:
        cut = int(len(m.test_x) * val_split)
        val_x.append(m.test_x[:cut])
        val_y.append(m.test_y[:cut])
        te_x.append(m.test_x[cut:])
        te_y.append(m.test_y[cut:])

    train_ds = SMDWindowDataset(
        [m.train_x for m in series],
        [m.train_y for m in series],
        win_size=win_size,
        stride=stride,
    )
    val_ds = SMDWindowDataset(val_x, val_y, win_size=win_size, stride=stride)
    test_ds = SMDWindowDataset(te_x, te_y, win_size=win_size, stride=stride)
    n_vars = series[0].train_x.shape[1] if series else N_VARS
    return SMDBundle(train_ds, val_ds, test_ds, n_vars, win_size, stride, names)
