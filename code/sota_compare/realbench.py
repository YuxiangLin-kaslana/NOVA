"""真实背景版 benchmark:用真实 train(正常)序列做背景,在其上注入同样的 6 类概念签名。

动机:真实 AD 数据(SMD/SWaT…)只有二分类异常标签,无"异常类型"标签,而开放词表闭环需类型 ground-truth。
标准做法 = **真实正常背景 + 可控注入类型**:既有真实纹理/相关结构,又保留类型标签,可跨多机器/多数据集评测。

实现:`activate(entity, dataset=...)` 把 `sigla_exp.ovbench.base_normal` monkeypatch 成"从真实正常窗采样",
于是所有下游(ovbench.make_window / make_window_strength / normal_stats,以及复用它们的 exp_detection_tie /
exp_early_warning)**无需改动**即改用真实背景。注入器与证据统计量沿用 ovbench(对通道数通用)。

通道:每个数据集取**方差最高的 12 维**(NVARS=12,与现有 CNN/证据对齐),再逐通道 z-score(使注入幅度可比)。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import sigla_exp.ovbench as OV

WIN, NVARS = OV.WIN, OV.NVARS
DEFAULT_DATA = Path("/u/ylin30/sigLA/data")

_BG = {"Z": None, "dataset": None, "entity": None}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_background(raw):
    """返回真实正常背景 [T, 12](方差最高 12 通道,逐通道 z-score)。"""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"expected 2D time-series array, got shape={raw.shape}")
    if raw.shape[1] < NVARS:
        raise ValueError(f"need at least {NVARS} variables, got shape={raw.shape}")
    sel = np.argsort(raw.var(0))[::-1][:NVARS]                            # 方差最高 12 通道
    X = raw[:, np.sort(sel)]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)                               # 逐通道标准化
    return Z.astype(np.float32)


def load_entity(dataset, entity, data_dir=DEFAULT_DATA):
    """加载真实正常背景.

    Supported layouts:
    - SMD: data/ServerMachineDataset/preprocessed/machine-<entity>_train.pkl
    - PSM: data/PSM/preprocessed/psm_train.pkl
    - Generic future datasets: data/<dataset>/preprocessed/<entity>_train.pkl
    """
    ds = str(dataset).upper()
    root = Path(data_dir)
    if ds in {"SMD", "SERVERMACHINEDATASET"}:
        ent = str(entity)
        if ent.startswith("machine-"):
            ent = ent.removeprefix("machine-")
        p = root / "ServerMachineDataset" / "preprocessed" / f"machine-{ent}_train.pkl"
    elif ds == "PSM":
        ent = str(entity).lower()
        if ent not in {"psm", "0", "1", "default"}:
            raise ValueError(f"PSM has one entity; use entity='psm', got {entity!r}")
        p = root / "PSM" / "preprocessed" / "psm_train.pkl"
    else:
        p = root / str(dataset) / "preprocessed" / f"{entity}_train.pkl"
    if not p.exists():
        raise FileNotFoundError(p)
    return _normalize_background(_load_pickle(p))


def load_machine(entity, data_dir=DEFAULT_DATA):
    """Backward-compatible SMD loader."""
    return load_entity("SMD", entity, data_dir)


def _real_base_normal(rng):
    """从真实正常序列随机截取一个 WIN 窗(替换 ovbench 的合成 base_normal)。"""
    Z = _BG["Z"]
    if Z is None:
        raise RuntimeError("realbench.activate(...) must be called before sampling")
    s = int(rng.integers(0, len(Z) - WIN))
    return Z[s:s + WIN].copy()


def activate(entity, data_dir=DEFAULT_DATA, dataset="SMD"):
    """加载数据集实体并 monkeypatch ovbench.base_normal → 全链路改用真实背景。"""
    _BG["Z"] = load_entity(dataset, entity, data_dir)
    _BG["dataset"] = dataset
    _BG["entity"] = entity
    OV.base_normal = _real_base_normal                                   # 关键:下游 make_window 用它
    return _BG["Z"]


def deactivate():
    """恢复合成背景(谨慎:OV 原 base_normal 来自 exp_novel_concept)。"""
    import scripts.exp_novel_concept as NC
    OV.base_normal = NC.base_normal
    _BG["Z"] = None; _BG["dataset"] = None; _BG["entity"] = None
