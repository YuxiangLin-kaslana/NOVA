from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .actions import weak_action_labels
from .profiles import ConceptProfileExtractor


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray


@dataclass
class DatasetBundle:
    train: SplitData
    val: SplitData
    test: SplitData
    n_vars: int


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D time-series array, got shape {x.shape}")
    return x


def _split_train_val(train: np.ndarray, labels: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_ratio >= 1.0:
        return train, train.copy(), labels, labels.copy()
    cut = max(1, int(len(train) * train_ratio))
    return train[:cut], train[cut:], labels[:cut], labels[cut:]


def _standardize(bundle: DatasetBundle) -> DatasetBundle:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(bundle.train.x)
    val_x = scaler.transform(bundle.val.x)
    test_x = scaler.transform(bundle.test.x)
    return DatasetBundle(
        train=SplitData(train_x.astype(np.float32), bundle.train.y.astype(np.int64)),
        val=SplitData(val_x.astype(np.float32), bundle.val.y.astype(np.int64)),
        test=SplitData(test_x.astype(np.float32), bundle.test.y.astype(np.int64)),
        n_vars=bundle.n_vars,
    )


def load_smd(base_dir: str | Path, dataset_name: str, train_ratio: float = 0.8) -> DatasetBundle:
    base = Path(base_dir)
    entity = dataset_name.split("_", 1)[1] if dataset_name.startswith("SMD_") else dataset_name
    data_dir = base / "ServerMachineDataset" / "preprocessed"
    with open(data_dir / f"machine-{entity}_train.pkl", "rb") as f:
        train = _as_2d(pickle.load(f))
    with open(data_dir / f"machine-{entity}_test.pkl", "rb") as f:
        test = _as_2d(pickle.load(f))
    with open(data_dir / f"machine-{entity}_test_label.pkl", "rb") as f:
        test_labels = np.asarray(pickle.load(f), dtype=np.int64).reshape(-1)

    train_labels = np.zeros(len(train), dtype=np.int64)
    train_x, val_x, train_y, val_y = _split_train_val(train, train_labels, train_ratio)
    return _standardize(
        DatasetBundle(
            train=SplitData(train_x, train_y),
            val=SplitData(val_x, val_y),
            test=SplitData(test, test_labels),
            n_vars=train.shape[1],
        )
    )


def load_swat(base_dir: str | Path, train_ratio: float = 0.8) -> DatasetBundle:
    data_dir = Path(base_dir) / "SWaT"
    train_csv = data_dir / "SWaT_Dataset_Normal_v1.csv"
    test_csv = data_dir / "SWaT_Dataset_Attack_v0.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("SWaT csv files were not found under the data directory.")

    train_df = pd.read_csv(train_csv).iloc[1:, 1:-1].astype(np.float32)
    test_df_raw = pd.read_csv(test_csv)
    test_labels = (test_df_raw.iloc[:, -1].astype(str).str.lower() == "attack").to_numpy(np.int64)
    test_df = test_df_raw.iloc[1:, 1:-1].astype(np.float32)
    test_labels = test_labels[1:]

    train = train_df.to_numpy()
    test = test_df.to_numpy()
    train_labels = np.zeros(len(train), dtype=np.int64)
    train_x, val_x, train_y, val_y = _split_train_val(train, train_labels, train_ratio)
    return _standardize(
        DatasetBundle(
            train=SplitData(train_x, train_y),
            val=SplitData(val_x, val_y),
            test=SplitData(test, test_labels),
            n_vars=train.shape[1],
        )
    )


def load_npz(path: str | Path, train_ratio: float = 0.8) -> DatasetBundle:
    data = np.load(path)
    train = _as_2d(data["train"])
    test = _as_2d(data["test"])
    test_labels = np.asarray(data.get("test_label", data.get("test_labels")), dtype=np.int64).reshape(-1)
    train_labels = np.asarray(data.get("train_label", np.zeros(len(train))), dtype=np.int64).reshape(-1)
    train_x, val_x, train_y, val_y = _split_train_val(train, train_labels, train_ratio)
    return _standardize(
        DatasetBundle(
            train=SplitData(train_x, train_y),
            val=SplitData(val_x, val_y),
            test=SplitData(test, test_labels),
            n_vars=train.shape[1],
        )
    )


def make_synthetic(length: int = 4000, n_vars: int = 5, seed: int = 0, train_ratio: float = 0.8) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = []
    for idx in range(n_vars):
        freq = 0.008 + 0.002 * idx
        base.append(np.sin(2 * np.pi * freq * t + idx) + 0.15 * rng.normal(size=length))
    x = np.stack(base, axis=1).astype(np.float32)
    labels = np.zeros(length, dtype=np.int64)
    for onset in [int(length * 0.45), int(length * 0.72)]:
        precursor = slice(max(0, onset - 120), onset)
        event = slice(onset, min(length, onset + 80))
        x[precursor, 0] += np.linspace(0.0, 1.2, precursor.stop - precursor.start)
        x[event, 0] += 2.5
        x[event, 1] -= 1.5
        labels[event] = 1

    split = length // 2
    train = x[:split]
    train_labels = np.zeros(split, dtype=np.int64)
    test = x[split:]
    test_labels = labels[split:]
    train_x, val_x, train_y, val_y = _split_train_val(train, train_labels, train_ratio)
    return _standardize(
        DatasetBundle(
            train=SplitData(train_x, train_y),
            val=SplitData(val_x, val_y),
            test=SplitData(test, test_labels),
            n_vars=n_vars,
        )
    )


def load_dataset(
    dataset: str,
    data_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> DatasetBundle:
    if dataset == "synthetic":
        return make_synthetic(seed=seed, train_ratio=train_ratio)
    if dataset.startswith("SMD"):
        return load_smd(data_dir, dataset, train_ratio=train_ratio)
    if dataset == "SWaT":
        return load_swat(data_dir, train_ratio=train_ratio)
    if dataset.endswith(".npz"):
        return load_npz(dataset, train_ratio=train_ratio)
    raise ValueError(f"Unsupported dataset: {dataset}")


class WindowDataset(Dataset):
    def __init__(
        self,
        split: SplitData,
        win_size: int,
        step: int,
        profile_extractor: ConceptProfileExtractor,
        l_min: int,
        l_max: int,
    ) -> None:
        self.x = split.x
        self.y = split.y
        self.win_size = win_size
        self.step = step
        self.profile_extractor = profile_extractor
        self.starts = np.arange(0, max(0, len(self.x) - win_size + 1), step, dtype=np.int64)
        self.ends = self.starts + win_size - 1
        if len(self.starts) == 0:
            raise ValueError(f"Split is shorter than win_size={win_size}: length={len(self.x)}")
        self.window_labels = np.asarray(
            [int(np.any(self.y[start : start + win_size] == 1)) for start in self.starts],
            dtype=np.int64,
        )
        self.action_labels = weak_action_labels(self.ends, self.window_labels, self.y, l_min=l_min, l_max=l_max)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.win_size
        window = self.x[start:end].astype(np.float32)
        score = np.sqrt(np.mean(window * window, axis=1, keepdims=True)).astype(np.float32)
        profile = self.profile_extractor.transform(window)
        arg_label = int(np.argmax(np.max(np.abs(window), axis=0)))
        return {
            "signal": torch.from_numpy(window),
            "score": torch.from_numpy(score),
            "profile": torch.from_numpy(profile),
            "label": torch.tensor(int(self.window_labels[idx]), dtype=torch.float32),
            "action": torch.tensor(int(self.action_labels[idx]), dtype=torch.long),
            "arg": torch.tensor(arg_label, dtype=torch.long),
            "end_idx": torch.tensor(int(self.ends[idx]), dtype=torch.long),
        }

