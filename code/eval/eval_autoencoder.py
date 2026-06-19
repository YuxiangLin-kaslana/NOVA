#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.data import SplitData, load_dataset
from sigla_exp.models import MLPAutoEncoder


@dataclass
class ScoreResult:
    starts: np.ndarray
    ends: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class SignalWindowDataset(Dataset):
    def __init__(self, split: SplitData, win_size: int, step: int) -> None:
        self.x = split.x.astype(np.float32)
        self.y = split.y.astype(np.int64)
        self.win_size = win_size
        self.step = step
        self.starts = np.arange(0, max(0, len(self.x) - win_size + 1), step, dtype=np.int64)
        self.ends = self.starts + win_size - 1
        if len(self.starts) == 0:
            raise ValueError(f"Split is shorter than win_size={win_size}: length={len(self.x)}")
        self.labels = np.asarray(
            [int(np.any(self.y[start : start + win_size] == 1)) for start in self.starts],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.win_size
        return {
            "signal": torch.from_numpy(self.x[start:end]),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "end": torch.tensor(int(self.ends[idx]), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores using only a trained autoencoder.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory containing checkpoint_best.pt.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path.")
    parser.add_argument("--dataset", default=None, help="Dataset name/path. Defaults to checkpoint args.")
    parser.add_argument("--data_dir", default=None, help="Data directory. Defaults to checkpoint args.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--threshold_source", choices=("train", "val", "test"), default="val", help="Deprecated; ignored.")
    parser.add_argument("--threshold_percentile", type=float, default=99.0, help="Deprecated; ignored.")
    parser.add_argument("--threshold_all_windows", action="store_true", help="Deprecated; ignored.")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--win_size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Metrics JSON path.")
    parser.add_argument("--scores_csv", type=Path, default=None, help="Per-window scores CSV path.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def value_from_args(cli_value: Any, ckpt_args: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(key, default)


def get_split(bundle: Any, name: str) -> SplitData:
    return getattr(bundle, name)


def score_split(
    model: MLPAutoEncoder,
    split: SplitData,
    win_size: int,
    step: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> ScoreResult:
    dataset = SignalWindowDataset(split, win_size, step)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    scores: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            batch_scores = model.anomaly_score(signal).detach().cpu().numpy()
            scores.append(batch_scores.astype(np.float64))
            starts.append(batch["start"].numpy().astype(np.int64))
            ends.append(batch["end"].numpy().astype(np.int64))
            labels.append(batch["label"].numpy().astype(np.int64))
    return ScoreResult(
        starts=np.concatenate(starts),
        ends=np.concatenate(ends),
        scores=np.concatenate(scores),
        labels=np.concatenate(labels),
    )


def point_scores_from_windows(length: int, result: ScoreResult) -> np.ndarray:
    point_scores = np.full(length, -np.inf, dtype=np.float64)
    for start, end, score in zip(result.starts, result.ends, result.scores):
        point_scores[int(start) : int(end) + 1] = np.maximum(point_scores[int(start) : int(end) + 1], score)
    finite = np.isfinite(point_scores)
    fill_value = float(np.min(result.scores)) if len(result.scores) else 0.0
    point_scores[~finite] = fill_value
    return point_scores


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int64)
    out: dict[str, Any] = {
        "count": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "score_min": float(np.min(scores)) if len(scores) else 0.0,
        "score_median": float(np.median(scores)) if len(scores) else 0.0,
        "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "score_max": float(np.max(scores)) if len(scores) else 0.0,
    }
    if len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["roc_auc"] = float(roc_auc_score(y_true, scores))
            out["average_precision"] = float(average_precision_score(y_true, scores))
        except Exception as exc:  # pragma: no cover - metric availability is environment-specific.
            out["score_metric_error"] = str(exc)
    return out


def write_scores_csv(path: Path, result: ScoreResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "score", "label"])
        for start, end, score, label in zip(result.starts, result.ends, result.scores, result.labels):
            writer.writerow([int(start), int(end), float(score), int(label)])


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or args.run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = choose_device(args.device)
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = dict(ckpt.get("args", {}))

    dataset_name = value_from_args(args.dataset, ckpt_args, "dataset", "SMD_1-7")
    data_dir = value_from_args(args.data_dir, ckpt_args, "data_dir", "/u/ylin30/sigLA/data")
    train_ratio = float(value_from_args(args.train_ratio, ckpt_args, "train_ratio", 0.8))
    seed = int(value_from_args(args.seed, ckpt_args, "seed", 0))
    win_size = int(value_from_args(args.win_size, ckpt_args, "win_size", 50))
    step = int(value_from_args(args.step, ckpt_args, "step", 5))
    latent_dim = int(value_from_args(args.latent_dim, ckpt_args, "latent_dim", 128))
    hidden_dim = int(value_from_args(args.hidden_dim, ckpt_args, "hidden_dim", 128))

    bundle = load_dataset(dataset_name, data_dir, train_ratio=train_ratio, seed=seed)
    n_vars = int(ckpt.get("n_vars", bundle.n_vars))
    if n_vars != bundle.n_vars:
        raise ValueError(f"Checkpoint n_vars={n_vars} does not match dataset n_vars={bundle.n_vars}.")

    model = MLPAutoEncoder(win_size, bundle.n_vars, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])

    eval_split = get_split(bundle, args.split)
    eval_result = score_split(model, eval_split, win_size, step, args.batch_size, args.num_workers, device)
    point_scores = point_scores_from_windows(len(eval_split.x), eval_result)

    metrics = {
        "dataset": dataset_name,
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "split": args.split,
        "score_summary": {
            "min": float(np.min(eval_result.scores)),
            "median": float(np.median(eval_result.scores)),
            "mean": float(np.mean(eval_result.scores)),
            "max": float(np.max(eval_result.scores)),
        },
        "window_metrics": ranking_metrics(eval_result.labels, eval_result.scores),
        "point_metrics": ranking_metrics(eval_split.y.astype(np.int64), point_scores),
    }
    if any(math.isnan(metrics["score_summary"][key]) for key in metrics["score_summary"]):
        raise RuntimeError("Unexpected NaN score summary.")

    output = args.output or args.run_dir / f"autoencoder_eval_{args.split}.json"
    scores_csv = args.scores_csv or args.run_dir / f"autoencoder_window_scores_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    write_scores_csv(scores_csv, eval_result)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved metrics to {output}")
    print(f"saved window scores to {scores_csv}")


if __name__ == "__main__":
    main()
