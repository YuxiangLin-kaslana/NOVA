from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from ..actions import weak_action_labels
from ..data import SplitData, load_dataset
from ..model import MLPAnomalyDetector, MLPConceptExtractor
from ..profiles import ConceptProfileExtractor, extract_raw_evidence


TRAIN_SOURCE = "test"


class FrameworkWindowDataset(Dataset):
    """Window dataset for the current detector -> concept -> policy framework."""

    def __init__(
        self,
        split: SplitData,
        win_size: int,
        step: int,
        profile_extractor: ConceptProfileExtractor,
        l_min: int,
        l_max: int,
    ) -> None:
        self.x = split.x.astype(np.float32)
        self.y = split.y.astype(np.int64)
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.win_size
        window = self.x[start:end].astype(np.float32)
        fallback_score = np.sqrt(np.mean(window * window, axis=1, keepdims=True)).astype(np.float32)
        raw_evidence = extract_raw_evidence(window).astype(np.float32)
        profile = self.profile_extractor.transform(window).astype(np.float32)
        arg_label = int(np.argmax(np.max(np.abs(window), axis=0)))
        return {
            "signal": torch.from_numpy(window),
            "score": torch.from_numpy(fallback_score),
            "raw_evidence": torch.from_numpy(raw_evidence),
            "profile": torch.from_numpy(profile),
            "label": torch.tensor(int(self.window_labels[idx]), dtype=torch.float32),
            "action": torch.tensor(int(self.action_labels[idx]), dtype=torch.long),
            "arg": torch.tensor(arg_label, dtype=torch.long),
            "end_idx": torch.tensor(int(self.ends[idx]), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SigLA framework components from the labeled test split.")
    parser.add_argument(
        "--task",
        choices=("detector", "autoencoder", "concept", "pipeline"),
        default="detector",
        help="autoencoder is an alias for detector; pipeline trains detector + concept.",
    )
    parser.add_argument("--dataset", default="SMD_1-7", help="synthetic, SMD_1-7, SWaT, or a .npz file")
    parser.add_argument("--data_dir", default="/u/ylin30/sigLA/data")
    parser.add_argument("--output_dir", default="/u/ylin30/sigLA/code/runs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--win_size", type=int, default=50)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--l_min", type=int, default=20)
    parser.add_argument("--l_max", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--profile_max_windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--policy_split",
        choices=("train", "val", "test"),
        default=None,
        help="Deprecated compatibility option. Training now always uses the dataset test split.",
    )
    return parser.parse_args()


def normalize_task(task: str) -> str:
    return "detector" if task == "autoencoder" else task


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def split_windows(dataset: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if len(dataset) == 1:
        return dataset, dataset
    val_len = max(1, int(round(len(dataset) * val_ratio)))
    val_len = min(val_len, len(dataset) - 1)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def detector_point_scores(model: MLPAnomalyDetector, signal: torch.Tensor) -> torch.Tensor:
    recon = model(signal)
    return torch.mean((recon - signal) ** 2, dim=2, keepdim=True)


def run_detector(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_vars: int,
    device: torch.device,
) -> tuple[MLPAnomalyDetector, dict[str, Any]]:
    model = MLPAnomalyDetector(args.win_size, n_vars, args.latent_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = clone_state_dict(model)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_idx, batch in enumerate(train_loader):
            signal = batch["signal"].to(device)
            recon = model(signal)
            loss = F.mse_loss(recon, signal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if args.limit_batches and batch_idx + 1 >= args.limit_batches:
                break

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                signal = batch["signal"].to(device)
                val_losses.append(float(F.mse_loss(model(signal), signal).cpu()))
                if args.limit_batches and batch_idx + 1 >= args.limit_batches:
                    break

        row = {
            "epoch": epoch,
            "train_loss": mean_or_zero(train_losses),
            "val_loss": mean_or_zero(val_losses),
        }
        history.append(row)
        print(f"detector epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f}")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = clone_state_dict(model)

    model.load_state_dict(best_state)
    return model, {"best_val_loss": float(best_val), "history": history}


def run_concept(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[MLPConceptExtractor, dict[str, Any]]:
    model = MLPConceptExtractor(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = clone_state_dict(model)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_idx, batch in enumerate(train_loader):
            raw_evidence = batch["raw_evidence"].to(device)
            profile = batch["profile"].to(device)
            logits = model(raw_evidence)
            loss = F.binary_cross_entropy_with_logits(logits, profile)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if args.limit_batches and batch_idx + 1 >= args.limit_batches:
                break

        model.eval()
        val_losses: list[float] = []
        val_mae: list[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                raw_evidence = batch["raw_evidence"].to(device)
                profile = batch["profile"].to(device)
                logits = model(raw_evidence)
                val_losses.append(float(F.binary_cross_entropy_with_logits(logits, profile).cpu()))
                val_mae.append(float(torch.mean(torch.abs(torch.sigmoid(logits) - profile)).cpu()))
                if args.limit_batches and batch_idx + 1 >= args.limit_batches:
                    break

        row = {
            "epoch": epoch,
            "train_loss": mean_or_zero(train_losses),
            "val_loss": mean_or_zero(val_losses),
            "val_mae": mean_or_zero(val_mae),
        }
        history.append(row)
        print(
            f"concept epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_mae={row['val_mae']:.6f}"
        )
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = clone_state_dict(model)

    model.load_state_dict(best_state)
    return model, {"best_val_loss": float(best_val), "history": history}


def extractor_payload(extractor: ConceptProfileExtractor) -> dict[str, Any]:
    return {
        "class": "ConceptProfileExtractor",
        "median": extractor.median.tolist(),
        "mad": extractor.mad.tolist(),
    }


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    n_vars: int,
    extractor: ConceptProfileExtractor,
    detector: MLPAnomalyDetector | None = None,
    concept_extractor: MLPConceptExtractor | None = None,
    primary_component: str | None = None,
) -> None:
    checkpoint: dict[str, Any] = {
        "format": "sigla_framework_v1",
        "task": normalize_task(args.task),
        "args": vars(args),
        "train_source": TRAIN_SOURCE,
        "n_vars": n_vars,
        "model_config": {
            "win_size": args.win_size,
            "n_vars": n_vars,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
        },
        "profile_extractor": extractor_payload(extractor),
    }
    component_models = {
        "detector": detector,
        "concept_extractor": concept_extractor,
    }
    for name, model in component_models.items():
        if model is not None:
            checkpoint[name] = model.state_dict()
    if primary_component is not None and component_models[primary_component] is not None:
        checkpoint["component"] = primary_component
        checkpoint["model"] = component_models[primary_component].state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    args.task = normalize_task(args.task)
    if args.policy_split is not None:
        print(f"warning: --policy_split={args.policy_split} is ignored; training source is always test.")

    set_seed(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or f"{args.task}_{args.dataset}_test_w{args.win_size}_s{args.step}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(args.dataset, args.data_dir, train_ratio=args.train_ratio, seed=args.seed)
    source_split = bundle.test
    extractor = ConceptProfileExtractor.fit(
        source_split.x,
        args.win_size,
        args.step,
        max_windows=args.profile_max_windows,
    )
    dataset = FrameworkWindowDataset(source_split, args.win_size, args.step, extractor, args.l_min, args.l_max)
    train_ds, val_ds = split_windows(dataset, args.val_ratio, args.seed)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)

    config = vars(args) | {
        "train_source": TRAIN_SOURCE,
        "source_points": int(len(source_split.x)),
        "source_positive_points": int(np.sum(source_split.y == 1)),
        "n_windows": int(len(dataset)),
        "train_windows": int(len(train_ds)),
        "val_windows": int(len(val_ds)),
        "n_vars": int(bundle.n_vars),
        "device": str(device),
    }
    write_json(run_dir / "config.json", config)

    print(f"dataset={args.dataset} n_vars={bundle.n_vars} device={device}")
    print(
        f"train_source={TRAIN_SOURCE} source={source_split.x.shape} "
        f"windows={len(dataset)} train_windows={len(train_ds)} val_windows={len(val_ds)}"
    )

    metrics: dict[str, Any] = {
        "task": args.task,
        "train_source": TRAIN_SOURCE,
        "dataset": args.dataset,
        "n_windows": int(len(dataset)),
        "train_windows": int(len(train_ds)),
        "val_windows": int(len(val_ds)),
    }
    detector: MLPAnomalyDetector | None = None
    concept_extractor: MLPConceptExtractor | None = None

    if args.task == "detector":
        detector, metrics["detector"] = run_detector(args, train_loader, val_loader, bundle.n_vars, device)
        save_checkpoint(run_dir / "checkpoint_best.pt", args, bundle.n_vars, extractor, detector=detector, primary_component="detector")
    elif args.task == "concept":
        concept_extractor, metrics["concept_extractor"] = run_concept(args, train_loader, val_loader, device)
        save_checkpoint(
            run_dir / "checkpoint_best.pt",
            args,
            bundle.n_vars,
            extractor,
            concept_extractor=concept_extractor,
            primary_component="concept_extractor",
        )
    elif args.task == "pipeline":
        detector, metrics["detector"] = run_detector(args, train_loader, val_loader, bundle.n_vars, device)
        concept_extractor, metrics["concept_extractor"] = run_concept(args, train_loader, val_loader, device)
        save_checkpoint(
            run_dir / "checkpoint_best.pt",
            args,
            bundle.n_vars,
            extractor,
            detector=detector,
            concept_extractor=concept_extractor,
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    write_json(run_dir / "metrics.json", metrics)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
