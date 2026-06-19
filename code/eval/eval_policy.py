#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.actions import ACTION_NAMES
from sigla_exp.data import WindowDataset, load_dataset
from sigla_exp.model import MLPActionPolicy, MLPAnomalyDetector, MLPConceptExtractor
from sigla_exp.models import SigLAPolicy
from sigla_exp.profiles import ConceptProfileExtractor
from sigla_exp.train.cli import FrameworkWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SigLA policy against weak action labels.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory containing checkpoint_best.pt.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path.")
    parser.add_argument("--dataset", default=None, help="Dataset name/path. Defaults to checkpoint args.")
    parser.add_argument("--data_dir", default=None, help="Data directory. Defaults to checkpoint args.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--win_size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--l_min", type=int, default=None)
    parser.add_argument("--l_max", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--profile_max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Metrics JSON path.")
    parser.add_argument("--predictions_csv", type=Path, default=None, help="Per-window prediction CSV path.")
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


def is_framework_checkpoint(ckpt: dict[str, Any]) -> bool:
    return ckpt.get("format") == "sigla_framework_v1" or "policy" in ckpt


def profile_extractor_from_checkpoint(
    ckpt: dict[str, Any],
    bundle: Any,
    win_size: int,
    step: int,
    max_windows: int,
) -> ConceptProfileExtractor:
    payload = ckpt.get("profile_extractor")
    if isinstance(payload, dict) and "median" in payload and "mad" in payload:
        return ConceptProfileExtractor(
            median=np.asarray(payload["median"], dtype=np.float32),
            mad=np.asarray(payload["mad"], dtype=np.float32),
        )
    source = bundle.test.x if ckpt.get("train_source") == "test" else bundle.train.x
    return ConceptProfileExtractor.fit(source, win_size, step, max_windows=max_windows)


@torch.no_grad()
def detector_point_scores(detector: MLPAnomalyDetector, signal: torch.Tensor) -> torch.Tensor:
    recon = detector(signal)
    return torch.mean((recon - signal) ** 2, dim=2, keepdim=True)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    out: dict[str, Any] = {
        "count": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "predicted_positives": int(np.sum(y_pred == 1)),
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["roc_auc"] = float(roc_auc_score(y_true, scores))
            out["average_precision"] = float(average_precision_score(y_true, scores))
        except Exception as exc:  # pragma: no cover - metric availability is environment-specific.
            out["score_metric_error"] = str(exc)
    return out


def action_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    n_actions = len(ACTION_NAMES)
    confusion = np.zeros((n_actions, n_actions), dtype=np.int64)
    for true_item, pred_item in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        confusion[int(true_item), int(pred_item)] += 1

    per_action = {}
    f1_with_support = []
    for idx, name in enumerate(ACTION_NAMES):
        tp = int(confusion[idx, idx])
        fp = int(np.sum(confusion[:, idx]) - tp)
        fn = int(np.sum(confusion[idx, :]) - tp)
        support = int(np.sum(confusion[idx, :]))
        pred_count = int(np.sum(confusion[:, idx]))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        if support > 0:
            f1_with_support.append(f1)
        per_action[name] = {
            "support": support,
            "predicted": pred_count,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1_present_actions": float(np.mean(f1_with_support)) if f1_with_support else 0.0,
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        "per_action": per_action,
    }


def write_predictions_csv(
    path: Path,
    end_idx: np.ndarray,
    labels: np.ndarray,
    action_true: np.ndarray,
    action_pred: np.ndarray,
    action_prob: np.ndarray,
    arg_true: np.ndarray,
    arg_pred: np.ndarray,
    risk_prob: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "end_idx",
                "label",
                "action_true",
                "action_true_name",
                "action_pred",
                "action_pred_name",
                "action_confidence",
                "arg_true",
                "arg_pred",
                "risk_prob",
            ]
        )
        for row in zip(end_idx, labels, action_true, action_pred, action_prob, arg_true, arg_pred, risk_prob):
            end_item, label_item, action_item, pred_item, prob_item, arg_item, arg_pred_item, risk_item = row
            writer.writerow(
                [
                    int(end_item),
                    int(label_item),
                    int(action_item),
                    ACTION_NAMES[int(action_item)],
                    int(pred_item),
                    ACTION_NAMES[int(pred_item)],
                    float(prob_item),
                    int(arg_item),
                    int(arg_pred_item),
                    float(risk_item),
                ]
            )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or args.run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = choose_device(args.device)
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = dict(ckpt.get("args", {}))
    framework_checkpoint = is_framework_checkpoint(ckpt)
    if ckpt_args.get("task") not in (None, "policy", "pipeline"):
        raise ValueError(f"Checkpoint task={ckpt_args.get('task')} is not a policy checkpoint.")

    dataset_name = value_from_args(args.dataset, ckpt_args, "dataset", "SMD_1-7")
    data_dir = value_from_args(args.data_dir, ckpt_args, "data_dir", "/u/ylin30/sigLA/data")
    train_ratio = float(value_from_args(args.train_ratio, ckpt_args, "train_ratio", 0.8))
    seed = int(value_from_args(args.seed, ckpt_args, "seed", 0))
    win_size = int(value_from_args(args.win_size, ckpt_args, "win_size", 50))
    step = int(value_from_args(args.step, ckpt_args, "step", 5))
    l_min = int(value_from_args(args.l_min, ckpt_args, "l_min", 20))
    l_max = int(value_from_args(args.l_max, ckpt_args, "l_max", 120))
    hidden_dim = int(value_from_args(args.hidden_dim, ckpt_args, "hidden_dim", 128))
    latent_dim = int(value_from_args(None, ckpt_args, "latent_dim", 128))
    profile_max_windows = int(value_from_args(args.profile_max_windows, ckpt_args, "profile_max_windows", 512))

    bundle = load_dataset(dataset_name, data_dir, train_ratio=train_ratio, seed=seed)
    n_vars = int(ckpt.get("n_vars", bundle.n_vars))
    if n_vars != bundle.n_vars:
        raise ValueError(f"Checkpoint n_vars={n_vars} does not match dataset n_vars={bundle.n_vars}.")

    extractor = profile_extractor_from_checkpoint(ckpt, bundle, win_size, step, profile_max_windows)
    split = getattr(bundle, args.split)
    dataset_cls = FrameworkWindowDataset if framework_checkpoint else WindowDataset
    dataset = dataset_cls(split, win_size, step, extractor, l_min, l_max)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    detector = None
    concept_extractor = None
    if framework_checkpoint:
        model = MLPActionPolicy(win_size, bundle.n_vars, hidden_dim=hidden_dim).to(device)
        policy_state = ckpt["policy"] if "policy" in ckpt else ckpt["model"]
        model.load_state_dict(policy_state)
        if "detector" in ckpt:
            detector = MLPAnomalyDetector(win_size, bundle.n_vars, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
            detector.load_state_dict(ckpt["detector"])
            detector.eval()
        if "concept_extractor" in ckpt:
            concept_extractor = MLPConceptExtractor(hidden_dim=hidden_dim).to(device)
            concept_extractor.load_state_dict(ckpt["concept_extractor"])
            concept_extractor.eval()
        model_class = "MLPActionPolicy"
    else:
        model = SigLAPolicy(bundle.n_vars, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model_class = "SigLAPolicy"
    model.eval()

    losses: list[float] = []
    end_idx_all: list[np.ndarray] = []
    label_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    action_pred_all: list[np.ndarray] = []
    action_prob_all: list[np.ndarray] = []
    arg_all: list[np.ndarray] = []
    arg_pred_all: list[np.ndarray] = []
    risk_prob_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            score = batch["score"].to(device)
            profile = batch["profile"].to(device)
            if detector is not None:
                score = detector_point_scores(detector, signal)
            if concept_extractor is not None:
                profile = torch.sigmoid(concept_extractor(batch["raw_evidence"].to(device)))
            action = batch["action"].to(device)
            arg = batch["arg"].to(device)
            label = batch["label"].to(device)

            out = model(signal, score, profile)
            loss = (
                F.cross_entropy(out["action_logits"], action)
                + 0.1 * F.cross_entropy(out["arg_logits"], arg)
                + 0.2 * F.binary_cross_entropy_with_logits(out["risk_logit"], label)
            )
            losses.append(float(loss.cpu()))

            action_probs = torch.softmax(out["action_logits"], dim=-1)
            action_pred = torch.argmax(action_probs, dim=-1)
            arg_pred = torch.argmax(out["arg_logits"], dim=-1)
            risk_prob = torch.sigmoid(out["risk_logit"])

            end_idx_all.append(batch["end_idx"].numpy().astype(np.int64))
            label_all.append(label.cpu().numpy().astype(np.int64))
            action_all.append(action.cpu().numpy().astype(np.int64))
            action_pred_all.append(action_pred.cpu().numpy().astype(np.int64))
            action_prob_all.append(torch.max(action_probs, dim=-1).values.cpu().numpy().astype(np.float64))
            arg_all.append(arg.cpu().numpy().astype(np.int64))
            arg_pred_all.append(arg_pred.cpu().numpy().astype(np.int64))
            risk_prob_all.append(risk_prob.cpu().numpy().astype(np.float64))

    end_idx = np.concatenate(end_idx_all)
    labels = np.concatenate(label_all)
    actions = np.concatenate(action_all)
    action_preds = np.concatenate(action_pred_all)
    action_probs = np.concatenate(action_prob_all)
    args_true = np.concatenate(arg_all)
    args_pred = np.concatenate(arg_pred_all)
    risk_probs = np.concatenate(risk_prob_all)
    risk_pred = (risk_probs > 0.5).astype(np.int64)

    metrics = {
        "dataset": dataset_name,
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "framework_checkpoint": bool(framework_checkpoint),
        "model_class": model_class,
        "split": args.split,
        "count": int(len(actions)),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "action_metrics": action_metrics(actions, action_preds),
        "arg_accuracy": float(np.mean(args_true == args_pred)) if len(args_true) else 0.0,
        "risk_metrics": binary_metrics(labels, risk_pred, risk_probs),
    }

    output = args.output or args.run_dir / f"policy_eval_{args.split}.json"
    predictions_csv = args.predictions_csv or args.run_dir / f"policy_predictions_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    write_predictions_csv(
        predictions_csv,
        end_idx,
        labels,
        actions,
        action_preds,
        action_probs,
        args_true,
        args_pred,
        risk_probs,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved metrics to {output}")
    print(f"saved predictions to {predictions_csv}")


if __name__ == "__main__":
    main()
