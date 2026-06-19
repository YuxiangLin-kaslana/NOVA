#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.agent import GPTInstantAgent, LocalSigLAAgent
from sigla_exp.agent.gpt_instant import DEFAULT_AGENT_MODEL
from sigla_exp.data import SplitData, load_dataset
from sigla_exp.pipeline import PipelineConfig, SigLATrajectoryPipeline, TrajectoryPrediction


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_TEST_AGENT_MODEL = "gpt-5.4-mini"


@dataclass
class HTTPAgentResponse:
    output_text: str


class HTTPResponsesResource:
    def __init__(self, api_key: str, timeout: float, max_output_tokens: int) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def create(self, *, model: str, instructions: str, input: list[dict[str, Any]]) -> HTTPAgentResponse:
        payload: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input,
        }
        if self.max_output_tokens > 0:
            payload["max_output_tokens"] = self.max_output_tokens

        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail[:1000]}") from exc

        output_text = extract_response_text(data)
        if not output_text:
            raise RuntimeError(f"OpenAI response did not contain output text: {data}")
        return HTTPAgentResponse(output_text=output_text)


class HTTPResponsesClient:
    def __init__(self, api_key: str, timeout: float, max_output_tokens: int) -> None:
        self.responses = HTTPResponsesResource(api_key, timeout, max_output_tokens)


def extract_response_text(data: dict[str, Any]) -> str:
    top_level = data.get("output_text")
    if isinstance(top_level, str):
        return top_level

    chunks: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the full SigLA trajectory pipeline in fallback mode.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name/path accepted by sigla_exp.data.load_dataset.")
    parser.add_argument("--data_dir", default="/u/ylin30/sigLA/data")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--win_size", type=int, default=50)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--risk_decay", type=float, default=0.85)
    parser.add_argument("--start_point", type=int, default=0, help="Start index inside the selected split.")
    parser.add_argument("--max_points", type=int, default=1200, help="Use 0 to evaluate the full split.")
    parser.add_argument("--max_windows", type=int, default=0, help="Use 0 for all windows in the selected points.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--agent", choices=("local", "gpt"), default="local")
    parser.add_argument("--online", action="store_true", help="Enable online retraining of detector + concept detector.")
    parser.add_argument(
        "--agent_model",
        default=DEFAULT_TEST_AGENT_MODEL,
        help=f"GPT model for --agent gpt. Core GPTInstantAgent default is {DEFAULT_AGENT_MODEL!r}.",
    )
    parser.add_argument("--strict_agent", action="store_true", help="Raise if the GPT agent call fails.")
    parser.add_argument("--openai_timeout", type=float, default=60.0)
    parser.add_argument("--openai_max_output_tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "fallback_pipeline_test" / "summary.json",
        help="Metrics JSON path.",
    )
    parser.add_argument(
        "--predictions_csv",
        type=Path,
        default=None,
        help="Per-window prediction CSV path. Defaults next to --output.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def get_split(bundle: Any, split_name: str) -> SplitData:
    return getattr(bundle, split_name)


def select_split(
    split: SplitData,
    start_point: int,
    max_points: int,
    max_windows: int,
    win_size: int,
    step: int,
) -> SplitData:
    start = max(0, start_point)
    if start >= len(split.x):
        raise ValueError(f"start_point={start_point} is outside split length={len(split.x)}")

    if max_windows > 0:
        n_points = win_size + step * (max_windows - 1)
    elif max_points > 0:
        n_points = max(win_size, max_points)
    else:
        n_points = len(split.x) - start

    end = min(len(split.x), start + n_points)
    if end - start < win_size:
        raise ValueError(f"Selected segment length={end - start} is shorter than win_size={win_size}")
    return SplitData(x=split.x[start:end], y=split.y[start:end])


def make_agent(args: argparse.Namespace) -> LocalSigLAAgent | GPTInstantAgent:
    if args.agent == "local":
        return LocalSigLAAgent()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --agent gpt is used.")

    client = HTTPResponsesClient(
        api_key=api_key,
        timeout=args.openai_timeout,
        max_output_tokens=args.openai_max_output_tokens,
    )
    return GPTInstantAgent(model=args.agent_model, enabled=True, strict=args.strict_agent, client=client)


def window_labels(labels: np.ndarray, prediction: TrajectoryPrediction) -> np.ndarray:
    return np.asarray(
        [int(np.any(labels[window.start : window.end + 1] == 1)) for window in prediction.windows],
        dtype=np.int64,
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
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


def validate_prediction(prediction: TrajectoryPrediction) -> None:
    if not prediction.windows:
        raise RuntimeError("Pipeline returned no windows.")

    for idx, window in enumerate(prediction.windows):
        if not isinstance(window.judgment.is_anomaly, bool):
            raise RuntimeError(f"Window {idx} returned a non-boolean is_anomaly.")
        values = [window.detector_score, window.risk_state, window.judgment.anomaly_score]
        values.extend(window.concept_state.profile.values())
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Window {idx} contains a non-finite fallback value.")


def summarize_prediction(
    args: argparse.Namespace,
    split: SplitData,
    prediction: TrajectoryPrediction,
    device: torch.device,
) -> dict[str, Any]:
    labels = window_labels(split.y, prediction)
    predicted_anomaly = np.asarray(prediction.anomaly_flags, dtype=np.int64)
    detector_scores = np.asarray([window.detector_score for window in prediction.windows], dtype=np.float64)
    risk_states = np.asarray([window.risk_state for window in prediction.windows], dtype=np.float64)
    concept_counts = Counter(window.concept_state.primary_concept or "none" for window in prediction.windows)
    judged_concept_counts: Counter = Counter()
    for window in prediction.windows:
        judged_concept_counts.update(window.judgment.concepts or ["none"])
    source_counts = Counter(window.judgment.source for window in prediction.windows)

    return {
        "dataset": args.dataset,
        "data_dir": str(args.data_dir),
        "split": args.split,
        "device": str(device),
        "mode": "fallback",
        "start_point": args.start_point,
        "fallback_modules": {
            "detector": "RMSFallbackDetector",
            "concept_extractor": "RawEvidenceConceptFallback",
            "agent": "GPTInstantAgent" if args.agent == "gpt" else "LocalSigLAAgent",
        },
        "agent_model": args.agent_model if args.agent == "gpt" else None,
        "strict_agent": bool(args.strict_agent),
        "n_points": prediction.n_points,
        "n_vars": prediction.n_vars,
        "win_size": args.win_size,
        "step": args.step,
        "max_windows_requested": args.max_windows,
        "n_windows": len(prediction.windows),
        "window_label_positive_count": int(np.sum(labels == 1)),
        "predicted_anomaly_count": int(np.sum(predicted_anomaly == 1)),
        "judged_concept_counts": dict(sorted(judged_concept_counts.items())),
        "primary_concept_counts": dict(sorted(concept_counts.items())),
        "decision_source_counts": dict(sorted(source_counts.items())),
        "online_stats": prediction.online_stats,
        "detector_score": {
            "min": float(np.min(detector_scores)),
            "median": float(np.median(detector_scores)),
            "mean": float(np.mean(detector_scores)),
            "max": float(np.max(detector_scores)),
        },
        "risk_state": {
            "min": float(np.min(risk_states)),
            "median": float(np.median(risk_states)),
            "mean": float(np.mean(risk_states)),
            "max": float(np.max(risk_states)),
        },
        "anomaly_metrics": binary_metrics(labels, predicted_anomaly),
    }


def write_predictions_csv(path: Path, split: SplitData, prediction: TrajectoryPrediction) -> None:
    labels = window_labels(split.y, prediction)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start",
                "end",
                "label",
                "detector_score",
                "risk_state",
                "primary_concept",
                "is_anomaly",
                "anomaly_score",
                "concepts",
                "confidence",
                "source",
                "rationale",
            ]
        )
        for label, window in zip(labels, prediction.windows):
            writer.writerow(
                [
                    window.start,
                    window.end,
                    int(label),
                    float(window.detector_score),
                    float(window.risk_state),
                    window.concept_state.primary_concept or "",
                    int(window.judgment.is_anomaly),
                    float(window.judgment.anomaly_score),
                    "|".join(window.judgment.concepts),
                    float(window.judgment.confidence),
                    window.judgment.source,
                    window.judgment.rationale,
                ]
            )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    bundle = load_dataset(args.dataset, args.data_dir, train_ratio=args.train_ratio, seed=args.seed)
    split = select_split(
        get_split(bundle, args.split),
        args.start_point,
        args.max_points,
        args.max_windows,
        args.win_size,
        args.step,
    )
    agent = make_agent(args)

    pipeline = SigLATrajectoryPipeline(
        detector=None,
        concept_extractor=None,
        agent=agent,
        config=PipelineConfig(win_size=args.win_size, step=args.step, risk_decay=args.risk_decay),
        device=device,
    )
    prediction = pipeline.predict_traj(
        split.x,
        language_context="Fallback smoke test: no trained detector, concept extractor, or LLM agent.",
        online=args.online,
    )
    validate_prediction(prediction)

    output = args.output
    predictions_csv = args.predictions_csv or output.with_name("predictions.csv")
    summary = summarize_prediction(args, split, prediction, device)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_predictions_csv(predictions_csv, split, prediction)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved summary to {output}")
    print(f"saved predictions to {predictions_csv}")


if __name__ == "__main__":
    main()
