#!/usr/bin/env python3
"""无真值在线训练 runner。

逐窗消费一条时间序列流：
    window -> anomaly detector + concept detector -> agent 判异常 + 给概念伪标签
          -> OnlineTrainer 在线重训两个小模型(detector 自监督重建 / concept BCE 伪标签)

训练**全程不使用真值**。流里的标签 y 只在最后用于**离线评测**
(整体 + 分漂移区间的 precision/recall/F1，展示抗漂移/恢复能力)。

模型可从已训 checkpoint 加载("复用离线模型 -> 直接在线")，
也可随机初始化 + warmup 冷启动(纯在线)。

用法见 scripts/run_online_training.sh。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.agent import GPTInstantAgent, LocalSigLAAgent
from sigla_exp.calibrator import CalibratorConfig
from sigla_exp.model import CNNConceptDetector, MLPAnomalyDetector, MLPConceptDetector
from sigla_exp.online import OnlineTrainConfig
from sigla_exp.pipeline import PipelineConfig, SigLATrajectoryPipeline
from sigla_exp.precursor import precursor_metrics

DEFAULT_CONCEPTS = ("spike", "level_shift", "oscillation", "variance_burst", "trend", "correlation_break")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass
class _Resp:
    output_text: str


class _ResponsesResource:
    """urllib-based OpenAI Responses client (no `openai` package dependency)."""

    def __init__(self, api_key: str, timeout: float, max_output_tokens: int) -> None:
        self.api_key, self.timeout, self.max_output_tokens = api_key, timeout, max_output_tokens

    def create(self, *, model: str, instructions: str, input: list[dict[str, Any]]) -> _Resp:
        payload: dict[str, Any] = {"model": model, "instructions": instructions, "input": input}
        if self.max_output_tokens > 0:
            payload["max_output_tokens"] = self.max_output_tokens
        req = urllib.request.Request(
            OPENAI_RESPONSES_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}") from exc
        text = data.get("output_text")
        if not isinstance(text, str):
            chunks = []
            for item in data.get("output", []):
                for c in (item.get("content", []) if isinstance(item, dict) else []):
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        chunks.append(c["text"])
            text = "\n".join(chunks)
        if not text:
            raise RuntimeError(f"empty OpenAI response: {str(data)[:300]}")
        return _Resp(output_text=text)


class _HTTPClient:
    def __init__(self, api_key: str, timeout: float, max_output_tokens: int) -> None:
        self.responses = _ResponsesResource(api_key, timeout, max_output_tokens)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="无真值在线训练 runner")
    p.add_argument("--stream", type=Path, required=True, help="输入流 .npz (含 x, y, regime, drift_points)")
    p.add_argument("--detector_ckpt", type=Path, default=None, help="可选：anomaly detector checkpoint")
    p.add_argument("--concept_ckpt", type=Path, default=None, help="可选：concept detector checkpoint")
    p.add_argument("--concept_model", choices=("cnn", "mlp"), default="cnn")
    p.add_argument("--win_size", type=int, default=100)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--kernel_size", type=int, default=7)
    # online training
    p.add_argument("--retrain_every", type=int, default=25)
    p.add_argument("--updates_per_round", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--buffer_size", type=int, default=512)
    p.add_argument("--detector_lr", type=float, default=1e-4)
    p.add_argument("--concept_lr", type=float, default=1e-4)
    p.add_argument("--update_scope", choices=("full", "head_only", "norm_only"), default="full",
                   help="full=全量; head_only=只更新head/decoder(冻结backbone); norm_only=只更新Norm(TENT)")
    p.add_argument("--min_confidence", type=float, default=0.5)
    p.add_argument("--warmup_windows", type=int, default=200, help="前 K 窗口无条件喂 detector(冷启动)")
    p.add_argument("--freeze_after", type=int, default=0,
                   help="抗漂移对照:>0 时在第 K 窗后停止一切在线更新(冻结臂)。0=全程适应")
    p.add_argument("--detector_track_all", action="store_true",
                   help="漂移跟踪:detector 缓冲每个近期窗(不按异常判定门控),让重建跟住漂移正常流形")
    p.add_argument("--detector_buffer_stride", type=int, default=1,
                   help="稀疏覆盖缓冲:每 K 个合格窗只收 1 个,使有界缓冲覆盖 K× 时间跨度(减重叠)")
    p.add_argument("--no_online", action="store_true", help="关闭在线训练(冻结基线，用于对照)")
    # decision routing (the redesign)
    p.add_argument("--decision", choices=("agent_raw", "calibrated_threshold", "calibrated_agent"),
                   default="agent_raw",
                   help="agent_raw=旧法(LLM 对裸分数判0/1); "
                        "calibrated_threshold=校准检测分数决策(无LLM); "
                        "calibrated_agent=校准提候选,agent 仅在候选+采样正常上确认")
    p.add_argument("--cal_quantile", type=float, default=0.95, help="正常参考的分位阈值")
    p.add_argument("--cal_window", type=int, default=512, help="校准参考缓冲(有界,抗漂移)")
    p.add_argument("--cal_warmup", type=int, default=100, help="前 K 窗无条件作为正常参考种子")
    p.add_argument("--cal_margin", type=float, default=1.0, help="候选判定阈值放大系数(>1 更保守)")
    p.add_argument("--normal_sample_rate", type=float, default=0.05,
                   help="calibrated_agent: 额外送给 agent 的正常窗比例(用于概念标签)")
    # agent
    p.add_argument("--agent", choices=("local", "gpt"), default="local")
    p.add_argument("--agent_model", default="gpt-4o-mini")
    p.add_argument("--anomaly_prior", type=float, default=None,
                   help="告知 LLM 异常基率(如 0.09),证据弱时倾向判正常(保守 agent)")
    p.add_argument("--strict_agent", action="store_true")
    p.add_argument("--openai_timeout", type=float, default=60.0)
    p.add_argument("--openai_max_output_tokens", type=int, default=256)
    # io
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "online" / "metrics.json")
    p.add_argument("--predictions_csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    # 前兆窗口感知早预警评测(论文第一核心组成):有效前兆窗 = [onset-l_max, onset-l_min]
    p.add_argument("--l_min", type=int, default=25, help="有效前兆窗下界(距 onset 最少提前点数,小于则迟滞)")
    p.add_argument("--l_max", type=int, default=150, help="有效前兆窗上界(距 onset 最多提前点数,大于则过早)")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_detector(args, n_vars: int, device) -> MLPAnomalyDetector:
    det = MLPAnomalyDetector(args.win_size, n_vars, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    if args.detector_ckpt is not None:
        ck = torch.load(args.detector_ckpt, map_location="cpu", weights_only=False)
        state = ck.get("detector", ck.get("model"))
        det.load_state_dict(state)
        print(f"loaded detector from {args.detector_ckpt}")
    return det.to(device)


def load_concept(args, n_vars: int, device):
    concepts = DEFAULT_CONCEPTS
    if args.concept_ckpt is not None:
        ck = torch.load(args.concept_ckpt, map_location="cpu", weights_only=False)
        cargs = ck.get("args", {})
        concepts = tuple(cargs.get("concepts", DEFAULT_CONCEPTS))
        model_kind = cargs.get("model", args.concept_model)
        kernel = int(cargs.get("kernel_size", args.kernel_size))
        hidden = int(cargs.get("hidden_dim", args.hidden_dim))
        if model_kind == "cnn":
            con = CNNConceptDetector(args.win_size, n_vars, n_concepts=len(concepts), kernel_size=kernel)
        else:
            con = MLPConceptDetector(args.win_size, n_vars, n_concepts=len(concepts), hidden_dim=hidden)
        con.load_state_dict(ck["model"])
        print(f"loaded concept ({model_kind}) from {args.concept_ckpt}")
    else:
        if args.concept_model == "cnn":
            con = CNNConceptDetector(args.win_size, n_vars, n_concepts=len(concepts), kernel_size=args.kernel_size)
        else:
            con = MLPConceptDetector(args.win_size, n_vars, n_concepts=len(concepts), hidden_dim=args.hidden_dim)
    return con.to(device), concepts


def make_agent(args, concepts):
    decider = args.decision == "calibrated_agent"
    if args.agent == "local":
        return LocalSigLAAgent(concept_names=concepts)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --agent gpt.")
    client = _HTTPClient(api_key, args.openai_timeout, args.openai_max_output_tokens)
    return GPTInstantAgent(
        model=args.agent_model, enabled=True, strict=args.strict_agent,
        client=client, concept_names=concepts, anomaly_rate=args.anomaly_prior,
        decider=decider,
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"count": int(len(y_true)), "positives": int(y_true.sum()),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = choose_device(args.device)

    data = np.load(args.stream)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64) if "y" in data else np.zeros(len(x), dtype=np.int64)
    regime = data["regime"].astype(np.int64) if "regime" in data else np.zeros(len(x), dtype=np.int64)
    n_vars = x.shape[1]
    print(f"stream={args.stream} shape={x.shape} positive={int(y.sum())} ({y.mean():.2%}) device={device}")

    detector = load_detector(args, n_vars, device)
    concept, concepts = load_concept(args, n_vars, device)
    agent = make_agent(args, concepts)
    print(f"concepts={list(concepts)} agent={args.agent} online={not args.no_online}")

    online_cfg = OnlineTrainConfig(
        enabled=not args.no_online,
        retrain_every=args.retrain_every,
        updates_per_round=args.updates_per_round,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        detector_lr=args.detector_lr,
        concept_lr=args.concept_lr,
        update_scope=args.update_scope,
        min_confidence=args.min_confidence,
        warmup_windows=args.warmup_windows,
        freeze_after=args.freeze_after,
        detector_track_all=args.detector_track_all,
        detector_buffer_stride=args.detector_buffer_stride,
    )
    cal_cfg = CalibratorConfig(
        quantile=args.cal_quantile,
        window=args.cal_window,
        warmup=args.cal_warmup,
        margin=args.cal_margin,
    )
    pipe = SigLATrajectoryPipeline(
        detector, concept, agent=agent,
        config=PipelineConfig(
            win_size=args.win_size, step=args.step,
            decision_mode=args.decision,
            calibrator=cal_cfg,
            normal_sample_rate=args.normal_sample_rate,
        ),
        device=device, online_config=online_cfg, concept_names=concepts,
    )

    pred = pipe.predict_traj(x, online=not args.no_online)

    # ---- 评测(标签仅此处使用，未进训练) ---- #
    win_label = np.asarray(
        [int(np.any(y[w.start:w.end + 1] == 1)) for w in pred.windows], dtype=np.int64)
    win_pred = np.asarray(pred.anomaly_flags, dtype=np.int64)
    win_regime = np.asarray([int(regime[w.end]) for w in pred.windows], dtype=np.int64)

    overall = binary_metrics(win_label, win_pred)
    source_counts: dict[str, int] = {}
    for w in pred.windows:
        source_counts[w.judgment.source] = source_counts.get(w.judgment.source, 0) + 1
    per_regime = {}
    for r in sorted(set(win_regime.tolist())):
        m = win_regime == r
        per_regime[str(r)] = binary_metrics(win_label[m], win_pred[m])

    # Calibrated proposer quality (decision before any agent override) + cost.
    win_candidate = np.asarray([int(w.candidate_anomaly) for w in pred.windows], dtype=np.int64)
    agent_calls = int(sum(1 for w in pred.windows if w.agent_called))
    candidate_metrics = binary_metrics(win_label, win_candidate)

    # ---- 前兆窗口感知早预警(论文第一核心组成):用事件 onset + lead-time 口径 ---- #
    win_ends = np.asarray([w.end for w in pred.windows], dtype=np.int64)
    precursor = precursor_metrics(y, win_ends, win_pred, l_min=args.l_min, l_max=args.l_max)

    metrics = {
        "stream": str(args.stream),
        "device": str(device),
        "online": not args.no_online,
        "update_scope": args.update_scope,
        "decision": args.decision,
        "agent": args.agent,
        "agent_source_counts": source_counts,
        "agent_calls": agent_calls,
        "agent_call_rate": float(agent_calls / max(1, len(pred.windows))),
        "calibrator": {"quantile": args.cal_quantile, "window": args.cal_window,
                       "warmup": args.cal_warmup, "margin": args.cal_margin},
        "candidate_only": candidate_metrics,
        "precursor": precursor,
        "concepts": list(concepts),
        "n_windows": len(pred.windows),
        "overall": overall,
        "per_regime": per_regime,
        "online_stats": pred.online_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    csv_path = args.predictions_csv or args.output.with_name("online_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["start", "end", "regime", "label", "candidate", "detector_percentile",
                     "detector_score", "detector_threshold", "score_over_threshold", "max_concept_prob",
                     "agent_called", "is_anomaly", "anomaly_score", "concepts", "confidence", "source"])
        for lab, rg, w in zip(win_label, win_regime, pred.windows):
            thr = float(w.detector_threshold)
            sot = float(w.detector_score) / thr if thr > 0 else 0.0
            max_cp = max(w.concept_state.profile.values()) if w.concept_state.profile else 0.0
            wr.writerow([w.start, w.end, int(rg), int(lab), int(w.candidate_anomaly),
                         float(w.detector_percentile), float(w.detector_score), thr, round(sot, 4),
                         round(float(max_cp), 4), int(w.agent_called),
                         int(w.judgment.is_anomaly), float(w.judgment.anomaly_score),
                         "|".join(w.judgment.concepts), float(w.judgment.confidence), w.judgment.source])

    print(json.dumps(metrics, indent=2, sort_keys=True))
    pm = precursor
    print("\n==== 前兆窗口感知早预警 (l_min=%d l_max=%d, %d events) ====" % (pm["l_min"], pm["l_max"], pm["n_events"]))
    print("  普通 AD:   P/R/F1 = %.3f/%.3f/%.3f  (窗内有异常点即算)" % (overall["precision"], overall["recall"], overall["f1"]))
    print("  早预警:    EW-P/R/F1 = %.3f/%.3f/%.3f  lead-time(mean/med)=%.0f/%.0f" % (
        pm["ew_precision"], pm["ew_recall"], pm["ew_f1"], pm["lead_time_mean"], pm["lead_time_median"]))
    print("  事件归类:  有效=%d 迟滞=%d 事件后=%d 漏=%d" % tuple(pm["event_outcomes"][k] for k in ("valid", "late", "post", "missed")))
    print("  虚高暴露:  普通 recall 把迟滞+事后也算成功 → 比早预警 recall 高 %.3f" % pm["inflation"])
    print("  报警分布:  %s" % pm["alarms_by_region"])
    print(f"saved metrics      -> {args.output}")
    print(f"saved predictions  -> {csv_path}")


if __name__ == "__main__":
    main()
