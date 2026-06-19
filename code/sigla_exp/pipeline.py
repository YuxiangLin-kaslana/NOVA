from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from .agent import AgentContext, AnomalyJudgment, LocalSigLAAgent
from .calibrator import CalibratorConfig, ScoreCalibrator
from .model import (
    ConceptDisentanglerConfig,
    ConceptState,
    RMSFallbackDetector,
    RawEvidenceConceptFallback,
    RuleConceptDisentangler,
)
from .online import OnlineTrainConfig, OnlineTrainer
from .profiles import CONCEPT_NAMES, extract_raw_evidence


class JudgeAgent(Protocol):
    def judge(self, context: AgentContext) -> AnomalyJudgment:
        ...


@dataclass
class PipelineConfig:
    win_size: int = 50
    step: int = 5
    risk_decay: float = 0.85
    primary_threshold: float = 0.65
    secondary_threshold: float = 0.55
    suppressed_threshold: float = 0.45
    max_history_items: int = 8
    # Anomaly decision routing (the redesign):
    #   "agent_raw"            -> legacy: agent judges 0/1 off the raw detector score.
    #   "calibrated_threshold" -> calibrated detector decides; agent NOT called.
    #   "calibrated_agent"     -> calibrated detector proposes a candidate; the agent
    #                             is called only on candidates (+ a sampled fraction of
    #                             normals) to confirm and to label concepts.
    decision_mode: str = "agent_raw"
    calibrator: CalibratorConfig | None = None
    normal_sample_rate: float = 0.05   # fraction of non-candidate windows sent to the agent
    concept_label_threshold: float = 0.5  # threshold for non-agent concept pseudo-labels


@dataclass
class WindowPrediction:
    start: int
    end: int
    detector_score: float
    risk_state: float
    concept_state: ConceptState
    judgment: AnomalyJudgment
    candidate_anomaly: bool = False
    detector_percentile: float = 0.0
    detector_threshold: float = 0.0
    agent_called: bool = False


@dataclass
class TrajectoryPrediction:
    n_points: int
    n_vars: int
    windows: list[WindowPrediction]
    online_stats: dict[str, Any] | None = None

    @property
    def anomaly_flags(self) -> list[int]:
        return [int(window.judgment.is_anomaly) for window in self.windows]

    @property
    def anomaly_scores(self) -> list[float]:
        return [float(window.judgment.anomaly_score) for window in self.windows]


class SigLATrajectoryPipeline:
    """End-to-end inference flow for one trajectory.

    The action policy has been removed: the flow is detector -> concept -> agent,
    where the agent both judges anomalies and supplies concept pseudo-labels.
    When ``online`` is enabled, an :class:`OnlineTrainer` retrains the detector
    and concept detector from those judgments as the stream is processed.

    Expected trajectory input is a standardized multivariate time series shaped
    [time, variables]. If a fitted scaler is provided, this class applies it
    before windowing.
    """

    def __init__(
        self,
        detector: torch.nn.Module | None,
        concept_extractor: Any,
        agent: JudgeAgent | None = None,
        scaler: Any | None = None,
        config: PipelineConfig | None = None,
        device: torch.device | str | None = None,
        online_config: OnlineTrainConfig | None = None,
        concept_names: Sequence[str] | None = None,
    ) -> None:
        self.detector = detector
        self.concept_extractor = concept_extractor
        self.concept_names = tuple(concept_names) if concept_names is not None else CONCEPT_NAMES
        self.agent = agent or LocalSigLAAgent(concept_names=self.concept_names)
        self.scaler = scaler
        self.config = config or PipelineConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online_config = online_config or OnlineTrainConfig()
        self.fallback_detector = RMSFallbackDetector()
        self.fallback_concept_extractor = RawEvidenceConceptFallback()
        self.disentangler = RuleConceptDisentangler(
            ConceptDisentanglerConfig(
                primary_threshold=self.config.primary_threshold,
                secondary_threshold=self.config.secondary_threshold,
                suppressed_threshold=self.config.suppressed_threshold,
            )
        )
        self._trainer: OnlineTrainer | None = None

        if self.detector is not None:
            self.detector.to(self.device).eval()
        if isinstance(self.concept_extractor, torch.nn.Module):
            self.concept_extractor.to(self.device).eval()

    def _ensure_trainer(self) -> OnlineTrainer:
        if self._trainer is None:
            self._trainer = OnlineTrainer(
                self.detector,
                self.concept_extractor,
                device=self.device,
                config=self.online_config,
                concept_names=self.concept_names,
            )
        return self._trainer

    def predict_traj(
        self,
        traj: np.ndarray,
        language_context: str = "",
        retrieved_cases: Sequence[Mapping[str, Any] | str] = (),
        initial_history: Sequence[Mapping[str, Any]] = (),
        online: bool = False,
    ) -> TrajectoryPrediction:
        x = self._prepare_traj(traj)
        starts = np.arange(0, max(0, len(x) - self.config.win_size + 1), self.config.step, dtype=np.int64)
        if len(starts) == 0:
            raise ValueError(f"Trajectory is shorter than win_size={self.config.win_size}: length={len(x)}")

        trainer = self._ensure_trainer() if online else None
        calibrated = self.config.decision_mode in ("calibrated_threshold", "calibrated_agent")
        calibrator = ScoreCalibrator(self.config.calibrator) if calibrated else None
        rng = np.random.default_rng(0)
        risk_state = 0.0
        history = [dict(item) for item in initial_history]
        predictions: list[WindowPrediction] = []

        for start_item in starts:
            start = int(start_item)
            end = start + self.config.win_size
            window = x[start:end].astype(np.float32)
            signal_t = torch.from_numpy(window).unsqueeze(0).to(self.device)

            _, detector_score = self._score_window(signal_t)
            concept_input, concept_state = self._build_concept_state(window)

            local_risk = max(concept_state.profile.values()) if concept_state.profile else 0.0
            risk_state = self.config.risk_decay * risk_state + (1.0 - self.config.risk_decay) * local_risk

            # ---- calibrated detector proposes the candidate (proposer) ---- #
            threshold: float | None = None
            if calibrator is not None:
                decision = calibrator.decide(float(detector_score))
                candidate = decision.is_candidate
                percentile = decision.percentile
                detector_is_normal = decision.is_normal
                threshold = decision.threshold
            else:
                candidate = False
                percentile = 0.0
                detector_is_normal = None

            context = AgentContext(
                time_index=end - 1,
                window_start=start,
                window_end=end - 1,
                detector_score=float(detector_score),
                risk_state=float(risk_state),
                profile=concept_state.profile,
                primary_concept=concept_state.primary_concept,
                secondary_concepts=concept_state.secondary_concepts,
                co_existing_concepts=concept_state.co_existing_concepts,
                suppressed_concepts=concept_state.suppressed_concepts,
                language_context=language_context,
                retrieved_cases=retrieved_cases,
                judgment_history=tuple(history[-self.config.max_history_items :]),
                calibrated=calibrated,
                detector_percentile=percentile if calibrated else None,
                candidate_anomaly=candidate if calibrated else None,
                detector_threshold=threshold if calibrated else None,
            )

            # ---- route the decision (decider) ---- #
            agent_called = False
            if self.config.decision_mode == "calibrated_threshold":
                # Pure calibrated detector: no LLM. Concepts thresholded for labels.
                judgment = self._calibrated_judgment(context, candidate, percentile, concept_state)
            elif self.config.decision_mode == "calibrated_agent":
                # Call the agent only on candidates (+ a sampled fraction of normals)
                # so it confirms/labels where it matters, cutting LLM cost ~10x.
                call = candidate or (rng.random() < self.config.normal_sample_rate)
                if call:
                    judgment = self.agent.judge(context)
                    agent_called = True
                else:
                    judgment = self._calibrated_judgment(context, candidate, percentile, concept_state)
            else:  # "agent_raw" (legacy)
                judgment = self.agent.judge(context)
                agent_called = True

            if calibrator is not None:
                # Ground the reference in the *calibrated* proposer, not the agent.
                calibrator.update(float(detector_score), is_normal=detector_is_normal)

            if trainer is not None:
                trainer.observe(window, concept_input, judgment, detector_is_normal=detector_is_normal)

            predictions.append(
                WindowPrediction(
                    start=start,
                    end=end - 1,
                    detector_score=float(detector_score),
                    risk_state=float(risk_state),
                    concept_state=concept_state,
                    judgment=judgment,
                    candidate_anomaly=bool(candidate),
                    detector_percentile=float(percentile),
                    detector_threshold=float(threshold) if threshold is not None else 0.0,
                    agent_called=bool(agent_called),
                )
            )
            history.append(
                {
                    "time_index": end - 1,
                    "is_anomaly": bool(judgment.is_anomaly),
                    "concepts": list(judgment.concepts),
                    "confidence": float(judgment.confidence),
                }
            )

        return TrajectoryPrediction(
            n_points=int(len(x)),
            n_vars=int(x.shape[1]),
            windows=predictions,
            online_stats=trainer.stats() if trainer is not None else None,
        )

    def _prepare_traj(self, traj: np.ndarray) -> np.ndarray:
        x = np.asarray(traj, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"Expected trajectory shaped [time, variables], got {x.shape}")
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        return x

    @torch.no_grad()
    def _score_window(self, signal_t: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.detector is None:
            return self.fallback_detector.score_window(signal_t)

        recon = self.detector(signal_t)
        point_scores = torch.mean((recon - signal_t) ** 2, dim=2, keepdim=True)
        detector_score = float(torch.mean(point_scores).detach().cpu())
        return point_scores, detector_score

    @torch.no_grad()
    def _concept_profile(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (concept_input, profile).

        ``concept_input`` is whatever the concept model consumes — the raw window
        for a window-based detector (CNN/MLPConceptDetector), or the hand-crafted
        evidence vector for the evidence-based MLPConceptExtractor — and is what
        the OnlineTrainer retrains the concept model on.
        """
        if self.concept_extractor is None:
            return self.fallback_concept_extractor.transform_with_raw(window)
        # Window-based concept detector (e.g. CNNConceptDetector / MLPConceptDetector).
        if hasattr(self.concept_extractor, "predict_proba"):
            signal_t = torch.from_numpy(window).unsqueeze(0).to(self.device)
            profile = self.concept_extractor.predict_proba(signal_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
            return window.astype(np.float32), profile
        raw = extract_raw_evidence(window).astype(np.float32)
        if hasattr(self.concept_extractor, "transform"):
            return raw, np.asarray(self.concept_extractor.transform(window), dtype=np.float32)
        if isinstance(self.concept_extractor, torch.nn.Module):
            evidence_t = torch.from_numpy(raw).unsqueeze(0).to(self.device)
            logits = self.concept_extractor(evidence_t)
            profile = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
            return raw, profile
        raise TypeError(f"Unsupported concept_extractor type: {type(self.concept_extractor)!r}")

    def _build_concept_state(self, window: np.ndarray) -> tuple[np.ndarray, ConceptState]:
        concept_input, profile = self._concept_profile(window)
        names = self.concept_names
        profile_map = {name: float(value) for name, value in zip(names, profile)}
        # Evidence map is only meaningful for the evidence-based path; align to names.
        raw_map = profile_map if concept_input.ndim > 1 else {
            name: float(value) for name, value in zip(names, concept_input)
        }
        return concept_input, self.disentangler.disentangle(profile_map, raw_map)

    def _calibrated_judgment(
        self,
        context: AgentContext,
        candidate: bool,
        percentile: float,
        concept_state: ConceptState,
    ) -> AnomalyJudgment:
        """Build a verdict without an LLM call: the calibrated detector decides,
        concepts come from thresholding the concept profile.

        Used by the ``calibrated_threshold`` arm and for the (cheap) non-sampled
        normal windows in ``calibrated_agent``. Confidence stays low for normals
        so the online concept buffer is not fed unconfirmed pseudo-labels.
        """
        thr = self.config.concept_label_threshold
        concepts = tuple(name for name, val in concept_state.profile.items() if float(val) >= thr)
        return AnomalyJudgment(
            is_anomaly=bool(candidate),
            anomaly_score=float(percentile),
            concepts=concepts if candidate else (),
            confidence=float(percentile if candidate else min(percentile, 0.49)),
            rationale="Calibrated detector decision (no LLM call).",
            source="calibrated",
        )
