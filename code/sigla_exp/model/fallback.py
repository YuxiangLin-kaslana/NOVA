from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from ..actions import ACTION_NAMES
from ..profiles import CONCEPT_NAMES, extract_raw_evidence


@dataclass
class ConceptState:
    profile: dict[str, float]
    primary_concept: str | None
    secondary_concepts: list[str] = field(default_factory=list)
    co_existing_concepts: list[str] = field(default_factory=list)
    suppressed_concepts: list[str] = field(default_factory=list)
    raw_evidence: dict[str, float] = field(default_factory=dict)


@dataclass
class PolicyState:
    candidate_action: str
    candidate_argument: int | None
    action_probabilities: dict[str, float]
    risk_probability: float


@dataclass
class ConceptDisentanglerConfig:
    primary_threshold: float = 0.65
    secondary_threshold: float = 0.55
    suppressed_threshold: float = 0.45


class RMSFallbackDetector:
    """Pure score fallback used when no trained bottom detector is available."""

    @torch.no_grad()
    def score_window(self, signal: torch.Tensor) -> tuple[torch.Tensor, float]:
        point_scores = torch.sqrt(torch.mean(signal * signal, dim=2, keepdim=True))
        return point_scores, float(torch.mean(point_scores).detach().cpu())


class RawEvidenceConceptFallback:
    """Pure concept fallback: use raw hand-crafted evidence as the profile."""

    def transform_with_raw(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = extract_raw_evidence(window).astype(np.float32)
        return raw, raw


class HeuristicPolicyFallback:
    """Pure action fallback from concept profile to candidate policy state."""

    def decide(self, profile: Mapping[str, float]) -> PolicyState:
        if not profile:
            action = "wait"
            confidence = 1.0
        else:
            primary_name, primary_score = max(profile.items(), key=lambda item: item[1])
            if primary_score >= 0.85 and primary_name in {"level_shift", "correlation_break"}:
                action = "escalate"
            elif primary_score >= 0.65:
                action = "alarm"
            elif primary_score >= 0.50:
                action = "request_evidence"
            else:
                action = "wait"
            confidence = float(primary_score)
        return PolicyState(
            candidate_action=action,
            candidate_argument=None,
            action_probabilities={name: float(name == action) * confidence for name in ACTION_NAMES},
            risk_probability=confidence,
        )


class RuleConceptDisentangler:
    """Pure rule-based disentangler for current coarse concept profiles."""

    def __init__(self, config: ConceptDisentanglerConfig | None = None) -> None:
        self.config = config or ConceptDisentanglerConfig()
        self.response_map = {
            "spike": {"level_shift", "seasonal_break", "contextual_deviation"},
            "level_shift": {"contextual_deviation"},
            "seasonal_break": {"contextual_deviation"},
            "contextual_deviation": set(),
            "correlation_break": {"contextual_deviation"},
        }

    def disentangle(
        self,
        profile: Mapping[str, float],
        raw_evidence: Mapping[str, float] | None = None,
    ) -> ConceptState:
        if not profile:
            return ConceptState(profile={}, primary_concept=None)

        primary_name, primary_score = max(profile.items(), key=lambda item: item[1])
        primary = primary_name if primary_score >= self.config.primary_threshold else None
        explainable = self.response_map.get(primary, set()) if primary is not None else set()

        secondary: list[str] = []
        co_existing: list[str] = []
        suppressed: list[str] = []

        for name, score in profile.items():
            if name == primary:
                continue
            if score >= self.config.secondary_threshold:
                if name in explainable and primary_score - score >= 0.05:
                    secondary.append(name)
                else:
                    co_existing.append(name)
            elif score >= self.config.suppressed_threshold and name in explainable:
                suppressed.append(name)

        return ConceptState(
            profile={name: float(value) for name, value in profile.items()},
            primary_concept=primary,
            secondary_concepts=secondary,
            co_existing_concepts=co_existing,
            suppressed_concepts=suppressed,
            raw_evidence={name: float(value) for name, value in (raw_evidence or {}).items()},
        )
