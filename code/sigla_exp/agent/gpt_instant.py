from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..profiles import CONCEPT_NAMES


DEFAULT_AGENT_MODEL = "gpt5.2instant"


@dataclass
class AnomalyJudgment:
    """Agent verdict for one window.

    Carries both of the agent's responsibilities:
      1. ``is_anomaly`` / ``anomaly_score`` -> the anomaly decision.
      2. ``concepts`` -> per-window concept pseudo-labels used by the online
         supervision path to retrain the concept detector.
    """

    is_anomaly: bool
    anomaly_score: float = 0.0
    concepts: tuple[str, ...] = ()
    confidence: float = 0.0
    rationale: str = ""
    source: str = "local"
    raw_response: str | None = None

    def concept_multihot(self, concept_names: Sequence[str] | None = None) -> list[float]:
        names = concept_names if concept_names is not None else CONCEPT_NAMES
        present = set(self.concepts)
        return [1.0 if name in present else 0.0 for name in names]


@dataclass
class AgentDecision:
    """Legacy action-policy verdict. Retained for backward-compatible imports."""

    action: str
    argument: int | str | None = None
    confidence: float = 0.0
    rationale: str = ""
    source: str = "local"
    raw_response: str | None = None


@dataclass
class AgentContext:
    """Evidence shown to the agent for a single window.

    The action-policy candidate has been removed: the agent now reasons over the
    detector score and the concept detector profile directly to decide whether
    the window is anomalous and which concepts are present.
    """

    time_index: int
    window_start: int
    window_end: int
    detector_score: float
    risk_state: float
    profile: Mapping[str, float]
    primary_concept: str | None = None
    secondary_concepts: Sequence[str] = field(default_factory=tuple)
    co_existing_concepts: Sequence[str] = field(default_factory=tuple)
    suppressed_concepts: Sequence[str] = field(default_factory=tuple)
    language_context: str = ""
    retrieved_cases: Sequence[Mapping[str, Any] | str] = field(default_factory=tuple)
    judgment_history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # Calibrated detector signal (proposer->decider). When ``calibrated`` is True
    # the agent is a *decider* confirming a candidate flagged by a calibrated
    # detector, not judging a bare reconstruction error.
    calibrated: bool = False
    detector_percentile: float | None = None  # score's rank within the normal reference (0..1)
    candidate_anomaly: bool | None = None      # whether the calibrated detector flagged this window
    detector_threshold: float | None = None    # current calibrated threshold the score is compared to

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "time_index": self.time_index,
            "window": {"start": self.window_start, "end": self.window_end},
            "detector_score": self.detector_score,
            "risk_state": self.risk_state,
            "concept_profile": dict(self.profile),
            "candidate_concepts": list(self.profile.keys()),
            "primary_concept": self.primary_concept,
            "secondary_concepts": list(self.secondary_concepts),
            "co_existing_concepts": list(self.co_existing_concepts),
            "suppressed_concepts": list(self.suppressed_concepts),
            "language_context": self.language_context,
            "retrieved_cases": list(self.retrieved_cases),
            "judgment_history": list(self.judgment_history),
        }
        if self.calibrated:
            # Discriminative signals the decider needs to separate true anomalies
            # from false alarms. percentile_in_normal saturates at ~1.0 for every
            # candidate (p95.1 and p99.9 look identical), so on its own it carries
            # no separating power at a loose threshold. score_over_threshold (how
            # far above the bar) and the concept signals are what actually separate.
            profile_values = list(self.profile.values()) if self.profile else [0.0]
            max_concept_prob = max(profile_values)
            score_over_threshold: float | None = None
            if self.detector_threshold is not None and self.detector_threshold > 0:
                score_over_threshold = round(self.detector_score / self.detector_threshold, 3)
            payload["calibrated_detector"] = {
                "percentile_in_normal": self.detector_percentile,
                "exceeds_threshold": self.candidate_anomaly,
                "score_over_threshold": score_over_threshold,
                "max_concept_prob": round(float(max_concept_prob), 3),
                "concept_persistence": round(float(self.risk_state), 3),
            }
        return payload


def _local_concepts(context: AgentContext, concept_names: Sequence[str], threshold: float) -> tuple[str, ...]:
    return tuple(name for name in concept_names if float(context.profile.get(name, 0.0)) >= threshold)


class LocalSigLAAgent:
    """Deterministic fallback agent used when no LLM call should be made.

    Thresholds the concept profile / risk state instead of calling an LLM, so
    the pipeline (and the online supervision path) stays fully reproducible.
    """

    def __init__(self, threshold: float = 0.5, concept_names: Sequence[str] | None = None) -> None:
        self.threshold = threshold
        self.concept_names = tuple(concept_names) if concept_names is not None else CONCEPT_NAMES

    def judge(self, context: AgentContext) -> AnomalyJudgment:
        profile_values = list(context.profile.values()) if context.profile else [0.0]
        max_concept = max(profile_values)
        concepts = _local_concepts(context, self.concept_names, self.threshold)
        if context.calibrated and context.candidate_anomaly is not None:
            # Decision belongs to the calibrated detector; this agent only labels
            # concepts. anomaly_score/confidence reflect the calibrated percentile.
            is_anomaly = bool(context.candidate_anomaly)
            pct = context.detector_percentile if context.detector_percentile is not None else 0.0
            anomaly_score = float(pct)
            rationale = "Calibrated detector decided; local agent supplied concept labels only."
        else:
            anomaly_score = max(max_concept, float(context.risk_state))
            is_anomaly = bool(concepts) or anomaly_score >= self.threshold
            rationale = (
                "Local fallback agent thresholded the concept profile / risk state "
                "because it does not call an external LLM."
            )
        return AnomalyJudgment(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            concepts=concepts,
            confidence=float(max(anomaly_score, 0.5 if is_anomaly else anomaly_score)),
            rationale=rationale,
            source="local",
        )


class GPTInstantAgent:
    """GPT-backed SigLA anomaly-judgment agent.

    The OpenAI SDK is imported lazily so the package remains usable in training
    environments that do not install API dependencies. Set strict=True if a
    missing SDK or malformed model response should raise instead of falling back
    to the deterministic local judgment.
    """

    def __init__(
        self,
        model: str = DEFAULT_AGENT_MODEL,
        enabled: bool = True,
        strict: bool = False,
        client: Any | None = None,
        concept_names: Sequence[str] | None = None,
        anomaly_rate: float | None = None,
        decider: bool = False,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.strict = strict
        self.client = client
        self.decider = decider
        self.concept_names = tuple(concept_names) if concept_names is not None else CONCEPT_NAMES
        if decider:
            self.instructions = build_decider_instructions(self.concept_names, anomaly_rate)
        else:
            self.instructions = build_instructions(self.concept_names, anomaly_rate)
        self.fallback = LocalSigLAAgent(concept_names=self.concept_names)

    def judge(self, context: AgentContext) -> AnomalyJudgment:
        if not self.enabled:
            return self.fallback.judge(context)

        try:
            client = self.client or self._make_client()
            response = client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=[
                    {
                        "role": "user",
                        "content": json.dumps(context.to_payload(), sort_keys=True),
                    }
                ],
            )
            text = response.output_text
            parsed = _extract_json(text)
            local = self.fallback.judge(context)
            # In decider mode the calibrated detector's candidate is the default
            # decision; the agent only overrides it with an explicit is_anomaly.
            default_anom = (
                bool(context.candidate_anomaly)
                if self.decider and context.candidate_anomaly is not None
                else local.is_anomaly
            )
            return AnomalyJudgment(
                is_anomaly=bool(parsed.get("is_anomaly", default_anom)),
                anomaly_score=float(parsed.get("anomaly_score", local.anomaly_score)),
                concepts=_clean_concepts(parsed.get("concepts", local.concepts), self.concept_names),
                confidence=float(parsed.get("confidence", 0.0)),
                rationale=str(parsed.get("rationale", "")),
                source=self.model,
                raw_response=text,
            )
        except Exception as exc:
            if self.strict:
                raise
            judgment = self.fallback.judge(context)
            judgment.rationale = f"{judgment.rationale} LLM fallback reason: {type(exc).__name__}: {exc}"
            judgment.source = "local_fallback"
            return judgment

    @staticmethod
    def _make_client() -> Any:
        from openai import OpenAI

        return OpenAI()


def build_instructions(concept_names: Sequence[str], anomaly_rate: float | None = None) -> str:
    conservative = ""
    if anomaly_rate is not None:
        pct = anomaly_rate * 100.0
        conservative = (
            f"\n\nAnomalies are RARE here (~{pct:.1f}% of windows); most windows are "
            "normal. Set is_anomaly=true only when the detector score AND concept "
            "evidence are clearly abnormal. When evidence is weak or ambiguous, "
            "prefer is_anomaly=false with an empty concepts list."
        )
    return f"""
You are the SigLA time-series monitoring agent. For the current window you must:
  1. Decide whether the window is anomalous.
  2. List which concepts are present, chosen only from:
     {", ".join(concept_names)}.

Use only the provided current and historical context (detector score, concept
profile, history). Do not assume future observations or labels. Your concept
list is used as a supervision signal, so include a concept only when the
evidence supports it.{conservative}

Return a JSON object with these keys:
  is_anomaly (boolean),
  anomaly_score (float in [0, 1]),
  concepts (array of concept names from the list above),
  confidence (float in [0, 1]),
  rationale (string).
""".strip()


def build_decider_instructions(concept_names: Sequence[str], anomaly_rate: float | None = None) -> str:
    """Decider-mode instructions: the agent VETOES weak calibrated candidates.

    The calibrated detector runs at a deliberately LOOSE threshold chosen for
    high recall, so it over-flags: roughly half of its candidates are false
    alarms. A single scalar threshold cannot raise precision without dropping
    recall, so the agent's job is the one thing the threshold cannot do — look
    at the discriminative signals and veto the false alarms, keeping recall while
    recovering precision. The agent is a SKEPTIC, not a rubber stamp.
    """
    prior = ""
    if anomaly_rate is not None:
        prior = (
            f" True anomalies are rare overall (~{anomaly_rate * 100.0:.1f}% of windows), "
            "yet most windows shown to you are flagged candidates — that is exactly "
            "why so many candidates must be false alarms."
        )
    return f"""
You are the SigLA time-series monitoring agent acting as a DECIDER operating at a
deliberately LOOSE detector threshold. The threshold was set low on purpose to
catch nearly every real anomaly (high recall), which means the detector
OVER-FLAGS: roughly half of the candidates it proposes are false alarms. A
scalar threshold alone cannot tell them apart. You can — that is your entire job.

Under "calibrated_detector" you are given, for the current window:
  - exceeds_threshold: whether the detector flagged this window as a candidate,
  - score_over_threshold: detector score divided by the threshold. ~1.0 means the
    score is BARELY over the bar (a typical false alarm); >=1.5 means clearly,
    decisively above it (a typical true anomaly). This is your strongest signal.
  - percentile_in_normal: rank vs. recent normal windows. SATURATES near 1.0 for
    almost every candidate, so it barely separates true anomalies from false
    alarms — do not lean on it.
  - max_concept_prob: strongest concept-detector probability for this window,
  - concept_persistence: how sustained the concept activation has been (a real
    fault tends to persist; a one-window blip is usually noise).

Your tasks:
  1. DECIDE is_anomaly. The detector over-flags by design, so VETO is your
     normal action, not the exception. Apply this CONFIRM GATE — confirm
     (is_anomaly=true) a candidate ONLY when at least one holds:
       (a) score_over_threshold >= 1.3  (the score is decisively, not marginally,
           above the bar), OR
       (b) STRONG, SPECIFIC, PERSISTENT concept evidence — i.e. high
           max_concept_prob (>= ~0.6) backed by concept_persistence that shows the
           activation is sustained, not a one-window blip.
     Otherwise VETO (is_anomaly=false). In particular, a candidate that only
     marginally exceeds the threshold (score_over_threshold < 1.3) AND lacks
     strong concept support is a false alarm — veto it. Do NOT confirm just
     because exceeds_threshold is true; that is exactly the false-alarm population.
     Clause (b) is the point of using an LLM: it lets you KEEP a genuine anomaly
     whose score is only marginally over the bar but whose concept evidence is
     strong — something a higher score threshold would wrongly drop.
     For a non-candidate (exceeds_threshold=false), default to is_anomaly=false;
     promote it only under clause (b).
     Expect to veto a large share of candidates (often roughly half). A run where
     you veto almost nothing means you added nothing over a bare threshold.
  2. LABEL which concepts are present, chosen only from:
     {", ".join(concept_names)}.
     This list is the supervision signal for online training, so include a
     concept only when the evidence supports it.{prior}

Use only the provided context; do not assume future observations or labels.

Return a JSON object with these keys:
  is_anomaly (boolean),
  anomaly_score (float in [0, 1]),
  concepts (array of concept names from the list above),
  confidence (float in [0, 1]),
  rationale (string).
""".strip()


# Default instructions for the legacy 5-concept taxonomy.
AGENT_INSTRUCTIONS = build_instructions(CONCEPT_NAMES)


def _clean_concepts(value: Any, concept_names: Sequence[str] = CONCEPT_NAMES) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    allowed = set(concept_names)
    return tuple(str(item) for item in value if str(item) in allowed)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("Agent response did not contain a JSON object.")
