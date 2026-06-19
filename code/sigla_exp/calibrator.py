from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class CalibratorConfig:
    """Knobs for the streaming detector-score calibrator.

    The calibrator turns a *raw, uncalibrated* reconstruction error into an
    anomaly **decision** by comparing it against a running reference of scores
    drawn from the (believed-)normal distribution. This is the fix for the
    failure mode where the LLM judged 0/1 off a bare scalar: the decision is
    now grounded in where the score falls within the normal distribution.
    """

    quantile: float = 0.95          # threshold = this quantile of the normal reference
    window: int = 512               # bounded reference buffer -> forgets old distribution (drift)
    warmup: int = 100               # first K windows seed the reference unconditionally
    min_ref: int = 32               # need this many reference scores before deciding
    margin: float = 1.0             # candidate if score > threshold * margin (>=1 raises the bar)
    update_normal_only: bool = True  # only scores judged normal re-enter the reference


class ScoreCalibrator:
    """Rolling-quantile calibrator: raw detector score -> candidate anomaly decision.

    Maintains a bounded buffer of detector scores believed to be normal and
    derives a percentile threshold from it. Because the buffer is bounded and
    only fed normal-looking scores, the threshold tracks the *current* normal
    distribution as the stream drifts — this is what lets a frozen detector keep
    a usable threshold under covariate drift, and what an adapting detector
    rides on top of.

    Usage per window:
        cal.observe_warmup(score)                       # during warmup, or
        decision = cal.decide(score)                    # candidate + percentile
        cal.update(score, is_normal=decision.is_normal) # feed reference
    """

    def __init__(self, config: CalibratorConfig | None = None) -> None:
        self.config = config or CalibratorConfig()
        self._ref: deque[float] = deque(maxlen=self.config.window)
        self.seen = 0

    @property
    def ready(self) -> bool:
        return len(self._ref) >= self.config.min_ref

    @property
    def in_warmup(self) -> bool:
        return self.seen < self.config.warmup

    def threshold(self) -> float | None:
        if not self.ready:
            return None
        ref = sorted(self._ref)
        # nearest-rank quantile; clamp index into range
        idx = min(len(ref) - 1, max(0, int(round(self.config.quantile * (len(ref) - 1)))))
        return ref[idx] * self.config.margin

    def percentile_of(self, score: float) -> float:
        """Fraction of the normal reference at or below ``score`` (0..1)."""
        if not self._ref:
            return 0.0
        below = sum(1.0 for r in self._ref if r <= score)
        return below / len(self._ref)

    def decide(self, score: float) -> "CalibrationDecision":
        self.seen += 1
        thr = self.threshold()
        pct = self.percentile_of(score)
        if thr is None:
            # Not enough reference yet: assume normal so the reference can fill.
            return CalibrationDecision(False, pct, thr, ready=False)
        candidate = score > thr
        return CalibrationDecision(candidate, pct, thr, ready=True)

    def update(self, score: float, is_normal: bool) -> None:
        """Feed the score back into the normal reference if appropriate.

        During warmup every score is admitted (early stream assumed mostly
        normal). Afterwards, only scores judged normal re-enter, so the
        reference does not get contaminated by anomalies.
        """
        if self.in_warmup or not self.config.update_normal_only or is_normal:
            self._ref.append(float(score))


@dataclass
class CalibrationDecision:
    is_candidate: bool      # detector proposes this window as anomalous
    percentile: float       # where the score falls within the normal reference (0..1)
    threshold: float | None  # current calibrated threshold (None before ready)
    ready: bool             # whether the reference had enough samples to decide

    @property
    def is_normal(self) -> bool:
        return not self.is_candidate
