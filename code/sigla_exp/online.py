from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .agent import AnomalyJudgment
from .profiles import CONCEPT_NAMES

_NORM_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)


def _trainable_modules(model: torch.nn.Module, scope: str) -> list[torch.nn.Module]:
    """Return the submodules to keep trainable for the given update scope.

    full      -> the whole model.
    head_only -> the `head` submodule (concept detectors) or `decoder`
                 (reconstruction detector); falls back to the last Linear.
    norm_only -> every normalization layer (BatchNorm/LayerNorm/GroupNorm).
    """
    if scope == "full":
        return [model]
    if scope == "norm_only":
        return [m for m in model.modules() if isinstance(m, _NORM_TYPES)]
    if scope == "head_only":
        if hasattr(model, "head") and isinstance(model.head, torch.nn.Module):
            return [model.head]
        if hasattr(model, "decoder") and isinstance(model.decoder, torch.nn.Module):
            return [model.decoder]
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        return [linears[-1]] if linears else [model]
    raise ValueError(f"Unknown update_scope: {scope!r}")


def _setup_scope(model: torch.nn.Module | None, scope: str):
    """Freeze everything outside the trainable modules; return (modules, params).

    Frozen params get requires_grad=False so backward skips them; the optimizer
    is built only over the trainable params. Returns ([], []) if nothing trains.
    """
    if model is None:
        return [], []
    modules = _trainable_modules(model, scope)
    keep = set()
    for m in modules:
        keep.update(id(p) for p in m.parameters())
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    params = [p for p in model.parameters() if p.requires_grad]
    return modules, params


@dataclass
class OnlineTrainConfig:
    """Knobs for the streaming retraining of detector + concept detector."""

    enabled: bool = True
    retrain_every: int = 25          # run an update after this many observed windows
    updates_per_round: int = 1       # gradient steps per update round
    batch_size: int = 32
    buffer_size: int = 512           # max samples kept per replay buffer
    detector_lr: float = 1e-4
    concept_lr: float = 1e-4
    update_scope: str = "full"       # "full" | "head_only" | "norm_only"
                                     #   head_only: freeze backbone, train only the
                                     #     head (concept) / decoder (detector) — less
                                     #     forgetting, cheaper, common for online adapt.
                                     #   norm_only: train only Norm affine params
                                     #     (TENT-style test-time adaptation).
    min_confidence: float = 0.5      # only trust agent judgments at/above this confidence
    detector_normal_only: bool = True  # reconstruction detector learns the normal manifold
    detector_buffer_stride: int = 1    # sparse-coverage buffer: keep only every K-th eligible
                                       # window. A bounded buffer then SPANS K× more time with
                                       # little overlap (consecutive windows overlap ~95% at
                                       # small step), so a burst refit sees the whole recent
                                       # regime instead of a narrow redundant slice — matching
                                       # the offline full-regime refit. 1 = keep every window.
    detector_track_all: bool = False   # drift-tracking mode: buffer EVERY recent window for
                                       # detector reconstruction, regardless of the anomaly
                                       # verdict. Anomalies are rare, so a bounded recent buffer
                                       # is dominated by (drifted) normals — this lets the
                                       # detector track the drifting normal manifold instead of
                                       # being starved when the calibrator flags everything
                                       # post-drift. Pair with a small buffer_size to forget the
                                       # old regime quickly.
    warmup_windows: int = 0          # first K windows feed the detector unconditionally
                                     # (assume early stream is mostly normal; breaks the
                                     # cold-start where an untrained concept net marks
                                     # every window anomalous and starves the detector)
    freeze_after: int = 0            # if >0, stop all online updates after this many
                                     # observed windows. Used for the anti-drift control:
                                     # both arms learn the initial regime identically, then
                                     # the "frozen" arm freezes here while the adapting arm
                                     # keeps tracking the drift. 0 = never freeze (adapt all).


@dataclass
class OnlineTrainState:
    observed: int = 0
    detector_updates: int = 0
    concept_updates: int = 0
    detector_buffer_size: int = 0
    concept_buffer_size: int = 0
    last_detector_loss: float | None = None
    last_concept_loss: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class OnlineTrainer:
    """Streaming supervisor that retrains the two small models from agent verdicts.

    Supervision sources (decided with the user):
      * detector  -> self-supervised reconstruction on windows the agent judged
        normal (keeps the reconstruction model anchored to the current normal
        manifold as the stream drifts).
      * concept   -> BCE against the agent's concept list, used as pseudo-labels.

    Models that are not ``torch.nn.Module`` (i.e. the rule-based fallbacks) are
    skipped, so the online loop degrades gracefully.
    """

    def __init__(
        self,
        detector: torch.nn.Module | None,
        concept_extractor: Any,
        device: torch.device | str | None = None,
        config: OnlineTrainConfig | None = None,
        concept_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config or OnlineTrainConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.concept_names = tuple(concept_names) if concept_names is not None else CONCEPT_NAMES

        self.detector = detector if isinstance(detector, torch.nn.Module) else None
        self.concept_extractor = concept_extractor if isinstance(concept_extractor, torch.nn.Module) else None

        scope = self.config.update_scope
        self._det_modules, det_params = _setup_scope(self.detector, scope)
        self._con_modules, con_params = _setup_scope(self.concept_extractor, scope)
        self.detector_opt = (
            torch.optim.AdamW(det_params, lr=self.config.detector_lr) if det_params else None
        )
        self.concept_opt = (
            torch.optim.AdamW(con_params, lr=self.config.concept_lr) if con_params else None
        )

        self._detector_buf: deque[np.ndarray] = deque(maxlen=self.config.buffer_size)
        self._concept_buf: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.config.buffer_size)
        self.state = OnlineTrainState()

    @property
    def active(self) -> bool:
        return self.config.enabled and (self.detector is not None or self.concept_extractor is not None)

    def observe(
        self,
        window: np.ndarray,
        concept_input: np.ndarray,
        judgment: AnomalyJudgment,
        detector_is_normal: bool | None = None,
    ) -> dict[str, Any] | None:
        """Buffer one window's supervision and run an update when it is due.

        ``concept_input`` is whatever the concept model consumes (raw window for
        a window-based detector, evidence vector for the evidence-based one).

        ``detector_is_normal`` lets the caller ground the detector's normal
        signal in the *calibrated detector* (score below threshold) rather than
        the agent verdict. This breaks the self-reinforcement loop where the
        agent's 0/1 both drove the decision and supervised the detector. When
        ``None`` the legacy agent-verdict gate is used.
        """
        if not self.config.enabled:
            return None

        self.state.observed += 1
        # Anti-drift control: once frozen, observe nothing further (no buffering,
        # no gradient steps) so the detector stays exactly as it was at the freeze
        # point while the stream keeps drifting.
        if self.config.freeze_after and self.state.observed > self.config.freeze_after:
            return None
        in_warmup = self.state.observed <= self.config.warmup_windows
        trust = judgment.confidence >= self.config.min_confidence
        # Detector: self-supervised reconstruction. During warmup feed every window
        # (early stream assumed normal); afterwards feed only windows believed
        # normal. Prefer the calibrated detector's verdict when supplied.
        if detector_is_normal is None:
            is_normal = not (self.config.detector_normal_only and judgment.is_anomaly)
            normal_trusted = trust and is_normal
        else:
            normal_trusted = detector_is_normal
        if self.detector is not None:
            # track_all: buffer every recent window (drift tracking). Otherwise feed
            # only warmup / believed-normal windows (anomaly-clean manifold).
            eligible = self.config.detector_track_all or in_warmup or normal_trusted
            stride = max(1, self.config.detector_buffer_stride)
            # Sparse coverage: subsample eligible windows (except during warmup, which
            # stays dense for cold start) so the bounded buffer spans a wider time range.
            if eligible and (in_warmup or self.state.observed % stride == 0):
                self._detector_buf.append(np.asarray(window, dtype=np.float32))
        # Concept: needs agent pseudo-labels, so only when the agent is trusted.
        if self.concept_extractor is not None and trust:
            target = np.asarray(judgment.concept_multihot(self.concept_names), dtype=np.float32)
            self._concept_buf.append((np.asarray(concept_input, dtype=np.float32), target))

        self.state.detector_buffer_size = len(self._detector_buf)
        self.state.concept_buffer_size = len(self._concept_buf)

        if self.config.retrain_every > 0 and self.state.observed % self.config.retrain_every == 0:
            return self.update()
        return None

    def update(self) -> dict[str, Any] | None:
        """Run ``updates_per_round`` gradient steps on whatever buffers have data."""
        if not self.active:
            return None
        det_losses: list[float] = []
        con_losses: list[float] = []
        for _ in range(max(1, self.config.updates_per_round)):
            det = self._detector_step()
            con = self._concept_step()
            if det is not None:
                det_losses.append(det)
            if con is not None:
                con_losses.append(con)

        if not det_losses and not con_losses:
            return None

        row: dict[str, Any] = {"observed": self.state.observed}
        if det_losses:
            self.state.detector_updates += 1
            self.state.last_detector_loss = float(np.mean(det_losses))
            row["detector_loss"] = self.state.last_detector_loss
        if con_losses:
            self.state.concept_updates += 1
            self.state.last_concept_loss = float(np.mean(con_losses))
            row["concept_loss"] = self.state.last_concept_loss
        self.state.history.append(row)
        return row

    @staticmethod
    def _train_mode(model: torch.nn.Module, modules: list[torch.nn.Module]) -> None:
        """Put `model` in eval, then flip only the trainable modules to train.

        This keeps a frozen backbone's BatchNorm running stats fixed under
        head_only, while letting norm_only update batch statistics (TENT-style).
        """
        model.eval()
        for m in modules:
            m.train()

    def _sample(self, buf: deque, batch_size: int) -> list[int]:
        n = len(buf)
        if n == 0:
            return []
        k = min(batch_size, n)
        return torch.randperm(n)[:k].tolist()

    def _detector_step(self) -> float | None:
        if self.detector is None or self.detector_opt is None or not self._detector_buf:
            return None
        idx = self._sample(self._detector_buf, self.config.batch_size)
        batch = np.stack([self._detector_buf[i] for i in idx], axis=0)
        signal = torch.from_numpy(batch).to(self.device)
        self._train_mode(self.detector, self._det_modules)
        recon = self.detector(signal)
        loss = F.mse_loss(recon, signal)
        self.detector_opt.zero_grad()
        loss.backward()
        self.detector_opt.step()
        self.detector.eval()
        return float(loss.detach().cpu())

    def _concept_step(self) -> float | None:
        if self.concept_extractor is None or self.concept_opt is None or not self._concept_buf:
            return None
        idx = self._sample(self._concept_buf, self.config.batch_size)
        if len(idx) < 2:
            # BatchNorm-based concept detectors need >= 2 samples in train mode.
            return None
        inputs = np.stack([self._concept_buf[i][0] for i in idx], axis=0)
        targets = np.stack([self._concept_buf[i][1] for i in idx], axis=0)
        evidence_t = torch.from_numpy(inputs).to(self.device)
        target_t = torch.from_numpy(targets).to(self.device)
        self._train_mode(self.concept_extractor, self._con_modules)
        logits = self.concept_extractor(evidence_t)
        loss = F.binary_cross_entropy_with_logits(logits, target_t)
        self.concept_opt.zero_grad()
        loss.backward()
        self.concept_opt.step()
        self.concept_extractor.eval()
        return float(loss.detach().cpu())

    def stats(self) -> dict[str, Any]:
        return {
            "observed": self.state.observed,
            "detector_updates": self.state.detector_updates,
            "concept_updates": self.state.concept_updates,
            "detector_buffer_size": self.state.detector_buffer_size,
            "concept_buffer_size": self.state.concept_buffer_size,
            "last_detector_loss": self.state.last_detector_loss,
            "last_concept_loss": self.state.last_concept_loss,
            "n_updates": len(self.state.history),
        }
