from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CONCEPT_NAMES = (
    "spike",
    "level_shift",
    "seasonal_break",
    "contextual_deviation",
    "correlation_break",
)


def _mad(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    out = np.median(np.abs(x - med), axis=axis, keepdims=keepdims)
    return out


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0, keepdims=True)
    scale = 1.4826 * _mad(x, axis=0, keepdims=True) + 1e-6
    return np.abs((x - med) / scale)


def _safe_corrcoef(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=0)) + 1e-6
    return (centered.T @ centered) / (norm[:, None] * norm[None, :])


def extract_raw_evidence(window: np.ndarray) -> np.ndarray:
    """Extract the five coarse SigLA concept evidences from one window.

    The implementation is intentionally lightweight. It follows the paper
    structure by treating statistical masks as evidence features, not final
    explanations.
    """
    x = np.asarray(window, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    length, n_var = x.shape

    z = _robust_z(x)
    amp = float(np.max(z))

    if length > 1:
        dx = np.diff(x, axis=0)
        diff = float(np.max(_robust_z(dx)))
    else:
        diff = 0.0
    spike = float(_sigmoid(0.7 * max(amp, diff) - 2.5))

    mid = max(1, length // 2)
    left = x[:mid]
    right = x[mid:] if mid < length else x[mid - 1 :]
    global_scale = 1.4826 * float(np.median(_mad(x, axis=0))) + 1e-6
    level_delta = float(np.max(np.abs(np.median(right, axis=0) - np.median(left, axis=0))) / global_scale)
    level_shift = float(_sigmoid(level_delta - 1.5))

    detrended = x - np.mean(x, axis=0, keepdims=True)
    if length >= 4:
        fft_mag = np.abs(np.fft.rfft(detrended, axis=0))[1:]
        energy = np.sum(fft_mag, axis=0) + 1e-6
        prob = fft_mag / energy
        entropy = -np.sum(prob * np.log(prob + 1e-8), axis=0) / np.log(max(2, fft_mag.shape[0]))
        high_freq = np.sum(fft_mag[fft_mag.shape[0] // 2 :], axis=0) / energy
        seasonal_raw = float(np.mean(0.5 * entropy + 0.5 * high_freq))
    else:
        seasonal_raw = 0.0
    seasonal_break = float(_sigmoid(4.0 * seasonal_raw - 2.0))

    if length > 3:
        history = x[:-1]
        last = x[-1:]
        q05 = np.quantile(history, 0.05, axis=0)
        q50 = np.quantile(history, 0.50, axis=0)
        q95 = np.quantile(history, 0.95, axis=0)
        spread = q95 - q05 + 1e-6
        context = float(np.max(np.abs(last - q50) / spread))
    else:
        context = amp
    contextual_deviation = float(_sigmoid(context - 1.5))

    if n_var > 1 and length >= 6:
        std_left = np.std(left, axis=0)
        std_right = np.std(right, axis=0)
        valid = (std_left > 1e-6) & (std_right > 1e-6)
        if int(np.sum(valid)) >= 2:
            first = _safe_corrcoef(left[:, valid])
            second = _safe_corrcoef(right[:, valid])
            corr_shift = float(np.mean(np.abs(second - first)))
        else:
            corr_shift = 0.0
    else:
        corr_shift = 0.0
    correlation_break = float(_sigmoid(5.0 * corr_shift - 1.5))

    return np.asarray(
        [spike, level_shift, seasonal_break, contextual_deviation, correlation_break],
        dtype=np.float32,
    )


@dataclass
class ConceptProfileExtractor:
    median: np.ndarray
    mad: np.ndarray

    @classmethod
    def fit(
        cls,
        series: np.ndarray,
        win_size: int,
        step: int,
        max_windows: int = 2048,
    ) -> "ConceptProfileExtractor":
        starts = np.arange(0, max(1, len(series) - win_size + 1), step)
        if len(starts) > max_windows:
            starts = np.linspace(0, len(starts) - 1, max_windows).astype(int)
            starts = np.arange(0, max(1, len(series) - win_size + 1), step)[starts]
        raw = [extract_raw_evidence(series[start : start + win_size]) for start in starts]
        values = np.stack(raw, axis=0)
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0) + 1e-4
        return cls(median=median.astype(np.float32), mad=mad.astype(np.float32))

    def transform(self, window: np.ndarray) -> np.ndarray:
        raw = extract_raw_evidence(window)
        calibrated = (raw - self.median) / self.mad
        return _sigmoid(calibrated).astype(np.float32)

    def transform_many(self, windows: Iterable[np.ndarray]) -> np.ndarray:
        return np.stack([self.transform(window) for window in windows], axis=0)
