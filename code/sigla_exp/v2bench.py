"""Lightweight V2 anomaly-taxonomy benchmark.

This module mirrors the six V2 concepts from ``time_rcd_concept_pipeline_v2.py``
without importing the full pipeline.  It is meant for fast, controlled paper
experiments: every concept has an interpretable evidence statistic that can be
used for open-vocabulary gating and naming.
"""
from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from typing import Optional

import numpy as np

WIN, NVARS = 128, 8

NORMAL = "normal"
CONCEPTS = [
    "spike_burst",
    "level_shift",
    "seasonal_break",
    "contextual_deviation",
    "shapelet",
    "correlation_break",
]

DEFS = {
    "spike_burst": "short-lived extreme amplitude point or burst",
    "level_shift": "persistent mean, level, or slope change over a substantial suffix or segment",
    "seasonal_break": "disruption of repeated periodic pattern, phase, frequency, or seasonal energy",
    "contextual_deviation": "values or short segments abnormal relative to local/phase/contextual expectation",
    "shapelet": "localized morphology or motif discord after primitive explanations are removed",
    "correlation_break": "multivariate dependency or pairwise relation break across channels",
}

STAT_OF = {
    "spike_burst": "burst_pointiness",
    "level_shift": "level_jump",
    "seasonal_break": "spectral_shift",
    "contextual_deviation": "contextual_residual",
    "shapelet": "shape_morphology",
    "correlation_break": "decorrelation",
}
STATS = list(STAT_OF.values())
CONCEPT_OF_STAT = {v: k for k, v in STAT_OF.items()}

STAT_MEANING = {
    "burst_pointiness": "isolated or very short extreme amplitude bursts",
    "level_jump": "persistent median/level discontinuity across adjacent blocks",
    "spectral_shift": "frequency, phase, or periodic-energy mismatch between window halves",
    "contextual_residual": "phase/local-neighbor residual that is abnormal but not a global spike",
    "shape_morphology": "localized z-normalized motif/curvature discord",
    "decorrelation": "loss of cross-channel synchronization in a local segment",
}

_GPT_CACHE: dict[tuple[str, tuple[tuple[str, float], ...]], Optional[str]] = {}


def _mad(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    if axis is None:
        return np.asarray(mad).squeeze()
    if keepdims:
        return mad
    return np.squeeze(mad, axis=axis)


def _z_norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return (y - np.nanmedian(y)) / (1.4826 * _mad(y) + 1e-6)


def _safe_quantile(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def _dims(rng: np.random.Generator, k: int) -> np.ndarray:
    return rng.choice(NVARS, size=min(k, NVARS), replace=False)


def base_normal(rng: np.random.Generator) -> np.ndarray:
    """Shared seasonal factors create a correlated multivariate normal window."""
    t = np.linspace(0.0, 1.0, WIN, endpoint=False)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    factors = np.stack(
        [
            np.sin(2.0 * np.pi * 2.0 * t + phase[0]),
            0.65 * np.sin(2.0 * np.pi * 5.0 * t + phase[1]),
            np.cos(2.0 * np.pi * 1.0 * t + phase[2]),
        ],
        axis=1,
    ).astype(np.float32)
    weights = rng.normal(0.0, 1.0, size=(3, NVARS)).astype(np.float32)
    x = factors @ weights
    x += rng.normal(0.0, 0.08, size=(WIN, NVARS)).astype(np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    return x.astype(np.float32)


def inject_spike_burst(x: np.ndarray, rng: np.random.Generator) -> None:
    variant = str(rng.choice(["single", "multi", "short_burst", "smooth_pulse"]))
    chs = _dims(rng, 3)
    if variant == "single":
        for d in chs:
            t = int(rng.integers(4, WIN - 4))
            x[t, d] += rng.choice([-1.0, 1.0]) * rng.uniform(7.0, 11.0)
    elif variant == "multi":
        for d in chs:
            start = int(rng.integers(4, WIN - 16))
            pts = rng.choice(np.arange(start, start + 12), size=4, replace=False)
            x[pts, d] += rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 8.5, size=len(pts))
    elif variant == "short_burst":
        dur = int(rng.integers(5, 11))
        start = int(rng.integers(4, WIN - dur - 4))
        pulse = np.hanning(dur).astype(np.float32)
        for d in chs:
            x[start : start + dur, d] += rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 8.0) * pulse
    else:
        dur = int(rng.integers(9, 16))
        start = int(rng.integers(4, WIN - dur - 4))
        tt = np.linspace(-1.0, 1.0, dur)
        pulse = np.exp(-5.0 * tt**2).astype(np.float32)
        for d in chs:
            x[start : start + dur, d] += rng.choice([-1.0, 1.0]) * rng.uniform(4.5, 7.0) * pulse


def inject_level_shift(x: np.ndarray, rng: np.random.Generator) -> None:
    start = int(rng.integers(WIN // 4, 2 * WIN // 3))
    variant = str(rng.choice(["abrupt", "slow_drift", "permanent"]))
    for d in _dims(rng, 4):
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(2.4, 4.6)
        if variant == "slow_drift":
            dur = int(rng.integers(20, 44))
            end = min(WIN, start + dur)
            x[start:end, d] += np.linspace(0.0, amp, end - start)
            if end < WIN:
                x[end:, d] += amp
        else:
            x[start:, d] += amp


def inject_seasonal_break(x: np.ndarray, rng: np.random.Generator) -> None:
    start = int(rng.integers(WIN // 3, WIN // 2))
    dur = WIN - start
    tt = np.arange(dur)
    variant = str(rng.choice(["frequency_drift", "phase_shift", "periodic_loss", "energy_change"]))
    for d in _dims(rng, 4):
        amp = rng.uniform(2.0, 3.4)
        period = rng.uniform(8.0, 20.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        envelope = np.linspace(0.25, 1.0, dur)
        if variant == "frequency_drift":
            drift = np.linspace(0.65, 1.90, dur)
            seasonal = amp * np.sin(2.0 * np.pi * tt * drift / period + phase) * envelope
            x[start:, d] += seasonal
        elif variant == "phase_shift":
            seasonal = amp * np.sin(2.0 * np.pi * tt / period + phase + rng.uniform(np.pi / 2.2, np.pi)) * envelope
            x[start:, d] += seasonal
        elif variant == "periodic_loss":
            centered = x[start:, d] - float(np.mean(x[start:, d]))
            x[start:, d] = float(np.mean(x[start:, d])) + centered * rng.uniform(0.08, 0.35)
        else:
            seasonal = amp * np.sin(2.0 * np.pi * tt / period + phase) * envelope
            x[start:, d] += seasonal


def inject_contextual_deviation(x: np.ndarray, rng: np.random.Generator) -> None:
    dur = int(rng.integers(22, 44))
    start = int(rng.integers(6, WIN - dur - 6))
    chs = _dims(rng, 2)
    variant = str(rng.choice(["phase_context_offset", "neighbor_residual", "range_context_swap"]))
    for ch in chs:
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 5.2)
        if variant == "phase_context_offset":
            mask = (np.arange(start, start + dur) % 8) == int(rng.integers(0, 8))
            seg = x[start : start + dur, ch].copy()
            seg[mask] += amp
            x[start : start + dur, ch] = seg
        elif variant == "neighbor_residual":
            tt = np.linspace(0.0, 1.0, dur)
            patch = np.sin(2.0 * np.pi * rng.uniform(1.0, 2.0) * tt + rng.uniform(0.0, 2.0 * np.pi))
            patch += 0.35 * np.sign(np.sin(2.0 * np.pi * 4.0 * tt))
            patch = patch - patch.mean()
            x[start : start + dur, ch] += 0.95 * amp * patch / (patch.std() + 1e-6)
        else:
            src = int(rng.integers(0, WIN - dur))
            repl = _z_norm(x[src : src + dur, ch]) * max(0.35, 0.75 * x[start : start + dur, ch].std())
            mask = (np.arange(dur) % 3) == int(rng.integers(0, 3))
            seg = x[start : start + dur, ch].copy()
            seg[mask] = np.median(seg) + repl[mask]
            x[start : start + dur, ch] = seg


def inject_shapelet(x: np.ndarray, rng: np.random.Generator) -> None:
    dur = 32
    start = int(rng.integers(8, WIN - dur - 8))
    variant = str(rng.choice(["double_peak", "motif_swap"]))
    tt = np.linspace(0.0, 1.0, dur)
    for ch in _dims(rng, 4):
        seg = x[start : start + dur, ch].copy()
        center = float(np.median(seg))
        scale = float(max(1.4826 * _mad(seg), 0.90))
        if variant == "double_peak":
            repl = np.exp(-0.5 * ((tt - 0.30) / 0.07) ** 2)
            repl += np.exp(-0.5 * ((tt - 0.70) / 0.07) ** 2)
            repl -= 1.45 * np.exp(-0.5 * ((tt - 0.50) / 0.08) ** 2)
            repl = center + _z_norm(repl) * scale * rng.uniform(2.10, 2.80)
        else:
            repl = np.sin(2.0 * np.pi * (1.0 * tt + 0.05)) - 0.75 * np.sin(2.0 * np.pi * (2.4 * tt + 0.30))
            repl += 0.55 * np.sign(np.sin(2.0 * np.pi * 3.0 * tt))
            repl = center + _z_norm(repl) * scale * rng.uniform(2.00, 2.55)
        edge = np.hanning(dur)
        alpha = np.clip(0.35 + 0.62 * edge, 0.0, 0.97)
        x[start : start + dur, ch] = (1.0 - alpha) * seg + alpha * repl


def inject_correlation_break(x: np.ndarray, rng: np.random.Generator) -> None:
    start = int(rng.integers(WIN // 6, WIN // 3))
    dur = int(rng.integers(WIN // 2, 2 * WIN // 3))
    end = min(WIN, start + dur)
    seg_len = end - start
    tt = np.arange(seg_len)
    ramp = np.clip(np.minimum(tt, (seg_len - 1) - tt) / 8.0, 0.0, 1.0).astype(np.float32)
    for d in _dims(rng, NVARS):
        lag = int(rng.integers(seg_len // 5, 4 * seg_len // 5))
        shifted = np.roll(x[start:end, d], lag)
        if rng.random() < 0.35:
            shifted = -shifted
        x[start:end, d] = ramp * shifted + (1.0 - ramp) * x[start:end, d]


INJECTORS = {
    "spike_burst": inject_spike_burst,
    "level_shift": inject_level_shift,
    "seasonal_break": inject_seasonal_break,
    "contextual_deviation": inject_contextual_deviation,
    "shapelet": inject_shapelet,
    "correlation_break": inject_correlation_break,
}


def make_window(concept: Optional[str], rng: np.random.Generator) -> np.ndarray:
    x = base_normal(rng)
    if concept is not None:
        INJECTORS[concept](x, rng)
    return x.astype(np.float32)


def make_window_strength(concept: Optional[str], rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    x = base_normal(rng)
    if concept is None or strength <= 0.0:
        return x.astype(np.float32)
    xf = x.copy()
    INJECTORS[concept](xf, rng)
    return (x + strength * (xf - x)).astype(np.float32)


def _local_step(y: np.ndarray, width: int = 28) -> float:
    best = 0.0
    for t in range(width, WIN - width + 1, 4):
        left = np.median(y[t - width : t])
        right = np.median(y[t : t + width])
        suffix = y[t:]
        prefix = y[:t]
        persist = abs(float(np.median(suffix) - np.median(prefix))) if len(suffix) >= width else 0.0
        best = max(best, 0.45 * abs(float(right - left)) + 0.55 * persist)
    half = abs(float(np.median(y[WIN // 2 :]) - np.median(y[: WIN // 2])))
    return max(best, half)


def _spectral_shift(y: np.ndarray) -> float:
    mid = WIN // 2
    left = y[:mid] - y[:mid].mean()
    right = y[mid:] - y[mid:].mean()
    p1 = np.abs(np.fft.rfft(left))[1:]
    p2 = np.abs(np.fft.rfft(right))[1:]
    n = min(len(p1), len(p2))
    if n <= 1:
        return 0.0
    q1 = p1[:n] / (p1[:n].sum() + 1e-6)
    q2 = p2[:n] / (p2[:n].sum() + 1e-6)
    peak_shift = abs(float(np.argmax(q2) - np.argmax(q1))) / n
    return float(0.65 * np.sum(np.abs(q2 - q1)) + 0.35 * peak_shift)


def _contextual_residual_score(y: np.ndarray) -> float:
    width = 13
    smooth = np.convolve(y, np.ones(width) / width, mode="same")
    resid = y - smooth
    global_scale = 1.4826 * _mad(y) + 1e-6
    rz = np.abs(resid / global_scale)
    phase_resid = np.zeros_like(y)
    for phase in range(8):
        idx = np.arange(WIN) % 8 == phase
        if idx.any():
            phase_resid[idx] = y[idx] - np.median(y[idx])
    pz = np.abs(phase_resid / global_scale)
    energies = []
    for start in range(0, WIN - 20 + 1, 4):
        energies.append(float(np.mean(rz[start : start + 20] ** 2)))
    locality = max(energies) / (np.median(energies) + 1e-6) if energies else 0.0
    return float(1.20 * _safe_quantile(rz, 0.98) + 1.00 * _safe_quantile(pz, 0.98) + 0.22 * locality)


def _shape_morphology_score(y: np.ndarray) -> float:
    y = _z_norm(y)
    curv = np.abs(np.diff(y, n=2))
    if curv.size == 0:
        return 0.0
    local = []
    width = 24
    for start in range(0, WIN - width + 1, 4):
        seg = y[start : start + width]
        cseg = np.abs(np.diff(_z_norm(seg), n=2))
        dseg = np.abs(np.diff(_z_norm(seg)))
        local.append(float(cseg.mean() + 0.40 * cseg.max() + 0.25 * dseg.max()))
    local_score = max(local) if local else float(curv.mean())
    global_score = float(_safe_quantile(curv, 0.98) + 0.30 * curv.max())
    return float(max(local_score, global_score))


def _decorrelation(x: np.ndarray, width: int = 40) -> float:
    best_corr = 1.0
    for start in range(0, WIN - width + 1, width // 3):
        seg = x[start : start + width]
        active = seg.std(axis=0) > 1e-6
        if int(active.sum()) < 2:
            continue
        corr = np.corrcoef(seg[:, active], rowvar=False)
        n = corr.shape[0]
        mean_abs = float((np.abs(corr).sum() - n) / (n * (n - 1)))
        best_corr = min(best_corr, mean_abs)
    return float(1.0 - best_corr)


def _shapelet_coherence(x: np.ndarray, width: int = 32) -> float:
    t = np.linspace(0.0, 1.0, width)
    double_peak = np.exp(-0.5 * ((t - 0.30) / 0.07) ** 2)
    double_peak += np.exp(-0.5 * ((t - 0.70) / 0.07) ** 2)
    double_peak -= 1.45 * np.exp(-0.5 * ((t - 0.50) / 0.08) ** 2)
    motif = np.sin(2.0 * np.pi * (1.0 * t + 0.05)) - 0.75 * np.sin(2.0 * np.pi * (2.4 * t + 0.30))
    motif += 0.55 * np.sign(np.sin(2.0 * np.pi * 3.0 * t))
    templates = [_z_norm(double_peak), _z_norm(motif)]
    best = 0.0
    for start in range(0, WIN - width + 1, 4):
        scores = []
        for d in range(x.shape[1]):
            seg = _z_norm(x[start : start + width, d])
            denom = np.linalg.norm(seg) + 1e-6
            scores.append(max(float(np.dot(seg, temp) / (denom * (np.linalg.norm(temp) + 1e-6))) for temp in templates))
        best = max(best, float(np.mean(np.sort(scores)[-4:])))
    return best


def evidence(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    z = (x - np.median(x, axis=0, keepdims=True)) / (1.4826 * _mad(x, axis=0, keepdims=True) + 1e-6)
    abs_z = np.abs(z)
    high_mask = abs_z > 3.0
    duration_ratio = float(high_mask.any(axis=1).mean())
    burst_raw = float(abs_z.max() / (_safe_quantile(abs_z, 0.95) + 1e-6))
    burst_pointiness = burst_raw * max(0.20, 1.0 - 2.4 * duration_ratio)
    level_raw = float(max(_local_step(x[:, d]) for d in range(x.shape[1])))
    spectral_shift = float(max(_spectral_shift(x[:, d]) for d in range(x.shape[1])))
    context_scores = np.array([_contextual_residual_score(x[:, d]) for d in range(x.shape[1])], dtype=float)
    shape_scores = np.array([_shape_morphology_score(x[:, d]) for d in range(x.shape[1])], dtype=float)
    context_raw = float(np.mean(np.sort(context_scores)[-2:]))
    shape_raw = float(np.mean(np.sort(shape_scores)[-4:]) + 80.0 * max(0.0, _shapelet_coherence(x) - 0.76))
    decorr = _decorrelation(x)
    burst_excess = max(0.0, burst_pointiness - 2.0)
    level_jump = level_raw / (1.0 + 0.35 * max(0.0, spectral_shift - 0.45))
    contextual_residual = context_raw / (
        1.0
        + 1.20 * burst_excess
        + 0.16 * max(0.0, level_raw - 2.8)
        + 0.35 * max(0.0, spectral_shift - 0.48)
        + 0.24 * max(0.0, shape_raw - 5.5)
    )
    shape_morphology = shape_raw / (
        1.0
        + 1.65 * burst_excess
        + 0.18 * max(0.0, level_raw - 2.8)
        + 0.75 * max(0.0, spectral_shift - 0.45)
    )
    out = {
        "burst_pointiness": burst_pointiness,
        "level_jump": level_jump,
        "spectral_shift": spectral_shift,
        "contextual_residual": contextual_residual,
        "shape_morphology": shape_morphology,
        "decorrelation": decorr,
    }
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in out.items()}


def normal_stats(rng: np.random.Generator, n: int = 300) -> tuple[dict[str, float], dict[str, float]]:
    evs = [evidence(make_window(None, rng)) for _ in range(n)]
    mu = {k: float(np.mean([ev[k] for ev in evs])) for k in STATS}
    sd = {k: float(np.std([ev[k] for ev in evs]) + 1e-6) for k in STATS}
    return mu, sd


def z_scores(ev: dict[str, float], mu: dict[str, float], sd: dict[str, float]) -> dict[str, float]:
    return {k: float((ev[k] - mu[k]) / (sd[k] + 1e-9)) for k in STATS}


def rule_namer(ev: dict[str, float], mu: dict[str, float], sd: dict[str, float], threshold: float = 2.0) -> Optional[str]:
    z = z_scores(ev, mu, sd)
    if z["burst_pointiness"] >= max(threshold, 5.0):
        return "spike_burst"
    if z["level_jump"] >= max(threshold, 4.0):
        return "level_shift"
    if z["contextual_residual"] >= max(threshold, 4.0):
        return "contextual_deviation"
    if z["shape_morphology"] >= threshold:
        return "shapelet"
    if z["spectral_shift"] >= 1.35 and z["level_jump"] < 3.0:
        return "seasonal_break"
    if z["contextual_residual"] >= threshold:
        return "contextual_deviation"
    if z["decorrelation"] >= threshold:
        return "correlation_break"
    dom = max(z, key=z.get)
    if z[dom] < threshold:
        return None
    return CONCEPT_OF_STAT[dom]


def signature_separation(rng: np.random.Generator, n: int = 100) -> dict[str, object]:
    mu, sd = normal_stats(rng, n=max(40, n))
    rows = {}
    for concept in CONCEPTS:
        vals = [z_scores(evidence(make_window(concept, rng)), mu, sd) for _ in range(n)]
        mean_z = {stat: float(np.mean([v[stat] for v in vals])) for stat in STATS}
        dom = max(mean_z, key=mean_z.get)
        rows[concept] = {
            "signature": STAT_OF[concept],
            "dominant": dom,
            "hit": dom == STAT_OF[concept],
            "mean_z": mean_z,
        }
    return {"normal_mu": mu, "normal_sd": sd, "rows": rows}


def gpt_recognize_top1(
    ev: dict[str, float],
    key: str,
    mu: dict[str, float],
    sd: dict[str, float],
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    z = {k: round(v, 1) for k, v in z_scores(ev, mu, sd).items()}
    cache_key = (model, tuple(sorted(z.items())))
    if cache_key in _GPT_CACHE:
        return _GPT_CACHE[cache_key]
    instr = (
        "You identify the SINGLE most likely time-series anomaly concept in a multivariate window. "
        "You are given per-statistic z-scores relative to normal; large positive means elevated. "
        "Taxonomy:\n"
        + "\n".join(f"- {k}: {v}" for k, v in DEFS.items())
        + "\nStatistic meanings:\n"
        + "\n".join(f"- {k}: {v}" for k, v in STAT_MEANING.items())
        + "\nProcedure: choose the concept whose signature statistic is most strongly elevated. "
        "If no statistic has z above about 2, answer null. Respond only as JSON "
        "{\"concept\":\"<taxonomy-name-or-null>\"}."
    )
    payload = {
        "model": model,
        "instructions": instr,
        "input": [{"role": "user", "content": "z-scores: " + json.dumps(z)}],
        "max_output_tokens": 80,
    }
    for _ in range(2):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/responses",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode())
            text = data.get("output_text")
            if not isinstance(text, str):
                text = "\n".join(c.get("text", "") for item in data.get("output", []) for c in item.get("content", []))
            start, end = text.find("{"), text.rfind("}")
            concept = json.loads(text[start : end + 1]).get("concept")
            result = concept if concept in CONCEPTS else None
            _GPT_CACHE[cache_key] = result
            return result
        except Exception:
            continue
    return "__ERROR__"


@lru_cache(maxsize=1)
def cached_normal_stats() -> tuple[dict[str, float], dict[str, float]]:
    return normal_stats(np.random.default_rng(2028), n=400)
