"""Long-tail open-vocabulary anomaly benchmark.

This module extends the six-concept ``ovbench`` setting into a larger
parameterized concept space.  The generated taxonomy contains ordinary
single-family anomalies and many compositional anomalies, such as
``trend + variance_burst`` or ``level_shift + correlation_break``.

The goal is not to replace the existing six-concept benchmark.  It is a stress
test for the paper claim that open-vocabulary anomaly learning becomes more
valuable when the anomaly space is long-tailed and only partially known.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

import sigla_exp.ovbench as OV


WIN, NVARS = OV.WIN, OV.NVARS

FAMILIES = (
    "spike",
    "level_shift",
    "oscillation",
    "variance_burst",
    "trend",
    "correlation_break",
)
LOCS = ("early", "mid", "late")
SCOPES = ("local", "broad")
SEVERITIES = ("mild", "strong")
STATS = OV.STATS


def _slug(parts: list[str]) -> str:
    return "__".join(p.replace("+", "_") for p in parts)


def generate_taxonomy(k: int, seed: int = 0) -> list[dict[str, Any]]:
    """Return the first ``k`` deterministic long-tail anomaly specifications."""
    if k < 1:
        raise ValueError("k must be positive")
    specs: list[dict[str, Any]] = []

    # Put the six ordinary concepts first, so K=6 is comparable to the original
    # mechanism benchmark.
    for fam in FAMILIES:
        specs.append(
            {
                "name": fam,
                "components": (fam,),
                "loc": "mid",
                "scope": "broad",
                "severity": "strong",
                "freq": "medium",
                "sign": "pos",
            }
        )

    # Then add ordinary variants, followed by compositional anomalies.  This
    # order makes K=20 already include long-tail combinations, while K=100 is
    # dominated by compositional types.
    for fam in FAMILIES:
        for loc in LOCS:
            for scope in SCOPES:
                if loc == "mid" and scope == "broad":
                    continue
                specs.append(
                    {
                        "name": _slug([fam, loc, scope]),
                        "components": (fam,),
                        "loc": loc,
                        "scope": scope,
                        "severity": "mild" if loc == "early" else "strong",
                        "freq": "high" if loc == "late" else "medium",
                        "sign": "neg" if scope == "local" and loc == "late" else "pos",
                    }
                )

    for a, b in combinations(FAMILIES, 2):
        for loc in LOCS:
            for scope in SCOPES:
                for severity in SEVERITIES:
                    specs.append(
                        {
                            "name": _slug([a, b, loc, scope, severity]),
                            "components": (a, b),
                            "loc": loc,
                            "scope": scope,
                            "severity": severity,
                            "freq": "high" if "oscillation" in (a, b) and loc != "early" else "medium",
                            "sign": "neg" if loc == "late" else "pos",
                        }
                    )

    # If a caller asks for more than the base construction, add deterministic
    # triple-composition variants.  K=100 does not need this, but it keeps the
    # generator extensible.
    if k > len(specs):
        rng = np.random.default_rng(seed)
        triples = list(combinations(FAMILIES, 3))
        i = 0
        while len(specs) < k:
            comps = triples[i % len(triples)]
            loc = LOCS[(i // len(triples)) % len(LOCS)]
            scope = SCOPES[(i // (len(triples) * len(LOCS))) % len(SCOPES)]
            severity = SEVERITIES[int(rng.integers(0, len(SEVERITIES)))]
            specs.append(
                {
                    "name": _slug([*comps, loc, scope, severity, str(i)]),
                    "components": comps,
                    "loc": loc,
                    "scope": scope,
                    "severity": severity,
                    "freq": "high",
                    "sign": "neg" if i % 2 else "pos",
                }
            )
            i += 1

    return specs[:k]


def spec_by_name(specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(s["name"]): s for s in specs}


def component_family(name: str) -> str:
    if name not in FAMILIES:
        raise KeyError(name)
    return name


def _dims(rng: np.random.Generator, scope: str) -> np.ndarray:
    n = 3 if scope == "local" else 8
    n = min(n, NVARS)
    return rng.choice(NVARS, n, replace=False)


def _loc_bounds(loc: str, width: int) -> tuple[int, int]:
    centers = {"early": int(0.25 * WIN), "mid": int(0.50 * WIN), "late": int(0.72 * WIN)}
    c = centers[loc]
    s = max(2, c - width // 2)
    e = min(WIN - 2, s + width)
    s = max(2, e - width)
    return s, e


def _amp(spec: dict[str, Any], mild: float, strong: float) -> float:
    return mild if spec["severity"] == "mild" else strong


def _sign(spec: dict[str, Any]) -> float:
    return -1.0 if spec["sign"] == "neg" else 1.0


def _inject_spike(x: np.ndarray, rng: np.random.Generator, spec: dict[str, Any], strength: float) -> None:
    s, e = _loc_bounds(str(spec["loc"]), 10)
    loc = int(rng.integers(s, e))
    amp = strength * _amp(spec, 4.0, 6.0) * _sign(spec)
    for d in _dims(rng, str(spec["scope"])):
        x[loc, d] += amp * float(rng.uniform(0.85, 1.15))


def _inject_level_shift(x: np.ndarray, rng: np.random.Generator, spec: dict[str, Any], strength: float) -> None:
    s, _ = _loc_bounds(str(spec["loc"]), 8)
    amp = strength * _amp(spec, 2.0, 3.5) * _sign(spec)
    for d in _dims(rng, str(spec["scope"])):
        x[s:, d] += amp


def _inject_oscillation(x: np.ndarray, rng: np.random.Generator, spec: dict[str, Any], strength: float) -> None:
    freq = 9 if spec["freq"] == "high" else 6
    amp = strength * _amp(spec, 1.1, 1.8)
    t = np.arange(WIN, dtype=np.float32)
    phase = float(rng.uniform(0, 2 * np.pi))
    wave = amp * np.sin(2 * np.pi * freq * t / WIN + phase)
    s, e = _loc_bounds(str(spec["loc"]), 70)
    env = np.zeros(WIN, dtype=np.float32)
    env[s:e] = 1.0
    if s > 0:
        ramp = min(8, s)
        env[s - ramp : s] = np.linspace(0, 1, ramp, endpoint=False)
    if e < WIN:
        ramp = min(8, WIN - e)
        env[e : e + ramp] = np.linspace(1, 0, ramp, endpoint=False)
    for d in _dims(rng, str(spec["scope"])):
        x[:, d] += (wave * env).astype(np.float32)


def _inject_variance_burst(x: np.ndarray, rng: np.random.Generator, spec: dict[str, Any], strength: float) -> None:
    width = 18 if spec["severity"] == "mild" else 28
    s, e = _loc_bounds(str(spec["loc"]), width)
    amp = strength * _amp(spec, 0.9, 1.4)
    for d in _dims(rng, str(spec["scope"])):
        x[s:e, d] += rng.normal(0, amp, e - s).astype(np.float32)


def _inject_trend(x: np.ndarray, rng: np.random.Generator, spec: dict[str, Any], strength: float) -> None:
    s, _ = _loc_bounds(str(spec["loc"]), 12)
    amp = strength * _amp(spec, 2.2, 3.8) * _sign(spec)
    ramp = np.zeros(WIN, dtype=np.float32)
    ramp[s:] = np.linspace(0, amp, WIN - s)
    for d in _dims(rng, str(spec["scope"])):
        x[:, d] += ramp


def _inject_correlation_break(x: np.ndarray, rng: np.random.Generator, spec: dict[str, Any], strength: float) -> None:
    width = 35 if spec["severity"] == "mild" else 55
    s, e = _loc_bounds(str(spec["loc"]), width)
    seg_len = e - s
    dims = _dims(rng, "broad")
    mix = float(np.clip(strength * _amp(spec, 0.55, 0.85), 0.0, 1.0))
    for d in dims:
        lag = int(rng.integers(max(2, seg_len // 5), max(3, 4 * seg_len // 5)))
        shifted = np.roll(x[s:e, d], lag)
        x[s:e, d] = ((1 - mix) * x[s:e, d] + mix * shifted).astype(np.float32)


INJECTORS = {
    "spike": _inject_spike,
    "level_shift": _inject_level_shift,
    "oscillation": _inject_oscillation,
    "variance_burst": _inject_variance_burst,
    "trend": _inject_trend,
    "correlation_break": _inject_correlation_break,
}


def make_window(spec: dict[str, Any] | None, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    x = OV.base_normal(rng).astype(np.float32)
    if spec is None:
        return x
    for comp in spec["components"]:
        INJECTORS[component_family(str(comp))](x, rng, spec, strength)
    return x.astype(np.float32)


def _step_location_and_sign(x: np.ndarray, w: int = 10) -> tuple[float, float]:
    best = 0.0
    best_t = WIN // 2
    best_sign = 0.0
    for d in range(NVARS):
        y = x[:, d]
        for t in range(w, WIN - w + 1):
            diff = float(np.median(y[t : t + w]) - np.median(y[t - w : t]))
            if abs(diff) > abs(best):
                best = diff
                best_t = t
                best_sign = np.sign(diff)
    return best_t / WIN, float(best_sign)


def _segment_location(values: np.ndarray) -> float:
    idx = int(np.argmax(values))
    return (idx + 0.5) / len(values)


def _variance_location(x: np.ndarray, nseg: int = 5) -> float:
    seg_scores = []
    for seg in np.array_split(x, nseg, axis=0):
        seg_scores.append(float(np.mean(np.var(seg, axis=0))))
    return _segment_location(np.asarray(seg_scores))


def _spike_location(x: np.ndarray) -> float:
    y = np.abs(x - np.median(x, axis=0, keepdims=True))
    idx = int(np.argmax(np.max(y, axis=1)))
    return idx / max(1, WIN - 1)


def _corr_location(x: np.ndarray, w: int = 25) -> float:
    starts = list(range(0, WIN - w + 1, max(1, w // 2)))
    vals = []
    for s in starts:
        seg = x[s : s + w]
        m = seg.std(0) > 1e-6
        if int(m.sum()) < 2:
            vals.append(1.0)
            continue
        c = np.corrcoef(seg[:, m], rowvar=False)
        n = c.shape[0]
        vals.append(float((np.abs(c).sum() - n) / (n * (n - 1))))
    return (starts[int(np.argmin(vals))] + w / 2) / WIN


def _spectral_freq(x: np.ndarray) -> float:
    det = x - x.mean(0, keepdims=True)
    mag = np.abs(np.fft.rfft(det, axis=0))[1:]
    if mag.size == 0:
        return 0.0
    idx = np.unravel_index(int(np.argmax(mag)), mag.shape)[0] + 1
    return idx / (WIN / 2)


def _scope_estimate(x: np.ndarray) -> float:
    peak = np.max(np.abs(x - np.median(x, axis=0, keepdims=True)), axis=0)
    return float(np.mean(peak > np.quantile(peak, 0.60)))


def _slope_sign(x: np.ndarray) -> float:
    t = np.arange(WIN, dtype=np.float32)
    y = x.mean(axis=1)
    slope = float(np.polyfit(t, y, 1)[0])
    return float(np.sign(slope))


def normal_stats(rng: np.random.Generator, n: int = 400) -> tuple[dict[str, float], dict[str, float]]:
    return OV.normal_stats(rng, n=n)


def features(x: np.ndarray, mu: dict[str, float], sd: dict[str, float]) -> np.ndarray:
    """Feature vector used by the long-tail runner.

    The first six coordinates are normalized signature evidence values.  The
    remaining coordinates are coarse localization/scope/sign descriptors that
    help separate parameterized variants and compositional anomalies.
    """
    ev = OV.evidence(x)
    z = np.asarray([(ev[s] - mu[s]) / (sd[s] + 1e-9) for s in STATS], dtype=np.float32)
    z = np.clip(z, -2.0, 10.0)
    step_loc, step_sign = _step_location_and_sign(x)
    extra = np.asarray(
        [
            _spike_location(x),
            step_loc,
            _variance_location(x),
            _corr_location(x),
            _spectral_freq(x),
            _scope_estimate(x),
            _slope_sign(x),
            step_sign,
        ],
        dtype=np.float32,
    )
    return np.concatenate([z, extra]).astype(np.float32)


def anomaly_score(feat: np.ndarray) -> float:
    return float(np.max(feat[: len(STATS)]))


def component_signature(feat: np.ndarray, top: int = 2, threshold: float = 2.0) -> tuple[str, ...]:
    z = feat[: len(STATS)]
    idx = np.argsort(-z)
    comps: list[str] = []
    stat_to_family = {OV.STAT_OF[c]: c for c in FAMILIES}
    for i in idx[:top]:
        if float(z[i]) < threshold:
            continue
        comps.append(stat_to_family[STATS[int(i)]])
    return tuple(comps)


def _bucket_loc(value: float) -> str:
    if value < 0.38:
        return "early"
    if value < 0.64:
        return "mid"
    return "late"


def memory_signature(feat: np.ndarray, threshold: float = 1.6) -> tuple[str, str, str, str]:
    """Infer a hierarchical subtype key from one raw feature vector.

    The key is deliberately coarse: component family set, location bucket, scope
    bucket, and severity bucket.  It is meant for memory indexing and merge/split
    control, not as an oracle label.
    """
    comps = component_signature(feat, top=2, threshold=threshold)
    if not comps:
        return ("normal", "none", "none", "none")

    loc_values = []
    for comp in comps:
        if comp == "spike":
            loc_values.append(float(feat[len(STATS) + 0]))
        elif comp == "level_shift":
            loc_values.append(float(feat[len(STATS) + 1]))
        elif comp == "variance_burst":
            loc_values.append(float(feat[len(STATS) + 2]))
        elif comp == "correlation_break":
            loc_values.append(float(feat[len(STATS) + 3]))
        else:
            loc_values.append(0.50)
    loc = _bucket_loc(float(np.mean(loc_values)))

    scope_value = float(feat[len(STATS) + 5])
    scope = "broad" if scope_value >= 0.38 else "local"

    max_z = float(np.max(feat[: len(STATS)]))
    severity = "strong" if max_z >= 4.5 else "mild"

    return ("+".join(sorted(comps)), loc, scope, severity)


def spec_signature(spec: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        "+".join(sorted(str(c) for c in spec["components"])),
        str(spec["loc"]),
        str(spec["scope"]),
        str(spec["severity"]),
    )
