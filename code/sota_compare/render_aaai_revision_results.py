#!/usr/bin/env python3
"""Render the key AAAI-revision results from frozen experiment JSON files.

The script is intentionally read-only with respect to experiment outputs.  It
loads the four audited JSON artifacts and writes one publication-style PNG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAIR = (
    ROOT
    / "docs"
    / "fair_openvocab_ablation_2026-07-09"
    / "fair_openvocab_ablation_result.json"
)
DEFAULT_FEATURE = (
    ROOT
    / "docs"
    / "feature_leakage_online_2026-07-09"
    / "feature_leakage_online_result.json"
)
DEFAULT_TEP = (
    ROOT
    / "docs"
    / "tep_native_typed_loto_2026-07-09"
    / "tep_native_typed_loto_result.json"
)
DEFAULT_NAMING = (
    ROOT
    / "docs"
    / "strict_naming_level_audit_2026-07-09"
    / "strict_naming_level_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "aaai_revision_2026-07-09"
    / "assets"
    / "fig_revision_key_results.png"
)


COLORS = {
    "gray": "#8B929A",
    "charcoal": "#30343B",
    "teal": "#2F7473",
    "blue": "#4C78A8",
    "rust": "#A8674A",
    "gold": "#B38B3E",
    "light": "#E6E8EA",
    "grid": "#D7DADD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fair", type=Path, default=DEFAULT_FAIR)
    parser.add_argument("--feature", type=Path, default=DEFAULT_FEATURE)
    parser.add_argument("--tep", type=Path, default=DEFAULT_TEP)
    parser.add_argument("--naming", type=Path, default=DEFAULT_NAMING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    return array[np.isfinite(array)]


def mean_ci95(values: Iterable[Any]) -> tuple[float, float, int]:
    array = finite(values)
    if array.size == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, 0.0, 1
    half_width = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(array.size)
    return mean, half_width, int(array.size)


def method_values(
    rows: list[dict[str, Any]], method: str, metric: str
) -> np.ndarray:
    return finite(
        row.get(metric) for row in rows if row.get("method") == method
    )


def style_axis(ax: plt.Axes, *, percent: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["charcoal"])
    ax.spines["bottom"].set_color(COLORS["charcoal"])
    ax.tick_params(colors=COLORS["charcoal"], labelsize=8.5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def label_bars(
    ax: plt.Axes,
    bars: Iterable[Any],
    *,
    y_offset: float = 0.018,
    fontsize: float = 7.8,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLORS["charcoal"],
        )


def panel_shared_gate(ax: plt.Axes, fair: dict[str, Any]) -> None:
    rows = fair["rows"]
    detector_specs = [
        (
            "AT proxy",
            "anomaly_transformer_proxy_shared_namer_flag_only",
            "anomaly_transformer_proxy_shared_namer_flag_or_novelty_gate",
        ),
        (
            "MemStream\nproxy",
            "memstream_proxy_shared_namer_flag_only",
            "memstream_proxy_shared_namer_flag_or_novelty_gate",
        ),
        (
            "CNN",
            "cnn_shared_namer_flag_only",
            "cnn_shared_namer_flag_or_novelty_gate",
        ),
    ]
    metric = "novel_typed_accuracy_including_queries"
    flag_stats = [mean_ci95(method_values(rows, flag, metric)) for _, flag, _ in detector_specs]
    gate_stats = [mean_ci95(method_values(rows, gate, metric)) for _, _, gate in detector_specs]

    x = np.arange(len(detector_specs), dtype=float)
    width = 0.34
    flag_bars = ax.bar(
        x - width / 2,
        [item[0] for item in flag_stats],
        width,
        yerr=[item[1] for item in flag_stats],
        color=COLORS["gray"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="Detector flag",
    )
    gate_bars = ax.bar(
        x + width / 2,
        [item[0] for item in gate_stats],
        width,
        yerr=[item[1] for item in gate_stats],
        color=COLORS["teal"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="+ shared novelty gate",
    )
    label_bars(ax, flag_bars)
    label_bars(ax, gate_bars)
    ax.set_xticks(x, [item[0] for item in detector_specs])
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Novel typed accuracy", fontsize=9)
    ax.set_title(
        "(a) Shared gate lifts detector+namer baselines",
        loc="left",
        fontsize=10.0,
        fontweight="semibold",
        pad=8,
    )
    ax.text(
        0.01,
        0.98,
        "shared structured evidence/namer; 2 backgrounds x 5 seeds",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#626970",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.89),
        frameon=False,
        fontsize=7.8,
        ncol=2,
        columnspacing=1.0,
        handlelength=1.5,
    )
    style_axis(ax)


def panel_memory(ax: plt.Axes, fair: dict[str, Any]) -> None:
    rows = fair["rows"]
    method_specs = [
        ("No vocab.\ngrowth", "nova_no_growth"),
        ("No memory\nreuse", "nova_no_reuse"),
        ("Nearest\nprototype", "nearest_prototype"),
        ("Memory\nreference", "nova_memory_reference"),
    ]
    query_stats = [
        mean_ci95(method_values(rows, method, "namer_call_rate_post"))
        for _, method in method_specs
    ]
    reuse_stats = [
        mean_ci95(method_values(rows, method, "post_discovery_future_reuse_rate"))
        for _, method in method_specs
    ]

    x = np.arange(len(method_specs), dtype=float)
    width = 0.34
    query_bars = ax.bar(
        x - width / 2,
        [item[0] for item in query_stats],
        width,
        yerr=[item[1] for item in query_stats],
        color=COLORS["rust"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="Post-stream namer query rate",
    )
    reuse_bars = ax.bar(
        x + width / 2,
        [item[0] for item in reuse_stats],
        width,
        yerr=[item[1] for item in reuse_stats],
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="Future reuse rate",
    )
    label_bars(ax, query_bars, fontsize=7.3)
    label_bars(ax, reuse_bars, fontsize=7.3)
    ax.set_xticks(x, [item[0] for item in method_specs])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_title(
        "(b) Memory changes the query/reuse balance",
        loc="left",
        fontsize=10.0,
        fontweight="semibold",
        pad=8,
    )
    ax.text(
        0.01,
        0.98,
        "reuse conditional on correct discovery; mean +/- 95% CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#626970",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.89),
        frameon=False,
        fontsize=7.6,
        ncol=2,
        columnspacing=0.9,
        handlelength=1.4,
    )
    style_axis(ax)


def panel_feature_loo(ax: plt.Axes, feature: dict[str, Any]) -> None:
    direct = {
        row["target"]: row
        for row in feature["loo_effects"]
        if row.get("contrast") == "specialized-specialized_loo"
        and row.get("metric") == "future_reuse_accuracy"
    }
    target_specs = [
        ("Oscillation", "oscillation"),
        ("Variance burst", "variance_burst"),
        ("Trend", "trend"),
        ("Correlation break", "correlation_break"),
    ]
    missing = [key for _, key in target_specs if key not in direct]
    if missing:
        raise KeyError(f"Missing direct-feature LOO rows: {missing}")

    y = np.arange(len(target_specs), dtype=float)
    means = np.array([100.0 * direct[key]["mean"] for _, key in target_specs])
    lows = np.array([100.0 * direct[key]["ci95_low"] for _, key in target_specs])
    highs = np.array([100.0 * direct[key]["ci95_high"] for _, key in target_specs])
    significant = np.array(
        [direct[key].get("holm_adjusted_p", 1.0) < 0.05 for _, key in target_specs]
    )
    colors = [COLORS["rust"] if value else COLORS["gray"] for value in significant]

    ax.hlines(y, 0, means, color=colors, linewidth=2.0, alpha=0.85)
    for index, (mean, low, high, color) in enumerate(
        zip(means, lows, highs, colors)
    ):
        ax.errorbar(
            mean,
            y[index],
            xerr=np.array([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.0,
            capsize=3,
            elinewidth=1.0,
        )
        ax.text(
            min(mean + 2.6, 101.5),
            y[index],
            f"{mean:.1f}",
            ha="left",
            va="center",
            fontsize=7.8,
            color=COLORS["charcoal"],
        )
    ax.axvline(0, color=COLORS["charcoal"], linewidth=0.8)
    ax.set_yticks(y, [label for label, _ in target_specs])
    ax.set_ylim(len(target_specs) - 0.5, -0.8)
    ax.set_xlim(-1, 106)
    ax.set_xlabel("Drop in future reuse accuracy (percentage points)", fontsize=9)
    ax.set_title(
        "(c) Direct-feature removal exposes\nrepresentation dependence",
        loc="left",
        fontsize=10.0,
        fontweight="semibold",
        pad=8,
    )
    ax.text(
        0.01,
        0.98,
        "paired full - leave-one-feature-out effect; n=10 seeds",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#626970",
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["rust"],
            markeredgecolor="white",
            label="Holm-adjusted p < .05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["gray"],
            markeredgecolor="white",
            label="not significant",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["charcoal"])
    ax.tick_params(colors=COLORS["charcoal"], labelsize=8.5, axis="y", length=0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def panel_tep(ax: plt.Axes, tep: dict[str, Any]) -> None:
    per_fault = tep["per_fault"]
    metric_specs = [
        ("Novel\nrejection", "open_novel_rejection_recall"),
        ("One-query\nreuse", "one_query_novel_reuse_accuracy"),
        ("Batch naming\nsuccess", "batch_query_naming_success"),
        ("Batch\nreuse", "batch_novel_reuse_accuracy"),
    ]
    values = [
        finite(row[metric]["mean"] for row in per_fault) for _, metric in metric_specs
    ]
    means = [float(np.mean(items)) for items in values]
    x = np.arange(len(metric_specs), dtype=float)

    bars = ax.bar(
        x,
        means,
        width=0.58,
        color=[COLORS["gray"], COLORS["blue"], COLORS["gold"], COLORS["rust"]],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    rng = np.random.default_rng(11)
    for index, items in enumerate(values):
        jitter = rng.uniform(-0.16, 0.16, size=items.size)
        ax.scatter(
            np.full(items.size, x[index]) + jitter,
            items,
            s=12,
            facecolors="white",
            edgecolors=COLORS["charcoal"],
            linewidths=0.45,
            alpha=0.78,
            zorder=3,
        )
    label_bars(ax, bars, y_offset=0.022)
    ax.set_xticks(x, [label for label, _ in metric_specs])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Fault-level rate", fontsize=9)
    known = float(tep["macro"]["closed_known_accuracy"]["mean"])
    ax.set_title(
        "(d) Native TEP leave-one-type-out transfer\nremains unreliable",
        loc="left",
        fontsize=10.0,
        fontweight="semibold",
        pad=8,
    )
    ax.text(
        0.01,
        0.98,
        f"20 held-out faults x 10 seeds; closed known-class accuracy={known:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#626970",
    )
    style_axis(ax)


def naming_audit_line(naming: dict[str, Any]) -> str:
    results = {row["background"]: row for row in naming["results"]}
    order = ["synthetic", "SMD:1-1", "SMD:2-5"]
    missing = [background for background in order if background not in results]
    if missing:
        raise KeyError(f"Missing naming-audit backgrounds: {missing}")
    scores = ", ".join(
        f"{background} {results[background]['macro']['llm_correct']['mean']:.1%}"
        for background in order
    )
    raw = "available" if naming.get("raw_generations_available") else "unavailable"
    rerating = "possible" if naming.get("human_rerating_possible") else "not possible"
    return (
        f"Naming audit context (historical Level {naming['evaluation_level']}; "
        f"{naming.get('new_api_calls', 0)} new API calls): macro keyword-match proxy "
        f"{scores}. Raw generations {raw}; human rerating {rerating}."
    )


def render(
    fair: dict[str, Any],
    feature: dict[str, Any],
    tep: dict[str, Any],
    naming: dict[str, Any],
    output: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": COLORS["charcoal"],
            "axes.titlecolor": COLORS["charcoal"],
            "text.color": COLORS["charcoal"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2))
    panel_shared_gate(axes[0, 0], fair)
    panel_memory(axes[0, 1], fair)
    panel_feature_loo(axes[1, 0], feature)
    panel_tep(axes[1, 1], tep)

    fig.suptitle(
        "Controlled revision experiments: gate, memory, feature dependence, and native transfer",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=COLORS["charcoal"],
    )
    fig.text(
        0.07,
        0.027,
        naming_audit_line(naming),
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#565D64",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.915,
        bottom=0.12,
        wspace=0.31,
        hspace=0.50,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fair = load_json(args.fair)
    feature = load_json(args.feature)
    tep = load_json(args.tep)
    naming = load_json(args.naming)
    render(fair, feature, tep, naming, args.output, args.dpi)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
