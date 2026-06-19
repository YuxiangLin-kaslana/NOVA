#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path("/u/ylin30/sigLA")
OUT_DIR = ROOT / "slide_figures"
RUNS = ROOT / "code" / "runs"


COLORS = {
    "blue": "#3A6EA5",
    "teal": "#1B998B",
    "green": "#57A773",
    "yellow": "#F2C14E",
    "red": "#D95D39",
    "gray": "#5F6C72",
    "light": "#F7F9FB",
    "dark": "#1F2D3D",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C2CC")
    ax.spines["bottom"].set_color("#B8C2CC")
    ax.tick_params(colors=COLORS["dark"], labelsize=10)


def figure(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(14, 7.875))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    fig.text(0.04, 0.94, title, fontsize=24, weight="bold", color=COLORS["dark"])
    if subtitle:
        fig.text(0.04, 0.895, subtitle, fontsize=13, color=COLORS["gray"])
    return fig, ax


def draw_framework() -> None:
    fig, ax = figure(
        "SigLA Framework",
        "Windowed monitoring pipeline: signal perception, concept profiling, policy decision, and agent refinement.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    nodes = [
        ("Multivariate\nTime Series", 0.06, COLORS["blue"]),
        ("Sliding\nWindows", 0.20, COLORS["teal"]),
        ("Anomaly\nDetector", 0.34, COLORS["green"]),
        ("Concept\nProfile", 0.48, COLORS["yellow"]),
        ("Action\nPolicy", 0.62, COLORS["red"]),
        ("Local / GPT\nAgent", 0.76, COLORS["blue"]),
        ("Final\nAction", 0.90, COLORS["dark"]),
    ]
    y = 0.48
    width = 0.105
    height = 0.18
    for label, x, color in nodes:
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color,
            alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=13, weight="bold", color="white")

    for idx in range(len(nodes) - 1):
        x1 = nodes[idx][1] + width / 2 + 0.012
        x2 = nodes[idx + 1][1] - width / 2 - 0.012
        arrow = FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color="#8AA0B2",
        )
        ax.add_patch(arrow)

    notes = [
        ("score", 0.34),
        ("concept evidence", 0.48),
        ("action + risk", 0.62),
        ("context + rationale", 0.76),
    ]
    for text, x in notes:
        ax.text(x, 0.26, text, ha="center", fontsize=11, color=COLORS["gray"])

    save(fig, "01_framework_structure.png")


def draw_training_setup(detector_config: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.875))
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.94, "Training Setup", fontsize=24, weight="bold", color=COLORS["dark"])
    fig.text(
        0.04,
        0.895,
        "All training and validation windows were derived from the SMD_1-7 test split.",
        fontsize=13,
        color=COLORS["gray"],
    )

    ax.set_position([0.08, 0.18, 0.84, 0.52])
    style_axis(ax)
    train_windows = detector_config["train_windows"]
    val_windows = detector_config["val_windows"]
    total = detector_config["n_windows"]
    ax.barh([0], [train_windows], color=COLORS["blue"], height=0.34, label="Train windows")
    ax.barh([0], [val_windows], left=[train_windows], color=COLORS["yellow"], height=0.34, label="Validation windows")
    ax.set_xlim(0, total * 1.08)
    ax.set_yticks([])
    ax.set_xlabel("Windows", fontsize=12, color=COLORS["dark"])
    ax.legend(loc="upper right", frameon=False)
    ax.text(train_windows / 2, 0, f"Train\n{train_windows:,}", ha="center", va="center", fontsize=13, color="white", weight="bold")
    ax.text(
        train_windows + val_windows / 2,
        0,
        f"Val\n{val_windows:,}",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["dark"],
        weight="bold",
    )

    info = [
        ("Dataset", detector_config["dataset"]),
        ("Source split", detector_config["train_source"]),
        ("Points", f"{detector_config['source_points']:,}"),
        ("Variables", str(detector_config["n_vars"])),
        ("Window / step", f"{detector_config['win_size']} / {detector_config['step']}"),
        ("Total windows", f"{total:,}"),
    ]
    x_positions = np.linspace(0.08, 0.86, len(info))
    for x, (label, value) in zip(x_positions, info):
        fig.text(x, 0.78, label, fontsize=10, color=COLORS["gray"])
        fig.text(x, 0.745, value, fontsize=15, weight="bold", color=COLORS["dark"])

    save(fig, "02_training_setup.png")


def draw_detector(detector_metrics: dict, detector_eval: dict) -> None:
    history = detector_metrics["detector"]["history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.875), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.78, wspace=0.22)
    fig.text(0.04, 0.94, "Detector Results", fontsize=24, weight="bold", color=COLORS["dark"])
    fig.text(0.04, 0.895, "Reconstruction detector outputs a continuous anomaly score; labels are used only for ranking evaluation.", fontsize=13, color=COLORS["gray"])

    ax = axes[0]
    style_axis(ax)
    ax.plot(epochs, train_loss, marker="o", color=COLORS["blue"], linewidth=2.5, label="Train loss")
    ax.plot(epochs, val_loss, marker="o", color=COLORS["red"], linewidth=2.5, label="Val loss")
    ax.set_title("Training Curve", fontsize=15, weight="bold", color=COLORS["dark"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(frameon=False)

    ax = axes[1]
    style_axis(ax)
    metrics = {
        "Window\nROC-AUC": detector_eval["window_metrics"]["roc_auc"],
        "Window\nAP": detector_eval["window_metrics"]["average_precision"],
        "Point\nROC-AUC": detector_eval["point_metrics"]["roc_auc"],
        "Point\nAP": detector_eval["point_metrics"]["average_precision"],
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [COLORS["teal"], COLORS["green"], COLORS["blue"], COLORS["yellow"]]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ranking Metrics", fontsize=15, weight="bold", color=COLORS["dark"])
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=10, color=COLORS["dark"])
    ax.text(
        0.02,
        -0.22,
        "No detector threshold is applied in the framework; downstream policy/agent consume the score.",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["gray"],
    )

    save(fig, "03_detector_results.png")


def draw_policy(policy_metrics: dict, policy_eval: dict) -> None:
    history = policy_metrics["policy"]["history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.875), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.78, wspace=0.22)
    fig.text(0.04, 0.94, "Policy Results", fontsize=24, weight="bold", color=COLORS["dark"])
    fig.text(0.04, 0.895, "Action policy learned weak labels well under the test-derived training setup.", fontsize=13, color=COLORS["gray"])

    ax = axes[0]
    style_axis(ax)
    ax.plot(epochs, train_loss, marker="o", color=COLORS["blue"], linewidth=2.5, label="Train loss")
    ax.plot(epochs, val_loss, marker="o", color=COLORS["red"], linewidth=2.5, label="Val loss")
    ax.set_title("Training Curve", fontsize=15, weight="bold", color=COLORS["dark"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Policy loss")
    ax2 = ax.twinx()
    ax2.plot(epochs, val_acc, marker="s", color=COLORS["green"], linewidth=2.3, label="Val accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Validation accuracy")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="center right")

    ax = axes[1]
    style_axis(ax)
    metrics = {
        "Action\nAccuracy": policy_eval["action_metrics"]["accuracy"],
        "Macro F1\nPresent": policy_eval["action_metrics"]["macro_f1_present_actions"],
        "Risk\nPrecision": policy_eval["risk_metrics"]["precision"],
        "Risk\nRecall": policy_eval["risk_metrics"]["recall"],
        "Risk\nF1": policy_eval["risk_metrics"]["f1"],
        "Argument\nAccuracy": policy_eval["arg_accuracy"],
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["yellow"], COLORS["red"], COLORS["gray"]]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Evaluation Metrics", fontsize=15, weight="bold", color=COLORS["dark"])
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=10, color=COLORS["dark"])

    save(fig, "04_policy_results.png")


def draw_agent(agent_summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.875), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.78, wspace=0.20)
    fig.text(0.04, 0.94, "GPT Agent Fallback Test", fontsize=24, weight="bold", color=COLORS["dark"])
    fig.text(0.04, 0.895, "100 fallback-pipeline windows with GPT agent decisions.", fontsize=13, color=COLORS["gray"])

    ax = axes[0]
    style_axis(ax)
    actions = list(agent_summary["action_counts"].keys())
    counts = list(agent_summary["action_counts"].values())
    colors = [COLORS["blue"], COLORS["red"], COLORS["yellow"]]
    ax.bar(actions, counts, color=colors[: len(actions)])
    ax.set_title("Agent Action Distribution", fontsize=15, weight="bold", color=COLORS["dark"])
    ax.set_ylabel("Window count")
    for idx, value in enumerate(counts):
        ax.text(idx, value + 1.5, str(value), ha="center", fontsize=12, weight="bold", color=COLORS["dark"])
    ax.text(
        0.02,
        -0.18,
        "GPT decisions: 100 / 100, local fallback: 0",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["gray"],
    )

    ax = axes[1]
    style_axis(ax)
    risk = agent_summary["risk_action_metrics"]
    metrics = {
        "Precision": risk["precision"],
        "Recall": risk["recall"],
        "F1": risk["f1"],
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    ax.bar(labels, values, color=[COLORS["green"], COLORS["teal"], COLORS["red"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("Risk-Action Metrics", fontsize=15, weight="bold", color=COLORS["dark"])
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=12, weight="bold", color=COLORS["dark"])
    ax.text(
        0.02,
        -0.18,
        f"Positive windows: {risk['positives']} / {risk['count']}; predicted risk: {risk['predicted_positives']} / {risk['count']}",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["gray"],
    )

    save(fig, "05_agent_results.png")


def main() -> None:
    detector_config = load_json(RUNS / "detector_SMD_1-7_test_w50_s5" / "config.json")
    detector_metrics = load_json(RUNS / "detector_SMD_1-7_test_w50_s5" / "metrics.json")
    detector_eval = load_json(RUNS / "detector_SMD_1-7_test_w50_s5" / "autoencoder_eval_test.json")
    policy_metrics = load_json(RUNS / "policy_SMD_1-7_test_w50_s5" / "metrics.json")
    policy_eval = load_json(RUNS / "policy_SMD_1-7_test_w50_s5" / "policy_eval_test.json")
    agent_summary = load_json(RUNS / "fallback_pipeline_gpt_test" / "summary_100.json")

    draw_framework()
    draw_training_setup(detector_config)
    draw_detector(detector_metrics, detector_eval)
    draw_policy(policy_metrics, policy_eval)
    draw_agent(agent_summary)


if __name__ == "__main__":
    main()
