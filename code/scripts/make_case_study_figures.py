#!/usr/bin/env python3
from __future__ import annotations

import csv
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path("/u/ylin30/sigLA")
OUT_DIR = ROOT / "case_studies"
RUNS = ROOT / "code" / "runs"


COLORS = {
    "wait": "#3A6EA5",
    "alarm": "#F2C14E",
    "request_evidence": "#1B998B",
    "escalate": "#D95D39",
    "inspect": "#57A773",
    "suppress": "#7B8794",
    "recalibrate": "#8E44AD",
    "anomaly": "#F7D6CC",
    "detector": "#3A6EA5",
    "risk": "#D95D39",
    "dark": "#1F2D3D",
    "gray": "#5F6C72",
}


ACTION_ORDER = ["wait", "alarm", "request_evidence", "escalate", "inspect", "suppress", "recalibrate"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C2CC")
    ax.spines["bottom"].set_color("#B8C2CC")
    ax.tick_params(colors=COLORS["dark"], labelsize=10)


def shade_positive_windows(ax: plt.Axes, x: np.ndarray, labels: np.ndarray) -> None:
    positive = np.where(labels == 1)[0]
    if len(positive) == 0:
        return
    for _, group in groupby(enumerate(positive), key=lambda item: item[1] - item[0]):
        idx = np.array([item[1] for item in group], dtype=int)
        left = x[idx[0]]
        right = x[idx[-1]]
        ax.axvspan(left, right, color=COLORS["anomaly"], alpha=0.55, linewidth=0)


def action_y_values(actions: np.ndarray) -> np.ndarray:
    mapping = {name: idx for idx, name in enumerate(ACTION_ORDER)}
    return np.asarray([mapping.get(item, -1) for item in actions], dtype=float)


def action_legend(actions: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[action], markersize=9, label=action)
        for action in actions
    ]


def choose_smd_case(det_rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
    starts = np.asarray([int(row["start"]) for row in det_rows])
    ends = np.asarray([int(row["end"]) for row in det_rows])
    labels = np.asarray([int(row["label"]) for row in det_rows])
    scores = np.asarray([float(row["score"]) for row in det_rows])

    positive_idx = np.where(labels == 1)[0]
    groups: list[np.ndarray] = []
    for _, group in groupby(enumerate(positive_idx), key=lambda item: item[1] - item[0]):
        groups.append(np.asarray([item[1] for item in group], dtype=int))
    if not groups:
        raise RuntimeError("No positive SMD windows were found.")

    best_positive_idx = int(positive_idx[np.argmax(scores[positive_idx])])
    best_group = next(group for group in groups if best_positive_idx in group)
    left = max(0, int(best_group[0]) - 25)
    right = min(len(det_rows) - 1, int(best_group[-1]) + 25)
    return left, right, int(starts[best_group[0]]), int(ends[best_group[-1]])


def draw_smd_detector_policy_case() -> None:
    det_rows = load_csv(RUNS / "detector_SMD_1-7_test_w50_s5" / "autoencoder_window_scores_test.csv")
    pol_rows = load_csv(RUNS / "policy_SMD_1-7_test_w50_s5" / "policy_predictions_test.csv")
    left, right, event_start, event_end = choose_smd_case(det_rows)
    sl = slice(left, right + 1)

    starts = np.asarray([int(row["start"]) for row in det_rows])[sl]
    ends = np.asarray([int(row["end"]) for row in det_rows])[sl]
    labels = np.asarray([int(row["label"]) for row in det_rows])[sl]
    scores = np.asarray([float(row["score"]) for row in det_rows])[sl]
    risk_prob = np.asarray([float(row["risk_prob"]) for row in pol_rows])[sl]
    action_pred = np.asarray([row["action_pred_name"] for row in pol_rows])[sl]
    action_true = np.asarray([row["action_true_name"] for row in pol_rows])[sl]
    action_conf = np.asarray([float(row["action_confidence"]) for row in pol_rows])[sl]

    x = ends
    score_y = np.log10(scores + 1.0)

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0, 1.0, 0.7]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Case Study 1: SMD Trained Detector + Policy", x=0.02, y=0.985, ha="left", fontsize=23, weight="bold", color=COLORS["dark"])
    fig.text(
        0.02,
        0.94,
        f"Selected anomaly episode: window-label event around time {event_start}-{event_end}. Shaded region = windows containing dataset anomaly labels.",
        fontsize=12,
        color=COLORS["gray"],
    )

    ax = axes[0]
    shade_positive_windows(ax, x, labels)
    ax.plot(x, score_y, color=COLORS["detector"], linewidth=2.2, marker="o", markersize=4, label="Detector score")
    ax.set_ylabel("log10(score+1)")
    ax.set_title("Detector: reconstruction score spikes inside the abnormal episode", fontsize=13, weight="bold", color=COLORS["dark"])
    ax.legend(loc="upper right", frameon=False)
    style(ax)

    ax = axes[1]
    shade_positive_windows(ax, x, labels)
    ax.plot(x, risk_prob, color=COLORS["risk"], linewidth=2.2, marker="o", markersize=4, label="Policy risk probability")
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.3, label="risk=0.5")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("risk prob")
    ax.set_title("Policy risk: high risk through the anomaly region", fontsize=13, weight="bold", color=COLORS["dark"])
    ax.legend(loc="lower right", frameon=False)
    style(ax)

    ax = axes[2]
    shade_positive_windows(ax, x, labels)
    y_pred = action_y_values(action_pred)
    for action in ACTION_ORDER:
        idx = np.where(action_pred == action)[0]
        if len(idx):
            ax.scatter(x[idx], y_pred[idx], s=95, color=COLORS[action], marker="s", edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(ACTION_ORDER)))
    ax.set_yticklabels(ACTION_ORDER)
    ax.set_title("Policy prediction: alarm before the event, escalate during event", fontsize=13, weight="bold", color=COLORS["dark"])
    ax.set_ylabel("pred action")
    style(ax)

    ax = axes[3]
    shade_positive_windows(ax, x, labels)
    y_true = action_y_values(action_true)
    ax.scatter(x, y_true, s=72, color="#2D3748", marker="|", linewidth=2.8, label="weak target action")
    ax.set_yticks(range(len(ACTION_ORDER)))
    ax.set_yticklabels(ACTION_ORDER)
    ax.set_xlabel("time index / window end")
    ax.set_ylabel("target action")
    ax.set_title("Weak target action and policy confidence", fontsize=13, weight="bold", color=COLORS["dark"])
    style(ax)
    ax2 = ax.twinx()
    ax2.plot(x, action_conf, color="#8AA0B2", linewidth=1.8, label="policy confidence")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_ylabel("confidence", color=COLORS["gray"])
    ax2.tick_params(colors=COLORS["gray"], labelsize=10)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, loc="upper right", frameon=False, ncol=2)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.08, hspace=0.38)
    save(fig, "01_smd_detector_policy_case.png")


def draw_gpt_agent_case() -> None:
    rows = load_csv(RUNS / "fallback_pipeline_gpt_test" / "predictions_100.csv")
    x = np.asarray([int(row["end"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    detector_score = np.asarray([float(row["detector_score"]) for row in rows])
    risk_state = np.asarray([float(row["risk_state"]) for row in rows])
    policy_action = np.asarray([row["policy_action"] for row in rows])
    final_action = np.asarray([row["final_action"] for row in rows])
    confidence = np.asarray([float(row["confidence"]) for row in rows])
    primary = np.asarray([row["primary_concept"] or "none" for row in rows])
    changed = policy_action != final_action

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.15, 0.75]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Case Study 2: GPT Agent on Fallback Pipeline", x=0.02, y=0.985, ha="left", fontsize=23, weight="bold", color=COLORS["dark"])
    fig.text(
        0.02,
        0.94,
        "Existing 100-window GPT run. Shaded region = windows containing positive labels. GPT followed the policy candidate action in this run.",
        fontsize=12,
        color=COLORS["gray"],
    )

    ax = axes[0]
    shade_positive_windows(ax, x, labels)
    ax.plot(x, detector_score, color=COLORS["detector"], linewidth=2.2, marker="o", markersize=4)
    ax.set_ylabel("fallback score")
    ax.set_title("Fallback detector score", fontsize=13, weight="bold", color=COLORS["dark"])
    style(ax)

    ax = axes[1]
    shade_positive_windows(ax, x, labels)
    ax.plot(x, risk_state, color=COLORS["risk"], linewidth=2.2, marker="o", markersize=4, label="risk state")
    ax.plot(x, confidence, color=COLORS["teal"] if "teal" in COLORS else "#1B998B", linewidth=1.7, alpha=0.8, label="GPT confidence")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("state / conf")
    ax.set_title("Agent context: smoothed risk state and GPT confidence", fontsize=13, weight="bold", color=COLORS["dark"])
    ax.legend(loc="lower right", frameon=False)
    style(ax)

    ax = axes[2]
    shade_positive_windows(ax, x, labels)
    y_policy = action_y_values(policy_action)
    y_final = action_y_values(final_action) + 0.22
    for action in ACTION_ORDER:
        idx_policy = np.where(policy_action == action)[0]
        if len(idx_policy):
            ax.scatter(x[idx_policy], y_policy[idx_policy], s=78, color=COLORS[action], marker="s", alpha=0.55, edgecolor="white", linewidth=0.5)
        idx_final = np.where(final_action == action)[0]
        if len(idx_final):
            ax.scatter(x[idx_final], y_final[idx_final], s=38, color=COLORS[action], marker="o", edgecolor="black", linewidth=0.45)
    if np.any(changed):
        ax.scatter(x[changed], y_final[changed], s=130, facecolors="none", edgecolors="black", linewidth=1.8, label="GPT changed action")
    ax.set_yticks(range(len(ACTION_ORDER)))
    ax.set_yticklabels(ACTION_ORDER)
    ax.set_ylabel("action")
    ax.set_title("Policy candidate vs GPT final action: no action overrides in this record", fontsize=13, weight="bold", color=COLORS["dark"])
    legend_items = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#8AA0B2", markersize=9, label="policy candidate"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#8AA0B2", markersize=7, label="GPT final action"),
        Patch(facecolor=COLORS["anomaly"], edgecolor="none", alpha=0.55, label="positive label window"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False, ncol=3)
    style(ax)

    ax = axes[3]
    shade_positive_windows(ax, x, labels)
    concept_to_y = {name: idx for idx, name in enumerate(sorted(set(primary)))}
    y = np.asarray([concept_to_y[name] for name in primary])
    ax.scatter(x, y, s=48, color="#5F6C72")
    ax.set_yticks(list(concept_to_y.values()))
    ax.set_yticklabels(list(concept_to_y.keys()))
    ax.set_xlabel("time index / window end in selected fallback segment")
    ax.set_title("Primary concept seen by the agent", fontsize=13, weight="bold", color=COLORS["dark"])
    style(ax)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.08, hspace=0.40)
    save(fig, "02_gpt_agent_fallback_case.png")


def main() -> None:
    draw_smd_detector_policy_case()
    draw_gpt_agent_case()


if __name__ == "__main__":
    main()
