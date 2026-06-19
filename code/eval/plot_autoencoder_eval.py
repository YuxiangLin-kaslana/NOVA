#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.data import load_dataset


COLORS = {
    "score": "#2563eb",
    "truth": "#111827",
    "grid": "#e5e7eb",
    "text": "#111827",
    "muted": "#6b7280",
    "band_truth": "#fee2e2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trained autoencoder eval curves as SVG.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset", default="SMD_1-7")
    parser.add_argument("--data_dir", default="/u/ylin30/sigLA/data")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--full_output", type=Path, default=None)
    parser.add_argument("--zoom_output", type=Path, default=None)
    parser.add_argument("--max_points", type=int, default=1400)
    parser.add_argument("--zoom_radius", type=int, default=900)
    return parser.parse_args()


def read_window_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts: list[int] = []
    ends: list[int] = []
    scores: list[float] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            starts.append(int(row["start"]))
            ends.append(int(row["end"]))
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def point_scores_from_windows(length: int, starts: np.ndarray, ends: np.ndarray, scores: np.ndarray) -> np.ndarray:
    point_scores = np.full(length, -np.inf, dtype=np.float64)
    for start, end, score in zip(starts, ends, scores):
        point_scores[int(start) : int(end) + 1] = np.maximum(point_scores[int(start) : int(end) + 1], score)
    point_scores[~np.isfinite(point_scores)] = float(np.min(scores))
    return point_scores


def downsample_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(np.int64)
    return x[idx], y[idx]


def compress_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    for idx, value in enumerate(mask.astype(bool)):
        if value and not in_region:
            start = idx
            in_region = True
        elif not value and in_region:
            regions.append((start, idx - 1))
            in_region = False
    if in_region:
        regions.append((start, len(mask) - 1))
    return regions


def path_line(x: np.ndarray, y: np.ndarray, x0: int, x1: int, y_min: float, y_max: float, box: tuple[int, int, int, int]) -> str:
    left, top, width, height = box
    x_range = max(1, x1 - x0)
    y_range = max(1e-12, y_max - y_min)
    pts = []
    for xi, yi in zip(x, y):
        px = left + width * (float(xi) - x0) / x_range
        py = top + height * (1.0 - (float(yi) - y_min) / y_range)
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def rect_for_region(region: tuple[int, int], x0: int, x1: int, box: tuple[int, int, int, int], fill: str, opacity: float) -> str:
    left, top, width, height = box
    start, end = region
    x_range = max(1, x1 - x0)
    rx = left + width * (start - x0) / x_range
    rw = width * max(1, end - start + 1) / x_range
    return f'<rect x="{rx:.2f}" y="{top}" width="{rw:.2f}" height="{height}" fill="{fill}" opacity="{opacity}"/>'


def tick_lines(x0: int, x1: int, box: tuple[int, int, int, int], n: int = 6) -> str:
    left, top, width, height = box
    pieces = []
    for value in np.linspace(x0, x1, n).astype(int):
        px = left + width * (value - x0) / max(1, x1 - x0)
        pieces.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + height}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        pieces.append(f'<text x="{px:.2f}" y="{top + height + 24}" text-anchor="middle" font-size="12" fill="{COLORS["muted"]}">{value}</text>')
    return "\n".join(pieces)


def label_track(truth: np.ndarray, x0: int, x1: int, box: tuple[int, int, int, int]) -> str:
    left, top, width, height = box
    x = np.arange(x0, x1 + 1)
    truth_slice = truth[x0 : x1 + 1]
    truth_y = np.where(truth_slice > 0, top + 42, top + 92)
    tx, ty = downsample_series(x, truth_y, 1800)
    truth_pts = path_line(tx, ty, x0, x1, float(top), float(top + height), (left, top, width, height))
    return "\n".join(
        [
            f'<text x="{left}" y="{top + 24}" font-size="13" fill="{COLORS["truth"]}">Dataset anomaly label</text>',
            f'<polyline points="{truth_pts}" fill="none" stroke="{COLORS["truth"]}" stroke-width="2"/>',
        ]
    )


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="42" font-size="24" font-weight="700" fill="{COLORS["text"]}">{html.escape(title)}</text>',
        f'<text x="40" y="68" font-size="14" fill="{COLORS["muted"]}">{html.escape(subtitle)}</text>',
    ]


def write_full_svg(
    path: Path,
    scores: np.ndarray,
    truth: np.ndarray,
    max_points: int,
) -> None:
    width, height = 1500, 860
    x0, x1 = 0, len(scores) - 1
    score_box = (80, 110, 1360, 420)
    track_box = (80, 610, 1360, 150)
    x = np.arange(len(scores))
    plot_y = np.log10(scores + 1.0)
    y_min = 0.0
    y_max = float(np.percentile(plot_y, 99.8)) * 1.08
    dx, dy = downsample_series(x, plot_y, max_points)
    score_pts = path_line(dx, dy, x0, x1, y_min, y_max, score_box)

    pieces = svg_header(
        width,
        height,
        "AutoEncoder Score Curves - Full Test Split",
        "Score is log10(reconstruction_error + 1). Red bands are dataset anomaly labels; no detector threshold is applied.",
    )
    pieces.append(tick_lines(x0, x1, score_box))
    for region in compress_regions(truth):
        pieces.append(rect_for_region(region, x0, x1, score_box, COLORS["band_truth"], 0.55))
    pieces.extend(
        [
            f'<rect x="{score_box[0]}" y="{score_box[1]}" width="{score_box[2]}" height="{score_box[3]}" fill="none" stroke="#d1d5db"/>',
            f'<polyline points="{score_pts}" fill="none" stroke="{COLORS["score"]}" stroke-width="1.6"/>',
            f'<text x="{score_box[0]}" y="{score_box[1] - 16}" font-size="14" fill="{COLORS["text"]}">AE anomaly score</text>',
            f'<text x="{score_box[0] + score_box[2] - 210}" y="{score_box[1] + 20}" font-size="13" fill="{COLORS["score"]}">blue: AE score</text>',
            tick_lines(x0, x1, track_box),
            f'<rect x="{track_box[0]}" y="{track_box[1]}" width="{track_box[2]}" height="{track_box[3]}" fill="none" stroke="#d1d5db"/>',
            label_track(truth, x0, x1, track_box),
            f'<text x="{track_box[0]}" y="{track_box[1] - 16}" font-size="14" fill="{COLORS["text"]}">Dataset labels for ranking evaluation</text>',
            f'<text x="{track_box[0] + track_box[2] / 2}" y="{height - 32}" font-size="13" text-anchor="middle" fill="{COLORS["muted"]}">Test time index</text>',
        ]
    )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_zoom_svg(
    path: Path,
    x_data: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    center: int,
    radius: int,
) -> None:
    width, height = 1500, 940
    x0 = max(0, center - radius)
    x1 = min(len(scores) - 1, center + radius)
    x = np.arange(x0, x1 + 1)
    score_box = (80, 110, 1360, 260)
    signal_box = (80, 455, 1360, 250)
    track_box = (80, 780, 1360, 120)
    local = x_data[x0 : x1 + 1]
    top_vars = np.argsort(np.max(np.abs(local), axis=0))[-3:][::-1]
    signal_min = float(np.percentile(local[:, top_vars], 1))
    signal_max = float(np.percentile(local[:, top_vars], 99))
    if signal_min == signal_max:
        signal_min -= 1.0
        signal_max += 1.0
    score_y = np.log10(scores[x0 : x1 + 1] + 1.0)
    score_max = float(np.max(score_y)) * 1.08
    score_pts = path_line(x, score_y, x0, x1, 0.0, score_max, score_box)

    pieces = svg_header(
        width,
        height,
        "AutoEncoder Score Curves - Zoom",
        f"Zoom centered at max AE score index {center}. Raw standardized signals show the three largest-amplitude variables in this window.",
    )
    for region in compress_regions(truth[x0 : x1 + 1]):
        shifted = (region[0] + x0, region[1] + x0)
        pieces.append(rect_for_region(shifted, x0, x1, score_box, COLORS["band_truth"], 0.55))
        pieces.append(rect_for_region(shifted, x0, x1, signal_box, COLORS["band_truth"], 0.45))
    pieces.extend(
        [
            tick_lines(x0, x1, score_box),
            f'<rect x="{score_box[0]}" y="{score_box[1]}" width="{score_box[2]}" height="{score_box[3]}" fill="none" stroke="#d1d5db"/>',
            f'<polyline points="{score_pts}" fill="none" stroke="{COLORS["score"]}" stroke-width="2"/>',
            f'<text x="{score_box[0]}" y="{score_box[1] - 16}" font-size="14" fill="{COLORS["text"]}">AE anomaly score</text>',
            tick_lines(x0, x1, signal_box),
            f'<rect x="{signal_box[0]}" y="{signal_box[1]}" width="{signal_box[2]}" height="{signal_box[3]}" fill="none" stroke="#d1d5db"/>',
            f'<text x="{signal_box[0]}" y="{signal_box[1] - 16}" font-size="14" fill="{COLORS["text"]}">Actual standardized signal curves</text>',
        ]
    )
    signal_colors = ["#111827", "#059669", "#7c3aed"]
    for color, var_idx in zip(signal_colors, top_vars):
        pts = path_line(x, local[:, var_idx], x0, x1, signal_min, signal_max, signal_box)
        pieces.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    legend_x = signal_box[0] + signal_box[2] - 260
    for idx, (color, var_idx) in enumerate(zip(signal_colors, top_vars)):
        y = signal_box[1] + 20 + idx * 22
        pieces.append(f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 28}" y2="{y - 4}" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<text x="{legend_x + 36}" y="{y}" font-size="13" fill="{COLORS["text"]}">variable {int(var_idx)}</text>')
    pieces.extend(
        [
            tick_lines(x0, x1, track_box),
            f'<rect x="{track_box[0]}" y="{track_box[1]}" width="{track_box[2]}" height="{track_box[3]}" fill="none" stroke="#d1d5db"/>',
            label_track(truth, x0, x1, track_box),
        ]
    )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    eval_path = run_dir / "autoencoder_eval_test.json"
    scores_path = run_dir / "autoencoder_window_scores_test.csv"
    full_output = args.full_output or run_dir / "autoencoder_eval_curves_test.svg"
    zoom_output = args.zoom_output or run_dir / "autoencoder_eval_zoom_test.svg"

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found: {eval_path}")
    starts, ends, scores, _ = read_window_scores(scores_path)
    bundle = load_dataset(args.dataset, args.data_dir, train_ratio=args.train_ratio, seed=args.seed)
    test = bundle.test
    point_scores = point_scores_from_windows(len(test.x), starts, ends, scores)
    truth = test.y.astype(np.int64)

    write_full_svg(full_output, point_scores, truth, args.max_points)
    center = int(np.argmax(point_scores))
    write_zoom_svg(zoom_output, test.x, point_scores, truth, center, args.zoom_radius)
    print(f"saved {full_output}")
    print(f"saved {zoom_output}")


if __name__ == "__main__":
    main()
