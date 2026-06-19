#!/usr/bin/env python3
"""把某台 SMD 机器的 test 序列导出成在线训练用的流 .npz。

用于「加载离线 checkpoint -> 直接在线」这条路径：离线 detector / concept 都是
在 SMD 上训的，所以在线流也必须是 SMD 同分布(同机器最佳),否则分布不匹配、
检测失真。这里直接用该机器的 test_x / test_label。

x      float32 [T, 38]   该机器 test 序列(SMD 已 min-max 归一化到 [0,1])
y      int64   [T]       逐点异常标签 —— 仅评测用，不进训练
regime int64   [T]       时间四分段(SMD 无显式漂移，这里仅作分段评测)

用法:
    python make_smd_stream.py --machine machine-1-1 --out streams/smd_machine-1-1.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIGLA_ROOT = HERE.parent.parent


def _load_smd_data():
    path = SIGLA_ROOT / "specific_data" / "Anomaly_detector" / "data.py"
    spec = importlib.util.spec_from_file_location("smd_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smd_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 SMD 机器 test 序列为在线流")
    p.add_argument("--machine", default="machine-1-1")
    p.add_argument("--out", type=Path, default=HERE / "streams" / "smd_machine-1-1.npz")
    p.add_argument("--n_segments", type=int, default=4, help="时间分段数(仅分段评测用)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    smd = _load_smd_data()
    series = smd.load_machine(args.machine)
    x = np.ascontiguousarray(series.test_x, dtype=np.float32)
    y = np.ascontiguousarray(series.test_y, dtype=np.int64).reshape(-1)
    T = len(x)

    bounds = np.linspace(0, T, args.n_segments + 1, dtype=np.int64)
    regime = np.zeros(T, dtype=np.int64)
    for r in range(args.n_segments):
        regime[bounds[r]:bounds[r + 1]] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, y=y, regime=regime, drift_points=bounds[1:-1])
    meta = {"machine": args.machine, "length": int(T), "n_vars": int(x.shape[1]),
            "positive_points": int(y.sum()), "positive_rate": float(y.mean()),
            "n_segments": args.n_segments, "note": "SMD test stream; labels eval-only"}
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved -> {args.out}")
    print(f"machine={args.machine} shape={x.shape} positive={int(y.sum())} ({y.mean():.2%})")


if __name__ == "__main__":
    main()
