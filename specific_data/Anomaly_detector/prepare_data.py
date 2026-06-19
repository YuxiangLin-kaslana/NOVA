#!/usr/bin/env python3
"""数据准备自检脚本。

不训练模型，只验证数据能正确加载、切窗、组成 batch，并打印关键统计。
直接运行：

    python prepare_data.py
    python prepare_data.py --win_size 100 --stride 10
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from data import build_smd_datasets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMD 异常检测数据准备自检")
    p.add_argument("--root", default=None, help="SMD 根目录（默认用 data.py 中的常量）")
    p.add_argument("--win_size", type=int, default=100)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kw = {"win_size": args.win_size, "stride": args.stride}
    if args.root:
        kw["root"] = args.root

    bundle = build_smd_datasets(**kw)

    print(f"机器数量      : {len(bundle.machines)}")
    print(f"特征维度 n_vars: {bundle.n_vars}")
    print(f"窗口长度      : {bundle.win_size}   步长: {bundle.stride}")
    print("-" * 48)

    n_train = len(bundle.train)
    n_test = len(bundle.test)
    test_anom = bundle.test.anomaly_window_count()
    print(f"train 窗口数  : {n_train:>8d}  (全部视为正常)")
    print(f"test  窗口数  : {n_test:>8d}")
    print(f"test  异常窗口: {test_anom:>8d}  ({test_anom / max(n_test, 1):.2%})")
    print("-" * 48)

    # 取一个 batch 验证形状
    loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    print("一个 train batch:")
    print(f"  signal : {tuple(batch['signal'].shape)}  dtype={batch['signal'].dtype}")
    print(f"  label  : {tuple(batch['label'].shape)}   值域={batch['label'].unique().tolist()}")
    print(f"  series : {tuple(batch['series'].shape)}")
    print(f"  signal 值域: [{batch['signal'].min():.3f}, {batch['signal'].max():.3f}]")

    test_loader = DataLoader(bundle.test, batch_size=args.batch_size, shuffle=False)
    tbatch = next(iter(test_loader))
    print("一个 test batch:")
    print(f"  signal : {tuple(tbatch['signal'].shape)}")
    print(f"  label  : 异常数={int(tbatch['label'].sum())}/{len(tbatch['label'])}")
    print("\n数据准备 OK。")


if __name__ == "__main__":
    main()
