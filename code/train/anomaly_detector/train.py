#!/usr/bin/env python3
"""训练 SMD 异常检测器 (MLPAnomalyDetector)。

模型：sigla_exp.model.mlp.MLPAnomalyDetector
    自编码器结构，重构输入窗口，用重构 MSE 作为异常分数（无监督）。

数据：specific_data/Anomaly_detector/data.py 的 build_smd_datasets()
    合并全部 28 台机器，滑动窗口 [win_size, n_vars]，默认 win=100 / stride=10。

训练设定：
    * 只用 train 窗口（默认全部正常）做无监督重构训练，损失 = MSE(recon, signal)。
    * 从 train 切出一部分做验证集，监控重构损失、早停保留最优权重。
    * 在 test 窗口上算异常分数（窗口重构 MSE），与真实窗口标签比较，
      报告 AUROC / AP / best-F1 等排序指标（纯 numpy，无 sklearn 依赖）。

输出（写到 --output_dir/--run_name/）：
    checkpoint_best.pt   与 sigla_exp 约定兼容：{"args", "n_vars", "model", "component"}
    metrics.json         训练历史 + test 集指标
    config.json          本次运行的超参

用法见 scripts/train_anomaly_detector.sh。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- 路径设置：定位 sigLA/code (ROOT) 与 sigLA 根目录 (SIGLA_ROOT) --------------- #
ROOT = Path(__file__).resolve().parents[2]          # .../sigLA/code
SIGLA_ROOT = ROOT.parent                            # .../sigLA
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.model.mlp import MLPAnomalyDetector  # noqa: E402

# 数据准备模块在 specific_data/Anomaly_detector/data.py，用 importlib 按路径加载，
# 避免与 sigla_exp.data 等同名模块冲突。
_DATA_PATH = SIGLA_ROOT / "specific_data" / "Anomaly_detector" / "data.py"
_spec = importlib.util.spec_from_file_location("smd_data", _DATA_PATH)
smd_data = importlib.util.module_from_spec(_spec)
sys.modules["smd_data"] = smd_data  # dataclass 解析需要模块已注册到 sys.modules
_spec.loader.exec_module(smd_data)


# --------------------------------------------------------------------------- #
#  指标（纯 numpy）                                                            #
# --------------------------------------------------------------------------- #
def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """基于秩 (Mann-Whitney U) 的 AUROC，处理并列分数取平均秩。"""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # 平均秩处理并列
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0  # 1-based 平均秩
        i = j
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision (PR 曲线下面积，阶梯式)。"""
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # AP = sum over thresholds of (R_k - R_{k-1}) * P_k
    prev_recall = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_recall) * p
        prev_recall = r
    return float(ap)


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """扫描阈值，找使 F1 最大的工作点。"""
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return {"best_f1": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "threshold": float("nan")}
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.argmax(f1))
    return {
        "best_f1": float(f1[k]),
        "precision": float(precision[k]),
        "recall": float(recall[k]),
        "threshold": float(s[k]),
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int64)
    out: dict[str, Any] = {
        "count": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "score_min": float(np.min(scores)) if len(scores) else 0.0,
        "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "score_median": float(np.median(scores)) if len(scores) else 0.0,
        "score_max": float(np.max(scores)) if len(scores) else 0.0,
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc(y_true, scores)
        out["average_precision"] = average_precision(y_true, scores)
        out.update(best_f1(y_true, scores))
    return out


# --------------------------------------------------------------------------- #
#  训练 / 评估                                                                 #
# --------------------------------------------------------------------------- #
def mean_or_zero(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        signal = batch["signal"].to(device)
        optimizer.zero_grad()
        recon = model(signal)
        loss = F.mse_loss(recon, signal)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return mean_or_zero(losses)


@torch.no_grad()
def eval_recon_loss(model, loader, device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        signal = batch["signal"].to(device)
        losses.append(float(F.mse_loss(model(signal), signal).cpu()))
    return mean_or_zero(losses)


@torch.no_grad()
def score_dataset(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """返回 (anomaly_scores, labels)，每个元素对应一个窗口。"""
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        signal = batch["signal"].to(device)
        s = model.anomaly_score(signal)            # [B]
        scores.append(s.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(scores), np.concatenate(labels)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 SMD MLP 异常检测器")
    # 数据
    p.add_argument("--data_root", default=str(smd_data.DEFAULT_SMD_ROOT),
                   help="ServerMachineDataset 根目录")
    p.add_argument("--win_size", type=int, default=100)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--machines", nargs="*", default=None,
                   help="只用指定机器（默认全部 28 台合并）")
    p.add_argument("--val_split", type=float, default=0.5,
                   help="把每台机器的 test 文件按时间切出多少比例做 val（其余做 test）")
    p.add_argument("--threshold_percentile", type=float, default=99.0,
                   help="val 全为正常时，用 val 异常分数的此分位数作为判定阈值")
    # 模型
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    # 优化
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    # 输出
    p.add_argument("--output_dir", default=str(ROOT / "runs"))
    p.add_argument("--run_name", default=None)
    # wandb（可选）
    p.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 记录")
    p.add_argument("--wandb_project", default="sigla-anomaly-detector")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None, help="默认用 --run_name")
    return p.parse_args()


def init_wandb(args: argparse.Namespace, run_name: str):
    """按需初始化 wandb；失败时打印警告并返回 None，不中断训练。"""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] 未安装 wandb，跳过记录（pip install wandb）")
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )
        return run
    except Exception as exc:  # 网络/鉴权失败不应让训练崩掉
        print(f"[wandb] 初始化失败，继续无记录训练: {exc}")
        return None


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    run_name = args.run_name or f"anomaly_detector_SMD_all_w{args.win_size}_s{args.stride}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb(args, run_name)

    # ---- 数据（干净无监督设定）---- #
    # train ← train 文件（纯正常，无异常）；val/test ← test 文件按时间 1:1 切分。
    bundle = smd_data.build_smd_datasets(
        root=args.data_root,
        win_size=args.win_size,
        stride=args.stride,
        machines=args.machines,
        val_split=args.val_split,
    )
    n_vars = bundle.n_vars

    train_loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(bundle.val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(bundle.test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    print(f"machines={len(bundle.machines)}  n_vars={n_vars}  "
          f"win={args.win_size}  stride={args.stride}")
    print(f"train_windows={len(bundle.train)} (纯正常)  "
          f"val_windows={len(bundle.val)} (异常 {bundle.val.anomaly_window_count()})  "
          f"test_windows={len(bundle.test)} (异常 {bundle.test.anomaly_window_count()})  device={device}")

    # ---- 模型 ---- #
    model = MLPAnomalyDetector(args.win_size, n_vars,
                               latent_dim=args.latent_dim,
                               hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ---- 训练循环 ---- #
    # 无监督训练：只用 train（纯正常）的重构 MSE 反向传播。
    # 模型选择标准（对所有 epoch 固定，因为 val 组成不变）：
    #   - val 含异常 → 用 val AUROC（越大越好，检测目标的诚实代理）
    #   - val 全正常 → 用 val 重构损失（越小越好，标准的无监督选择方式）
    _, val_labels0 = score_dataset(model, val_loader, device)
    val_has_pos = bool((val_labels0 == 1).any())
    print(f"val 含异常: {val_has_pos} -> 模型选择依据: "
          f"{'val AUROC' if val_has_pos else 'val 重构损失'}")

    history: list[dict[str, float]] = []
    best_score = -float("inf")   # 统一成「越大越好」
    best_state = clone_state_dict(model)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_scores, val_labels = score_dataset(model, val_loader, device)
        val_recon = eval_recon_loss(model, val_loader, device)
        val_auc = roc_auc(val_labels, val_scores) if val_has_pos else float("nan")
        select = val_auc if val_has_pos else -val_recon
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_recon_loss": val_recon, "val_roc_auc": val_auc}
        history.append(row)
        print(f"epoch={epoch:>3d}  train_loss={train_loss:.6f}  "
              f"val_recon={val_recon:.6f}  val_auc={val_auc:.4f}")
        if wandb_run is not None:
            wandb_run.log(row)
        if select > best_score:
            best_score = select
            best_state = clone_state_dict(model)

    model.load_state_dict(best_state)

    # ---- 用最优模型：在 val 上定阈值，套到 test 上评估 ---- #
    val_scores, val_labels = score_dataset(model, val_loader, device)
    val_metrics = ranking_metrics(val_labels, val_scores)
    if val_has_pos:
        # val 有异常：用 best-F1 阈值
        thr = best_f1(val_labels, val_scores)["threshold"]
        thr_method = "val_best_f1"
    else:
        # val 全正常：用正常分数的高分位数当阈值（纯无监督，不需要异常标签）
        thr = float(np.percentile(val_scores, args.threshold_percentile))
        thr_method = f"val_p{args.threshold_percentile:g}"

    test_scores, test_labels = score_dataset(model, test_loader, device)
    test_metrics = ranking_metrics(test_labels, test_scores)          # threshold-free（AUROC/AP）
    # 用 val 定的阈值在 test 上算 precision/recall/f1（诚实工作点，不偷看 test 标签）
    pred = (test_scores >= thr).astype(np.int64)
    tp = int(((pred == 1) & (test_labels == 1)).sum())
    fp = int(((pred == 1) & (test_labels == 0)).sum())
    fn = int(((pred == 0) & (test_labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    test_metrics["f1_at_threshold"] = float(f1)
    test_metrics["precision_at_threshold"] = float(prec)
    test_metrics["recall_at_threshold"] = float(rec)
    test_metrics["threshold"] = float(thr)
    test_metrics["threshold_method"] = thr_method
    # 注：test_metrics 里的 "best_f1" 是 test 上的 oracle 上界，仅作参考。

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else str(x)
    print(f"val:  roc_auc={fmt(val_metrics.get('roc_auc'))}  "
          f"threshold={thr:.6f} ({thr_method})")
    print("test metrics:")
    for k in ("roc_auc", "average_precision", "f1_at_threshold",
              "precision_at_threshold", "recall_at_threshold", "best_f1"):
        print(f"  {k}: {fmt(test_metrics.get(k))}")
    if wandb_run is not None:
        wandb_run.summary["best_select_score"] = best_score
        wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()
                       if isinstance(v, (int, float))})
        wandb_run.log({f"test/{k}": v for k, v in test_metrics.items()
                       if isinstance(v, (int, float))})

    # ---- 保存 ---- #
    ckpt_args = {
        "dataset": "SMD_all",
        "data_dir": args.data_root,
        "win_size": args.win_size,
        "step": args.stride,
        "stride": args.stride,
        "val_split": args.val_split,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "machines": bundle.machines,
    }
    checkpoint = {
        "args": ckpt_args,
        "n_vars": n_vars,
        "component": "detector",
        "model": best_state,        # 与 eval_autoencoder.py 期望的 "model" 键兼容
        "detector": best_state,
    }
    torch.save(checkpoint, run_dir / "checkpoint_best.pt")
    write_json(run_dir / "config.json", vars(args))
    write_json(run_dir / "metrics.json", {
        "best_select_score": best_score,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    })
    print(f"saved run to {run_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
