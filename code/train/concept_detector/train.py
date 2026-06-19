#!/usr/bin/env python3
"""训练 concept detector（多标签，合成数据监督）。

模型：sigla_exp.model.mlp.MLPConceptDetector
    原始窗口 [B, win, n_vars] -> 6 个 concept 的 logits（多标签）。

数据：specific_data/Concept_detector/concept_synth.py
    从 SMD **正常**窗口实时合成「带 concept 标签」的样本：
    往正常窗口注入 spike / level_shift / oscillation / variance_burst /
    trend / correlation_break（可多个共现），注入了什么标签就是什么。

为什么用合成：真实异常没有「形态」标注，但正常数据 + 程序化注入 =
    海量带标签训练数据。detector 学会识别形态后，可用在真实异常上做解释。

切分（干净）：
    train ← 各机器 train 文件（纯正常）合成
    val   ← 各机器 test 文件前半的【正常窗口】合成（同分布、独立窗口）
    （concept 标签全部来自合成，与 SMD 的 test_label 无关，故无标签泄露。）

评估：每个 concept 的 ROC-AUC + 阈值 0.5 下的 F1，以及 macro 平均。

用法见 scripts/train_concept_detector*.sh。
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

ROOT = Path(__file__).resolve().parents[2]          # .../sigLA/code
SIGLA_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigla_exp.model.cnn import CNNConceptDetector  # noqa: E402
from sigla_exp.model.mlp import MLPConceptDetector  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


smd_data = _load(SIGLA_ROOT / "specific_data" / "Anomaly_detector" / "data.py", "smd_data")
concept_synth = _load(SIGLA_ROOT / "specific_data" / "Concept_detector" / "concept_synth.py", "concept_synth")
CONCEPT_NAMES = concept_synth.CONCEPT_NAMES


# --------------------------------------------------------------------------- #
#  指标（纯 numpy，多标签）                                                    #
# --------------------------------------------------------------------------- #
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def f1_at(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"f1": f1, "precision": prec, "recall": rec}


# --------------------------------------------------------------------------- #
#  正常窗口池 + 合成数据集                                                     #
# --------------------------------------------------------------------------- #
def stack_windows(ds) -> np.ndarray:
    """把一个 SMDWindowDataset 里只含正常(label==0)的窗口堆成 [N, win, n_vars]。"""
    keep = [ds[i]["signal"].numpy() for i in range(len(ds)) if int(ds[i]["label"]) == 0]
    return np.stack(keep) if keep else np.empty((0,))


def build_pools(args):
    """train 用各机器 train 文件全部窗口；val 用 test 文件前半的正常窗口。"""
    bundle = smd_data.build_smd_datasets(
        root=args.data_root, win_size=args.win_size, stride=args.stride,
        machines=args.machines, val_split=0.5,
    )
    train_pool = np.stack([bundle.train[i]["signal"].numpy() for i in range(len(bundle.train))])
    val_pool = stack_windows(bundle.val)   # val 段里仅取正常窗口做合成
    return bundle, train_pool, val_pool


# --------------------------------------------------------------------------- #
#  训练 / 评估                                                                 #
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    for batch in loader:
        signal = batch["signal"].to(device)
        label = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(signal)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        p = model.predict_proba(batch["signal"].to(device)).cpu().numpy()
        P.append(p); Y.append(batch["label"].numpy())
    return np.concatenate(P), np.concatenate(Y)


def per_concept_metrics(P: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"per_concept": {}}
    aucs, f1s = [], []
    for i, name in enumerate(CONCEPT_NAMES):
        auc = roc_auc(Y[:, i], P[:, i])
        f = f1_at(Y[:, i], P[:, i], 0.5)
        out["per_concept"][name] = {"roc_auc": auc, **f}
        if auc == auc:
            aucs.append(auc)
        f1s.append(f["f1"])
    out["macro_roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_f1"] = float(np.mean(f1s))
    return out


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def init_wandb(args, run_name):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] 未安装，跳过"); return None
    try:
        return wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                          name=args.wandb_run_name or run_name, config=vars(args))
    except Exception as exc:
        print(f"[wandb] 初始化失败: {exc}"); return None


def parse_args():
    p = argparse.ArgumentParser(description="训练 concept detector（多标签 / 合成监督）")
    p.add_argument("--data_root", default=str(smd_data.DEFAULT_SMD_ROOT))
    p.add_argument("--machines", nargs="*", default=None)
    p.add_argument("--win_size", type=int, default=100)
    p.add_argument("--stride", type=int, default=10)
    # 合成
    p.add_argument("--p_normal", type=float, default=0.2)
    p.add_argument("--max_concepts", type=int, default=3)
    # 模型 / 优化
    p.add_argument("--model", choices=("mlp", "cnn"), default="cnn",
                   help="cnn=1D-CNN(擅长局部形态)；mlp=扁平 MLP")
    p.add_argument("--hidden_dim", type=int, default=256, help="MLP 隐层宽度")
    p.add_argument("--kernel_size", type=int, default=7, help="CNN 卷积核大小")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    # 输出 / wandb
    p.add_argument("--output_dir", default=str(ROOT / "runs"))
    p.add_argument("--run_name", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="sigla-concept-detector")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or "concept_detector_w%d_s%d" % (args.win_size, args.stride)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, run_name)

    # ---- 数据：正常窗口池 -> 合成多标签样本 ---- #
    bundle, train_pool, val_pool = build_pools(args)
    n_vars = bundle.n_vars
    train_ds = concept_synth.SyntheticConceptDataset(
        train_pool, seed=args.seed, p_normal=args.p_normal, max_concepts=args.max_concepts)
    # val 用不同 seed、固定 epoch，保证可复现且与 train 注入不同
    val_ds = concept_synth.SyntheticConceptDataset(
        val_pool, seed=args.seed + 999, p_normal=args.p_normal, max_concepts=args.max_concepts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print(f"machines={len(bundle.machines)} n_vars={n_vars} win={args.win_size} stride={args.stride}")
    print(f"train_normal_pool={len(train_pool)}  val_normal_pool={len(val_pool)}  "
          f"concepts={list(CONCEPT_NAMES)}  device={device}")

    # ---- 模型 ---- #
    if args.model == "cnn":
        model = CNNConceptDetector(args.win_size, n_vars, n_concepts=len(CONCEPT_NAMES),
                                   kernel_size=args.kernel_size, dropout=args.dropout).to(device)
    else:
        model = MLPConceptDetector(args.win_size, n_vars, n_concepts=len(CONCEPT_NAMES),
                                   hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    print(f"model={args.model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_macro_auc = -1.0
    best_state = clone_state(model)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, None)
        P, Y = collect_preds(model, val_loader, device)
        vm = per_concept_metrics(P, Y)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_macro_roc_auc": vm["macro_roc_auc"], "val_macro_f1": vm["macro_f1"]}
        history.append(row)
        print(f"epoch={epoch:>3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_macroAUC={vm['macro_roc_auc']:.4f} val_macroF1={vm['macro_f1']:.4f}")
        if wandb_run is not None:
            wandb_run.log(row)
        if vm["macro_roc_auc"] > best_macro_auc:
            best_macro_auc = vm["macro_roc_auc"]
            best_state = clone_state(model)

    model.load_state_dict(best_state)
    P, Y = collect_preds(model, val_loader, device)
    final = per_concept_metrics(P, Y)
    print(f"\nbest val macro ROC-AUC={best_macro_auc:.4f}")
    print(f"{'concept':18s} {'roc_auc':>8s} {'f1':>7s} {'prec':>7s} {'rec':>7s}")
    for name in CONCEPT_NAMES:
        m = final["per_concept"][name]
        print(f"{name:18s} {m['roc_auc']:8.4f} {m['f1']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f}")
    print(f"{'MACRO':18s} {final['macro_roc_auc']:8.4f} {final['macro_f1']:7.4f}")

    if wandb_run is not None:
        wandb_run.summary["best_val_macro_roc_auc"] = best_macro_auc
        for name in CONCEPT_NAMES:
            for k, v in final["per_concept"][name].items():
                wandb_run.log({f"val/{name}/{k}": v})

    # ---- 保存 ---- #
    checkpoint = {
        "args": {"model": args.model, "win_size": args.win_size, "stride": args.stride,
                 "hidden_dim": args.hidden_dim, "kernel_size": args.kernel_size,
                 "concepts": list(CONCEPT_NAMES), "machines": bundle.machines, "seed": args.seed},
        "n_vars": n_vars,
        "component": "concept_detector",
        "model": best_state,
    }
    torch.save(checkpoint, run_dir / "checkpoint_best.pt")
    write_json(run_dir / "config.json", vars(args))
    write_json(run_dir / "metrics.json",
               {"best_val_macro_roc_auc": best_macro_auc, "history": history, "val": final})
    print(f"saved run to {run_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
