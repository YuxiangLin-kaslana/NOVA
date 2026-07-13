#!/usr/bin/env python3
"""【学习式融合(严格不偷看):加 LLM/文本 是否比纯数值更好?】

等权融合太naïve(弱通道稀释强信号)。正确测法:**5 折 CV 逻辑回归**,比不同特征集的**留出 AUC**——
模型自动给文本合适权重(最差学成 0→退回 numeric),所以若 [num+text] 的 CV-AUC > [num] → 文本带正交增量。

通道分(对全部多模态点各算一次,存盘备查):numeric / text(LLM只看文本) / llm_full(LLM看数值+文本)。
特征集:[num] / [num,text] / [num,llm_full] / [num,text,llm_full]。报 OOF AUC(多 random_state 误差棒)。
判据:[num,text] 或 [num,*] 的 CV-AUC 显著 > [num] → 加 LLM 有用(多模态互补);否则冗余/无用。
用法: sbatch sota_compare/run_mmdfuse2.sh
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sota_compare.exp_mmd_earlywarning as M    # noqa: E402
import sota_compare.exp_mmd_fusion as F          # gpt_text_only  # noqa: E402

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"


def oof_auc(X, y, n_rep=20):
    """多次 5 折 CV 的 out-of-fold AUC,返回 mean,std。"""
    aucs = []
    for rs in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=rs)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    ot, dates, rep = M.load()
    pts, thr = M.build_points(ot, dates, rep)
    if SMOKE:
        pts = pts[:50]
    y = np.array([p[1] for p in pts])
    num = np.array([(ot[i] - thr) + (ot[i] - ot[i - 3]) for i, _, _ in pts])
    # 更丰富的数值特征(给纯数值最强机会,确保对比公平)
    lvl = np.array([ot[i] for i, _, _ in pts])
    slope = np.array([ot[i] - ot[i - 3] for i, _, _ in pts])
    accel = np.array([(ot[i] - ot[i - 1]) - (ot[i - 3] - ot[i - 4]) for i, _, _ in pts])
    NUM = np.column_stack([num, lvl, slope, accel])

    text = np.full(len(pts), np.nan); full = np.full(len(pts), np.nan)
    if net_ok:
        for j, (i, _, rtxt) in enumerate(pts):
            t = F.gpt_text_only(rtxt, key); f = M.gpt_risk(ot[i - M.K + 1:i + 1], rtxt, key)
            text[j] = np.nan if t is None else t
            full[j] = np.nan if f is None else f
        text = np.nan_to_num(text, nan=np.nanmean(text)); full = np.nan_to_num(full, nan=np.nanmean(full))
    json.dump({"num": NUM.tolist(), "text": text.tolist(), "full": full.tolist(), "y": y.tolist()},
              open(ROOT / "runs" / "mmd_fusion2_scores.json", "w"))

    print(f"Time-MMD ILI 学习式融合(5折CV)  net_ok={net_ok} n={len(pts)} 正例率={y.mean():.0%} H={M.H}\n")
    feats = {"[num]": NUM}
    if net_ok:
        feats["[num,text]"] = np.column_stack([NUM, text])
        feats["[num,llm_full]"] = np.column_stack([NUM, full])
        feats["[num,text,llm_full]"] = np.column_stack([NUM, text, full])
        feats["[text only]"] = text.reshape(-1, 1)
    print(f"{'特征集':24s}{'CV-AUC':>16s}")
    print("-" * 42)
    base = None
    for name, X in feats.items():
        m, s = oof_auc(X, y, n_rep=3 if SMOKE else 20)
        if name == "[num]":
            base = m
        delta = "" if name == "[num]" else f"  Δvs[num]={m-base:+.3f}"
        print(f"{name:24s}{m:>11.3f}±{s:<4.3f}{delta}")
    print("\n判据:[num,text] 或 [num,*] 的 CV-AUC 显著高于 [num](Δ>0 且超过 ±std)→ 加 LLM/文本带来正交增量(多模态有用);")
    print("     Δ≈0 → 文本与数值冗余,加 LLM 无改善(但也不更差,因模型可自动降权)。")


if __name__ == "__main__":
    main()
