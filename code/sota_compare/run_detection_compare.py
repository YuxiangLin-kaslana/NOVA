#!/usr/bin/env python3
"""【新异常类型检测 · SOTA 同口径对比】

回答审稿人:出现**从未见过的新异常类型**(correlation_break 涌现)时,前人 SOTA 检测器够不够用?
所有臂跑**同一条流、同一套指标**(复用 scripts.exp_detection_tie 的 build_stream / detect_metrics):

  Frozen-CNN          闭集分类器(normal+3已知类),从不扩词表/无 LLM —— 我们的弱基线锚点
  AnomalyTransformer  ICLR'22 无监督重构+关联差异 SOTA(冻结)
  MemStream           WWW'22 记忆+在线更新,显式抗漂移 SOTA(在线吸收漂移)
  Ours (bootstrap)    证据门控→LLM 命名→在线扩词表+类平衡重放

指标(涌现段):新类**检测召回**(pred≠normal)、整体检测 F1/精度/召回、新类**分类**准确率(pred==NOVEL)。
要点:无监督 SOTA 能(对强信号)把 novel 判"异常",但 nov_classacc≡0(没有类型概念)——这是 ours 补的空白。
用法: sbatch sota_compare/run_compare.sh  (env CMP_NSEED 默认 5)
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.exp_detection_tie as DT          # 复用已验证的流/指标/frozen+bootstrap 原语  # noqa: E402
import sigla_exp.ovbench as CB                  # noqa: E402
from sota_compare.baselines import MemStream, AnomalyTransformer  # noqa: E402

NSEED = int(os.environ.get("CMP_NSEED", "5"))
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"  # 冒烟:小规模、跳过 LLM,仅验证 shape/流程
device = DT.device
ANOM = "anomaly"                                # 无类型检测器的"非正常"哨兵预测
N_PT = 300 if SMOKE else 3000                    # 闭集预训练样本数
N_NORM_TR = 200 if SMOKE else 1500               # 无监督 SOTA 正常训练集
UNSUP_EP = 3 if SMOKE else 40                    # 无监督 SOTA 训练轮数


# --------- frozen + bootstrap:复用 DT 原语,但喂入**本 runner 构造的同一条流** --------- #
def run_cnn_arms(stream, onset, pre_state, replay, mu, sd, key, net_ok, rng):
    det = DT.make_detector(len(DT.BASE_VOCAB)); det.load_state_dict(pre_state)
    froz_pred = [DT.BASE_VOCAB[int(np.argmax(DT.proba(det, [x])[0]))] for x, _ in stream]

    det = DT.make_detector(len(DT.BASE_VOCAB)); det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(DT.BASE_VOCAB); buf = []; boot_pred = []; llm_flags = []; pending = 0
    for x, _ in stream:
        ev = CB.evidence(x)
        p = DT.proba(det, [x])[0]
        mx = float(p.max()); pred_idx = int(np.argmax(p))
        susp = DT.suspect_novel(ev, mu, sd)
        mislabel = (vocab[pred_idx] in (DT.KNOWN_ANOM + [DT.NORMAL])) and susp
        audit = rng.random() < DT.AUDIT
        if mx >= DT.TAU and not mislabel and not audit:
            pred, llm = vocab[pred_idx], 0
        else:
            c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            c = None if c == "__ERROR__" else c
            pred = c if c else vocab[pred_idx]; llm = 1
            if pred not in vocab:
                if susp:
                    vocab.append(pred); DT.grow_head(det, len(vocab))
                    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                else:
                    pred = vocab[pred_idx]
            buf.append((x, vocab.index(pred))); pending += 1
            if pending >= DT.RETRAIN_EVERY:
                per_class = {}
                for w, li in replay:
                    per_class.setdefault(li, []).append(w)
                for w, li in buf:
                    per_class.setdefault(li, []).append(w)
                K = 40; Xb, Yb = [], []
                for li, ws in per_class.items():
                    for j in rng.integers(0, len(ws), K):
                        Xb.append(ws[j]); Yb.append(DT.onehot(li, len(vocab)))
                DT.train_on(det, opt, Xb, Yb, epochs=2); pending = 0
        boot_pred.append(pred); llm_flags.append(llm)
    return froz_pred, boot_pred, vocab, float(np.mean(llm_flags[onset:]))


# --------- 无监督 SOTA 检测器:正常窗预训练 → 留出正常标定 q95 阈 → 逐窗判 --------- #
def run_unsup_arm(model, normal_train, normal_cal, stream):
    model.fit(normal_train)
    cal = model.score_stream(normal_cal, update=False)
    thresh = float(np.quantile(cal, 0.95))                         # FAR≈5%
    scores = model.score_stream([x for x, _ in stream], update=True)
    preds = [ANOM if s > thresh else DT.NORMAL for s in scores]
    return preds


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)

    # 预训练闭集 CNN(normal+3 已知),供 frozen+bootstrap
    det = DT.make_detector(len(DT.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(N_PT):
        c = DT.BASE_VOCAB[rng.integers(len(DT.BASE_VOCAB))]
        Xpt.append(DT.make_labeled(c, rng)); Ypt.append(DT.onehot(DT.BASE_VOCAB.index(c), len(DT.BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))

    stream, onset = DT.build_stream(rng)                           # 同一条评测流喂所有臂
    trues = [t for _, t in stream]

    froz_pred, boot_pred, vocab, llm_rate = run_cnn_arms(
        stream, onset, pre_state, replay, mu, sd, key, net_ok, rng)

    # 无监督 SOTA 用**独立 rng** 生成正常训练/标定集(不扰动评测流)
    rng2 = np.random.default_rng(10_000 + seed)
    normal_train = [CB.make_window(None, rng2) for _ in range(N_NORM_TR)]
    normal_cal = [CB.make_window(None, rng2) for _ in range(400)]
    at_pred = run_unsup_arm(AnomalyTransformer(CB.WIN, CB.NVARS, device, epochs=UNSUP_EP, seed=seed),
                            normal_train, normal_cal, stream)
    ms_pred = run_unsup_arm(MemStream(CB.WIN, CB.NVARS, device, epochs=UNSUP_EP, seed=seed),
                            normal_train, normal_cal, stream)

    arms = {"frozen": froz_pred, "anomaly_transformer": at_pred,
            "memstream": ms_pred, "bootstrap": boot_pred}
    out = {name: DT.detect_metrics(p[onset:], trues[onset:], None) for name, p in arms.items()}
    out["llm_rate"] = llm_rate
    out["grew"] = int(DT.NOVEL in vocab)
    return out


def ms(xs):
    a = np.array(xs, float); return float(np.nanmean(a)), float(np.nanstd(a))


def output_path(default_name):
    explicit = os.environ.get("CMP_OUTPUT_JSON")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else ROOT / p
    tag = os.environ.get("CMP_RUN_TAG", "").strip()
    if tag:
        stem, suffix = Path(default_name).stem, Path(default_name).suffix
        return ROOT / "runs" / f"{stem}_{tag}{suffix}"
    return ROOT / "runs" / default_name


# 人工标签成本 + 是否有"类型"概念(money table 的关键两列)
META = {
    "frozen":              ("closed-set 分类器", 0, "已知类型"),
    "anomaly_transformer": ("ICLR'22 无监督SOTA(冻结)", 0, "无(仅分数)"),
    "memstream":           ("WWW'22 抗漂移SOTA(在线)", 0, "无(仅分数)"),
    "bootstrap":           ("Ours: LLM开放词表闭环", 0, "已知+自举新类型"),
}
ARMS = list(META)


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key) and not SMOKE               # 冒烟跳过 LLM(省钱/快)
    print(f"device={device} net_ok={net_ok} SMOKE={SMOKE} NSEED={NSEED}  novel={DT.NOVEL}\n"
          f"  base_vocab={DT.BASE_VOCAB}  检测=pred≠normal; known_stats={sorted(DT.KNOWN_STATS)}")
    res = []
    for s in range(NSEED):
        r = run_seed(s, key, net_ok)
        res.append(r)
        line = " | ".join(f"{a[:4]} novR={r[a]['nov_recall']:.0%} F1={r[a]['f1']:.2f}" for a in ARMS)
        print(f"[seed {s}] {line}  (grew={r['grew']})")

    print("\n" + "=" * 96)
    print(f"涌现后 SOTA 对比(mean±std over {NSEED} seeds, novel={DT.NOVEL}):\n")
    hdr = f"{'method':22s}{'人工标签':>8s}{'类型概念':>14s}{'新类检测召回':>14s}{'整体F1':>10s}{'新类分类':>10s}"
    print(hdr); print("-" * 96)
    for a in ARMS:
        desc, labels, hastype = META[a]
        nr, ns = ms([r[a]["nov_recall"] for r in res])
        f1, f1s = ms([r[a]["f1"] for r in res])
        ca, cas = ms([r[a]["nov_classacc"] for r in res])
        print(f"{a:22s}{labels:>8d}{hastype:>14s}{nr*100:>9.0f}±{ns*100:<3.0f}{f1:>8.2f}±{f1s:<4.2f}{ca*100:>6.0f}±{cas*100:<3.0f}")
    lm, _ = ms([r["llm_rate"] for r in res])
    print("-" * 96)
    print(f"Ours LLM 调用率(涌现后): {lm:.0%}   词表长全率: {np.mean([r['grew'] for r in res]):.0%}")
    print("\n判读:无监督 SOTA(AnomalyTransformer/MemStream)能把强信号 novel 判'异常'(检测召回或不低),")
    print("但**新类分类准确率≡0**——它们没有类型概念,永远命名不出新类型,做不了类型化早预警;")
    print("MemStream 即便在线吸收漂移,也只学'新正常',学不出'新异常命名'。Ours 在零人工标签下同时拿到检测+命名。")
    print("=" * 96)
    outp = output_path("sota_detection_compare.json")
    json.dump(dict(nseed=NSEED, novel=DT.NOVEL, per_seed=res, meta=META), open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
