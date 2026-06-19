#!/usr/bin/env python3
"""路线B 的桥:把开放词表闭环接到**二分类异常检测**(normal vs anomaly),证明它不只改善"分类",
更让检测器**检测出从未见过的新异常类型**——回答审稿人核心问题:闭集检测器对新类盲不盲?

设定:检测器含一个 **normal 类** + 3 个已知异常类(spike/level_shift/oscillation)。
检测 = argmax 类别 ≠ normal。流里 50% normal + 50% 异常;后段**新异常类型** correlation_break 涌现。
对照:
  frozen     闭集检测器(永不扩词表,无 LLM)→ 新类要么被误判 normal(漏检)要么误判已知(检到但类型错)。
  bootstrap  证据门控发现新类(z 超阈且主导非已知签名,自然排除 normal)→ LLM 命名 → 扩类 → 在线学
             → 新类被正确检测**并**正确命名。
指标:新类**检测召回**(flagged=argmax≠normal)、整体检测 F1(anomaly vs normal)、新类分类准确率。
复用 sigla_exp.ovbench 正交基准。用法: sbatch scripts/exp_detection_tie.sh (env OVD_NSEED, 默认5)
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sigla_exp.model import CNNConceptDetector  # noqa: E402
import sigla_exp.ovbench as CB                  # noqa: E402

WIN, NVARS = CB.WIN, CB.NVARS
NORMAL = "normal"
KNOWN_ANOM = ["spike", "level_shift", "oscillation"]
NOVEL = "correlation_break"
BASE_VOCAB = [NORMAL] + KNOWN_ANOM                          # normal=0,已知异常 1..3
KNOWN_STATS = {CB.STAT_OF[c] for c in KNOWN_ANOM}
TAU = 0.5
AUDIT = 0.08
NOVEL_Z = 2.3                                              # 主导非已知签名 z 超此 → 疑似新异常(normal 的 z 小,自然排除)
RETRAIN_EVERY = 15
N_WARM, SEG = 200, 400
NSEED = int(os.environ.get("OVD_NSEED", "5"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_detector(n_out):
    return CNNConceptDetector(WIN, NVARS, n_concepts=n_out, kernel_size=7).to(device)


@torch.no_grad()
def proba(det, X):
    det.eval()
    return torch.sigmoid(det(torch.tensor(np.stack(X)).to(device))).cpu().numpy()


def train_on(det, opt, X, Y, epochs):
    det.train()
    Xt = torch.tensor(np.stack(X)).to(device); Yt = torch.tensor(np.stack(Y)).to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(det(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    det.eval()


def grow_head(det, new_n):
    old = det.head[-1]
    new = nn.Linear(old.in_features, new_n).to(device)
    with torch.no_grad():
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
    det.head[-1] = new


def onehot(idx, n):
    v = np.zeros(n, np.float32); v[idx] = 1.0; return v


def suspect_novel(ev, mu, sd):
    devz = {k: abs((ev[k] - mu[k]) / sd[k]) for k in mu}
    dom = max(devz, key=devz.get)
    return (dom not in KNOWN_STATS) and (devz[dom] > NOVEL_Z)


def make_labeled(concept_or_normal, rng):
    c = None if concept_or_normal == NORMAL else concept_or_normal
    return CB.make_window(c, rng)


def build_stream(rng):
    """50% normal + 50% 异常;段A 异常仅已知,段B 起加入 NOVEL。"""
    stream = []
    for _ in range(N_WARM):
        c = NORMAL if rng.random() < 0.5 else KNOWN_ANOM[rng.integers(len(KNOWN_ANOM))]
        stream.append((make_labeled(c, rng), c))
    onset = len(stream)
    pool = KNOWN_ANOM + [NOVEL]
    for _ in range(SEG):
        c = NORMAL if rng.random() < 0.5 else pool[rng.integers(len(pool))]
        stream.append((make_labeled(c, rng), c))
    return stream, onset


def detect_metrics(preds, trues, vocab):
    """preds: 预测类名;trues: 真类名。检测=pred≠normal。返回整体 F1 与新类召回。"""
    tp = sum(1 for p, t in zip(preds, trues) if t != NORMAL and p != NORMAL)
    fp = sum(1 for p, t in zip(preds, trues) if t == NORMAL and p != NORMAL)
    fn = sum(1 for p, t in zip(preds, trues) if t != NORMAL and p == NORMAL)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    nov = [(p, t) for p, t in zip(preds, trues) if t == NOVEL]
    nov_recall = np.mean([p != NORMAL for p, t in nov]) if nov else float("nan")
    nov_classacc = np.mean([p == NOVEL for p, t in nov]) if nov else float("nan")
    return dict(f1=f1, prec=prec, rec=rec, nov_recall=float(nov_recall), nov_classacc=float(nov_classacc))


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)

    # ---- 预训练:normal + 3 已知异常类 ---- #
    det = make_detector(len(BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(3000):
        c = BASE_VOCAB[rng.integers(len(BASE_VOCAB))]
        Xpt.append(make_labeled(c, rng)); Ypt.append(onehot(BASE_VOCAB.index(c), len(BASE_VOCAB)))
    train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))

    stream, onset = build_stream(rng)

    # ---- frozen 基线 ---- #
    froz_pred = [BASE_VOCAB[int(np.argmax(proba(det, [x])[0]))] for x, _ in stream]

    # ---- bootstrap 闭环 ---- #
    det = make_detector(len(BASE_VOCAB)); det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(BASE_VOCAB); buf = []; boot_pred = []; llm_flags = []; pending = 0
    for i, (x, true_c) in enumerate(stream):
        ev = CB.evidence(x)
        p = proba(det, [x])[0]
        mx = float(p.max()); pred_idx = int(np.argmax(p))
        susp = suspect_novel(ev, mu, sd)
        mislabel = (vocab[pred_idx] in (KNOWN_ANOM + [NORMAL])) and susp
        audit = rng.random() < AUDIT
        if mx >= TAU and not mislabel and not audit:
            pred, llm = vocab[pred_idx], 0
        else:
            c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            c = None if c == "__ERROR__" else c
            pred = c if c else vocab[pred_idx]
            llm = 1
            if pred not in vocab:
                if susp:
                    vocab.append(pred); grow_head(det, len(vocab))
                    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                else:
                    pred = vocab[pred_idx]
            buf.append((x, vocab.index(pred))); pending += 1
            if pending >= RETRAIN_EVERY:
                per_class = {}
                for w, li in replay:
                    per_class.setdefault(li, []).append(w)
                for w, li in buf:
                    per_class.setdefault(li, []).append(w)
                K = 40; Xb, Yb = [], []
                for li, ws in per_class.items():
                    for j in rng.integers(0, len(ws), K):
                        Xb.append(ws[j]); Yb.append(onehot(li, len(vocab)))
                train_on(det, opt, Xb, Yb, epochs=2); pending = 0
        boot_pred.append(pred); llm_flags.append(llm)

    trues = [t for _, t in stream]
    fm = detect_metrics(froz_pred[onset:], trues[onset:], BASE_VOCAB)
    bm = detect_metrics(boot_pred[onset:], trues[onset:], vocab)
    # 新类检测召回随时间(涌现后 4 段)
    nov_idx = [i for i in range(onset, len(stream)) if trues[i] == NOVEL]
    segs = [s for s in np.array_split(nov_idx, 4) if len(s)]
    nov_rec_curve = [float(np.mean([boot_pred[i] != NORMAL for i in s])) for s in segs]
    return dict(frozen=fm, bootstrap=bm, grew=int(NOVEL in vocab),
                llm_rate=float(np.mean(llm_flags[onset:])), nov_rec_curve=nov_rec_curve)


def ms(xs):
    a = np.array(xs, float); return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    print(f"device={device} net_ok={net_ok} NSEED={NSEED}\n  base_vocab={BASE_VOCAB}  novel={NOVEL}\n"
          f"  检测=argmax≠normal; known_stats={sorted(KNOWN_STATS)}")
    res = []
    for s in range(NSEED):
        r = run_seed(s, key, net_ok)
        res.append(r)
        print(f"[seed {s}] novel检测召回 froz={r['frozen']['nov_recall']:.0%} boot={r['bootstrap']['nov_recall']:.0%}"
              f" | 整体F1 froz={r['frozen']['f1']:.2f} boot={r['bootstrap']['f1']:.2f}"
              f" | novel类型对 boot={r['bootstrap']['nov_classacc']:.0%} grew={r['grew']}")

    print("\n" + "=" * 80)
    print(f"涌现后(mean±std over {NSEED} seeds):")
    for name, key2, sub in [("新类检测召回", "nov_recall", None), ("整体检测 F1", "f1", None),
                            ("整体检测精度", "prec", None), ("整体检测召回", "rec", None),
                            ("新类**分类**准确率", "nov_classacc", None)]:
        fm_, fs_ = ms([r["frozen"][key2] for r in res])
        bm_, bs_ = ms([r["bootstrap"][key2] for r in res])
        print(f"  {name:16s}: frozen {fm_:.2f}±{fs_:.2f}   bootstrap {bm_:.2f}±{bs_:.2f}")
    lm, lsd = ms([r["llm_rate"] for r in res])
    curve = np.mean([r["nov_rec_curve"] for r in res], 0)
    print(f"  LLM 调用率: {lm:.0%}±{lsd:.0%}   词表长全率: {np.mean([r['grew'] for r in res]):.0%}")
    print(f"  新类检测召回随时间(4段,mean): {[round(float(v), 2) for v in curve]}")
    fnr, _ = ms([r["frozen"]["nov_recall"] for r in res])
    bnr, _ = ms([r["bootstrap"]["nov_recall"] for r in res])
    print(f"\n判读:frozen 新类检测召回={fnr:.0%}(闭集对新类{'盲→漏检' if fnr < 0.5 else '仍能报警但类型错'}),"
          f"bootstrap={bnr:.0%}。")
    print("=" * 80)
    json.dump(dict(nseed=NSEED, per_seed=res), open(ROOT / "runs" / "detection_tie_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
