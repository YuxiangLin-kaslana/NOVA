#!/usr/bin/env python3
"""路线B 强化:**多个**新异常类型**错峰**涌现 —— 证明自举开放词表不是为单一类调出来的。

headline(exp_openvocab_loop)只留出 1 类(correlation_break)。审稿人会问:是不是只对这一个类奏效?
本实验把已知类减到 3,留出 3 类(variance_burst / trend / correlation_break),在流里**先后**涌现:
  段A 只已知 → 段B 引入新类#1 → 段C 再引入新类#2 → 段D 再引入新类#3。
看闭环能否**逐个**长出:词表 3→4→5→6,每个新类各自从~0 爬升,LLM 调用率在每次涌现处**多峰**后衰减。

关键修复(对比首版多留出崩溃):
  (1) 用 **clean_bench** 干净可分基准(6 概念在 6 统计量上一一对应,见 clean_bench.py 诊断),
      新颖门控 = z-score argmax 不在已知签名集 → trend/variance_burst/corr_break 都被正确判新颖。
  (2) LLM **强制 top-1**(gpt_recognize_top1) → 杜绝过度列举(首版多留出时"取首个非已知"≈随机的元凶)。

对照: frozen 闭集检测器(永 3 类,无 LLM)→ 所有新类永远错。
用法: sbatch scripts/exp_openvocab_multi.sh
"""
from __future__ import annotations

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
import sigla_exp.ovbench as CB                  # noqa: E402  证据正交基准:窗/证据/门控/LLM(见 diag_separation_v2)

WIN, NVARS = CB.WIN, CB.NVARS
KNOWN = ["spike", "level_shift", "oscillation"]                 # 只这 3 类进预训练
NOVELS = ["variance_burst", "trend", "correlation_break"]       # 留出的 3 类,错峰涌现
KNOWN_STATS = {CB.STAT_OF[c] for c in KNOWN}                    # 已知类的签名统计量集合
TAU = 0.5                         # 检测器置信门:max prob < TAU → 不确定 → 叫 LLM
AUDIT = 0.08                      # 审计抽样:即便自信也按此比例叫 LLM
NOVEL_Z = 2.3                     # 新颖门控幅度阈:主导非已知统计量 z 须超过此值(滤噪声误判)
RETRAIN_EVERY = 15                # 每积累这么多新伪标签,在线重训一次
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
    """对每个统计量做正常分布 z-score,取偏离最大者;若它不属于已知签名**且**幅度超阈 → 疑似新类型。
    幅度阈 NOVEL_Z 滤掉已知类的噪声误判(否则 warm-up 期会冒出伪类污染词表)。label-free。"""
    devz = {k: abs((ev[k] - mu[k]) / sd[k]) for k in mu}
    dom = max(devz, key=devz.get)
    return (dom not in KNOWN_STATS) and (devz[dom] > NOVEL_Z)


def build_stream(rng, n_warm, seg):
    """错峰流:段A 只已知;之后每段把一个新类加入活跃集,与已知按比例混采。"""
    stream, onset = [], {}
    for _ in range(n_warm):
        c = KNOWN[rng.integers(len(KNOWN))]
        stream.append((CB.make_window(c, rng), c))
    active = []
    for nov in NOVELS:
        active.append(nov)
        onset[nov] = len(stream)
        for _ in range(seg):
            c = active[rng.integers(len(active))] if rng.random() < 0.5 else KNOWN[rng.integers(len(KNOWN))]
            stream.append((CB.make_window(c, rng), c))
    return stream, onset


N_WARM, SEG = 150, 200
NSEED = int(os.environ.get("OVM_NSEED", "5"))


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


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)

    # ---- 预训练:检测器只在 3 个已知类上训(+ replay 池防遗忘) ---- #
    det = make_detector(len(KNOWN))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(2500):
        c = KNOWN[rng.integers(len(KNOWN))]
        Xpt.append(CB.make_window(c, rng)); Ypt.append(onehot(KNOWN.index(c), len(KNOWN)))
    train_on(det, opt, Xpt, Ypt, epochs=30)
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))

    # ---- 错峰流 ---- #
    stream, onset = build_stream(rng, N_WARM, SEG)

    # ---- frozen 基线(在 bootstrap 改动 det 之前算) ---- #
    froz_correct = []
    for x, true_c in stream:
        p = proba(det, [x])[0]
        froz_correct.append(int(KNOWN[int(np.argmax(p))] == true_c))   # 只能在 3 类里选 → 新类必错

    # ---- bootstrap 闭环 agent ---- #
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(KNOWN)
    buf, rec, vocab_trace = [], [], []
    pending = 0
    for i, (x, true_c) in enumerate(stream):
        ev = CB.evidence(x)
        p = proba(det, [x])[0]
        mx = float(p.max()); pred_idx = int(np.argmax(p))
        susp = suspect_novel(ev, mu, sd)
        mislabel = (vocab[pred_idx] in KNOWN) and susp                 # 自信预测已知,但证据指向新类
        audit = rng.random() < AUDIT
        if mx >= TAU and not mislabel and not audit:                   # 有把握、非疑似新类、未抽审计
            pred, src, llm = vocab[pred_idx], "det", 0
        else:                                                          # 叫 LLM(强制 top-1)
            c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            c = None if c == "__ERROR__" else c
            pred = c if c else vocab[pred_idx]
            src, llm = "llm", 1
            # 长新类别需**证据门控背书**:仅当 susp(证据判新颖)且 LLM 给出未知名 → 才建类。
            # 否则(非疑似窗 LLM 误命名)退回检测器/已知,杜绝 warm-up 伪类污染词表。
            if pred not in vocab:
                if susp:
                    vocab.append(pred); grow_head(det, len(vocab))
                    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                else:
                    pred = vocab[pred_idx]                             # 不建类,退回检测器预测
            buf.append((x, vocab.index(pred))); pending += 1
            if pending >= RETRAIN_EVERY:                               # 类平衡在线重训
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
        rec.append(dict(true=true_c, pred=pred, src=src, correct=int(pred == true_c),
                        llm=llm, vsize=len(vocab)))
        vocab_trace.append(len(vocab))

    # ---- 本 seed 汇总 ---- #
    last_start = onset[NOVELS[-1]]
    fa = float(np.mean(froz_correct[last_start:]))
    ba = float(np.mean([r["correct"] for r in rec[last_start:]]))
    llm_overall = float(np.mean([r["llm"] for r in rec[N_WARM:]]))
    per_novel = {}
    for nov in NOVELS:
        idx = [i for i, (_, c) in enumerate(stream) if c == nov]
        segs = [s for s in np.array_split(idx, 4) if len(s)]
        curve = [float(np.mean([rec[i]["correct"] for i in s])) for s in segs]
        per_novel[nov] = dict(curve=curve, bootstrap=float(np.mean([rec[i]["correct"] for i in idx])),
                              frozen=float(np.mean([froz_correct[i] for i in idx])), n=len(idx))
    segs = np.array_split(np.arange(len(stream)), 10)
    llm_curve = [float(np.mean([rec[i]["llm"] for i in s])) for s in segs]
    voc_curve = [int(vocab_trace[s[-1]]) for s in segs]
    return dict(frozen_lastseg=fa, bootstrap_lastseg=ba, llm_overall=llm_overall,
                final_vocab_size=vocab_trace[-1], grew=int(vocab_trace[-1] >= len(KNOWN) + len(NOVELS)),
                per_novel=per_novel, llm_curve=llm_curve, vocab_curve=voc_curve, onsets=onset)


def ms(xs):
    a = np.array(xs, float); return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    print(f"device={device} net_ok={net_ok} NSEED={NSEED}\n  known(预训练)={KNOWN}\n"
          f"  novels(错峰涌现)={NOVELS}\n  known_stats={sorted(KNOWN_STATS)} TAU={TAU}")
    res = []
    for s in range(NSEED):
        r = run_seed(s, key, net_ok)
        res.append(r)
        pn = r["per_novel"]
        print(f"[seed {s}] frozen={r['frozen_lastseg']:.0%} boot={r['bootstrap_lastseg']:.0%} "
              f"grew={r['grew']} | " + " ".join(f"{n[:4]}={pn[n]['bootstrap']:.0%}" for n in NOVELS))

    fm, fs = ms([r["frozen_lastseg"] for r in res])
    bm, bs = ms([r["bootstrap_lastseg"] for r in res])
    lm, ls = ms([r["llm_overall"] for r in res])
    print("\n" + "=" * 80)
    print(f"全段(3 新类都活跃)分类准确率  (mean±std over {NSEED} seeds):")
    print(f"  frozen    = {fm:.1%} ± {fs:.1%}")
    print(f"  bootstrap = {bm:.1%} ± {bs:.1%}")
    print(f"涌现后整体 LLM 调用率 = {lm:.1%} ± {ls:.1%}   词表长全率 grew = "
          f"{np.mean([r['grew'] for r in res]):.0%}")
    print("\n%-20s %-16s %-22s" % ("新类型", "frozen", "bootstrap (mean±std)"))
    print("-" * 60)
    agg_novel = {}
    for nov in NOVELS:
        fnm, _ = ms([r["per_novel"][nov]["frozen"] for r in res])
        bnm, bns = ms([r["per_novel"][nov]["bootstrap"] for r in res])
        curve_m = np.mean([r["per_novel"][nov]["curve"] for r in res], 0)
        agg_novel[nov] = dict(frozen_mean=fnm, bootstrap_mean=bnm, bootstrap_std=bns,
                              curve_mean=[float(v) for v in curve_m])
        print("%-20s %-16s %-22s curve=%s" % (
            nov, f"{fnm:.0%}", f"{bnm:.0%} ± {bns:.0%}", [round(float(v), 2) for v in curve_m]))
    llm_curve_m = np.mean([r["llm_curve"] for r in res], 0)
    voc_curve_m = np.mean([r["vocab_curve"] for r in res], 0)
    print(f"\nLLM 调用率随时间(10 段, mean): {[round(float(v), 2) for v in llm_curve_m]}")
    print(f"词表大小随时间(10 段, mean):   {[round(float(v), 1) for v in voc_curve_m]}  "
          f"(应 {len(KNOWN)}→{len(KNOWN) + len(NOVELS)})")
    if net_ok and bm > fm + 0.1 and all(agg_novel[n]["bootstrap_mean"] > 0.3 for n in NOVELS):
        print("\n结论:✅ 多 seed 稳定:闭环无标注地逐个长出 3 个新类、词表阶梯增长 —— 开放词表自举对多类型成立。")
    print("=" * 80)

    out = dict(nseed=NSEED,
               frozen_lastseg=dict(zip(("mean", "std"), ms([r["frozen_lastseg"] for r in res]))),
               bootstrap_lastseg=dict(zip(("mean", "std"), ms([r["bootstrap_lastseg"] for r in res]))),
               llm_overall=dict(zip(("mean", "std"), ms([r["llm_overall"] for r in res]))),
               per_novel=agg_novel, llm_curve_mean=[float(v) for v in llm_curve_m],
               vocab_curve_mean=[float(v) for v in voc_curve_m], per_seed=res)
    json.dump(out, open(output_path("openvocab_multi_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
