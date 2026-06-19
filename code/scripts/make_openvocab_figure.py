#!/usr/bin/env python3
"""把 A(单新类分类)/ B(多新类分类)/ D(检测桥)三段结果做成一张论文图。
读取 runs/openvocab_loop_result.json、openvocab_multi_result.json、detection_tie_result.json,
输出 slide_figures/06_openvocab_results.png(6 子图)。
纯绘图(读 JSON),用 ragenv2 环境的 python 跑:
  /u/ylin30/.conda/envs/ragenv2/bin/python scripts/make_openvocab_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT.parent / "slide_figures" / "06_openvocab_results.png"   # 顶层 sigLA/slide_figures(与 01–05 同处)

C_FROZ, C_BOOT, C_ORA, C_LLM = "#9aa0a6", "#1a73e8", "#34a853", "#ea8600"


def load(name):
    return json.load(open(RUNS / name))


def ms(seq, f, b):
    """从 per_seed 取 frozen/bootstrap 某指标的 mean,std。"""
    fa = np.array([s["frozen"][f] for s in seq], float)
    ba = np.array([s["bootstrap"][b] for s in seq], float)
    return (np.nanmean(fa), np.nanstd(fa)), (np.nanmean(ba), np.nanstd(ba))


def main():
    A = load("openvocab_loop_result.json")
    B = load("openvocab_multi_result.json")
    D = load("detection_tie_result.json")
    E = load("early_warning_result.json")
    fig, ax = plt.subplots(2, 4, figsize=(21, 8.6))
    fig.suptitle("LLM-bootstrapped open-vocabulary anomaly learning: "
                 "classification (A,B) → detection (D) → early warning (E)",
                 fontsize=14, fontweight="bold")

    # ---- (1) Task A: 4-arm 后段整体准确率 ---- #
    a = ax[0, 0]
    arms = [("frozen", "frozen_p2", C_FROZ), ("bootstrap\n(ours)", "bootstrap_p2", C_BOOT),
            ("oracle\n(human UB)", "oracle_p2", C_ORA), ("LLM-only", "llm_only_acc", C_LLM)]
    xs = np.arange(len(arms))
    a.bar(xs, [A[k]["mean"] for _, k, _ in arms], yerr=[A[k]["std"] for _, k, _ in arms],
          color=[c for *_, c in arms], capsize=4)
    a.set_xticks(xs); a.set_xticklabels([n for n, *_ in arms], fontsize=9)
    a.set_ylim(0, 1); a.set_ylabel("post-emergence accuracy")
    a.set_title("(A) Single novel type: arm comparison\n(5 seeds, ±std)", fontsize=11)
    for x, (_, k, _) in zip(xs, arms):
        a.text(x, A[k]["mean"] + 0.03, f"{A[k]['mean']:.0%}", ha="center", fontsize=9)

    # ---- (2) Task A: 新类准确率爬升 + LLM 成本衰减 ---- #
    a = ax[0, 1]
    nov = A["nov_curve"]["mean"]; llm = A["llm_curve"]["mean"]
    t = np.arange(1, len(nov) + 1)
    a.plot(t, nov, "-o", color=C_BOOT, label="novel-class accuracy")
    a.plot(np.arange(1, len(llm) + 1), llm, "-s", color=C_LLM, label="LLM call rate")
    a.set_ylim(0, 1); a.set_xlabel("time segment (post-emergence)")
    a.set_title("(A) Detector learns → LLM cost decays", fontsize=11)
    a.legend(fontsize=9, loc="center right")

    # ---- (3) Task B: per-novel frozen vs bootstrap ---- #
    a = ax[0, 2]
    novels = ["variance_burst", "trend", "correlation_break"]
    xs = np.arange(len(novels)); w = 0.38
    fzn = [B["per_novel"][n]["frozen_mean"] for n in novels]
    bst = [B["per_novel"][n]["bootstrap_mean"] for n in novels]
    bse = [B["per_novel"][n]["bootstrap_std"] for n in novels]
    a.bar(xs - w / 2, fzn, w, color=C_FROZ, label="frozen")
    a.bar(xs + w / 2, bst, w, yerr=bse, color=C_BOOT, capsize=4, label="bootstrap (ours)")
    a.set_xticks(xs); a.set_xticklabels([n.replace("_", "\n") for n in novels], fontsize=9)
    a.set_ylim(0, 1.05); a.set_ylabel("classification accuracy")
    a.set_title("(B) Three staggered novel types\n(5 seeds, ±std)", fontsize=11)
    a.legend(fontsize=9)
    for x, v in zip(xs, bst):
        a.text(x + w / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)

    # ---- (4) Task B: 词表阶梯增长 + LLM 率 ---- #
    a = ax[1, 0]
    voc = B["vocab_curve_mean"]; lc = B["llm_curve_mean"]
    t = np.arange(1, len(voc) + 1)
    a.step(t, voc, where="mid", color="#673ab7", label="vocabulary size")
    a.set_ylabel("vocabulary size", color="#673ab7"); a.set_ylim(2.5, 6.5)
    a.set_xlabel("time segment"); a.set_title("(B) Vocabulary grows 3→6, no labels", fontsize=11)
    a2 = a.twinx()
    a2.plot(np.arange(1, len(lc) + 1), lc, "-s", color=C_LLM, label="LLM call rate")
    a2.set_ylabel("LLM call rate", color=C_LLM); a2.set_ylim(0, 1)
    a.legend(fontsize=8, loc="upper left"); a2.legend(fontsize=8, loc="center right")

    # ---- (5) Task D: 新类检测召回(桥的 headline) ---- #
    a = ax[1, 1]
    seq = D["per_seed"]
    (fnr_m, fnr_s), (bnr_m, bnr_s) = ms(seq, "nov_recall", "nov_recall")
    a.bar([0, 1], [fnr_m, bnr_m], yerr=[fnr_s, bnr_s], color=[C_FROZ, C_BOOT], capsize=5, width=0.55)
    a.set_xticks([0, 1]); a.set_xticklabels(["frozen\n(closed-set)", "bootstrap\n(ours)"], fontsize=9)
    a.set_ylim(0, 1.05); a.set_ylabel("novel-type DETECTION recall")
    a.set_title("(D) Closed-set detector is BLIND to novel\n(recall≈0); loop recovers it", fontsize=11)
    a.text(0, fnr_m + 0.04, f"{fnr_m:.0%}", ha="center", fontsize=10)
    a.text(1, bnr_m + 0.04, f"{bnr_m:.0%}", ha="center", fontsize=10)

    # ---- (6) Task D: 整体检测 P/R/F1 ---- #
    a = ax[1, 2]
    mets = [("precision", "prec"), ("recall", "rec"), ("F1", "f1")]
    xs = np.arange(len(mets)); w = 0.38
    fz = [np.nanmean([s["frozen"][k] for s in seq]) for _, k in mets]
    bz = [np.nanmean([s["bootstrap"][k] for s in seq]) for _, k in mets]
    fze = [np.nanstd([s["frozen"][k] for s in seq]) for _, k in mets]
    bze = [np.nanstd([s["bootstrap"][k] for s in seq]) for _, k in mets]
    a.bar(xs - w / 2, fz, w, yerr=fze, color=C_FROZ, capsize=4, label="frozen")
    a.bar(xs + w / 2, bz, w, yerr=bze, color=C_BOOT, capsize=4, label="bootstrap (ours)")
    a.set_xticks(xs); a.set_xticklabels([n for n, _ in mets], fontsize=9)
    a.set_ylim(0, 1.05); a.set_ylabel("overall detection (anomaly vs normal)")
    a.set_title("(D) Overall detection: recall↑ (precision is the cost)", fontsize=11)
    a.legend(fontsize=9, loc="lower left")

    # ---- (7) Task E: 类型化早预警 recall(headline) ---- #
    seqE = E["per_seed"]
    def em(k, sub=None):
        vals = [(s[k][sub] if sub else s[k]) for s in seqE]
        return float(np.mean(vals)), float(np.std(vals))
    a = ax[0, 3]
    et_f, et_fs = em("novel_froz", "ew_recall"); et_b, et_bs = em("novel_boot", "ew_recall")
    lead, _ = em("novel_boot", "lead_mean"); tfar, _ = em("typefar_boot")
    a.bar([0, 1], [et_f, et_b], yerr=[et_fs, et_bs], color=[C_FROZ, C_BOOT], capsize=5, width=0.55)
    a.set_xticks([0, 1]); a.set_xticklabels(["frozen\n(closed-set)", "bootstrap\n(ours)"], fontsize=9)
    a.set_ylim(0, 1.12); a.set_ylabel("TYPED early-warning recall\n(name novel type in precursor window)")
    a.set_title("(E) Closed-set can never NAME a novel\ntype early; loop warns it typed", fontsize=11)
    a.text(0, et_f + 0.04, f"{et_f:.0%}", ha="center", fontsize=10)
    a.text(1, et_b + 0.04, f"{et_b:.0%}", ha="center", fontsize=10)
    a.text(1, 0.5, f"lead-time\n{lead:.0f} windows\ntype-FAR {tfar:.0%}", ha="center", va="center",
           fontsize=8.5, color="white", fontweight="bold")

    # ---- (8) Task E: 类型化 vs 二分类(价值在"类型化") ---- #
    a = ax[1, 3]
    eb_f, _ = em("novel_froz_bin", "ew_recall"); eb_b, _ = em("novel_boot_bin", "ew_recall")
    xs = np.arange(2); w = 0.38
    a.bar(xs - w / 2, [et_f, eb_f], w, color=C_FROZ, label="frozen")
    a.bar(xs + w / 2, [et_b, eb_b], w, color=C_BOOT, label="bootstrap (ours)")
    a.set_xticks(xs); a.set_xticklabels(["TYPED\n(which type?)", "binary\n(any anomaly?)"], fontsize=9)
    a.set_ylim(0, 1.12); a.set_ylabel("early-warning recall")
    a.set_title("(E) Value is TYPED warning: closed-set\nbinary-detects but mislabels the type", fontsize=11)
    a.legend(fontsize=8, loc="lower left")
    for x, v in zip(xs - w / 2, [et_f, eb_f]):
        a.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)
    for x, v in zip(xs + w / 2, [et_b, eb_b]):
        a.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")
    # 同时打印关键数字便于核对
    print(f"A: frozen {A['frozen_p2']['mean']:.1%} boot {A['bootstrap_p2']['mean']:.1%} "
          f"oracle {A['oracle_p2']['mean']:.1%} llm_only {A['llm_only_acc']['mean']:.1%}")
    print(f"B: frozen {B['frozen_lastseg']['mean']:.1%} boot {B['bootstrap_lastseg']['mean']:.1%}")
    print(f"D: novel detection recall frozen {fnr_m:.0%}±{fnr_s:.0%} boot {bnr_m:.0%}±{bnr_s:.0%}")
    print(f"E: typed early-warning recall frozen {et_f:.0%} boot {et_b:.0%} "
          f"(lead {lead:.0f}w, type-FAR {tfar:.0%}); binary EW frozen {eb_f:.0%} boot {eb_b:.0%}")


if __name__ == "__main__":
    main()
