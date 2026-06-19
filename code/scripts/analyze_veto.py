#!/usr/bin/env python3
"""决定性实验分析:量化 GPT decider 的否决行为与对 precision 的拯救。

读取两臂的 metrics + agent 臂逐窗 CSV,回答论文核心问题:
  在高 recall 工作点上,改写后的 agent 是否真的否决了假阳性候选、保住了真阳性,
  把 precision 救回来 —— 这是单标量阈值做不到的。

用法: python scripts/analyze_veto.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEC = ROOT / "runs" / "online" / "decisive"


def load_metrics(name: str) -> dict:
    # Tolerate a stray trailing brace ("Extra data"): decode just the first object.
    text = open(DEC / f"{name}.json").read()
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


def read_preds(name: str) -> list[dict]:
    with open(DEC / f"pred_{name}.csv") as f:
        return list(csv.DictReader(f))


def fmt(m: dict) -> str:
    o = m["overall"]
    return (f"P/R/F1 = {o['precision']:.3f} / {o['recall']:.3f} / {o['f1']:.3f}  "
            f"(tp={o['tp']} fp={o['fp']} fn={o['fn']}, positives={o['positives']})")


def binmetrics(rows: list[dict], pred_fn) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        y = int(r["label"]); p = 1 if pred_fn(r) else 0
        if y and p: tp += 1
        elif not y and p: fp += 1
        elif not y and not p: tn += 1
        else: fn += 1
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


SOT_GATE = 1.3  # score_over_threshold confirm gate (matches the decider prompt)


def main() -> None:
    thr = load_metrics("thr_q97")
    agent = load_metrics("agent_q97")

    print("=" * 72)
    print("决定性实验 @ q=0.97 / margin=1.0 (高 recall 工作点)")
    print("=" * 72)
    print(f"[基线] 纯校准阈值 (无 GPT)      {fmt(thr)}")
    print(f"[决定] 校准 + GPT 怀疑者 decider  {fmt(agent)}")
    print(f"       GPT 调用 = {agent['agent_calls']}/{agent['n_windows']} "
          f"({agent['agent_call_rate']:.1%})")
    do, to = agent["overall"], thr["overall"]
    print(f"\nΔ precision = {do['precision'] - to['precision']:+.3f}   "
          f"Δ recall = {do['recall'] - to['recall']:+.3f}   "
          f"Δ F1 = {do['f1'] - to['f1']:+.3f}")

    # ---- 逐窗否决行为 ---- #
    rows = read_preds("agent_q97")
    called = [r for r in rows if int(r["agent_called"]) == 1]
    cand_called = [r for r in called if int(r["candidate"]) == 1]
    vetoes = [r for r in cand_called if int(r["is_anomaly"]) == 0]          # cand=1 -> normal
    confirms = [r for r in cand_called if int(r["is_anomaly"]) == 1]        # cand=1 -> anomaly
    promotions = [r for r in called if int(r["candidate"]) == 0 and int(r["is_anomaly"]) == 1]

    veto_fp = sum(1 for r in vetoes if int(r["label"]) == 0)   # 正确否决(杀掉真 FP)
    veto_tp = sum(1 for r in vetoes if int(r["label"]) == 1)   # 误杀(杀掉真 TP)
    conf_tp = sum(1 for r in confirms if int(r["label"]) == 1)
    conf_fp = sum(1 for r in confirms if int(r["label"]) == 0)

    n_cand = sum(1 for r in rows if int(r["candidate"]) == 1)
    print("\n" + "-" * 72)
    print(f"agent 在 {len(cand_called)} 个被调用的候选窗上的决策(总候选 {n_cand}):")
    print(f"  否决候选 (cand=1 -> normal): {len(vetoes):4d}   "
          f"其中 杀掉真FP={veto_fp}  误杀真TP={veto_tp}")
    print(f"  确认候选 (cand=1 -> anomaly):{len(confirms):4d}   "
          f"其中 真TP={conf_tp}  假FP={conf_fp}")
    print(f"  提拔正常 (cand=0 -> anomaly):{len(promotions):4d}")
    veto_rate = len(vetoes) / max(1, len(cand_called))
    print(f"  veto rate = {veto_rate:.1%}")
    if vetoes:
        precision_of_vetoes = veto_fp / len(vetoes)
        print(f"  否决精度(否决里确为FP的比例) = {precision_of_vetoes:.1%}  "
              f"-> 每误杀 1 个 TP 换掉 {veto_fp / max(1, veto_tp):.1f} 个 FP")

    # ---- 纯阈值反事实对照(关键:证明 agent 不只是更紧的阈值) ---- #
    has_sot = rows and "score_over_threshold" in rows[0]
    if has_sot:
        def sot(r): return float(r["score_over_threshold"])
        ctf = binmetrics(rows, lambda r: int(r["candidate"]) == 1 and sot(r) >= SOT_GATE)
        print("\n" + "-" * 72)
        print(f"纯阈值反事实 @ score_over_threshold>={SOT_GATE}(等价于把 margin 收到 {SOT_GATE},无概念信息):")
        print(f"  P/R/F1 = {ctf['precision']:.3f} / {ctf['recall']:.3f} / {ctf['f1']:.3f}  "
              f"(tp={ctf['tp']} fp={ctf['fp']} fn={ctf['fn']})")
        print(f"  agent 相对该反事实: ΔP={do['precision']-ctf['precision']:+.3f}  "
              f"ΔR={do['recall']-ctf['recall']:+.3f}  ΔF1={do['f1']-ctf['f1']:+.3f}")

        # agent 确认的候选里:高分(阈值也会留) vs 低分被概念救回(只有 LLM 会留)
        kept = [r for r in confirms]
        kept_hi = [r for r in kept if sot(r) >= SOT_GATE]
        kept_lo = [r for r in kept if sot(r) < SOT_GATE]   # 低分但被 agent 确认 = 概念救回
        rescue_tp = sum(1 for r in kept_lo if int(r["label"]) == 1)
        rescue_fp = sum(1 for r in kept_lo if int(r["label"]) == 0)
        print(f"  agent 确认的 {len(kept)} 候选: 高分(≥{SOT_GATE},阈值也会留) {len(kept_hi)}  |  "
              f"低分被概念救回(<{SOT_GATE}) {len(kept_lo)}  其中真TP={rescue_tp} 假FP={rescue_fp}")
        print(f"  -> 这 {rescue_tp} 个低分真异常正是纯阈值(收到{SOT_GATE})会漏掉、而 LLM 用概念证据保住的 recall。")

    # ---- 结论 ---- #
    print("\n" + "=" * 72)
    beats_threshold_ceiling = do["f1"] >= 0.72
    if beats_threshold_ceiling and do["recall"] > to["recall"] - 0.06:
        print(f"结论:✅✅ agent F1={do['f1']:.3f} 越过纯阈值天花板 0.72 —— "
              "「agent > 任何单阈值」坐实为硬结论。")
    elif do["precision"] > to["precision"] + 0.02 and do["recall"] > to["recall"] - 0.06:
        print(f"结论:✅ agent 否决假阳性、基本保住 recall,precision 救回(F1={do['f1']:.3f}),"
              "但尚未越过 0.72 天花板,见反事实对照判断概念救回是否足量。")
    elif len(vetoes) == 0:
        print("结论:❌ agent 仍未否决任何候选(橡皮图章未修复),需进一步加强 prompt/信号。")
    else:
        print(f"结论:⚠️ agent 有否决行为但净收益有限(F1={do['f1']:.3f}),见上方 Δ 与误杀数。")
    print("=" * 72)


if __name__ == "__main__":
    main()
