#!/usr/bin/env python3
"""【多数据集 / 多实体 · 新类型检测 SOTA 对比】

把单一(合成)benchmark 扩展到**真实数据背景**:每个实体的真实正常序列做背景,
注入同样的 6 类概念签名,novel=correlation_break 涌现。复用 run_detection_compare.run_seed
(realbench.activate 后 monkeypatch 生效,全链路自动改用真实背景)。

报告:每实体一行 + 跨实体汇总(mean±std)。证明"SOTA 对新类型盲、ours 全面胜出"在真实背景上稳健,
而非合成 benchmark 的产物。
用法:
  SMD: sbatch sota_compare/run_multidata.sh
  PSM: REAL_DATASET=PSM REAL_ENTITIES=psm sbatch sota_compare/run_multidata.sh
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sota_compare.realbench as RB              # noqa: E402
import sota_compare.run_detection_compare as RDC  # noqa: E402

SMOKE = RDC.SMOKE
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
DATASET = os.environ.get("REAL_DATASET", "SMD")
DEFAULT_ENTITIES = ["1-1"] if SMOKE else ["1-1", "2-1", "3-1", "1-6", "2-5", "3-7"]
if DATASET.upper() == "PSM":
    DEFAULT_ENTITIES = ["psm"]
ENTITIES = os.environ.get("REAL_ENTITIES", os.environ.get("REAL_MACHINES", ",".join(DEFAULT_ENTITIES))).split(",")
ARMS, META, ms = RDC.ARMS, RDC.META, RDC.ms


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


def agg_arm(res, arm, field):
    return ms([r[arm][field] for r in res])


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key) and not SMOKE
    print(f"device={RDC.device} net_ok={net_ok} SMOKE={SMOKE} NSEED={NSEED} novel={RDC.DT.NOVEL}")
    print(f"dataset={DATASET} entities({len(ENTITIES)}): {ENTITIES}\n")

    per_entity = {}
    for ent in ENTITIES:
        ent = ent.strip()
        Z = RB.activate(ent, dataset=DATASET)                    # 真实背景生效
        res = [RDC.run_seed(s, key, net_ok) for s in range(NSEED)]
        per_entity[ent] = res
        nr, _ = agg_arm(res, "bootstrap", "nov_recall")
        anr, _ = agg_arm(res, "anomaly_transformer", "nov_recall")
        mnr, _ = agg_arm(res, "memstream", "nov_recall")
        fnr, _ = agg_arm(res, "frozen", "nov_recall")
        print(f"[{DATASET}:{ent} | T={len(Z)}] novel检测召回  froz={fnr:.0%} anom={anr:.0%} "
              f"mems={mnr:.0%} boot={nr:.0%}  (新类分类 boot="
              f"{agg_arm(res,'bootstrap','nov_classacc')[0]:.0%})")

    # ---- 跨实体汇总:把所有 (entity,seed) 摊平 ---- #
    flat = {a: {f: [] for f in ("nov_recall", "f1", "nov_classacc")} for a in ARMS}
    for res in per_entity.values():
        for r in res:
            for a in ARMS:
                for f in flat[a]:
                    flat[a][f].append(r[a][f])

    print("\n" + "=" * 100)
    print(f"跨 {len(ENTITIES)} 个真实 {DATASET} 实体 × {NSEED} seeds 汇总(novel={RDC.DT.NOVEL}):\n")
    hdr = f"{'method':22s}{'类型概念':>14s}{'新类检测召回':>16s}{'整体F1':>12s}{'新类分类':>12s}"
    print(hdr); print("-" * 100)
    for a in ARMS:
        _, _, hastype = META[a]
        nr, ns = ms(flat[a]["nov_recall"]); f1, f1s = ms(flat[a]["f1"]); ca, cas = ms(flat[a]["nov_classacc"])
        print(f"{a:22s}{hastype:>14s}{nr*100:>11.0f}±{ns*100:<3.0f}{f1:>9.2f}±{f1s:<4.2f}{ca*100:>8.0f}±{cas*100:<3.0f}")
    print("-" * 100)
    print("判读(诚实版):在真实背景上,无监督 SOTA(尤其 AnomalyTransformer)**能把 novel 检测为'异常'**")
    print("(binary 检测召回不低,且实体间方差大)——'SOTA 检不出 novel'是合成 benchmark 的产物,真实数据不成立。")
    print("**唯一普遍成立的结构性差异:新类分类——所有 SOTA 恒 0(只有分数、无类型),只有 ours >0。**")
    print("即:别人能说'出事了',只有 ours 能说'是哪种没见过的事' → 类型化预警/可操作输出是不可替代的卖点。")
    print("=" * 100)
    suffix = DATASET.lower()
    default_name = f"sota_multidata_compare_{suffix}.json"
    if DATASET.upper() == "SMD":
        default_name = "sota_multidata_compare.json"
    outp = output_path(default_name)
    payload = dict(nseed=NSEED, dataset=DATASET, entities=ENTITIES, novel=RDC.DT.NOVEL,
                   per_entity=per_entity, meta=META)
    if DATASET.upper() == "SMD":
        payload["machines"] = ENTITIES
        payload["per_machine"] = per_entity
    json.dump(payload, open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
