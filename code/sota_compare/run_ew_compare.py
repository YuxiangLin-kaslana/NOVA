#!/usr/bin/env python3
"""【类型化早期预警 · SOTA 同口径对比】

回答:对**从未见过的新异常类型**,前人 SOTA 能否在前兆窗内提前报出**正确的新类型**?
复用 scripts.exp_early_warning 的事件流(前兆窗携弱化签名)、ew_eval、frozen/bootstrap 原语。

  Frozen-CNN / AnomalyTransformer / MemStream / Ours(bootstrap)

指标:
  - 类型化 EW recall(headline):前兆窗内报出**正确新类型 T**。无类型概念的检测器恒 0。
  - 二分类 EW recall(对照):前兆窗内报"有异常"(强信号 SOTA 也能,但报不出类型)。
  - lead-time、type-FAR(正常被误报为 T)、binary-FAR。
用法: sbatch sota_compare/run_ew.sh  (env CMP_NSEED 默认 3,OVE_NOVEL 默认 trend)
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
import scripts.exp_early_warning as EW          # 复用事件流/ew_eval/frozen+bootstrap  # noqa: E402
import sigla_exp.ovbench as CB                  # noqa: E402
from sota_compare.baselines import MemStream, AnomalyTransformer  # noqa: E402

NSEED = int(os.environ.get("CMP_NSEED", "3"))
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
device = EW.device
N_PT = 360 if SMOKE else 3600
N_NORM_TR = 200 if SMOKE else 1500
UNSUP_EP = 3 if SMOKE else 40


def bg_mask(lab, onsets):
    """正常背景窗掩码(排除前兆窗与事件窗),用于算 FAR(复用 EW 口径)。"""
    is_bg = (lab == 0).copy()
    for t, _ in onsets:
        is_bg[max(0, t - EW.L_MAX): t] = False
    return is_bg


def run_unsup_typed_binary(model, normal_train, normal_cal, W, is_bg):
    """无监督 SOTA:正常窗训练→q95 标定→逐窗分数。typed 报警恒 0(无类型),binary=score>thresh。"""
    model.fit(normal_train)
    thresh = float(np.quantile(model.score_stream(normal_cal, update=False), 0.95))
    scores = model.score_stream(W, update=True)
    bin_alarm = (scores > thresh).astype(int)
    typed_alarm = np.zeros(len(W), int)            # 无类型概念 → 永远报不出 T
    return typed_alarm, bin_alarm, float(np.mean(bin_alarm[is_bg])) if is_bg.any() else 0.0


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)

    det = EW.make_detector(len(EW.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(N_PT):
        c = EW.BASE_VOCAB[rng.integers(len(EW.BASE_VOCAB))]
        if c == EW.NORMAL:
            Xpt.append(CB.make_window_strength(None, rng))
        else:
            s = float(rng.uniform(EW.PREC_STR - 0.05, 1.0))
            Xpt.append(CB.make_window_strength(c, rng, s))
        Ypt.append(EW.onehot(EW.BASE_VOCAB.index(c), len(EW.BASE_VOCAB)))
    EW.train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))

    cal = [CB.make_window_strength(None, rng) for _ in range(400)]
    thresh = float(np.quantile(1.0 - EW.proba(det, cal)[:, 0], 0.95))

    W, lab, onsets = EW.build_stream(rng)
    is_bg = bg_mask(lab, onsets)
    far = lambda a: float(np.mean(a[is_bg])) if is_bg.any() else 0.0

    # frozen + bootstrap(EW 原语)
    fpred, fscore, _ = EW.run_detector_stream(pre_state, W, mu, sd, key, net_ok, False, replay, thresh)
    bpred, bscore, vocab = EW.run_detector_stream(pre_state, W, mu, sd, key, net_ok, True, replay, thresh)

    def typed(pred): return np.array([p == EW.NOVEL for p in pred], int)
    arms = {}
    # frozen / bootstrap:typed = 预测==NOVEL;binary = 校准 score 超阈
    for name, pred, score in [("frozen", fpred, fscore), ("bootstrap", bpred, bscore)]:
        ta, ba = typed(pred), (np.asarray(score) > thresh).astype(int)
        arms[name] = dict(typed=EW.ew_eval(ta, onsets, EW.NOVEL),
                          binary=EW.ew_eval(ba, onsets, EW.NOVEL),
                          type_far=far(ta), bin_far=far(ba))

    # 无监督 SOTA:正常窗训练(纯 normal)
    rng2 = np.random.default_rng(20_000 + seed)
    normal_train = [CB.make_window_strength(None, rng2) for _ in range(N_NORM_TR)]
    normal_cal = [CB.make_window_strength(None, rng2) for _ in range(400)]
    for cls, nm in [(AnomalyTransformer, "anomaly_transformer"), (MemStream, "memstream")]:
        m = cls(CB.WIN, CB.NVARS, device, epochs=UNSUP_EP, seed=seed)
        ta, ba, bfar = run_unsup_typed_binary(m, normal_train, normal_cal, W, is_bg)
        arms[nm] = dict(typed=EW.ew_eval(ta, onsets, EW.NOVEL),
                        binary=EW.ew_eval(ba, onsets, EW.NOVEL),
                        type_far=far(ta), bin_far=bfar)
    arms["grew"] = int(EW.NOVEL in vocab)
    return arms


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


META = {
    "frozen":              ("closed-set 分类器", 0, "已知类型"),
    "anomaly_transformer": ("ICLR'22 无监督SOTA(冻结)", 0, "无(仅分数)"),
    "memstream":           ("WWW'22 抗漂移SOTA(在线)", 0, "无(仅分数)"),
    "bootstrap":           ("Ours: LLM开放词表闭环", 0, "已知+自举新类型"),
}
ARMS = list(META)


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key) and not SMOKE
    print(f"device={device} net_ok={net_ok} SMOKE={SMOKE} NSEED={NSEED}  novel={EW.NOVEL}  "
          f"L=[{EW.L_MIN},{EW.L_MAX}] PREC_STR={EW.PREC_STR}")
    res = []
    for s in range(NSEED):
        r = run_seed(s, key, net_ok)
        res.append(r)
        line = " | ".join(f"{a[:4]} typedEW={r[a]['typed']['ew_recall']:.0%}" for a in ARMS)
        print(f"[seed {s}] {line}  (grew={r['grew']})")

    print("\n" + "=" * 96)
    print(f"前兆窗早预警 SOTA 对比(mean±std over {NSEED} seeds, novel={EW.NOVEL}):\n")
    hdr = f"{'method':22s}{'类型概念':>14s}{'类型化EW召回':>14s}{'lead(窗)':>10s}{'type-FAR':>10s}{'二分类EW':>10s}"
    print(hdr); print("-" * 96)
    for a in ARMS:
        _, _, hastype = META[a]
        tr, ts = ms([r[a]["typed"]["ew_recall"] for r in res])
        ld, _ = ms([r[a]["typed"]["lead_mean"] for r in res])
        tf, _ = ms([r[a]["type_far"] for r in res])
        br, _ = ms([r[a]["binary"]["ew_recall"] for r in res])
        print(f"{a:22s}{hastype:>14s}{tr*100:>9.0f}±{ts*100:<3.0f}{ld:>9.1f}{tf*100:>9.1f}%{br*100:>9.0f}%")
    print("-" * 96)
    print("判读:类型化早预警上,frozen 与无监督 SOTA 全为 0(没有 T 类,命名不出新类型);")
    print("二分类早预警上强信号 novel 大家都能报'有异常',但**只有 ours 报得出是哪种新类型** → 这正是价值所在。")
    print(f"词表长全率(ours): {np.mean([r['grew'] for r in res]):.0%}")
    print("=" * 96)
    outp = output_path("sota_ew_compare.json")
    json.dump(dict(nseed=NSEED, novel=EW.NOVEL, per_seed=res, meta=META), open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
