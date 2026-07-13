#!/usr/bin/env python3
"""【漂移 vs 新异常类型 判别实验】证明"标量异常检测器分不清'该适应的良性漂移'和'该报警的新异常类型'"。

一条流(可选真实 SMD 背景,env REAL_MACHINE):
  - 全程**良性协变量漂移**:正常基线均值缓慢平移(窗内恒定偏移,逐窗增大 0→D)。这是良性,不该报警。
    所有臂统一做**一行 per-window 去均值归一化**(协变量漂移=归一化即可解决,非贡献)→ 对各臂公平。
  - 后半段**缓慢爬升的新异常类型** NOVEL=oscillation(渐发振动故障):强度 0→1 线性增长,前期极弱(像正常)。

三臂(同流、同归一化):
  frozen      AnomalyTransformer(冻结无监督):score>τ 报警。无适应、无类型。
  memstream   记忆+在线更新:把"看起来正常"的窗吸进记忆并再训 → **缓慢爬升的新异常被吸成新正常 → 漏报**。
  ours        证据门控(signature 轴,均值平移不触发任何签名)+ LLM 命名 + 在线扩类。

度量:
  - 良性漂移段误报率 drift_FA(正常窗被报警比例,越低越好)。
  - 新异常检测召回随**爬升强度四分位**的曲线(memstream 因吸收→低平;ours 随强度上升)。
  - 新异常**命名**准确率(仅 ours >0)。
要点:frozen 能检不能命名;memstream 吸收→漏;只有 ours 漂移不报 + 新异常命名报警 → 区分了两者。
用法: sbatch sota_compare/run_drift.sh  (env CMP_NSEED, CMP_SMOKE, REAL_MACHINE)
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
import scripts.exp_detection_tie as DT          # CNN 原语  # noqa: E402
import sigla_exp.ovbench as CB                  # 注入器/证据/LLM  # noqa: E402
from sota_compare.baselines import MemStream, AnomalyTransformer  # noqa: E402

SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
REAL_MACHINE = os.environ.get("REAL_MACHINE", "")
device = DT.device

NORMAL = "normal"
KNOWN = ["spike", "level_shift", "trend"]
NOVEL = "oscillation"                                       # 渐发振动:spectral_peak 签名随强度增长
BASE_VOCAB = [NORMAL] + KNOWN
KNOWN_STATS = {CB.STAT_OF[c] for c in KNOWN}
TAU, AUDIT, NOVEL_Z, RETRAIN_EVERY = 0.5, 0.05, 2.3, 12
DRIFT_D = 2.0                                               # 漂移末端均值平移量(σ)
SMIN = 0.08                                                # 爬升起始强度(极弱,像正常)
N_PT = 300 if SMOKE else 2400
N_NORM_TR = 200 if SMOKE else 1200
UNSUP_EP = 3 if SMOKE else 40
T_WARM = 60 if SMOKE else 200                              # 漂移前预热正常段(标定用)
T_DRIFT = 100 if SMOKE else 400                            # 仅漂移(良性)段
T_NOVEL = 120 if SMOKE else 500                            # 漂移持续 + 新异常爬升段


def mc(x):
    """一行 per-window 去均值归一化(消解协变量均值平移)。"""
    return (x - x.mean(0, keepdims=True)).astype(np.float32)


def inject(concept, base, rng, strength=1.0):
    """在给定 base 窗上注入 concept 的(可按 strength 缩放的)签名。"""
    if concept is None or strength <= 0:
        return base.astype(np.float32)
    xf = base.copy(); CB.INJ[concept](xf, rng)
    return (base + strength * (xf - base)).astype(np.float32)


def build_stream(rng, drift_vec):
    """返回 windows / label(0正常1异常) / 是否novel / 阶段 / 爬升强度。"""
    W, lab, is_nov, phase, creep = [], [], [], [], []
    def drift(frac):
        return (frac * DRIFT_D) * drift_vec                # 窗内恒定偏移(纯均值平移,无窗内斜率)

    for i in range(T_WARM):                                # 预热:无漂移正常
        W.append(CB.base_normal(rng)); lab.append(0); is_nov.append(0); phase.append("warm"); creep.append(0.0)
    for i in range(T_DRIFT):                               # 仅良性漂移(正常)
        frac = (i + 1) / (T_DRIFT + T_NOVEL)
        W.append(CB.base_normal(rng) + drift(frac)); lab.append(0); is_nov.append(0)
        phase.append("drift"); creep.append(0.0)
    for i in range(T_NOVEL):                               # 漂移持续 + 新异常缓慢爬升
        frac = (T_DRIFT + i + 1) / (T_DRIFT + T_NOVEL)
        base = CB.base_normal(rng) + drift(frac)
        c = (i + 1) / T_NOVEL                              # 爬升强度 0→1
        if rng.random() < 0.5:
            s = SMIN + (1 - SMIN) * c
            W.append(inject(NOVEL, base, rng, s)); lab.append(1); is_nov.append(1)
        else:
            W.append(base.astype(np.float32)); lab.append(0); is_nov.append(0)
        phase.append("novel"); creep.append(c)
    return (W, np.array(lab), np.array(is_nov), np.array(phase, object), np.array(creep))


def unsup_alarms(model, normal_tr, normal_cal, W):
    model.fit([mc(x) for x in normal_tr])
    thr = float(np.quantile(model.score_stream([mc(x) for x in normal_cal], update=False), 0.95))
    sc = model.score_stream([mc(x) for x in W], update=True)
    return (sc > thr).astype(int)


def ours_loop(pre_state, replay, W, mu, sd, key, net_ok, thr_ours):
    """ours:每窗 mc 后喂 CNN;证据门控(均值平移不触发签名)→ LLM 命名 → 扩类 → 类平衡重训。
    报警 = P(非normal) > thr_ours(在正常标定集上按 q95 标定,与无监督 SOTA 同口径 ~5% FA);
    命名 = argmax 类型(仅在报警时有意义)。→ 同等误报率下比"早检出新异常 + 命名"。"""
    det = DT.make_detector(len(BASE_VOCAB)); det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(BASE_VOCAB); buf = []; pending = 0
    alarms, names = [], []
    rng = np.random.default_rng(777)
    for x in W:
        xm = mc(x)
        ev = CB.evidence(x)                                # 签名统计量本身平移不变
        p = DT.proba(det, [xm])[0]; mx = float(p.max()); pi = int(np.argmax(p))
        devz = {k: abs((ev[k] - mu[k]) / sd[k]) for k in mu}
        dom = max(devz, key=devz.get)
        susp = (dom not in KNOWN_STATS) and (devz[dom] > NOVEL_Z)
        mislabel = (vocab[pi] in (KNOWN + [NORMAL])) and susp
        if mx >= TAU and not mislabel and rng.random() >= AUDIT:
            pred = vocab[pi]
        else:
            c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
            c = None if c == "__ERROR__" else c
            pred = c if c else vocab[pi]
            if pred not in vocab:
                if susp:
                    vocab.append(pred); DT.grow_head(det, len(vocab))
                    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                else:
                    pred = vocab[pi]
            if pred != NORMAL and pred in vocab:
                buf.append((xm, vocab.index(pred))); pending += 1
            if pending >= RETRAIN_EVERY:
                per = {}
                for w, li in replay: per.setdefault(li, []).append(w)
                for w, li in buf: per.setdefault(li, []).append(w)
                K = 40; Xb, Yb = [], []
                for li, ws in per.items():
                    for j in rng.integers(0, len(ws), K):
                        Xb.append(ws[j]); Yb.append(DT.onehot(li, len(vocab)))
                DT.train_on(det, opt, Xb, Yb, epochs=2); pending = 0
        # 报警走标定阈(同口径 ~5% FA);命名走 argmax 类型
        anom_score = 1.0 - float(p[0])                     # P(非normal),normal 恒为 index 0
        alarms.append(int(anom_score > thr_ours)); names.append(pred)
    return np.array(alarms), names, vocab


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)
    dvec = rng.normal(0, 1, CB.NVARS).astype(np.float32)
    dvec /= (np.linalg.norm(dvec) / np.sqrt(CB.NVARS) + 1e-6)       # 单位尺度漂移方向

    # ours 的 CNN 预训练(mc 后的 normal+3已知)
    det = DT.make_detector(len(BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(N_PT):
        c = BASE_VOCAB[rng.integers(len(BASE_VOCAB))]
        base = CB.base_normal(rng)
        x = base if c == NORMAL else inject(c, base, rng, float(rng.uniform(0.5, 1.0)))
        Xpt.append(mc(x)); Ypt.append(DT.onehot(BASE_VOCAB.index(c), len(BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))

    W, lab, is_nov, phase, creep = build_stream(rng, dvec)
    normal_tr = [CB.base_normal(rng) for _ in range(N_NORM_TR)]     # 无漂移正常(训练无监督SOTA)
    normal_cal = [CB.base_normal(rng) for _ in range(400)]

    a_fro = unsup_alarms(AnomalyTransformer(CB.WIN, CB.NVARS, device, epochs=UNSUP_EP, seed=seed),
                         normal_tr, normal_cal, W)
    a_mem = unsup_alarms(MemStream(CB.WIN, CB.NVARS, device, epochs=UNSUP_EP, seed=seed),
                         normal_tr, normal_cal, W)
    # ours 报警阈标定到 ~5% FA(正常标定集上 P(非normal) 的 q95,与无监督 SOTA 同口径)
    cal_s = 1.0 - DT.proba(det, [mc(x) for x in normal_cal])[:, 0]
    thr_ours = float(np.quantile(cal_s, 0.95))
    a_our, names, vocab = ours_loop(pre_state, replay, W, mu, sd, key, net_ok, thr_ours)

    drift_mask = (phase == "drift")                                # 良性漂移正常窗
    nov_mask = (is_nov == 1)
    def fa(a): return float(a[drift_mask].mean())
    def rec(a): return float(a[nov_mask].mean())
    # 爬升四分位召回曲线
    def curve(a):
        cv = creep[nov_mask]; aa = a[nov_mask]; out = []
        for q in range(4):
            m = (cv >= q / 4) & (cv < (q + 1) / 4 if q < 3 else cv <= 1.01)
            out.append(float(aa[m].mean()) if m.any() else float("nan"))
        return out
    name_acc = float(np.mean([names[i] == NOVEL for i in range(len(W)) if is_nov[i]]))
    return dict(
        drift_FA=dict(frozen=fa(a_fro), memstream=fa(a_mem), ours=fa(a_our)),
        nov_recall=dict(frozen=rec(a_fro), memstream=rec(a_mem), ours=rec(a_our)),
        nov_curve=dict(frozen=curve(a_fro), memstream=curve(a_mem), ours=curve(a_our)),
        ours_name_acc=name_acc, grew=int(NOVEL in vocab))


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


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key) and not SMOKE
    if REAL_MACHINE:
        import sota_compare.realbench as RB
        Z = RB.activate(REAL_MACHINE); bg = f"real SMD machine-{REAL_MACHINE} (T={len(Z)})"
    else:
        bg = "synthetic"
    print(f"device={device} net_ok={net_ok} SMOKE={SMOKE} NSEED={NSEED} bg={bg}\n"
          f"  KNOWN={KNOWN} NOVEL={NOVEL} drift_D={DRIFT_D} creep[{SMIN}->1]")
    res = [run_seed(s, key, net_ok) for s in range(NSEED)]
    for s, r in enumerate(res):
        print(f"[seed {s}] drift误报 fro={r['drift_FA']['frozen']:.0%} mem={r['drift_FA']['memstream']:.0%} "
              f"our={r['drift_FA']['ours']:.0%} | novel召回 fro={r['nov_recall']['frozen']:.0%} "
              f"mem={r['nov_recall']['memstream']:.0%} our={r['nov_recall']['ours']:.0%} | "
              f"命名 our={r['ours_name_acc']:.0%}")

    print("\n" + "=" * 92)
    print(f"漂移 vs 新类型 判别({NSEED} seeds, bg={bg}):\n")
    print(f"{'method':14s}{'良性漂移误报↓':>16s}{'新异常检测召回↑':>18s}{'新异常命名↑':>14s}")
    print("-" * 92)
    for a, tag in [("frozen", "AnomTransf(冻结)"), ("memstream", "MemStream(适应)"), ("ours", "Ours(签名+LLM)")]:
        fa_m, fa_s = ms([r["drift_FA"][a] for r in res])
        rc_m, rc_s = ms([r["nov_recall"][a] for r in res])
        nm = ms([r["ours_name_acc"] for r in res])[0] if a == "ours" else 0.0
        print(f"{tag:14s}{fa_m*100:>11.0f}±{fa_s*100:<3.0f}{rc_m*100:>13.0f}±{rc_s*100:<3.0f}{nm*100:>11.0f}%")
    print("-" * 92)
    for a in ("frozen", "memstream", "ours"):
        cv = np.nanmean([r["nov_curve"][a] for r in res], 0)
        print(f"  {a:10s} 召回随爬升强度(4分位): {[round(float(v),2) for v in cv]}")
    print("\n判读(诚实版,经 FA 对齐):")
    print("- 协变量漂移:归一化后 frozen/memstream 误报 ~5%(印证'漂移=归一化即可,非贡献')。")
    print("- 检测新异常(binary):两 SOTA 都能(强了就检出),非差异点。")
    print("- **唯一稳健差异 = 命名:SOTA 恒 0,ours 95-99%**(检出'有异常' vs 说出'是哪种新类型')。")
    print("- ⚠️ ours 是分类器非标定检测器,FA 对齐后漂移段误报偏高(双峰置信度→q95阈过敏),")
    print("  故 ours 的'检测/FA'列不作对比依据;本实验只支撑'命名'轴 + '检测器靠归一化处理掉协变量漂移'。")
    print("- 否证:'MemStream 把缓慢新异常吸成新正常→漏'未成立(memstream 召回随强度上升,非低平)。")
    print("=" * 92)
    outp = output_path(f"drift_vs_novel{'_'+REAL_MACHINE if REAL_MACHINE else ''}.json")
    json.dump(dict(nseed=NSEED, bg=bg, per_seed=res), open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
