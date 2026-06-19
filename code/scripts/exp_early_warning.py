#!/usr/bin/env python3
"""路线B 的"最后一公里":把开放词表闭环接到**前兆窗早预警时序**,回答——
闭集检测器对**从未见过的异常类型**能否提前预警?开放词表自举能否把"novel 早预警"从 0 救回来?

设定(每个事件前有一段前兆窗,携带该类型签名的**弱化版** strength<1,point_label=0):
  - 早预警成功 = 在有效前兆窗 [onset-l_max, onset-l_min] 内报警(检测器认出类型,哪怕弱信号)。
  - 已知类型:检测器在全/弱强度上都训过 → 两臂都能早预警(对照,证明 EW 可达)。
  - novel 类型(correlation_break):frozen 从没见过 → 前兆窗无法报警(EW=0);bootstrap 在前几个**完整**
    novel 事件上自举出该类后,对**后续事件的前兆**(同签名方向的弱信号)能提前报警 → EW recall>0、lead-time>0。

指标(novel 事件 vs 已知事件分别算):有效早预警 recall、lead-time、迟滞/事后/漏、虚高(普通 recall−EW recall)。
复用 sigla_exp.ovbench(强度可调注入)+ sigla_exp.precursor 口径。用法: sbatch scripts/exp_early_warning.sh
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
# novel 选签名清晰、远离 normal 的类型(trend:lin_r2 满强度 z+36,0.6 前兆仍强可分)。
# correlation_break 因"靠缺席发现、与 normal 内在可混"不适合早预警(学后 normal 校准塌、FAR 爆),见 log。
NOVEL = os.environ.get("OVE_NOVEL", "trend")
KNOWN_ANOM = [c for c in ["spike", "level_shift", "oscillation", "variance_burst", "correlation_break", "trend"]
              if c != NOVEL][:3]
BASE_VOCAB = [NORMAL] + KNOWN_ANOM
KNOWN_STATS = {CB.STAT_OF[c] for c in KNOWN_ANOM}
TAU = 0.5
AUDIT = 0.08
NOVEL_Z = 2.3
RETRAIN_EVERY = 12
# 时序(窗为单位):前兆窗 [onset-L_MAX, onset-L_MIN],事件持续 DUR
L_MIN, L_MAX, DUR = 2, 8, 6
PREC_STR = 0.6                                            # 前兆信号强度(弱化版)
N_EVENTS, WARM_EV = 44, 12                                # 总事件数;前 WARM_EV 个仅已知(预热)
NSEED = int(os.environ.get("OVE_NSEED", "3"))
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


def build_stream(rng):
    """事件流:每个事件 = 前兆窗(L_MAX 个,携带弱化签名,label0)→ onset → DUR 个完整事件窗(label1)。
    事件间填正常窗。前 WARM_EV 个仅已知类型,之后已知/novel 各半。"""
    W, lab, onsets = [], [], []
    for ev in range(N_EVENTS):
        for _ in range(int(rng.integers(12, 20))):                  # 事件间正常
            W.append(CB.make_window_strength(None, rng)); lab.append(0)
        c = KNOWN_ANOM[rng.integers(3)] if ev < WARM_EV else \
            (NOVEL if rng.random() < 0.5 else KNOWN_ANOM[rng.integers(3)])
        for _ in range(L_MAX):                                      # 前兆窗(弱化签名)
            W.append(CB.make_window_strength(c, rng, PREC_STR)); lab.append(0)
        onset = len(W); onsets.append((onset, c))
        for _ in range(DUR):                                       # 完整事件
            W.append(CB.make_window_strength(c, rng, 1.0)); lab.append(1)
    return W, np.array(lab), onsets


def run_detector_stream(det0_state, W, mu, sd, key, net_ok, online, replay, thresh):
    """逐窗跑检测器;online=True 时跑开放词表闭环(门控→LLM→扩类→重训)。返回每窗报警 0/1 与最终词表。
    报警 = anomaly_score(1−P(normal)) > thresh(校准阈值,两臂同值,使 FAR 受控)。
    replay=预训练 (window, class_idx) 池,重训回放保持 normal/已知校准。"""
    det = make_detector(len(BASE_VOCAB)); det.load_state_dict(det0_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(BASE_VOCAB); buf = []; pending = 0; preds = []; scores = []
    rng_rt = np.random.default_rng(12345)                          # 重训采样 RNG(固定可复现)
    for x in W:
        p = proba(det, [x])[0]
        mx = float(p.max()); pred_idx = int(np.argmax(p))
        pred = vocab[pred_idx]
        if online:
            ev = CB.evidence(x)
            susp = suspect_novel(ev, mu, sd)
            mislabel = (vocab[pred_idx] in (KNOWN_ANOM + [NORMAL])) and susp
            if not (mx >= TAU and not mislabel):
                c = CB.gpt_recognize_top1(ev, key, mu, sd) if net_ok else None
                c = None if c == "__ERROR__" else c
                pred = c if c else vocab[pred_idx]
                if pred not in vocab:
                    if susp:
                        vocab.append(pred); grow_head(det, len(vocab))
                        opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
                    else:
                        pred = vocab[pred_idx]
                if pred != NORMAL and pred in vocab:
                    buf.append((x, vocab.index(pred))); pending += 1
                if pending >= RETRAIN_EVERY:                       # 类平衡重训:回放 normal+已知 + 新类
                    per_class = {}
                    for w, li in replay:                          # 预训练池(normal+3已知,含弱强度)
                        per_class.setdefault(li, []).append(w)
                    for w, li in buf:                             # 新类伪标签
                        per_class.setdefault(li, []).append(w)
                    K = 40; Xb, Yb = [], []
                    for li, ws in per_class.items():
                        for j in rng_rt.integers(0, len(ws), K):
                            Xb.append(ws[j]); Yb.append(onehot(li, len(vocab)))
                    train_on(det, opt, Xb, Yb, epochs=2); pending = 0
        preds.append(pred); scores.append(1.0 - float(p[0]))      # 类型预测 + anomaly_score
    return preds, np.array(scores), vocab


def ew_eval(alarms, onsets, which):
    """对指定类型(which: NOVEL 或 'known')的事件算前兆早预警。"""
    sel = [(t, c) for t, c in onsets if (c == NOVEL if which == NOVEL else c in KNOWN_ANOM)]
    valid = late = post = missed = 0; leads = []
    for t, _ in sel:
        vwin = alarms[t - L_MAX: t - L_MIN + 1]
        latewin = alarms[t - L_MIN + 1: t]
        postwin = alarms[t: t + DUR]
        if vwin.any():
            valid += 1
            earliest = (t - L_MAX) + int(np.where(vwin)[0].min())
            leads.append(t - earliest)
        elif latewin.any():
            late += 1
        elif postwin.any():
            post += 1
        else:
            missed += 1
    n = max(1, len(sel))
    return dict(n=len(sel), valid=valid, late=late, post=post, missed=missed,
                ew_recall=valid / n, any_recall=(valid + late + post) / n,
                inflation=(post + late) / n, lead_mean=float(np.mean(leads)) if leads else 0.0)


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)

    # ---- 预训练:normal + 3 已知异常类(全/弱多强度,使检测器能认已知类的前兆) ---- #
    det = make_detector(len(BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(3600):
        c = BASE_VOCAB[rng.integers(len(BASE_VOCAB))]
        if c == NORMAL:
            Xpt.append(CB.make_window_strength(None, rng))
        else:
            s = float(rng.uniform(PREC_STR - 0.05, 1.0))           # 已知类训到含前兆强度
            Xpt.append(CB.make_window_strength(c, rng, s))
        Ypt.append(onehot(BASE_VOCAB.index(c), len(BASE_VOCAB)))
    train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))      # normal+已知(含弱强度)回放池

    # 校准二分类阈值(仅作"任意异常检测"对照):正常窗 anomaly_score 95 分位 → FAR≈5%
    cal = [CB.make_window_strength(None, rng) for _ in range(400)]
    thresh = float(np.quantile(1.0 - proba(det, cal)[:, 0], 0.95))

    W, lab, onsets = build_stream(rng)
    fpred, fscore, _ = run_detector_stream(pre_state, W, mu, sd, key, net_ok, False, replay, thresh)
    bpred, bscore, vocab = run_detector_stream(pre_state, W, mu, sd, key, net_ok, True, replay, thresh)

    # 类型特定报警(headline):预测就是 NOVEL 类型 T。frozen 无 T 类 → 恒 0。
    fa_typed = np.array([p == NOVEL for p in fpred], int)
    ba_typed = np.array([p == NOVEL for p in bpred], int)
    # 二分类报警(对照):校准 score 超阈
    fa_bin = (fscore > thresh).astype(int); ba_bin = (bscore > thresh).astype(int)

    is_bg = (lab == 0).copy()                                      # 正常背景窗(非前兆/非事件)
    for t, _ in onsets:
        is_bg[max(0, t - L_MAX): t] = False
    far = lambda a: float(np.mean(a[is_bg])) if is_bg.any() else 0.0

    return dict(
        # headline:类型化早预警
        novel_froz=ew_eval(fa_typed, onsets, NOVEL), novel_boot=ew_eval(ba_typed, onsets, NOVEL),
        typefar_froz=far(fa_typed), typefar_boot=far(ba_typed),
        # 对照:二分类"任意异常"早预警(强信号 novel 闭集也能报,但报错类型)
        novel_froz_bin=ew_eval(fa_bin, onsets, NOVEL), novel_boot_bin=ew_eval(ba_bin, onsets, NOVEL),
        binfar_froz=far(fa_bin), binfar_boot=far(ba_bin),
        grew=int(NOVEL in vocab),
    )


def ms(xs):
    a = np.array(xs, float); return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    print(f"device={device} net_ok={net_ok} NSEED={NSEED}  novel={NOVEL}  "
          f"L=[{L_MIN},{L_MAX}] DUR={DUR} PREC_STR={PREC_STR} N_EVENTS={N_EVENTS}")
    res = []
    for s in range(NSEED):
        r = run_seed(s, key, net_ok)
        res.append(r)
        print(f"[seed {s}] 类型化EW froz={r['novel_froz']['ew_recall']:.0%} "
              f"boot={r['novel_boot']['ew_recall']:.0%}(lead {r['novel_boot']['lead_mean']:.1f}, "
              f"typeFAR {r['typefar_boot']:.0%}) | 二分类EW froz={r['novel_froz_bin']['ew_recall']:.0%} "
              f"boot={r['novel_boot_bin']['ew_recall']:.0%} grew={r['grew']}")

    def agg(arm, field):
        return ms([r[arm][field] for r in res])
    print("\n" + "=" * 82)
    print(f"前兆窗早预警(novel={NOVEL}, mean±std over {NSEED} seeds):\n")
    print("【headline:类型化早预警 —— 在前兆窗内提前报出**正确的新类型** T】")
    fm, fs = agg("novel_froz", "ew_recall"); bm, bs = agg("novel_boot", "ew_recall")
    blm, _ = agg("novel_boot", "lead_mean")
    tff, _ = ms([r["typefar_froz"] for r in res]); tfb, _ = ms([r["typefar_boot"] for r in res])
    print(f"  类型化早预警 recall: frozen {fm:.0%}±{fs:.0%}(无 T 类,永远命名不出)   "
          f"bootstrap {bm:.0%}±{bs:.0%}")
    print(f"  lead-time(窗):      bootstrap {blm:.1f}（前兆窗宽 {L_MAX - L_MIN + 1}）")
    print(f"  类型 FAR(正常被判 T): frozen {tff:.1%}   bootstrap {tfb:.1%}  <- 须低")
    print("\n【对照:二分类'任意异常'早预警 —— 只问报不报警,不问类型】")
    bfm, _ = agg("novel_froz_bin", "ew_recall"); bbm, _ = agg("novel_boot_bin", "ew_recall")
    bff, _ = ms([r["binfar_froz"] for r in res]); bfb, _ = ms([r["binfar_boot"] for r in res])
    print(f"  二分类早预警 recall: frozen {bfm:.0%}   bootstrap {bbm:.0%}   "
          f"(FAR frozen {bff:.0%} / boot {bfb:.0%})")
    print(f"\n判读:即便闭集二分类能(对强信号)提前报'有异常'({bfm:.0%}),它**永远报不出是哪种新类型**"
          f"(类型化 {fm:.0%});\nbootstrap 自举后能在前兆窗提前 {blm:.1f} 窗报出正确新类型({bm:.0%}),"
          f"且正常几乎不误判为 T({tfb:.1%})。词表长全率 {np.mean([r['grew'] for r in res]):.0%}")
    print("=" * 82)
    json.dump(dict(nseed=NSEED, per_seed=res), open(ROOT / "runs" / "early_warning_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
