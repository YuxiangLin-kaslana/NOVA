#!/usr/bin/env python3
"""路线B headline:LLM 自举的开放词表异常分类闭环(agent)—— 加固版(对照臂 + 多 seed 误差棒)。

故事:流式来异常窗。检测器只认训练过的"已知类别";当一个**新类型**涌现,检测器无能为力
→ agent 用证据判据发现疑似新类 → 叫 LLM zero-shot 命名 → 扩一个异常类别 → 用 LLM 判断当
**伪标签**类平衡在线重训 → 检测器自己学会 → 之后不再叫 LLM。**无需人工标注,系统自己长出新类。**

四个对照臂(都从同一预训练检测器出发,逐 seed 重复):
  frozen     闭集检测器,永不扩词表,无 LLM → 新类永远错(下界)。
  bootstrap  本方法:闸门触发→LLM 伪标签→扩词表→在线学。
  oracle     人工标注上界:同一闭环,但闸门触发时用**真值**当标签(无 LLM)→ 衡量 LLM 标注噪声的代价。
  llm_only   每窗都叫 LLM 直接分类,无检测器学习 → 准确率高但**恒 100% 成本**(成本对照)。

脚本复用 exp_novel_concept 的基底/注入器/证据/LLM。用法: sbatch scripts/exp_openvocab_loop.sh
seed 数由环境变量 OVL_NSEED 控制(默认 5);LLM-only 抽样窗数由 OVL_LLMONLY_N 控制(默认 100)。
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
import scripts.exp_novel_concept as NC          # noqa: E402  复用 base/injectors/evidence/LLM

WIN, NVARS = NC.WIN, NC.NVARS
KNOWN = list(NC.KNOWN)           # 进训练的 5 类
NOVEL = NC.NOVEL                 # 留出的新类(涌现)
TAU = 0.5                        # 检测器置信门:max prob < TAU → 视为"不确定 / 可能新类" → 触发标注
AUDIT = 0.08                     # 审计抽样:即便自信也按此比例触发标注(抓"自信误判的新类")
RETRAIN_EVERY = 15               # 每积累这么多新伪标签,在线重训一次
N1, N2 = 200, 500                # 流:前段只已知,后段新类涌现
NSEED = int(os.environ.get("OVL_NSEED", "5"))
LLMONLY_N = int(os.environ.get("OVL_LLMONLY_N", "100"))  # LLM-only 在后段抽样的窗数(控成本)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每个已知概念的"签名统计"。新颖性触发用:窗里没有任何已知签名升高 → 疑似新类型。与检测器置信度无关,对 OOD 可靠。
KNOWN_SIG = {"spike": "max_abs_zscore", "level_shift": "max_step_change",
             "oscillation": "high_freq_energy_frac", "variance_burst": "right_left_std_ratio",
             "trend": "max_linear_slope"}


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
    """把 concept 输出维 +1(增加一个异常类别),旧权重保留,新行随机初始化。"""
    old = det.head[-1]
    new = nn.Linear(old.in_features, new_n).to(device)
    with torch.no_grad():
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
    det.head[-1] = new


def onehot(idx, n):
    v = np.zeros(n, np.float32); v[idx] = 1.0; return v


def suspect_novel(ev, base):
    return not any(ev[s] > base[s] * 1.2 for s in KNOWN_SIG.values())


def llm_name(ev, base, key, net_ok):
    """bootstrap 的**新类命名器**:疑似新类窗(无任何已知签名升高)取 LLM 列表里的**非已知**概念
    (真新类),否则取首个。对微弱新类(如 correlation_break)比 top-1 更可靠。返回 pred 或 None。"""
    got = NC.gpt_recognize(ev, key, base) if net_ok else []
    got = [g for g in got if g != "__ERROR__"]
    if suspect_novel(ev, base):
        cands = [g for g in got if g not in KNOWN_SIG]
        return cands[0] if cands else (got[0] if got else None)
    return got[0] if got else None


def llm_classify_top1(ev, base, key, net_ok):
    """llm_only 基线的**逐窗单标签分类器**:强制 LLM 返回唯一最显著概念(杜绝过度列举)。返回 pred 或 None。"""
    if not net_ok:
        return None
    c = NC.gpt_recognize_top1(ev, key, base)
    return None if (c is None or c == "__ERROR__") else c


def online_arm(pre_state, stream, base, replay, labeler, rng):
    """通用在线闭环。labeler(ev, true_c) -> (pred_label or None, n_label_calls)。
    闸门(不确定/疑似新类/审计)触发时调用 labeler;扩词表 + 类平衡在线重训。返回逐窗 rec。"""
    det = make_detector(len(KNOWN)); det.load_state_dict(pre_state)
    opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
    vocab = list(KNOWN); buf = []; rec = []; pending = 0
    for i, (x, true_c) in enumerate(stream):
        ev = NC.evidence(x)
        p = proba(det, [x])[0]
        mx = float(p.max()); pred_idx = int(np.argmax(p))
        mislabel = (vocab[pred_idx] in KNOWN_SIG) and suspect_novel(ev, base)
        audit = rng.random() < AUDIT
        if mx >= TAU and not mislabel and not audit:          # 信检测器
            pred, src, ncall = vocab[pred_idx], "det", 0
        else:                                                 # 触发标注(LLM 或人工)
            lab, ncall = labeler(ev, true_c)
            pred = lab if lab is not None else vocab[pred_idx]
            src = "label"
            if pred not in vocab:
                vocab.append(pred); grow_head(det, len(vocab))
                opt = torch.optim.AdamW(det.parameters(), lr=3e-4, weight_decay=1e-4)
            buf.append((x, vocab.index(pred))); pending += 1
            if pending >= RETRAIN_EVERY:                      # 类平衡在线重训:每类等量
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
                        ncall=ncall, vsize=len(vocab), phase=int(i >= N1)))
    return rec


def run_seed(seed, key, net_ok):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    base = {k: float(np.mean([NC.evidence(NC.make_window(None, rng))[k] for _ in range(40)]))
            for k in NC.evidence(NC.make_window(None, rng))}

    # ---- 预训练:检测器只在已知 5 类上训 ---- #
    det = make_detector(len(KNOWN))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(2500):
        c = KNOWN[rng.integers(len(KNOWN))]
        Xpt.append(NC.make_window(c, rng)); Ypt.append(onehot(KNOWN.index(c), len(KNOWN)))
    train_on(det, opt, Xpt, Ypt, epochs=30)
    pre_state = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))  # (window, known_idx) 防遗忘 replay 池

    # ---- 构造流:前段只已知,后段新类涌现 ---- #
    stream = []
    for _ in range(N1):
        c = KNOWN[rng.integers(len(KNOWN))]; stream.append((NC.make_window(c, rng), c))
    for _ in range(N2):
        c = NOVEL if rng.random() < 0.5 else KNOWN[rng.integers(len(KNOWN))]
        stream.append((NC.make_window(c, rng), c))

    # ---- frozen 基线 ---- #
    froz = []
    for x, true_c in stream:
        pred = KNOWN[int(np.argmax(proba(det, [x])[0]))]
        froz.append(int(pred == true_c))

    # ---- bootstrap(LLM 伪标签) & oracle(真值上界) ---- #
    rec_b = online_arm(pre_state, stream, base, replay,
                       lambda ev, tc: (llm_name(ev, base, key, net_ok), 1), rng)
    rec_o = online_arm(pre_state, stream, base, replay,
                       lambda ev, tc: (tc, 0), rng)

    # ---- LLM-only:后段抽样,每窗直接叫 LLM 分类(恒 100% 成本) ---- #
    p2 = list(range(N1, N1 + N2))
    sub = rng.choice(p2, size=min(LLMONLY_N, len(p2)), replace=False)
    lo_correct = []
    for i in sub:
        x, true_c = stream[i]; ev = NC.evidence(x)
        lab = llm_classify_top1(ev, base, key, net_ok)
        lo_correct.append(int(lab == true_c))

    # ---- 汇总(本 seed) ---- #
    def p2_acc(rec):
        s = [r for r in rec if r["phase"] == 1]; return float(np.mean([r["correct"] for r in s]))

    def novel_acc(rec):
        s = [r for r in rec if r["true"] == NOVEL]; return float(np.mean([r["correct"] for r in s])) if s else float("nan")

    nov_idx = [i for i, (_, c) in enumerate(stream) if c == NOVEL]
    seg = [s for s in np.array_split(nov_idx, 5) if len(s)]
    llm_seg = np.array_split(p2, 5)
    return dict(
        frozen_p2=float(np.mean(froz[N1:])),
        bootstrap_p2=p2_acc(rec_b), oracle_p2=p2_acc(rec_o),
        llm_only_acc=float(np.mean(lo_correct)),
        novel_frozen=float(np.mean([froz[N1 + k] for k, (_, c) in enumerate(stream[N1:]) if c == NOVEL])),
        novel_bootstrap=novel_acc(rec_b), novel_oracle=novel_acc(rec_o),
        llm_rate_bootstrap=float(np.mean([r["ncall"] for r in rec_b if r["phase"] == 1])),
        label_rate_oracle=float(np.mean([1 if r["src"] == "label" else 0 for r in rec_o if r["phase"] == 1])),
        final_vocab=rec_b[-1]["vsize"],
        nov_curve=[float(np.mean([rec_b[i]["correct"] for i in s])) for s in seg],
        llm_curve=[float(np.mean([rec_b[i]["ncall"] for i in s])) for s in llm_seg],
    )


def ms(xs):
    a = np.array(xs, float); return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    net_ok = bool(key)
    print(f"device={device} net_ok={net_ok} known={KNOWN} novel={NOVEL} "
          f"NSEED={NSEED} TAU={TAU} LLMONLY_N={LLMONLY_N}")
    seeds = list(range(NSEED))
    res = []
    for s in seeds:
        r = run_seed(s, key, net_ok)
        res.append(r)
        print(f"[seed {s}] frozen={r['frozen_p2']:.1%} boot={r['bootstrap_p2']:.1%} "
              f"oracle={r['oracle_p2']:.1%} llm_only={r['llm_only_acc']:.1%} "
              f"novel_boot={r['novel_bootstrap']:.1%} llm_rate={r['llm_rate_bootstrap']:.1%}")

    def agg(k):
        return ms([r[k] for r in res])
    curve_stack = lambda k: np.array([r[k] for r in res], float)
    nov_m = curve_stack("nov_curve").mean(0); nov_s = curve_stack("nov_curve").std(0)
    llm_m = curve_stack("llm_curve").mean(0); llm_s = curve_stack("llm_curve").std(0)

    print("\n" + "=" * 74)
    print(f"后段(新类涌现后)分类准确率  (mean±std over {NSEED} seeds):")
    for k, name in [("frozen_p2", "frozen   (闭集下界)"), ("bootstrap_p2", "bootstrap(本方法 )"),
                    ("oracle_p2", "oracle   (人工上界)"), ("llm_only_acc", "llm_only (恒高成本)")]:
        m, sd = agg(k); print(f"  {name}: {m:.1%} ± {sd:.1%}")
    print(f"\n新类 {NOVEL} 上准确率:")
    for k, name in [("novel_frozen", "frozen"), ("novel_bootstrap", "bootstrap"), ("novel_oracle", "oracle")]:
        m, sd = agg(k); print(f"  {name:>9}: {m:.1%} ± {sd:.1%}")
    bm, bsd = agg("llm_rate_bootstrap"); om, osd = agg("label_rate_oracle")
    print(f"\n标注成本(后段调用率):  bootstrap LLM={bm:.1%}±{bsd:.1%}   "
          f"oracle 人工={om:.1%}±{osd:.1%}   llm_only=100%")
    print(f"新类准确率随时间(5 段, mean): {[round(v, 2) for v in nov_m]}   ±{[round(v, 2) for v in nov_s]}")
    print(f"LLM 调用率随时间(5 段, mean): {[round(v, 2) for v in llm_m]}   ±{[round(v, 2) for v in llm_s]}")
    fm, _ = agg("frozen_p2"); bp, _ = agg("bootstrap_p2"); nb, _ = agg("novel_bootstrap")
    if bp > fm + 0.1 and nb > 0.3 and llm_m[-1] < llm_m.max():
        print("结论:✅ 闭环无标注地长出新类→准确率回升、LLM 调用衰减,且多 seed 稳定 —— 自举开放词表成立。")
    print("=" * 74)

    out = {"nseed": NSEED, "per_seed": res}
    for k in ["frozen_p2", "bootstrap_p2", "oracle_p2", "llm_only_acc",
              "novel_frozen", "novel_bootstrap", "novel_oracle",
              "llm_rate_bootstrap", "label_rate_oracle"]:
        m, sd = agg(k); out[k] = {"mean": m, "std": sd}
    out["nov_curve"] = {"mean": [float(v) for v in nov_m], "std": [float(v) for v in nov_s]}
    out["llm_curve"] = {"mean": [float(v) for v in llm_m], "std": [float(v) for v in llm_s]}
    json.dump(out, open(output_path("openvocab_loop_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
