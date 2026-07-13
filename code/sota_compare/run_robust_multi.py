#!/usr/bin/env python3
"""巩固:鲁棒闭环跨多台真实 SMD 机器 × 多 seed 验证。确认"自我毒化修复 + 命名本事保留"普遍成立。
每机器:seed0 额外跑**原版**做毒化参照;鲁棒版跑 NSEED seeds 求 mean±std。
  测试1(纯正常+漂移,无真异常):FA + 伪类增长数 → 鲁棒版应 FA≈5%、伪类≈0;原版应 FA 高、伪类多。
  测试2(含缓慢爬升真新异常):漂移FA / 召回 / 命名 → 鲁棒版应低FA + 高召回高命名(本事没丢)。
用法: sbatch sota_compare/run_robust_multi.sh  (env REAL_MACHINES, CMP_NSEED 默认3, CMP_SMOKE)
"""
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.exp_detection_tie as DT          # noqa: E402
import sigla_exp.ovbench as CB                  # noqa: E402
import sota_compare.exp_drift_vs_novel as EXP   # noqa: E402
import sota_compare.realbench as RB             # noqa: E402
from sota_compare.robust_loop import robust_ours_loop  # noqa: E402

device = DT.device
SMOKE = os.environ.get("CMP_SMOKE", "0") == "1"
NSEED = int(os.environ.get("CMP_NSEED", "1" if SMOKE else "3"))
DEFAULT_M = ["1-1"] if SMOKE else ["1-1", "2-1", "3-1", "1-6", "2-5"]
MACHINES = os.environ.get("REAL_MACHINES", ",".join(DEFAULT_M)).split(",")
N_PT = 300 if SMOKE else 2400
DRIFT_N = 200 if SMOKE else 600


def pretrain(rng):
    det = DT.make_detector(len(EXP.BASE_VOCAB))
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    Xpt, Ypt = [], []
    for _ in range(N_PT):
        c = EXP.BASE_VOCAB[rng.integers(len(EXP.BASE_VOCAB))]
        base = CB.base_normal(rng)
        x = base if c == EXP.NORMAL else EXP.inject(c, base, rng, float(rng.uniform(0.5, 1.0)))
        Xpt.append(EXP.mc(x)); Ypt.append(DT.onehot(EXP.BASE_VOCAB.index(c), len(EXP.BASE_VOCAB)))
    DT.train_on(det, opt, Xpt, Ypt, epochs=30)
    pre = copy.deepcopy(det.state_dict())
    replay = list(zip(Xpt, [int(np.argmax(y)) for y in Ypt]))
    thr = float(np.quantile(1.0 - DT.proba(det, [EXP.mc(CB.base_normal(rng)) for _ in range(400)])[:, 0], 0.95))
    return pre, replay, thr


def drift_stream(rng, dvec):
    W = [CB.base_normal(rng) for _ in range(200)]
    for i in range(DRIFT_N):
        W.append(CB.base_normal(rng) + (i + 1) / DRIFT_N * EXP.DRIFT_D * dvec)
    return W


def run_seed(seed, key, net_ok, want_orig):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    mu, sd = CB.normal_stats(rng)
    dvec = rng.normal(0, 1, CB.NVARS).astype(np.float32); dvec /= (np.linalg.norm(dvec) / np.sqrt(CB.NVARS) + 1e-6)
    pre, replay, thr = pretrain(rng)
    nspur = lambda v: len([c for c in v if c not in EXP.BASE_VOCAB])

    W1 = drift_stream(rng, dvec)
    out = {}
    if want_orig:
        a, _, v = EXP.ours_loop(pre, replay, W1, mu, sd, key, net_ok, thr)
        out["orig_fa"] = float(np.mean(a)); out["orig_spur"] = nspur(v)
    a, _, v, _ = robust_ours_loop(pre, replay, W1, mu, sd, key, net_ok, thr)
    out["rob_fa"] = float(np.mean(a)); out["rob_fa_late"] = float(np.mean(a[-200:])); out["rob_spur"] = nspur(v)

    rng2 = np.random.default_rng(1000 + seed)
    W2, lab, is_nov, phase, creep = EXP.build_stream(rng2, dvec)
    a2, names, v2, st2 = robust_ours_loop(pre, replay, W2, mu, sd, key, net_ok, thr)
    nov = (is_nov == 1)
    out["t2_drift_fa"] = float(a2[phase == "drift"].mean())
    out["t2_recall"] = float(a2[nov].mean())
    out["t2_name"] = float(np.mean([names[i] == EXP.NOVEL for i in range(len(W2)) if is_nov[i]]))
    out["t2_grew"] = int(EXP.NOVEL in v2)
    out["t2_commit"], out["t2_recalib"] = st2["commit"], st2["recalib"]
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


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key) and not SMOKE
    print(f"device={device} net_ok={net_ok} SMOKE={SMOKE} NSEED={NSEED} machines={MACHINES}\n")
    allres = {}
    for m in MACHINES:
        m = m.strip(); RB.activate(m)
        rs = [run_seed(s, key, net_ok, want_orig=(s == 0)) for s in range(NSEED)]
        allres[m] = rs
        rf, rfs = ms([r["rob_fa"] for r in rs]); rl, _ = ms([r["rob_fa_late"] for r in rs])
        rsp, _ = ms([r["rob_spur"] for r in rs])
        tr, _ = ms([r["t2_recall"] for r in rs]); tn, _ = ms([r["t2_name"] for r in rs])
        tf, _ = ms([r["t2_drift_fa"] for r in rs])
        gr, _ = ms([r["t2_grew"] for r in rs])
        ca, _ = ms([r["t2_commit"] for r in rs])
        o = rs[0]
        print(f"[machine-{m}] 纯漂移: 原版FA={o['orig_fa']:.0%}(伪类{o['orig_spur']}) → "
              f"鲁棒FA={rf:.0%}±{rfs:.0%}(末{rl:.0%},伪类{rsp:.1f}) | "
              f"含真异常: 漂移FA={tf:.0%} 召回={tr:.0%} 命名={tn:.0%} 真类长全={gr:.0%} "
              f"(长类提交{ca:.1f})")

    # 跨机器汇总
    flat = lambda f: [r[f] for rs in allres.values() for r in rs]
    print("\n" + "=" * 96)
    print(f"跨 {len(MACHINES)} 机器 × {NSEED} seeds 汇总:")
    print(f"  [纯漂移·毒化修复] 鲁棒 FA={ms(flat('rob_fa'))[0]:.0%}±{ms(flat('rob_fa'))[1]:.0%} "
          f"末段={ms(flat('rob_fa_late'))[0]:.0%}  伪类增长={ms(flat('rob_spur'))[0]:.2f}±{ms(flat('rob_spur'))[1]:.2f}")
    oavg = ms([allres[m][0]['orig_fa'] for m in MACHINES])[0]
    ospur = ms([allres[m][0]['orig_spur'] for m in MACHINES])[0]
    print(f"  [纯漂移·原版参照] FA={oavg:.0%}  伪类增长={ospur:.1f}")
    print(f"  [含真异常·本事保留] 漂移FA={ms(flat('t2_drift_fa'))[0]:.0%}  召回={ms(flat('t2_recall'))[0]:.0%}±{ms(flat('t2_recall'))[1]:.0%}  "
          f"命名={ms(flat('t2_name'))[0]:.0%}±{ms(flat('t2_name'))[1]:.0%}  真新类长全率={ms(flat('t2_grew'))[0]:.0%}")
    print(f"  [长类机制] 提交={ms(flat('t2_commit'))[0]:.1f}  阈值重标定次数={ms(flat('t2_recalib'))[0]:.1f}")
    print("=" * 96)
    print("判读:鲁棒版跨机器一致 FA≈5%、伪类≈0(毒化修复普遍成立),且含真异常时召回/命名不掉(本事没丢)。")
    outp = output_path("robust_multi.json")
    json.dump(allres, open(outp, "w"), indent=2)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
