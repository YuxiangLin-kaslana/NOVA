#!/usr/bin/env python3
"""【MIMIC 多模态早预警:体征 + 临床笔记 学习式融合】骨架,数据到位即可跑。

判据(同 exp_mmd_fusion2,严格不偷看):5 折 CV 逻辑回归比 [vitals] vs [vitals,text] 的留出 AUC。
若 [vitals,text] 显著 > [vitals] → 临床笔记带**正交增量**(体征里没有的影像/症状/判断)→ 多模态真有用。

无泄漏铁律:决策时刻 t 只用 charttime≤t(笔记还需 storetime≤t)的体征与文本;不用出院小结。

数据接口(3 个 TODO 函数,按你下到的 MIMIC 路径/schema 填):见 load_cohort/load_vitals/load_notes。
用法(数据本地后): sbatch mimic/run_mimic.sh
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

MIMIC = Path(os.environ.get("MIMIC_DIR", "/u/ylin30/sigLA/data/mimic"))
K_HOURS = 8                  # 决策点回看体征小时数
H_HOURS = 6                  # 预警 horizon:未来 H 小时内是否 onset
VITALS = ["heart_rate", "mbp", "resp_rate", "spo2", "temperature"]

# ----------------------------------------------------------------------------- #
#  数据接口:按 MIMIC 实际 schema 填。下面给出**期望的返回结构**。
# ----------------------------------------------------------------------------- #
def load_cohort():
    """返回 DataFrame: 每行一个 ICU stay,列 = [stay_id, onset_hour 或 NaN(无事件), intime, ...]。
    建议直接用 MIMIC-Sepsis pipeline(github.com/yongh7/MIMIC-sepsis)产出的 Sepsis-3 onset 表。
    备选标签:vasopressor/vent 起始小时、院内死亡。"""
    raise NotImplementedError("TODO: 读 MIMIC-Sepsis 队列表 → [stay_id, onset_hour, intime]")


def load_vitals(stay_id):
    """返回该 stay 的逐小时体征 DataFrame,index=hour(自 intime 起 0,1,2,...),列=VITALS。
    来源:MIMIC-IV icu/chartevents(按 d_items 映射 itemid),重采样到小时。"""
    raise NotImplementedError("TODO: chartevents → 逐小时体征矩阵")


def load_notes(stay_id):
    """返回该 stay 的笔记列表 [(avail_hour, text), ...],avail_hour=storetime 相对 intime 的小时(因果可用时刻)。
    来源:MIMIC-IV-Note radiology(charttime/storetime);**排除 discharge**。MIMIC-III 可用 NOTEEVENTS 进度笔记。"""
    raise NotImplementedError("TODO: radiology 报告 → [(avail_hour, text)],按 storetime 因果对齐")


# ----------------------------------------------------------------------------- #
#  以下逻辑已就绪(数据接口填好即可跑)
# ----------------------------------------------------------------------------- #
def latest_note(notes, t):
    cand = [(h, x) for h, x in notes if h <= t]
    return cand[-1][1][:1200] if cand else None


def gpt_note_risk(text, key, model="gpt-4o-mini"):
    """LLM 读临床笔记 → 未来 H 小时内 sepsis/恶化的风险 0-100(只看文本,隔离正交贡献)。"""
    instr = ("You assess ICU early-warning risk from a single timestamped clinical note (e.g., a radiology report) "
             f"ALONE. Estimate the RISK (0-100) that the patient will deteriorate / develop sepsis within the next "
             f"{H_HOURS} hours based only on what this note implies (findings, infection signs, clinical concern). "
             'Use the full 0-100 range. Respond ONLY JSON {"risk": <0-100>}. No markdown.')
    payload = {"model": model, "instructions": instr,
               "input": [{"role": "user", "content": f"note: \"{text}\""}], "max_output_tokens": 40}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read().decode())
            txt = data.get("output_text")
            if not isinstance(txt, str):
                txt = "\n".join(c.get("text", "") for it in data.get("output", []) for c in it.get("content", []))
            s, e = txt.find("{"), txt.rfind("}")
            return float(json.loads(txt[s:e + 1]).get("risk"))
        except Exception:
            continue
    return None


def vitals_features(vdf, t):
    """决策时刻 t 的体征特征:每个体征的 当前值/均值/斜率(近 K_HOURS 小时,只用 ≤t)。"""
    seg = vdf[(vdf.index <= t) & (vdf.index > t - K_HOURS)]
    feats = []
    for v in VITALS:
        s = seg[v].dropna().values if v in seg else np.array([])
        if len(s) == 0:
            feats += [0.0, 0.0, 0.0]
        else:
            slope = (s[-1] - s[0]) if len(s) > 1 else 0.0
            feats += [float(s[-1]), float(np.mean(s)), float(slope)]
    return feats


def build_dataset(key, net_ok, max_stays=None):
    """遍历队列,在 onset 前(或对照 stay 全程)采决策点,产出 (vitals_feat, note_risk, label)。"""
    coh = load_cohort()
    if max_stays:
        coh = coh.head(max_stays)
    Xv, Xt, y = [], [], []
    for _, row in coh.iterrows():
        sid = row["stay_id"]; onset = row.get("onset_hour", np.nan)
        vdf = load_vitals(sid); notes = load_notes(sid)
        if vdf is None or len(vdf) < K_HOURS + H_HOURS:
            continue
        Tmax = int(vdf.index.max())
        for t in range(K_HOURS, Tmax - H_HOURS, 2):           # 每 2 小时一个决策点
            if not np.isnan(onset) and t >= onset:            # 只做 onset 前(早预警)
                continue
            label = int((not np.isnan(onset)) and (t < onset <= t + H_HOURS))
            nt = latest_note(notes, t)
            risk = gpt_note_risk(nt, key) if (net_ok and nt) else np.nan
            Xv.append(vitals_features(vdf, t)); Xt.append(risk); y.append(label)
    return np.array(Xv, float), np.array(Xt, float), np.array(y, int)


def oof_auc(X, y, n_rep=20):
    aucs = []
    for rs in range(n_rep):
        skf = StratifiedKFold(5, shuffle=True, random_state=rs); oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    key = os.environ.get("OPENAI_API_KEY", ""); net_ok = bool(key)
    Xv, Xt, y = build_dataset(key, net_ok, max_stays=int(os.environ.get("MIMIC_MAXSTAYS", "0")) or None)
    Xt = np.nan_to_num(Xt, nan=np.nanmean(Xt)).reshape(-1, 1)
    print(f"MIMIC 多模态早预警  n={len(y)} 正例率={y.mean():.0%} (vitals dim={Xv.shape[1]})\n")
    feats = {"[vitals]": Xv, "[vitals,note]": np.column_stack([Xv, Xt]), "[note only]": Xt}
    base = None
    print(f"{'特征集':18s}{'CV-AUC':>16s}")
    print("-" * 36)
    for nm, X in feats.items():
        m, s = oof_auc(X, y)
        if nm == "[vitals]":
            base = m
        d = "" if nm == "[vitals]" else f"  Δ={m-base:+.3f}"
        print(f"{nm:18s}{m:>11.3f}±{s:<4.3f}{d}")
    print("\n判据:[vitals,note] 的 CV-AUC 显著 > [vitals] → 临床笔记带正交增量(多模态真有用)。")


if __name__ == "__main__":
    main()
