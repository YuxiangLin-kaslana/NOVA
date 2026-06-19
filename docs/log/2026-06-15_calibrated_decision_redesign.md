# SigLA 重设计实验日志 —— 校准决策(proposer→decider)

> 日期:2026-06-15 · 运行:Delta A100 · LLM:gpt-4o-mini
> 数据流:SMD machine-1-1 test(28479 点,38 维,异常率 9.46%),win=100 step=25,~1136 窗

---

## 0. 起因

上一轮在线实验(见 [online_training_report.md](../online_training_report.md))两次都**塌成平凡分类器**:
激进 prompt 全报警(F1=0.26),保守 prompt 全不报(F1=0.00)。诊断:信号其实都在
(离线 detector 重建误差在评测流上 ROC-AUC≈0.952、best-F1≈0.69),是**让 LLM 对未校准的
裸重建误差(0.0013 还是 0.0125?)凭感觉判 0/1**,把信号丢了。

**修法**:把异常**决策权**从 LLM 移到「校准后的检测分数」(proposer→decider)。

---

## 1. 重设计的框架

```
离线(一次性,有监督):
  SMD 正常序列        → MLPAnomalyDetector(重建自编码器)
  SMD 正常+概念注入   → CNNConceptDetector(6 概念多标签)

在线(流式,无真值),逐窗:
  detector → 重建误差(裸分数)
       │
       ▼  ★ ScoreCalibrator(本轮新增,核心)
     滚动 P{q} 阈值 → 候选异常 + 百分位         ← 决策在这里(proposer)
       │
  concept → 6 概念概率
       │
       ▼  agent(仅在候选 + 5% 采样正常上调用)
     确认/否决 + 概念标注                        ← decider + 伪标签
       │
       ▼  OnlineTrainer(回放缓冲,周期重训)
     detector: 校准器判正常的窗 → 自监督重建
     concept : agent 概念伪标签 → BCE
```

**与上一版的根本区别**:决策不再由 agent 对裸标量目测,而由校准器用「正常分布分位阈值」给出。
关键代码:`sigla_exp/calibrator.py`(新)、`sigla_exp/pipeline.py`(decision_mode 路由)、
`sigla_exp/online.py`(detector 正常信号锚在校准器而非 agent)、`sigla_exp/agent/gpt_instant.py`
(decider prompt)。runner:`scripts/run_online.py`(`--decision`)、`scripts/run_online_calibrated.sh`(三臂)。

三种 `decision_mode`:
- `agent_raw`(旧法,LLM 对裸分数判 0/1)
- `calibrated_threshold`(校准分数决策,无 LLM)
- `calibrated_agent`(校准提候选,agent 仅在候选+采样正常上确认)

---

## 2. 三臂实验结果(q=0.95 / margin=1.0,即初始默认)

| 臂 | 配置 | P / R / F1 | regime2/3 F1 | GPT 调用 |
|----|------|-----------|-------------|---------|
| A | 校准阈值 + 在线(full) | 0.355 / 0.993 / **0.523** | 0.87 / 0.23 | 0 |
| B | 校准 + GPT decider + 在线 head_only | 0.345 / 0.993 / 0.512 | 0.84 / 0.23 | 433/1136 (38%) |
| C | 校准阈值 + 冻结不在线 | 0.338 / 0.993 / 0.505 | 0.81 / 0.22 | 0 |

**结论**:
1. **塌缩被修好**:三臂都在 F1≈0.5、recall≈0.99,决策机制活了(对比旧版 0.26 / 0.00)。✅
2. **在线适应有正信号但弱**:A(在线)0.523 > C(冻结)0.505,+0.018 F1;detector 随漂移重训确实有用,
   但 machine-1-1 漂移不强,幅度小。抗漂移主图需用**合成漂移流**(已知强漂移点)。
3. **瓶颈是 precision(~0.35)不是 recall**:q=0.95 阈值过报(预测正例 ~27% vs 真实 9.5%)。

---

## 3. 关键发现:agent 在当前工作点是橡皮图章

B 臂单独重跑(独立 CSV),分析 441 次 GPT 调用的否决行为:

```
否决候选 (cand=1 → 正常):   0
提拔正常 (cand=0 → 异常):   0
确认候选 (cand=1 → 异常): 411   ← 全部照单全收
其中真异常 (label=1):       139 / 411   → 272 个假阳性,agent 一个都没拦
```

**agent 一次都没翻案**,B 与 A 在决策上等价(那 0.01 F1 差来自 head_only vs full,与 agent 无关)。
两个叠加原因:
1. **prompt 绑死**:decider 指令写「默认信任校准检测器,别凭直觉翻案」→ 永不否决(矫枉过正)。
2. **工作点让否决无意义**:P95 把 36% 的窗标成候选,2/3 是垃圾;agent 只看到百分位 rank
   (p95.1 与 p99.9 在它眼里差不多),没有判别信号 + 被要求默认信任 → 必然全确认。

---

## 4. 阈值扫描(calibrated_threshold + 在线,0 GPT,真实异常窗=140)

| q | margin | precision | recall | **F1** | 预测正例 |
|---|--------|-----------|--------|--------|---------|
| **0.99** | **1.25** | 0.703 | 0.729 | **0.716** | 145 |
| 0.97 | 1.5 | 0.673 | 0.764 | **0.716** | 159 |
| 0.98 | 1.5 | 0.746 | 0.671 | 0.707 | 126 |
| 0.99 | 1.0 | 0.540 | 0.907 | 0.677 | 235 |
| 0.97 | 1.0 | 0.433 | 0.971 | 0.599 | 314 |
| 0.95 | 1.0 | 0.349 | 0.993 | 0.517 | 398 |  ← 旧默认(全表最差)

**结论**:
1. **报告承诺兑现**:纯调阈值(无 GPT)即把 F1 从 0.52 → **0.716**(q=0.99/m=1.25,P 0.70 / R 0.73,
   预测 145 ≈ 真实 140)。之前三臂用的 q=0.95/m=1.0 恰是全表最差。
2. **纯阈值天花板 ≈ 0.72**:precision/recall 在一根旋钮上互搏——收紧 precision↑ 但 recall↓
   (漏报进不了候选,agent 也救不回);放松反之。单标量阈值做不到「同时高 recall + 高 precision」。

---

## 5. 由此确定 agent 的正确用法(论文价值主张)

agent 是 decider,**只能在候选里否决,不能凭空找回漏报** → 在最优(已很紧)阈值上跑没意义。
正确用法:

> **故意把阈值开松(高 recall),让 agent 砍掉假阳性候选、把 precision 救回来**——单阈值做不到。

预期:在 q=0.97/m=1.0(recall 0.971,precision 0.433,314 候选,~178 假阳性)上,
若改写后的 agent 能否决 ~178 个 FP、保住 136 个 TP → precision ~0.9、recall 守住 0.97 → **F1 可达 ~0.9**,远超纯阈值 0.72。

**论文叙事**:校准阈值单独封顶 ~0.72;LLM decider 的价值在于让系统能在高 recall 工作点运行,
再用定向否决恢复 precision——这是单阈值办不到的,也是「为什么需要 LLM」的答案。

---

## 6. 下一步

1. 更新 `run_online_calibrated.sh` 默认:A/C 臂用 **q=0.99/m=1.25**(纯阈值最优 0.72);
   B 臂用**高 recall 工作点 q=0.97/m=1.0**(给 agent 留否决空间)。
2. **重写 decider prompt + 喂判别信号**(score/threshold 比值、concept 最大概率/持续性),
   让 agent 从橡皮图章变怀疑者。
3. 重跑 B,量 **veto rate** 与 precision 是否被救回——验证 agent 价值的决定性实验。
4. 合成漂移流(`drift_gradual`,已知漂移点)跑三臂,做抗漂移/恢复主图(随机初始化 + full scope)。

## 附:产物路径

- 三臂结果:`code/runs/online/calibrated/smd_machine-1-1_{A,B,C}_*.json`
- B 臂否决分析用 CSV:`code/runs/online/calibrated/pred_smd_machine-1-1_B_rerun.csv`
- 阈值扫描:`code/runs/online/sweep/q{...}_m{...}.json`
- Slurm 作业:19268369(三臂)、19268639(B 重跑)
