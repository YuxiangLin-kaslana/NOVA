# SigLA 无真值在线训练 —— 实验报告

> 时序异常监控:两个轻量模型 + LLM agent,在线、无真值地随数据流适应。
> 日期:2026-06-07 · 运行环境:Delta A100-SXM4-40GB · LLM:`gpt-4o-mini`

---

## 1. 背景与动机

原架构是四件套:`anomaly detector + concept detector + action policy + LLM`。
本轮工作做了两个决定:

1. **砍掉 action policy**,精简为 **两个小模型 + LLM agent**。agent 同时承担:
   ① 判断窗口是否异常;② 作为**监督来源**为在线训练提供概念伪标签。
2. **加入在线训练**:系统在处理数据流时,**不使用真值**地持续重训两个小模型,
   以适应分布/概念漂移。

核心论点(目标投稿 AAAI):**无人工标签的在线适应(label-free online adaptation
under drift)**——与现有时序 LLM-agent 工作(多为 training-free、静态)形成区分。

---

## 2. 框架

### 2.0 端到端流程图

```
┌─ 离线阶段(一次性,有监督) ────────────────────────────────────────┐
│  SMD 正常 train 序列            ──训练──> MLPAnomalyDetector (重建MSE)│
│  SMD 正常窗口 + 概念注入(§4.1) ─训练─> CNNConceptDetector (多标签BCE)│
│                                              │ 产出 checkpoints       │
└──────────────────────────────────────────────┼────────────────────┘
                                                │ (加载)
┌─ 在线阶段(流式,无真值) ──────────────────────▼────────────────────┐
│  数据流 x[T,38](A=合成漂移流§4.2 / B=SMD真实)                       │
│       ─逐窗→ detector(异常分数) + concept(6概念 profile)             │
│                              │                                        │
│                              ▼                                        │
│                        LLM agent (gpt-4o-mini)                        │
│                  judge → {is_anomaly, concepts[], confidence}         │
│                              │                                        │
│            异常判定 ◄────────┤                                        │
│                              └──► OnlineTrainer(回放缓冲,周期更新)   │
│                                    detector: 正常窗口 重建(自监督)    │
│                                    concept : agent概念伪标签 BCE       │
│                                    (有界缓冲遗忘旧分布; head_only 可选)│
└──────────────────────────────────────────────┬────────────────────┘
                                                ▼
        评测(标签仅此处使用):整体/分段 F1、ROC-AUC、online_stats
```

### 2.1 模型结构

**anomaly detector —— `MLPAnomalyDetector`(重建式自编码器)**

```
输入窗口 [B, win=100, n_vars=38]  ──展平──> [B, 3800]
        │
   encoder: Linear(3800→128) ReLU → Linear(128→128) ReLU   ─> latent z[B,128]
        │
   decoder: Linear(128→128) ReLU → Linear(128→3800)        ─> 重建 [B,100,38]
        │
   异常分数 = 重建误差 MSE(recon, 输入)   ← 偏离正常形态越大,分数越高
```

**concept detector —— `CNNConceptDetector`(1D-CNN 多标签)**

```
输入窗口 [B, 100, 38] ──转置──> [B, 38(通道), 100(时间)]
        │
   Conv1d(38→64, k=7) - BN - ReLU
   Conv1d(64→128, k=7) - BN - ReLU         ← 沿时间卷积,平移不变,擅长局部形态
        │
   时间维 全局池化: avg ⊕ max  ─────────> [B, 256]
        │
   head: Linear(256→128) ReLU Dropout → Linear(128→6)   ─> 6 概念 logits(多标签)
        │
   sigmoid → 6 个概念的独立概率
```
> 6 概念:spike / level_shift / oscillation / variance_burst / trend / correlation_break

### 2.2 推理流程(逐窗)

detector 出异常分数、concept 出 6 概念概率,一起给 LLM agent;agent 输出
`{is_anomaly, anomaly_score, concepts[], confidence, rationale}`——既做异常判定(职能①),
其 `concepts` 又作为在线训练的概念伪标签(职能②)。agent 经 OpenAI Responses API
调用(urllib 客户端,不依赖 openai 包),失败自动回退到确定性 local agent。

### 2.3 在线训练(`OnlineTrainer`)

**回放缓冲 + 周期性小批量更新**,而非逐样本 SGD:

| 模型 | 监督来源 | 损失 | 进缓冲条件 |
|------|---------|------|-----------|
| detector | **自监督**(重建自身) | MSE | agent 判为正常的窗口(+warmup) |
| concept | **agent 伪标签** | BCE | agent 置信度 ≥ 阈值 |

- 缓冲为**有界 deque**(默认 512)→ 隐式遗忘旧分布,是抗漂移的核心机制。
- 触发:每 `retrain_every`(25)窗做几步梯度。
- 三闸门:`warmup_windows`、`min_confidence`、`detector_normal_only`。
- **更新范围 `update_scope`**:`full` / `head_only`(冻结 backbone,只调 head/decoder)/
  `norm_only`(只调 BatchNorm,TENT 式)。
- **全程不使用真值**;标签仅在最后用于离线评测。

### 2.4 数据与代码组织(`specific_data/`)

| 目录 | 内容 |
|------|------|
| `Anomaly_detector/` | SMD 加载 + 切窗(`data.py`) |
| `Concept_detector/` | 概念注入数据集(`concept_synth.py`,6 概念) |
| `Online_training/` | 在线流生成器(`make_stream.py` 合成漂移 / `make_smd_stream.py` SMD)+ 数据 |

---

## 3. 离线模型训练(在线的基座)

两个小模型先离线训好,作为在线适应的起点。数据均来自 **SMD machine-1-1**,
窗口 `win=100, stride=10`,50 epochs,lr 1e-3,batch 256。

### 3.1 anomaly detector(重建自编码器)

- **监督方式**:只在**正常**序列上做重建(无异常标签参与训练)。
- **结果**:train_loss `0.0376 → 0.00061`,val 重建 loss `0.0297 → 0.0014`,平滑收敛。
- val ROC-AUC 显示为 NaN:训练/验证只含正常窗口,异常判定留给阈值/下游,
  模型选择以**重建 loss** 为准。

| | 起始 | 收敛 |
|---|------|------|
| train 重建 loss | 0.0376 | **0.00061** |
| val 重建 loss | 0.0297 | **0.0014** |

### 3.2 concept detector(1D-CNN 多标签)

- **监督方式**:在 SMD 正常窗口上**程序化注入** 6 种概念(可共现),注入了什么即标签
  (`p_normal=0.2`,每窗最多 3 个概念)。kernel=7。
- **结果**:best val macro ROC-AUC **0.846**,final val macro-F1 **0.654**。

| 概念 | ROC-AUC | F1 | Precision | Recall |
|------|---------|-----|-----------|--------|
| level_shift | **0.971** | 0.865 | 0.905 | 0.829 |
| oscillation | 0.917 | 0.760 | 0.805 | 0.720 |
| trend | 0.916 | 0.734 | 0.850 | 0.646 |
| variance_burst | 0.894 | 0.730 | 0.706 | 0.756 |
| spike | 0.828 | 0.555 | 0.622 | 0.501 |
| correlation_break | 0.548 | 0.250 | 0.322 | 0.204 |
| **macro** | **0.846** | **0.649** | — | — |

> 多数概念学得很好(level_shift/oscillation/trend/variance_burst,AUC 0.89~0.97);
> correlation_break 最难(AUC 0.55)——它只打乱维间相关、不改变各维边缘分布,信号最弱。

---

## 4. 合成异常与概念漂移设计

合成数据用于两处:**(a)** §3.2 概念检测器的离线监督(在 SMD 正常窗口上注入概念);
**(b)** 实验 A 的**合成漂移流**(在合成正常信号上注入 + 制造漂移)。
*实验 B(SMD)用的是真实异常标签,不做合成注入。*

### 4.1 合成异常:6 种概念注入(`concept_synth.py`)

约定:每次随机选 **1 ~ n_vars/3** 个维度,在窗口内一个**随机子区间**原地注入;
一个窗口可**多概念共现**(多标签);`p_normal` 概率不注入(纯正常,标签全 0)。

| 概念 | 注入方式 | 特点 |
|------|---------|------|
| spike | 随机位置加 1~3 个幅值 ±[0.3,0.9] 的尖峰 | 局部短促 |
| level_shift | 子区间整体平移 ±[0.2,0.6] | 均值持续偏移 |
| oscillation | 子区间叠加高频正弦(周期 2~6 步,幅 0.1~0.4) | 高频振荡 |
| variance_burst | 子区间加高斯噪声 σ∈[0.1,0.3] | 均值不变、方差暴增 |
| trend | 子区间叠加线性斜坡(斜率 ±[0.2,0.6]) | 缓慢漂移 |
| correlation_break | ≥2 维,子区间内各维独立打乱时间顺序 | 边缘分布几乎不变,破坏维间相关 |

> 设计对齐真实异常:多维(只影响部分维)、变时长、可共现;含纯正常样本让模型学"什么都没有"。

### 4.2 合成漂移流构造(`make_stream.py`)

**基础正常信号**:每维 `sin(2π·f_d·t + φ_d)`,频率 f_d∈[0.010,0.014],幅度 0.6~0.8,
均值 0.5,叠加高斯噪声 σ=0.03,归一化到 ~[0,1]。

**4 个等长区间,漂移点 `[3000, 6000, 9000]`**,同时施加两类漂移:

```
区间:        0              1              2              3
时间:  0────────3000──────6000──────9000──────12000
─ 协变量漂移 P(X)(基础信号统计量逐区间变化) ─────────────────────
   频率 × (1.0 → 1.8)   均值 + (0.0 → 0.25)   幅度 × (1.0 → 0.7)
   gradual: 区间内线性过渡到下一区间   |   abrupt: 区间间阶跃
─ 概念漂移(每区间允许出现的异常概念构成不同) ────────────────────
   区间0: spike,oscillation        区间2: variance_burst,correlation_break
   区间1: level_shift,trend        区间3: spike,level_shift,trend
```

**异常事件注入**:按 `anomaly_rate=0.08` 在流中散布若干个**窗口长**事件,事件窗内
按**当前区间允许的概念**注入,并把这些时刻标为异常(`y=1`)。最终 ~**7.5%** 的点为异常。

> 这样同时具备**协变量漂移**(输入分布变)和**概念漂移**(异常形态构成变),
> 且漂移点已知,便于做"抗漂移/恢复"曲线。`drift_abrupt.npz` / `drift_gradual.npz` 各一份。

---

## 5. 在线实验设置

| 项 | 实验 A(合成流) | 实验 B(SMD 流) |
|----|----------------|----------------|
| 数据流 | 合成漂移流 `drift_gradual`(12000点) | SMD machine-1-1 test(28479点) |
| 模型起点 | **随机初始化** | **加载 §3 离线 ckpt** |
| 更新范围 | `full` + warmup 200 | `head_only`(冻结离线 backbone) |
| agent | gpt-4o-mini | gpt-4o-mini + **保守 prior 0.09** |
| win / step | 100 / 10(~1191窗) | 100 / 25(~1136窗) |
| 硬件 / 时长 | A100 / 42 min | A100 / 42 min |

> 两次 LLM 都真实驱动:source 计数 `gpt-4o-mini` 占 1188/1191 与 1131/1136。

---

## 6. 结果

### 6.1 端到端在线训练(两次都塌成平凡分类器)

| 指标 | 实验 A(随机+full) | 实验 B(离线ckpt+head_only+保守) |
|------|-------------------|-------------------------------|
| precision | 0.15 | 0.00 |
| recall | **1.00** | **0.00** |
| F1 | 0.26 | 0.00 |
| 预测正例 / 总窗 | 1188 / 1191 | 0 / 1136 |
| 行为 | **几乎全判异常** | **全判正常** |
| 在线更新 | det 47 / con 47 次 | det 45 / con 45 次 |

→ 两端退化:激进 prompt 全报警,保守 prompt 全不报。机制都跑通(LLM 驱动、
两模型在线收敛、loss 下降),但**检测无区分力**。

### 6.2 诊断:离线模型自身信号极强(关键发现)

直接用离线模型在 SMD 流上算"模型分数 vs 真值"(**不经过 agent**):

| 信号 | ROC-AUC | 最佳阈值 F1 | 正常 vs 异常均值 |
|------|---------|-----------|----------------|
| detector 重建误差 | **0.952** | **0.689** | 0.0013 vs 0.0125 |
| concept 最大概率 | **0.935** | **0.732** | 0.174 vs 0.509 |

**信号完全在那(AUC≈0.95),是在线流水线把它丢了。**

---

## 7. 关键结论

1. **组件是好的**:离线 detector / concept 把异常分得很开(在线评测流上 AUC 0.95)。
2. **症结在"谁做异常决策"**:当前让 **LLM 对一个未校准的原始重建误差(0.0013 还是
   0.0125?)凭感觉判 0/1**。这一步随 prompt 偏置漂移 → 两次实验分别塌向全报警/全不报。
3. **自强化放大**:agent 的判断又回流去训两个模型,把"全异常/全正常"固化(loss→0.003)。

> 本质:重建式异常检测应**用正常数据定阈值**来决策,我们却跳过阈值、让 LLM 目测裸标量。

---

## 8. 下一步(修法)

把异常**决策**还给校准后的检测分数,agent 退回它擅长的"确认/解释 + 概念标注"
(回到 proposer→decider 范式,proposer = 校准的 detector):

1. **detector 分数校准**:用正常参考(warmup/运行时分位数)定阈值 → 候选异常(立等 F1≈0.69)。
2. **喂 agent 校准信号**:给"当前处于正常分布第 98 百分位 / 超过 95% 阈值",而非裸 MSE。
3. **三条对照实验**(抗漂移主图):`detector 阈值` / `detector+agent 在线 head_only` / `冻结不在线`。

预期:在线检测可从 F1=0 恢复到 ~0.7,且概念伪标签终于建立在可信异常判定之上。

---

## 附:关键文件

| 文件 | 作用 |
|------|------|
| `code/sigla_exp/pipeline.py` | 推理流水线(detector→concept→agent) |
| `code/sigla_exp/online.py` | OnlineTrainer(回放缓冲、update_scope、闸门) |
| `code/sigla_exp/agent/gpt_instant.py` | LLM/local agent,judge()、概念伪标签、保守 prompt |
| `code/scripts/run_online.py` | 在线 runner(加载 ckpt、跑流、评测) |
| `code/scripts/run_online_training.sh` | A100 提交脚本 |
| `code/train/concept_detector/train.py` | concept detector 离线训练(§3.2) |
| `specific_data/Online_training/` | 流生成器 + 数据 |
| `runs/online/*.json` | 各次在线结果(source 计数、分段 F1、online_stats) |
