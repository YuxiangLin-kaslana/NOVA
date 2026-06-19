# SMD 异常检测实验记录

数据集：Server Machine Dataset (SMD)，28 台机器、各 38 维指标，每台 train 文件（纯正常）+ test 文件（含逐点异常标签）。
窗口：`win_size=100`，`stride=10`，窗口标签 = 窗口内任一时刻异常即为 1。
环境：Delta 1×A100-SXM4-40GB；conda 环境 `/projects/bflz/ylin30/conda_envs/sigla`（torch 2.5.1）。
wandb 项目：`sigla-anomaly-detector`、`sigla-concept-detector`（entity `yuxianglin2025-northwestern-university`）。

相关代码：
- 数据准备：[specific_data/Anomaly_detector/data.py](../specific_data/Anomaly_detector/data.py)
- 合成器：[specific_data/Anomaly_detector/concept_synth.py](../specific_data/Anomaly_detector/concept_synth.py)
- 模型：[code/sigla_exp/model/mlp.py](../code/sigla_exp/model/mlp.py)（`MLPAnomalyDetector` / `MLPConceptDetector`）、[code/sigla_exp/model/cnn.py](../code/sigla_exp/model/cnn.py)（`CNNConceptDetector`）
- 训练：[code/train/anomaly_detector/train.py](../code/train/anomaly_detector/train.py)、[code/train/concept_detector/train.py](../code/train/concept_detector/train.py)

---

## 实验一：异常检测器（无监督自编码器）

**做法**：`MLPAnomalyDetector`（自编码器）只在正常数据上做无监督重构训练，用重构 MSE 当异常分数。
**评估口径**：test 报告 threshold-free 的 ROC-AUC / Average-Precision；best-F1 为 test 上 oracle 上界（仅参考）。

### 结果：28 台合并 vs 单台机器

| 设置 | 训练数据 | test ROC-AUC | Average-Precision | best-F1 (oracle) |
|------|---------|:---:|:---:|:---:|
| 28 台合并（job 18925977） | 全部机器 train | 0.7235 | 0.2493 | 0.3175 |
| **单台 machine-1-1（job 18926320）** | 仅该机 train | **0.9380** | **0.7959** | **0.7535** |

### 关键发现

- **合并 28 台机器是性能差的主因，不是 bug。** SMD 的归一化是各机器独立做的，同一 [0,1] 值在不同机器含义不同；一个全局 AE 要同时拟合 28 种分布 → 欠拟合，且跨机器异常分数不可比（实测正常窗口重构 MSE 跨机器差 **12.8×**），全局排序被搅乱。逐机器评估 AUROC 中位数即 0.83。
- **单台机器上这个朴素 AE 相当能打**：machine-1-1 AUROC 0.94、AP 0.80。
- **F1 偏低 ≠ 模型差**：machine-1-1 在 val 选的阈值（正常分数 99 分位）下 F1 仅 0.33（precision 0.84 / recall 0.20），因为 p99 太保守、漏报多；oracle best-F1 高达 0.75 说明模型本身分得开，调阈值即可。

### 数据切分（干净无监督，无泄露）

- train ← train 文件（纯正常）；val/test ← test 文件按时间 1:1 切分。
- 模型选择：val 有异常→val AUROC；val 全正常→val 重构损失。
- 阈值：val 有异常→best-F1；val 全正常→正常分数分位数（`--threshold_percentile`，默认 99）。
- 注意：machine-1-1 的异常聚集在 test 时间线 56%~97%，故时间 1:1 切后 **val 全为正常**，按上述逻辑自动走「重构损失选模型 + 分位数阈值」。

---

## 实验二：Concept Detector（多标签，合成监督）

**动机**：异常检测器只能说「不正常」，说不出「为什么」。concept detector 识别异常的**形态**。
**做法**：真实异常无形态标注 → 从正常窗口**程序化注入**已知形态，注入什么标签就是什么，得到带标签训练数据。
**concept（6 类，多标签）**：`spike / level_shift / oscillation / variance_burst / trend / correlation_break`（依据对 SMD 327 个真实异常段的形态统计选定）。
**切分**：train 用 train 文件正常窗口合成；val 用 test 文件前半的正常窗口合成。concept 标签全部来自合成，与 SMD 的 test_label 无关 → 无标签泄露。

### 结果：MLP vs 1D-CNN（machine-1-1）

每个 concept 的 ROC-AUC：

| concept | MLP (8 epoch) | **CNN (50 epoch, job 18926759)** |
|---|:---:|:---:|
| spike | 0.498 | **0.828** |
| level_shift | 0.906 | **0.971** |
| oscillation | 0.496 | **0.917** |
| variance_burst | 0.519 | **0.894** |
| trend | 0.752 | **0.916** |
| correlation_break | 0.510 | 0.548 |
| **MACRO ROC-AUC** | **0.613** | **0.846** |
| **MACRO F1 @0.5** | 0.162 | **0.649** |

CNN 最终各 concept F1：level_shift 0.865 / oscillation 0.760 / variance_burst 0.730 / trend 0.734 / spike 0.555 / correlation_break 0.250。

### 关键发现

- **「扁平 MLP 抓不住位置随机的局部形态」得到验证。** 全局/大范围 concept（level_shift, trend）MLP 学得好；局部/短时 concept（spike, oscillation, variance_burst）MLP 停在随机线 ~0.50——因为它们在 100×38=3800 维输入里占比极小且位置随机。
- **换平移不变的 1D-CNN 后三个局部 concept 全部救回**：spike 0.50→0.83、oscillation 0.50→0.92、variance_burst 0.52→0.89。Macro AUROC 0.61→0.85。
- **correlation_break 仍难（0.548）**：纯多变量结构概念（打乱维间相关、保持各维边缘分布），全局池化丢失联合信息，信号也易被 oscillation/variance_burst 吸收。待改进方向：增强注入强度 / 显式喂入维间相关特征。

---

## 待办 / 后续

- correlation_break 专项提升（增强注入方式 + 显式相关性特征）。
- 异常检测器：把阈值分位数调到更贴近真实异常率（如 p90）以改善 F1。
- 可选：把 concept detector 推广到多台机器 / 逐机器评估。
