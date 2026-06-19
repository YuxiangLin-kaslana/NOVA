# SigLA 两周成果汇总报告(2026-06-15 ~ 06-17)

> 范围:上周「校准决策 + LLM decider + 抗漂移」一线,与本周「LLM 自举开放词表持续学习 + 检测桥」一线。
> 目的:判定哪些想法**可行 / 不可行 / 仅作 motivating**,并给出未来方向。所有结论附实测数字与来源 log。

---

## 0. 一句话结论

上周把「LLM 当 decider 修 precision」与「在线适应抗漂移」做到见底:**decider 有真但适度的价值,抗漂移
(朴素在线适应)被证伪**。据此转向本周的新论文主张——**LLM 自举的开放词表持续学习**:agent 在
**无人工标注**下发现、命名、并长出从未见过的异常类型,使检测器从**对新类失明(检测召回 ~0%)**恢复到
**88% 检测召回**。这条线本周已用基线 + 5-seed 误差棒在 A(单新类分类)/ B(多新类分类)/ D(检测桥)三处验证。

---

## 1. 时间线与关键结果

### 上周(06-15 ~ 06-16):校准决策 + decider + 抗漂移
| 日期 | 实验 | 关键数字 | 判定 |
|---|---|---|---|
| 06-15 | 校准决策重设计(proposer→decider) | 修好"塌成平凡分类器";纯阈值扫描天花板 **F1 0.72** | 基础设施 ✓ |
| 06-16 | **LLM decider 价值**(高 recall 工作点否决 FP) | 匹配 recall=0.94 处,agent P=0.543 vs 阈值前沿 P≈0.49(**+0.05**);否决精度 94% | 真但**适度** ✓ |
| 06-16 | decider 调更狠(gate≥1.3) | F1 0.639 < 温和版 0.689;gpt-4o-mini 违反自己的 gate、过度否决、误杀 44 TP | **不可行** ✗ |
| 06-16b | **抗漂移**:在线适应 vs 冻结(合成漂移流) | 漂移击垮冻结(FP 5%→100%)✓;漂移可学(离线探针)✓;**但朴素在线适应 9 组配置全 90–100% FP** | **核心负结果** ✗ |
| 06-16c | 新类型识别 LLM vs 参数化(motivating) | 已知 5 类参数化 99.7%;留出类 参数化 **0%** vs LLM **86.7%** | 仅 motivating(by construction) |
| 06-16d | 前兆早预警评测接入 | 普通 AD recall 0.98 **完全空心**(有效早预警=0,全是事件中/后) | 评测协议 ✓ + motivating |

### 方向转折(06-16,用户拍板)
**放弃抗漂移/在线适应一线**(9 组死胡同)。新主张 = **(A) 前兆早预警 = 问题** + **(B) LLM 自举开放词表
持续学习 = 方法**:新异常**类型**涌现 → LLM zero-shot 命名 → 伪标签 → 在线扩检测器词表(无人工标注)。

### 本周(06-17):开放词表闭环 + 检测桥
| 实验 | 设定 | frozen | bootstrap(本方法) | 来源 |
|---|---|---|---|---|
| **A 单新类分类** | 留出 correlation_break,4 臂 5 seed | 48.9% | **74.7%**(逼近人工上界 97.0%;LLM-only 仅 21.2%) | log 17b |
| **B 多新类分类** | 3 新类错峰涌现,5 seed | 50.6% | **88.1%**(trend 100% / variance 67% / corr 56%) | log 17c |
| **D 检测桥** | normal+3已知异常类,检测=argmax≠normal,5 seed | 新类检测召回 **15%**(4/5 seed=0) | **88%**;整体召回 0.78→0.97 | log 17d |

本周还产出:证据正交基准 `sigla_exp/ovbench.py`、论文综合图 `slide_figures/06_openvocab_results.png`。

---

## 2. 哪些想法**可行**(已有证据支撑)

1. **LLM 自举开放词表持续学习(本周核心,强可行)。** 无人工标注下发现+命名+长出新异常类型:单新类分类
   49→75%、多新类 51→88%、**新类检测召回 ~0→88%**;词表阶梯增长 3→6;LLM 调用率随检测器学会而衰减
   (单新类 43%→17%)。这是当前最强、最可发表的贡献。

2. **LLM decider 作"高 recall 工作点的定向 FP 否决器"(适度可行)。** 在匹配 recall 处越过单阈值 PR 前沿
   (+0.05 precision,否决精度 94%)——这是"为什么需要 LLM"的直接证据。**但增益适度**,且 F1 绝对值未超纯
   阈值全表最优(0.69 vs 0.72),因为剩余硬 FP 是漂移诱发、决策层修不了。可作次要结果,不宜当 headline。

3. **关键工程手段(使 B 从崩溃到可用)。**
   - **证据正交基准**:6 个签名统计量重设计 + 循环移位式 correlation_break,使每概念恰好且仅触发自己的签名
     → 多新类 gate 才可用(打破之前 1/3 成功的 blocker)。
   - **LLM 命名喂 z-score 而非原始值**:gpt-4o-mini 不会比较原始数值(一律锚定 level_shift),新类命名
     40%→76%。
   - **trend 签名 kendall→lin_r2 + level_shift 去趋势**:消除"台阶被当斜坡"导致的提前建类。
   - **类平衡在线重训 + replay**:在 ~36% 标签噪声下仍学到 59–100%,防灾难性遗忘。

4. **前兆早预警评测协议(可行的评测口径)。** 事件级早预警 F1 / lead-time / "虚高暴露",实锤了重建检测器的
   0.98 recall 是空心的(0 有效早预警)——是诚实评测的基础设施。

---

## 3. 哪些想法**不可行 / 已证伪**

1. **朴素在线适应(重建 AE + 连续 SGD)抗漂移——证伪,已穷尽。** 9 组配置(gradual/abrupt × naive/track ×
   lr × step × 连续/突发 × 共享/自阈 × 密/稀疏缓冲)漂移段 FP 始终 90–100%。**机制性失败**:(a) 泛化滞后——
   in-sample 损失 0.006 但 out-of-sample 新到窗 0.025–0.033,gradual 漂移比适应快;(b) SGD 误差地板
   ~0.013 > 过拟合离线 detector 定的阈值 0.005,连 regime-0 都过不了。**不是超参问题,不要再调合成流。**

2. **把 decider prompt 调更激进——证伪。** gpt-4o-mini 会违反自己的 OR-gate、过度否决、误杀真异常,F1 反降。
   弱模型被逼成不一致的机械算术。**温和 v1 是要报告的版本,不要加码。**

3. **在已最优(很紧)的阈值上用 LLM decider——无意义。** decider 只能在候选里否决、不能找回漏报,紧阈值下
   411 次调用 0 次否决(橡皮图章)。必须故意开松到高 recall 才有价值空间。

4. **"闭集对新类失明"当 headline——不可。** 裸结果(留出类输出 0%)部分 by construction,审稿人会说
   OSR/ZSL 老结论。**只能当 motivating 小图**;真正贡献是 bootstrap 把检测/分类**恢复**回来。

---

## 4. 当前诚实 caveat(可行线上的已知短板)

- **全合成数据**:A/B/D 全在受控合成流上;尚未在真实数据(SMD machine-1-1)上演示开放词表闭环。
- **检测桥精度是代价**:bootstrap 整体 precision 0.91→0.80(在线重训引入少量 normal 误报),F1 仍升。
- **最弱新类**:correlation_break 分类 56%±11%、命名 64%(decorr 签名弱 +2.9);variance_burst 命名 64%。
- **早预警 END 指标尚未打通**:EW 口径已就位,但开放词表闭环还没接到"前兆窗内提前预警"——目前 EW 仍可能恒 0,
  因为系统还没从 onset 前的前兆信号学到可提前的判别力。这是「方法」与「问题(早预警)」之间**最后一公里**。

---

## 5. 未来方向(按优先级)

1. **【最高】接通早预警时序闭环(最后一公里)。** 把开放词表闭环接到带 onset 的流,目标指标=**新型事件前兆窗内
   早预警 F1 / lead-time**(而非纯分类率)。需让系统从前兆窗学到可提前的信号。这是把本周"方法"真正落到论文
   "问题(前兆早预警)"上的决定性实验,也是当前唯一仍悬空的核心环节。

2. **【高】真实数据演示。** 在 SMD(或其它多变量 AD 基准)上重做开放词表闭环(哪怕单新类),证明不是合成特调。
   可复用 decider 实验已有的 SMD detector 基础设施。

3. **【高】文献定位 / novelty 核查。** 对照 LLM-as-labeler、open-vocabulary AD、novel-class detection in streams——
   这三者各自已知,**赌注是"LLM 命名→自举→早预警"这一闭环组合**。建议 deep-research 一轮厘清差异点与风险。

4. **【中】检测桥精度回补。** 缓解在线重训对 normal 的误报(更强的 normal replay / 阈值校准 / 否决器复用 decider)。

5. **【中】最弱新类与命名增强。** 强化 correlation_break 的 decorr 签名与注入强度;再调 variance_burst 判别度。

6. **【低】多机器 / 更多 seed 误差棒;decider 与开放词表的统一叙事**(decider 修 precision + 开放词表修 novel-recall,
   都是"LLM 在闭集检测器之上补它结构上做不到的事")。

---

## 附:关键文件
- 实验脚本:`scripts/exp_openvocab_loop.py`(A)、`exp_openvocab_multi.py`(B)、`exp_detection_tie.py`(D)、
  `exp_novel_concept.py`(motivating)、`run_online_*.sh`(上周 decider/抗漂移)。
- 基准/工具:`sigla_exp/ovbench.py`(证据正交基准)、`scripts/diag_separation_v2.py`(可分性探针)、
  `diag_naming_acc.py`(命名混淆)、`make_openvocab_figure.py`(论文图)。
- 结果:`runs/openvocab_loop_result.json` / `openvocab_multi_result.json` / `detection_tie_result.json`。
- 详细 log:`docs/log/2026-06-15…` ~ `2026-06-17d…`。图:`slide_figures/06_openvocab_results.png`。
