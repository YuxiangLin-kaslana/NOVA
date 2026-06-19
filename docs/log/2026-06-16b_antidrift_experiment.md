# SigLA 抗漂移实验日志 —— 在线适应 vs 冻结(合成漂移流)

> 日期:2026-06-16 · 运行:Delta A100/A40(多次 sbatch)· 无 LLM(纯 detector 适应)
> 承接 [decider 决定性实验](2026-06-16_decider_value_experiment.md):precision 天花板=漂移诱发 FP,
> 本实验验证"在线适应压低漂移段 FP → precision 回升"。

## 0. 设置

数据 `drift_gradual.npz`(12000 点,4 区间各 3000,漂移点 3000/6000/9000;协变量+概念双漂移)。
共同起点 detector:在 regime-0 正常窗上专训(`scripts/train_drift_detector.py`)——SMD 预训练 detector
**不迁移**到合成流(异常分≈正常分,ROC-AUC 仅 0.65)。三臂:frozen(部署即冻结)、
online_naive(只学校准器判正常的窗)、online_track(学近期所有窗,小缓冲跟漂移)。
决策用固定阈值(regime-0 正常分 q0.95 = 0.0133,部署即冻结),隔离自适应校准器干扰。
新增代码:`OnlineTrainConfig.freeze_after`、`detector_track_all`;`run_drift.sh`、`make_drift_figure.py`、`probe_refit.sh`。

## 1. 已干净证实的结论 ✅

1. **漂移确实击垮冻结 detector**:regime-0 内 F1=0.86(P0.75/R1.0),跨入漂移区间后正常窗重建误差
   暴涨 ~7×(0.006→0.041),FP 率 5%→**100%**,F1 塌到 0.23。漂移 FP 机制坐实。
2. **漂移是可学的(离线探针 `probe_refit.sh`)**:把 regime-0 detector 在 regime-1/3 正常窗上重训
   150 epoch,误差 0.041→**0.011**、0.029→0.012,均降到阈值 0.0133 **以下** → 检测可恢复。

## 2. 核心负结果 ❌(7 组配置一致)

**系统当前的在线适应机制(连续重建重训)无法从漂移中恢复检测。** 跨以下配置,漂移区间 FP 率
始终 90–100%:naive vs track_all;lr 5e-4 vs 1e-3;step 25 vs 5(每区间 120 vs 595 窗);
连续 mini-batch vs 突发大批量重训(每 50 窗 200 步);共享阈值 vs 每臂自校准阈值。

- online_track **确实在适应**:漂移段重建误差稳定低于 frozen(如 regime1 0.030 vs 0.041),
  但**停在阈值的 ~2× 平台**(0.025–0.030 vs 阈值 0.0133),never 跌破 → FP 率不降。
- 加大力度(高 lr / 突发重训)边际收益递减,且开始**反噬 recall**(over-adapt 把异常也重建好了,R 0.96→0.85)。
- 离线全区间重拟合能到 0.011,在线流式停在 0.025–0.030。**~2–3× 的 gap 关不掉。**

**根因(推断)**:在线只能看近期、且 step=5 时缓冲内窗高度重叠(256 窗仅覆盖 ~1380 点的窄切片),
mini-batch 噪声 + 在线 eval + 紧阈值,使流式适应停在远高于离线全量重拟合的误差平台;
而把力度加到能压误差时,又过拟合近期窗、损伤 recall。

## 2b. 追加:稀疏缓冲假设被证伪 + 真正根因 + abrupt 也失败

- **稀疏覆盖缓冲(buffer 200 × stride 3 ≈ 全区间跨度)无效**:漂移段误差不降,regime-0 反而更差。
- **真正根因(决定性)**:online_track 的**训练损失(缓冲内 in-sample)= 0.006**(≈regime-0 质量,拟合得很好),
  但**逐窗记录的新到窗分数(out-of-sample)漂移段 = 0.025–0.033**。即**泛化滞后**:gradual 漂移每窗都在变,
  detector 拟合好缓冲时,新到的窗已漂得更远 → 永远落后。**探针的 0.011 是 in-sample,误导性**;
  流式真实的 out-of-sample 误差就是压不到阈值下。
- **abrupt(分段平稳)也失败**:本以为平稳区间内 online 能追上,但仍 100% FP。两个叠加原因:
  (1) 缓冲 600 窗跨度 ≈ 一个区间,abrupt 跳变后整段都在"旧+新"混合,直到区间末才纯净 → 没有干净的平稳段可收敛;
  (2) **阈值不可达**:regime-0 detector 离线训到 0.003,阈值仅 0.005;而 online-SGD 的误差地板 ~0.013,
  **连 regime-0 都过不了**(online regime-0 late FP 76–86%)。online SGD 无法保持离线 120-epoch 的过拟合质量。

**九组配置(gradual/abrupt × naive/track × lr × step × 连续/突发 × 共享/自阈 × 密/稀疏缓冲)全部 FP 90–100%。**

## 3. 最终结论(停止合成流调参)

**抗漂移主张部分成立但有重要反转**:漂移破坏冻结模型(✓)、漂移可学(✓),
但**朴素连续在线适应不足以恢复**(✗)——这本身是有价值的发现,不是失败。

**合成流这条路已穷尽,不再调参。** 失败是机制性的(泛化滞后 + SGD 地板 > 不可达阈值),非超参问题。

剩余可行方向(待定夺):
1. **(推荐)改用真实 machine-1-1 数据讲漂移**:decider 实验已证漂移 FP 在 regime 1/3,昨天 A(在线)vs C(冻结)
   +0.018——小而真,叙事最稳;且避开"阈值不可达"的合成流陷阱。
2. **重设协议**:别用过拟合的离线 detector 定阈(0.003→阈值0.005 不可达);改成两臂都在线 SGD 起步、
   各自部署校准,公平比"持续适应 vs 冻结"。
3. **重设机制**:reconstruction-AE + 连续 SGD 不适合;考虑更快的 test-time adaptation 或非重建检测器。
4. 把"朴素在线适应不足以恢复漂移"作为**论文的 motivating 负结果**,引出 decider / 更强适应。

## 附:产物
- 三臂结果与图:`code/runs/online/drift/{frozen,online_naive,online_track}.json` + `pred_*.csv` + `drift_main.png`
- regime-0 detector:`code/runs/drift_detector_regime0/checkpoint_best.pt`
- 探针:`scripts/probe_refit.sh`(A40,证明漂移可学)
- Slurm:19296641/19296956/19301027/19302535/19303395/19303514(detector 训练+三臂多配置)、19300358(探针)
