# SOTA 对比实验(novel anomaly TYPE / concept drift)

目标:证明出现**从未见过的新异常类型**时,SigLA 的 LLM 开放词表自举闭环相对前人 SOTA 有价值。
所有臂跑**同一条流、同一套指标**(复用 `scripts/exp_detection_tie.py` 与 `scripts/exp_early_warning.py` 口径)。

## 对比臂

| 臂 | 代表思路 | 机制 | 人工标签 | 类型概念 |
|---|---|---|---|---|
| Frozen-CNN | 闭集分类 | normal+3已知类,从不扩词表 | 0 | 已知类型 |
| AnomalyTransformer (ICLR'22) | 无监督重构 SOTA | 关联差异 + minimax,冻结 | 0 | 无(仅分数) |
| MemStream (WWW'22) | 抗漂移 SOTA | 记忆库 + 在线更新吸收漂移 | 0 | 无(仅分数) |
| **Ours (bootstrap)** | LLM 开放词表闭环 | 证据门控→LLM命名→在线扩词表+类平衡重放 | 0 | 已知+自举新类型 |
| (上界,待加) Oracle/iCaRL | 人工标签上界 | 全量人工新类标签 | 全量 | 已知+人工新类型 |

## faithfulness 说明(论文需交代)

baseline 是各方法**核心机制**的忠实紧凑再实现(faithful compact re-implementation),
适配到本项目窗口化多变量流 + 统一 `score_stream` 接口:
- **MemStream**:去噪 AE + 正常窗记忆库;**score=AE 重构误差**;score<β(看起来正常)时把窗写入记忆并周期性
  在记忆上轻量在线再训 AE(吸收漂移)。注:作者原版 score=嵌入到记忆最近邻距离,但**经诊断**(`diag_memstream.py`)
  该距离在本合成 benchmark 对注入异常**非判别**(各异常 recall@5%FAR ≈ 8–12% ≈ 瞎猜),故采用重构误差的
  faithful AE 变体(已知异常 recall 98–100%);抗漂移的 memory+在线更新机制保留。重构误差**仍对 correlation_break
  盲(8%)**——印证"靠缺席发现的新类型逃过重构检测器"。
- **AnomalyTransformer**:每层 series-association(学习注意力)与 prior-association(可学习高斯);
  对称 KL=关联差异;minimax 训练;窗级异常分 = mean_t[softmax(−AssDis)·重构误差]。

未用作者原始 repo 的原因:原 repo 绑定各自 SMD npy 加载、不产出逐窗 score+type、且计算节点无网。

## 运行(GPU via sbatch,勿在登录节点跑)

```bash
cd /u/ylin30/sigLA/code
sbatch sota_compare/smoke.sh        # 冒烟:1 seed、小规模、跳过 LLM,验证 shape/流程
sbatch sota_compare/run_compare.sh  # 全量:CMP_NSEED(默认5),含 LLM(读 .env 的 OPENAI_API_KEY)
```

结果 JSON:`code/runs/sota_detection_compare.json`。

## 指标

涌现段(novel 涌现后):新类**检测召回**(pred≠normal)、整体检测 F1/精度/召回、新类**分类**准确率(pred==NOVEL)。
预期故事:无监督 SOTA 检测召回或不低,但 **新类分类≡0**(无类型概念);ours 在零人工标签下同时拿到检测+命名。

## TODO

- [ ] 早期预警同口径对比(`run_ew_compare.py`,复用 `exp_early_warning`):类型化 EW recall + lead-time + type-FAR。
- [ ] 加 Oracle/iCaRL 人工标签上界臂。
- [ ] 加 TranAD / DCdetector(凑齐闭集 SOTA 阵列);GCD(新类发现,放最后)。
- [ ] 真实 SMD 数据的 novel-type 切分 demo。
