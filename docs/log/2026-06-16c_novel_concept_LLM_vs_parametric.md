# 路线B motivating 实验:新异常类型识别 —— LLM zero-shot vs 参数化

> 日期:2026-06-16 · A100(job 19312246)· gpt-4o-mini · 受控合成(可控注入器 + 相关基底)

## 设定
6 类概念,已知 5 类(spike/level_shift/oscillation/variance_burst/trend)进概念检测器训练;
**correlation_break 留出**(训练从不出现),仅测试出现。同一批窗上比"正确命名该类"的识别率:
- 参数化:CNNConceptDetector(6 输出,但 correlation_break 维从无正样本 → 死的)。
- LLM:同样的 6 个通用统计证据(含全窗平均跨通道相关性)+ 6 类定义 + 正常基线,zero-shot 命名。
公平性:① 已知类型参数化必须很好;② LLM 拿通用统计非答案。脚本 `scripts/exp_novel_concept.py`。

## 结果
| 概念 | 参数化 | LLM |
|---|---|---|
| spike | 100% | 100% |
| level_shift | 98.3% | 100% |
| oscillation | 100% | 100% |
| variance_burst | 100% | 75% |
| trend | 100% | 100% |
| **correlation_break(新类型)** | **0.0% (avg p=0.00)** | **86.7%** |

已知 5 类参数化平均 **99.7%**;新类型参数化 **0%** vs LLM **86.7%**。

## 结论与诚实定位
- ✅ **机制坐实**:闭集参数化检测器对训练时没有的新类型**结构性失明**(输出维死,无标签也救不回);
  LLM 用语义先验 + 通用证据 **zero-shot 86.7% 识别**。已知类型参数化 99.7% 证明它不是菜、是专门对新类型盲。
- ⚠️ **这不是论文的新贡献,是 motivating 证据**:裸结果("闭集输出不了留出类")部分是 by construction,
  审稿人会说 expected(OSR/ZSL 老结论)。**因此它只当 motivating 小图,不当 headline。**
- **真正的贡献仍待做**:LLM 自举闭环(zero-shot 命名 → 伪标签 → 在线扩词表)+ 绑早预警的 END 指标。
  这个实验只证明了"问题存在 + 闭集失明",还没证明"闭环能无标注地长出新类并提升早预警"。

## 下一步
1. **闭环实验**:correlation_break 涌现后,用 LLM 伪标签在线训练参数化检测器 → 后续同类识别率从 0% 爬升;
   对比 LLM-once / 人工标注上界。这才是 headline。
2. 绑早预警:在带 onset 的流上,量"新型事件的前兆窗口内早预警 F1 / lead-time"。

附:`code/runs/novel_concept_result.json`,Slurm 19312246(及失败前身 19307523/19311731,SMD 基底不可用,已弃)。
