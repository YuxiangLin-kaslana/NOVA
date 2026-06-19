# SigLA 决定性实验日志 —— 证明 LLM decider 的价值

> 日期:2026-06-16 · 运行:Delta A100(job 19289248,单 A100,~15min)· LLM:gpt-4o-mini
> 承接 [2026-06-15 重设计日志](2026-06-15_calibrated_decision_redesign.md) 第 6 节「下一步」第 2/3 项

---

## 0. 要回答的问题

昨天确认:agent 在最优(已很紧)阈值上是橡皮图章(411 次调用 0 次否决),纯阈值天花板 ≈ 0.72。
论文主张:**故意把阈值开松到高 recall 工作点,让 agent 否决假阳性候选、把 precision 救回来——单标量阈值做不到。**
本实验在 q=0.97/m=1.0(纯阈值 P0.43/R0.97/F1 0.60,178 个 FP)上验证。

## 1. 做的改动(让 agent 从橡皮图章变怀疑者)

1. **喂判别信号**(`agent/gpt_instant.py` `to_payload`):候选窗除了会饱和的 `percentile_in_normal`,
   新增 `score_over_threshold`(分数/阈值,~1.0=刚过线的典型 FP,≥1.5=远超线的典型真异常)、
   `max_concept_prob`、`concept_persistence`(EMA)。
2. **重写 decider prompt**(`build_decider_instructions`):从「默认信任检测器、别翻案」改成
   **怀疑者**——明确告知运行在故意开松的高 recall 阈值、约一半候选是 FP,要求**主动否决**
   刚过线且概念证据弱的候选,并写明「一次都不否决=没比裸阈值多做任何事」。
3. **wire threshold**(`pipeline.py`):把校准阈值传入 AgentContext,使 `score_over_threshold` 可计算。

runner:`scripts/run_online_decisive.sh`(sbatch,两臂);分析:`scripts/analyze_veto.py`。

## 2. 结果 @ q=0.97 / margin=1.0(除 agent 外两臂配置完全一致:同校准器、online full)

| 臂 | P | R | F1 | FP |
|----|---|---|----|----|
| 纯校准阈值(无 GPT) | 0.433 | 0.971 | 0.599 | 178 |
| **校准 + GPT 怀疑者 decider** | **0.543** | 0.943 | **0.689** | **111** |
| Δ | **+0.110** | −0.029 | **+0.090** | −67 |

GPT 调用 348/1136(30.6%)。

**逐窗否决行为**(314 个候选全部送 agent):
- 否决 71 个(veto rate 22.6%):**杀掉真 FP 67 个,误杀真 TP 仅 4 个**;否决精度 **94.4%**(每误杀 1 TP 换 16.8 FP)。
- 确认 243(真 TP 132 / 假 FP 111);提拔正常→异常 0(decider 不找回漏报,符合设计)。

## 3. 结论

1. **橡皮图章修好** ✅:旧版 0 次否决 → 现在 71 次、94% 精度。判别信号 + 怀疑者 prompt 起效。
2. **越过单阈值 PR 前沿** ✅(核心主张):在**匹配 recall=0.943** 处,纯阈值 sweep 前沿插值 P≈0.49,
   agent 实测 **P=0.543,高出 +0.053**。同样 recall 下 agent 给出更高 precision——这是单标量阈值办不到的,
   即「为什么需要 LLM」的直接证据。
3. **但绝对 F1 尚未超过纯阈值全表最优**(agent 0.689 < 纯阈值最优 0.716 @ q0.99/m1.0 区段)⚠️:
   因为 agent **仍偏保守**——只否决了 178 个 FP 中的 67 个,还留着 111 个 FP(veto rate 仅 22.6%)。
   否决精度高达 94%,说明**还有大量安全的否决空间没用**。

## 3b. 追加(同日):把怀疑者调更狠(gate=1.3)—— 反而更差,并揭示真正瓶颈

按"score_over_threshold≥1.3 或 强概念证据才确认"重写 prompt 后重跑(job 19291720):

| | P | R | F1 |
|---|---|---|----|
| 纯阈值基线 | 0.439 | 0.979 | 0.606 |
| **v2 激进 decider** | 0.616 | 0.664 | **0.639** |
| v1 温和 decider | 0.543 | 0.943 | **0.689** |
| 纯阈值反事实 @ sot≥1.3 | 0.502 | 0.921 | 0.650 |

- veto rate 52%,但否决精度从 94%→73%,**误杀 44 个真异常**,recall 崩到 0.66。F1=0.639 **低于** v1,也**低于** sot≥1.3 纯阈值反事实(0.650)。
- **LLM 违反自己的 gate**:被误杀的 44 个 TP 平均 sot=4.68(86% ≥1.3),clause(a) 本应无条件确认 —— gpt-4o-mini 因其概念信号弱(均值0.42)而误杀,过度依赖概念、忽略高分。激进 prompt 把模型逼成了不一致的机械算术,反受其害。
- **离线规则天花板**:在 v2 CSV 上网格搜 (sot_gate, concept_gate) 的 OR 规则,最优仅 F1=0.687。**任何 sot+concept 规则在高 recall 都过不了 ~0.69**。

**根因(决定性诊断)**:sot≥1.3 候选区有 129 TP / 128 FP,提 precision 的硬骨头是这 128 个 FP。它们的 regime 分布:
- 真 TP:regime 2 占 122/129。
- 硬 FP:regime 1(41)+ regime 3(68)= 109/128,几乎不在 regime 2。

即**这些"远超阈值却为假"的报警是漂移诱发的**:regime 1/3 是漂移正常段,(未充分在线适应的)detector 重建误差因良性漂移而偏高,且概念检测器在该段也误激活(concept 均值 0.498)。**没有任何决策信号能把它们与真异常分开**——这不是 decider 能修的,是 detector 抗漂移问题。昨天 per-regime F1 中 regime3≈0.22 与此完全一致。

## 3c. 修正后的结论

1. **可支持的主张(弱形式,v1)**:温和 decider 在匹配高 recall 处**优于单阈值 PR 前沿**——v1 实测 tp=132/fp=111,严格优于 sot≥1.3 阈值的 tp=129/fp=128(多 3 TP、少 17 FP)。这是真实、可发表的"agent>阈值"证据,**但增益是适度的(+0.05 P)**。
2. **不可支持的主张(强形式)**:在 machine-1-1 高 recall 工作点,**decider 无法把 F1 推过 0.72**;0.716 的天花板本就位于更低 recall 段。瓶颈是漂移诱发的硬 FP,decision-only(LLM 或规则)修不了。
3. **把 prompt 调更狠适得其反**(v2):弱模型不再遵守 OR-gate、过度否决,F1 反降。**v1 是要报告的工作点,不要再加码 prompt。**

## 4. 修正后的下一步(优先级已变)

1. ~~把怀疑者调更狠~~ **已做(3b),证伪**:harsher prompt 反而更差,瓶颈不在决策激进度而在漂移 FP。
   **采用 v1 温和 decider 作为"agent as decider"的报告结果**(上方 3c.1)。
2. **抗漂移成为主线(原第 4 项升为最高优先级)**:硬 FP 109/128 集中在 regime 1/3 漂移段,
   决定性证据表明 precision 天花板是 detector 抗漂移问题。跑合成漂移流(`drift_gradual`,已知漂移点)
   的在线适应 vs 冻结对照,直接验证"在线适应压低漂移段 FP→precision 回升"。这才是能真正提 F1 的杠杆。
3. (可选)给 agent 喂 regime/drift 上下文,让它在漂移段更激进否决——但属推测,优先级低于 2。
4. 多机器/多 seed 复现给误差棒(machine-1-x + 随机 seed)。

## 附:产物路径
- 两臂结果:`code/runs/online/decisive/{thr_q97,agent_q97}.json` + `pred_*.csv`
- 分析脚本:`code/scripts/analyze_veto.py`(veto rate / FP 否决 / 误杀 / PR 前沿对比)
- Slurm:job 19289248(单 A100,walltime 25min,实际 15min)
