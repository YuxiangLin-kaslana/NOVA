"""SOTA 对比实验包(独立于已验证的 scripts/，复用 sigla_exp.ovbench 同口径)。

证明:出现**从未见过的新异常类型**(concept drift)时,SigLA 的 LLM 开放词表自举闭环
相对前人 SOTA 的价值——同口径对比 novel-type 检测召回 + 类型化早预警。

baseline 均为各方法**核心机制**的忠实紧凑再实现(faithful compact re-implementation),
适配到本项目的窗口化多变量流 + 统一接口,非作者原始代码(原 repo 绑定各自 SMD npy 加载、
不产出逐窗 score+type,且计算节点无网)。见 README.md 的 faithfulness 说明。
"""
