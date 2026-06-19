# 最后一公里:开放词表闭环 → 前兆窗**类型化**早预警

> 日期:2026-06-18 · A100(job 19359345)· gpt-4o-mini · `scripts/exp_early_warning.py`(3 seeds)

## 目标
把开放词表闭环接到论文的「问题」——**前兆窗早预警时序**。每个事件前有前兆窗 [onset-L_MAX, onset-L_MIN](L=[2,8]),
携带该类型签名的**弱化版**(strength=0.6,point_label=0);早预警成功 = 在前兆窗内报警。问:闭集检测器能否
**提前预警从未见过的异常类型**?开放词表自举能否把它救回来?复用 `ovbench.make_window_strength` + `precursor` 口径。

## 关键设计教训(两个 artifact,均已修)
1. **在线重训必须回放预训练池**。首版漏了 normal+已知 replay → bootstrap 正常背景窗误报率(FAR)49% → "100%
   早预警"是"到处报"失真。加回 replay 后降到 ~23%,仍不够。
2. **二分类报警对强信号无区分力**。alarm=argmax≠normal 或校准 score>阈:对强信号 novel(trend),**闭集即便不
   认识也会报"有异常"(二分类 EW=100%)**,且在线重训会侵蚀 score 校准(bootstrap 二分类 FAR 38%)。
   → 改用**类型特定报警**:alarm = 预测**就是新类型 T**。这才是开放词表独有的能力,且 normal 不会被判成
   well-separated 的 T(type-FAR 受控)。
3. **novel 类型选择**:correlation_break 是"靠缺席发现、与 normal 内在可混"的类型,学了它会塌 normal 校准
   (FAR 60%),**不适合早预警**;改用签名清晰、远离 normal 的 **trend**(lin_r2 满强度 z+36,0.6 前兆仍强可分)。

## 结果(novel=trend,known=spike/level_shift/oscillation,3 seeds)
### Headline:类型化早预警(前兆窗内报出**正确新类型** T)
| | frozen | bootstrap |
|---|---|---|
| 类型化早预警 recall | **0% ± 0%**(无 T 类,永远命名不出) | **100% ± 0%** |
| lead-time(窗) | — | **8.0**(=满前兆窗宽 7+) |
| 类型 FAR(正常被判 T) | 0.0% | **4.8%**(受控) |

### 对照:二分类"任意异常"早预警(只问报不报,不问类型)
| | frozen | bootstrap |
|---|---|---|
| 二分类早预警 recall | 100% | 100% |
| 二分类 FAR | 5%(校准) | 38%(在线重训侵蚀 score 校准) |

## 结论
**闭集检测器即便能(对强信号)提前报"有异常",也永远报不出是哪种新类型(类型化 EW=0)。开放词表闭环自举后,
能在前兆窗提前 8 窗报出正确的新异常类型(100%),且正常几乎不误判为 T(type-FAR 4.8%),全程无人工标注。**
这把「方法(开放词表持续学习)」干净落到了「问题(前兆早预警)」上——本周三段实证(A 单新类分类 / B 多新类分类 /
D 检测桥)的时序闭环完成。

## 诚实 caveat
- "frozen 类型化 EW=0"部分 by construction(没有 T 类自然命名不出);其价值在于**结合 lead-time + 无标注自举 +
  低 type-FAR** 的端到端演示,而非裸"闭集盲"。
- 对 **normal-邻近**的弱信号类型(correlation_break),类型化 EW 也会因 type-FAR 升高而退化——是方法的已知边界。
- 在线重训会侵蚀**二分类** score 校准(boot 二分类 FAR 38%);类型化口径不受影响,但若要二分类早预警需在线重标定阈值。
- 单一 novel=trend、3 seed;可扩展到多 novel 类型 / 多 seed。

## 文件
`scripts/exp_early_warning.py`(env `OVE_NOVEL`、`OVE_NSEED`);结果 `runs/early_warning_result.json`。
`sigla_exp/ovbench.py` 新增 `make_window_strength`(强度可调注入,供前兆)。
