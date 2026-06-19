# 路线B 多新类涌现跑通:证据正交基准打破 multi-type blocker

> 日期:2026-06-17 · A100(job 19344653)· gpt-4o-mini · `sigla_exp/ovbench.py` + `scripts/exp_openvocab_multi.py`

## 背景:之前为什么崩
首版多新类(留出 variance_burst/trend/correlation_break)只学会 1/3(variance_burst ~1%、trend 27%、
correlation_break 0%)。根因是**证据非正交**:旧 evidence 里
- trend↔level_shift 纠缠(半窗中位数阶跃:斜坡也有大阶跃 → 都抬 step);
- variance_burst↔spike 纠缠(噪声爆发的方差混合=重尾 → 抬 max-z/峰度);
- variance_burst↔oscillation 纠缠(噪声是宽带 → 抬高频能量占比)。
新颖门控靠"主导统计量不在已知签名集"判新类,纠缠导致新类误触已知签名 → 不被判新颖 → 不建类。

## 修复:`sigla_exp/ovbench.py`(证据正交基准)
把 4 个签名换成**对纠缠维度鲁棒**的版本,并微调 2 个注入器,使**每概念恰好且仅触发自己的签名**:

| 概念 | 签名统计量 | 为什么正交 |
|---|---|---|
| spike | `kurtosis` | 稀疏极值 ^4 主导;爆发幅度调低后近高斯不触发 |
| level_shift | `local_step`(局部窗口中位数突变) | 斜坡逐窗仅 slope*w → 不触发 |
| oscillation | `spectral_peak`(**高频带**单频占比) | base 低频正弦不计;宽带噪声分散 → 不触发 |
| variance_burst | `var_localiz`(高通残差分段 max/median MAD) | spike 稀疏/平滑结构 → ≈1 |
| trend | `kendall`(值-时间 Kendall τ) | 台阶 τ≈0.5、正弦 τ≈0 → 不触发 |
| correlation_break | `decorr`(1−滑窗最小平均\|corr\|) | 靠"不触发任何已知签名"被发现 |

注入器关键改动:variance_burst 调低幅度(sd 0.7–1.0)避免重尾;**correlation_break 改用每通道不同的
循环移位 + 边界交叉淡化** —— 精确保留每通道边际/局部纹理(var_localiz/kurtosis 全不变),只打散跨通道
对齐。这一步把 correlation_break 误触 var_localiz 从 z=+9.4 压到 **+0.0**(否则 LLM 会把它误名为 variance_burst)。

**可分性 sanity(`diag_separation_v2.py`,job 19344597):** 每概念自身签名 z(spike kurtosis +62、level_shift
local_step +6.3、oscillation spectral_peak +37、variance_burst var_localiz +5.1、trend kendall +14.6、
correlation_break decorr +2.9),且**每个新类不触发任何已知签名**(其余列 z 均 <2)。✅ gate 可用。

## 两处追加修复(把 38–64% 抬到 91%)
**(a) trend 签名 kendall→lin_r2,level_shift 注入器去趋势。** 命名诊断(`diag_naming_acc.py`)发现:用
kendall 时 level_shift 的台阶单调,kendall z(+11.4)甚至高过它自己的 local_step(+6.3)→ gate 把 level_shift
误判 novel、LLM 把它命名 trend(48%)→ **trend 类在涌现前被提前建出**。改用线性拟合 R²(纯斜坡≈1,去趋势台阶≈0)
后,level_shift 的 lin_r2 = −0.2(原 +11.4),纠缠结构性消除。
**(b) LLM 命名喂 z-score 而非原始值。** gpt-4o-mini 不会比较原始数值,一律锚定 level_shift(新类命名仅 40%)。
改喂"每统计量偏离正常几个 sd"后只需找最大 z 再语义映射 → 新类命名 40%→76%(trend 100%、correlation_break 24%→64%)。

## 多新类结果(job 19344876,单 seed,known=3 类,3 新类错峰涌现 onset 150/350/550)
| 新类 | frozen | bootstrap | 涌现后曲线(4 段) |
|---|---|---|---|
| variance_burst | 0% | **67%** | [0.54, 0.65, 0.71, 0.77] ✓单调爬升 |
| trend | 0% | **100%** | [1.0, 1.0, 1.0, 1.0] |
| correlation_break | 0% | **59%** | [0.33, 0.5, 1.0, 0.6] |

全段(3 新类都活跃):frozen 59% → **bootstrap 91%**(首版 70%)。词表干净增长 3→4→6,**warm-up 期 LLM
调用率仅 0.05–0.08**(level_shift 不再误判 novel,无提前建类)。3 个新类全部无标注长出。

## 命名混淆现状(`diag_naming_acc.py` job 19344875)
spike/level_shift/oscillation/trend 命名 100%;variance_burst 64%(混 spike/oscillation)、correlation_break
64%(混 trend)。检测器在 ~36% 标签噪声下仍学到 59–100%(类平衡重训鲁棒)。

## 多 seed 误差棒(job 19344981,5 seeds,`OVM_NSEED`)
| | frozen | bootstrap |
|---|---|---|
| 全段(3 新类活跃) | 50.6% ± 4.8% | **88.1% ± 2.9%** |
| variance_burst | 0% | 67% ± 2%(curve 0.60→0.73) |
| trend | 0% | **100% ± 0%**(每 seed 满分) |
| correlation_break | 0% | 56% ± 11%(最抖) |

词表 **100% 长全**(3→6),warm-up 期 LLM 率仅 0.08–0.10(无提前建类),词表均值曲线
`[3.2,3.4,5.2,5.4,5.6,5.6,5.6,6.0,6.0,6.0]` 阶梯增长。**B 完成,与 task A 同等严谨度(多 seed)。**

## 仍待加强(诚实 caveat)
- correlation_break 56% ± 11% 最弱最抖(decorr 签名弱 +2.9、命名 64%);variance_burst 命名 64%。
  可选改进:再调这两个签名判别度 / 增强 correlation_break 注入强度。非阻塞。
- 后续:文献定位、把多新类接二分类检测端到端、与 LLM-as-labeler / open-vocab AD 对照。

## 文件
- 新基准 `code/sigla_exp/ovbench.py`(取代旧 `scripts/clean_bench.py`,后者证据未真正正交)。
- `exp_openvocab_multi.py` 已切到 `import sigla_exp.ovbench as CB`。
- 复现:`sbatch scripts/exp_openvocab_multi.sh`;可分性:`sbatch scripts/diag_separation_v2.sh`。
