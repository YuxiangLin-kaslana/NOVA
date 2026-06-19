# 前兆窗口感知早预警评测 —— 接入 + 验证

> 日期:2026-06-16 · 复活论文第一核心组成(前兆设定),接进 run_online.py

## 做了什么
- 新建 `sigla_exp/precursor.py`:复活 `actions.event_regions`(0稳定/1有效前兆/2迟滞/3事件中后),
  实现**事件级早预警**口径:报警落在有效前兆窗 [onset-l_max, onset-l_min] 内才算成功;
  分 有效/迟滞/事件后/漏;报 lead-time;算"虚高暴露"=普通recall − 早预警recall。
- 接进 `scripts/run_online.py`(--l_min/--l_max),所有实验自动输出普通AD + 早预警两套口径。
- 单元测试 5/5 通过(有效/迟滞/事后/误报区分 + lead-time + 虚高)。

## 真实 SMD 验证(machine-1-1,calibrated_threshold,8 events,l_min=25 l_max=200)
| 口径 | P/R/F1 |
|---|---|
| 普通 AD | 0.439/**0.979**/0.606 |
| 早预警 | 0.000/**0.000**/0.000 |

事件归类:有效=0 迟滞=0 事件后=5 漏=3;报警分布 stable=207 valid=0 late=0 post=105;虚高=0.625。

## 结论(对论文重要)
**普通 AD 的 0.98 recall 完全空心**:全部是事件中/后检测,**零有效早预警**。重建检测器只在异常
正发生时才 spike,onset 前不报 → 是 detector 不是 early-warning。这实锤了论文"避免靠临近 onset
刷虚高"的设定,并直接 motivate "需要一个真正会提前预警的系统"。

## 嵌入路线 B
早预警口径 = 路线 B 的 END 指标:新类型闭环实验要打到"新型事件前兆窗内早预警 F1 / lead-time",
而不是概念识别率。下一步需要:让系统从前兆窗学到可提前的信号(否则 EW 恒为 0)。
