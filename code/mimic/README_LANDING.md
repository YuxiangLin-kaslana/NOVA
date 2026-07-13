# MIMIC 临床笔记 → 多模态早预警落地方案

**目标**:在真实 ICU 数据上检验"加 LLM/文本是否在体征之外带来**正交增量**"——这是 Time-MMD(文本复述序列→冗余、无增量)之后,"LLM 加值"唯一未被证否的真实场景。MIMIC 的放射报告/笔记带**体征里没有的信息**(影像发现、症状、医生判断),理论上有正交增量。

---

## 0. 关键坑(必须遵守,否则结论无效)

- **不要用出院小结(discharge summaries)**:出院时写的 → 含结局 → 对院内早预警是**时间泄漏**。
- **只用住院期间带时间戳的文本**:MIMIC-IV-Note 的 **radiology 报告**(有 `charttime`/`storetime`);若用 MIMIC-III,可用 `NOTEEVENTS` 的 nursing/physician 进度笔记(更密集)。
- **严格时间因果**:决策时刻 t,只喂 `charttime ≤ t`(且报告**已 store**,`storetime ≤ t`)的文本与 `≤ t` 的体征。放射报告的 storetime 常晚于 charttime,**用 storetime 更保守**。

---

## 1. 数据获取(你本人做,我下不了)

1. 注册 PhysioNet 账号:https://physionet.org/register/
2. 完成 **CITI "Data or Specimens Only Research"** 培训(免费,~2–3h),上传完成报告申请 credentialing。
3. 在各数据集页面**签 DUA**:
   - **MIMIC-IV v3.1**:https://physionet.org/content/mimiciv/3.1/ (`hosp` + `icu` 模块:admissions/patients/chartevents/labevents/d_items)
   - **MIMIC-IV-Note v2.2**:https://physionet.org/content/mimic-iv-note/2.2/ (`radiology`, `radiology_detail`;discharge 不用)
   - (可选)**MIMIC-III v1.4** 若要更密集的进度笔记:`NOTEEVENTS`
4. 下载(示例):
   ```bash
   wget -r -N -c -np --user <USER> --ask-password \
     https://physionet.org/files/mimiciv/3.1/
   wget -r -N -c -np --user <USER> --ask-password \
     https://physionet.org/files/mimic-iv-note/2.2/
   ```
   下到 `/u/ylin30/sigLA/data/mimic/`。(也可走 GCP BigQuery,但本地 CSV 更省事。)

---

## 2. 队列 + 标签(强烈建议复用现成 pipeline,省几个月)

- **MIMIC-Sepsis**:https://github.com/yongh7/MIMIC-sepsis —— 35,239 ICU 病人、**Sepsis-3** 标签、时间对齐的体征/实验室/治疗、含预处理代码。直接产出 (stay_id, hourly features, onset time)。
- **任务**:Sepsis-3 onset 的**前兆早预警**——在 onset 前 H 小时窗内,用 ≤t 的体征 [+ ≤t 的笔记] 预测未来 [t, t+H] 是否 onset。对齐我们 Time-MMD 的设定(无泄漏、matched FA / CV-AUC)。
- **更简单的备选标签**(若 Sepsis-3 派生太重):院内死亡 / 升压药起始 / 机械通气起始 —— 都是有明确时间戳的"恶化"事件。

---

## 3. 实验(骨架已写:`exp_mimic_fusion.py`)

与 `sota_compare/exp_mmd_fusion2.py` **同方法论**(严格不偷看):
- **vitals 通道**:近 K 小时体征(HR/MAP/RR/SpO2/Temp 等)的统计特征。
- **text 通道**:决策时刻 ≤t 的最近放射报告 →(A)LLM 读报告给风险分,或(B)临床文本 embedding 特征。
- **学习式融合**:5 折 CV 逻辑回归(或 GBDT),比 **[vitals] vs [vitals,text]** 的**留出 AUC**。
- **判据**:`[vitals,text] 的 CV-AUC 显著 > [vitals]`(Δ 超 ±std)→ **笔记带正交增量,多模态真有用**(这正是 Time-MMD 给不出的)。

`exp_mimic_fusion.py` 里 `load_cohort()` / `load_vitals()` / `load_notes()` 是数据接口(标了 TODO,按你下到的路径/schema 填)。其余(决策点构建、通道打分、CV 融合、报表)已就绪。**数据一到我来跑/调。**

---

## 4. 预期与诚实预案

- **若 [vitals,text] > [vitals]**:拿到"LLM/文本正交增量"的**正面真实证据** → 这是论文的强支撑(且能把之前的负结果升级成"何时不行/何时行"的完整故事)。
- **若 Δ≈0**:即便真实临床笔记也冗余 → 那"LLM 加值"基本被彻底证否 → 转向贡献 B / 负结果论文。无论哪样都是干净结论。

Sources:
- MIMIC-IV-Note v2.2: https://physionet.org/content/mimic-iv-note/2.2/
- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- MIMIC-Sepsis benchmark + code: https://github.com/yongh7/MIMIC-sepsis (arXiv 2510.24500)
