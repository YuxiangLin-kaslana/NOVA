# Online_training —— 无真值在线训练数据流

为**无真值在线适应(label-free online adaptation under drift)**实验生成的
合成时间序列流。训练全程不使用标签；标签只用于离线评测。

## 内容

| 文件 | 说明 |
|------|------|
| `make_stream.py` | 生成器：合成带概念漂移的流，复用 [Concept_detector/concept_synth.py](../Concept_detector/concept_synth.py) 的 6 个概念注入函数 |
| `streams/drift_gradual.npz` | 渐变漂移流(已生成，T=12000, n_vars=38) |
| `streams/drift_abrupt.npz` | 突变漂移流(已生成) |
| `streams/*.json` | 每条流的漂移点 + 概念调度 + 生成参数 |

## 漂移设计

- **协变量漂移 P(X)**：基础正常信号的频率/均值/幅度逐区间漂移(渐变或突变)。
- **概念漂移**：每个区间异常的概念构成不同(区间1: spike/oscillation → 区间2:
  level_shift/trend → ...)，模拟"异常形态本身在变"。
- 4 个等长区间，漂移点 `[3000, 6000, 9000]`，异常占比 ~7.5%。

## npz 字段

```
x            float32 [T, 38]   流(归一化基础 ~[0,1]，注入处可越界)
y            int64   [T]       逐点异常标签 —— 仅评测用，不进训练
regime       int64   [T]       每点所属漂移区间
drift_points int64   [3]       漂移发生点
```

## 重新生成

```bash
python make_stream.py --out streams/drift_gradual.npz --drift gradual --seed 0
python make_stream.py --out streams/drift_abrupt.npz  --drift abrupt  --seed 1
# 可调：--length --n_vars --win_size --n_regimes --anomaly_rate
```

## 怎么被消费

由在线训练 runner([code/scripts/run_online.py](../../code/scripts/run_online.py))逐窗读取：
detector + concept detector → agent 判异常并给概念伪标签 → OnlineTrainer 在线重训
两个小模型(detector 自监督重建 / concept BCE 伪标签)。详见
[scripts/run_online_training.sh](../../code/scripts/run_online_training.sh)。
