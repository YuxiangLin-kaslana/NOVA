# 当前代码框架说明

最后更新：2026-05-30

本文档描述 `/u/ylin30/sigLA` 当前可运行代码的结构、输入输出、模型层数、训练方式和评估方式。主线代码是 `code/sigla_exp`；`CANDI/` 是保留在仓库中的参考/对比实现，不是当前 minimal SigLA 训练入口。

## 1. 总体结构

```text
/u/ylin30/sigLA
├── code/
│   ├── README.md
│   ├── sigla_exp/
│   │   ├── data.py        # 数据读取、标准化、滑窗 Dataset
│   │   ├── profiles.py    # 5 类 concept profile 特征
│   │   ├── actions.py     # 弱监督 action label 生成
│   │   ├── models.py      # MLP AutoEncoder 与 SigLA Policy
│   │   └── train.py       # 当前训练入口
│   ├── eval/
│   │   ├── eval_autoencoder.py       # AutoEncoder 阈值评估
│   │   └── plot_autoencoder_eval.py  # AutoEncoder 评估曲线 SVG
│   ├── scripts/
│   │   └── train_autoencoder_delta.sbatch
│   └── runs/              # 训练、评估输出
├── data/                  # SMD/SWaT/npz 数据目录
├── docs/                  # 本文档所在目录
└── CANDI/                 # 外部/参考 baseline 代码
```

当前 `code/sigla_exp` 实现的是一个早期实验框架，而不是论文中完整的 RL / preference-training 系统。它包含两条训练路线：

1. `autoencoder`：CANDI-style MLP reconstruction autoencoder，用重构误差做异常分数。
2. `policy`：Signal-Profile-Action policy，用弱监督 precursor-window 标签做 behavior cloning。

## 2. 数据输入与输出

### 2.1 支持的数据入口

训练入口为：

```bash
cd /u/ylin30/sigLA/code
python -m sigla_exp.train --task autoencoder --dataset SMD_1-7
python -m sigla_exp.train --task policy --dataset SMD_1-7 --policy_split test
```

`--dataset` 支持：

| dataset 参数 | 读取方式 | 原始输入要求 |
| --- | --- | --- |
| `synthetic` | `make_synthetic()` 生成 | 无需外部文件，默认长度 4000，变量数 5 |
| `SMD_1-7`、`SMD_1-8` 等 | `load_smd()` | `data/ServerMachineDataset/preprocessed/machine-{entity}_train.pkl`、`machine-{entity}_test.pkl`、`machine-{entity}_test_label.pkl` |
| `SWaT` | `load_swat()` | `data/SWaT/SWaT_Dataset_Normal_v1.csv` 与 `SWaT_Dataset_Attack_v0.csv` |
| `*.npz` 文件路径 | `load_npz()` | 必须包含 `train`、`test`、`test_label` 或 `test_labels`，可选 `train_label` |

`.npz` 格式的形状约定：

```text
train:      [time, variables]，或一维 [time]
test:       [time, variables]，或一维 [time]
test_label: [time]，0/1 point-level anomaly label
train_label:[time]，可选；默认全 0
```

如果输入是一维时间序列，代码会自动转换为 `[time, 1]`。除此之外，时间序列必须是二维 `[T, C]`，其中 `T` 是时间长度，`C` 是变量数。

### 2.2 数据标准化与 split

所有数据会被封装为：

```python
DatasetBundle(
    train=SplitData(x=train_x, y=train_y),
    val=SplitData(x=val_x, y=val_y),
    test=SplitData(x=test_x, y=test_y),
    n_vars=C,
)
```

处理流程：

1. train/val 按时间顺序切分，不 shuffle。默认 `train_ratio=0.8`。
2. `StandardScaler` 只在 train split 上 `fit`。
3. train、val、test 都用同一个 scaler 做 transform。
4. SMD/SWaT 的 train label 默认全 0，test label 来自数据集文件。
5. 输出 `x` 为 `float32`，输出 `y` 为 `int64`。

### 2.3 滑动窗口 Dataset

`WindowDataset` 把 point-level 时间序列转换成 window-level 训练样本。

默认参数：

```text
win_size = 50
step     = 5
l_min    = 20
l_max    = 120
```

窗口索引：

```text
starts = 0, step, 2*step, ...
ends   = starts + win_size - 1
```

窗口标签：

```text
window_label = 1 if 窗口内任意 point label 为 1 else 0
```

每个 `__getitem__` 返回：

| key | shape / type | 含义 | 用途 |
| --- | --- | --- | --- |
| `signal` | `[win_size, n_vars]`, `float32` | 标准化后的原始窗口 | AE 输入；Policy signal 输入 |
| `score` | `[win_size, 1]`, `float32` | 每个时间点的 RMS 分数：`sqrt(mean(window^2 over variables))` | Policy score 输入 |
| `profile` | `[5]`, `float32` | 5 类 concept profile 校准后特征 | Policy profile 输入 |
| `label` | scalar `float32` | window-level anomaly label | Policy risk head 训练 |
| `action` | scalar `long` | 弱监督 action class id | Policy action head 训练 |
| `arg` | scalar `long` | 绝对幅值最大变量 id | Policy argument head 训练 |
| `end_idx` | scalar `long` | 窗口末端 point index | 分析/对齐 |

注意：`autoencoder` 任务只使用 `signal`，但当前实现仍通过 `WindowDataset` 计算 `score/profile/action`，因为它和 policy 共用数据管线。

## 3. Concept Profile

代码位置：`code/sigla_exp/profiles.py`

当前 profile 不是深模型，而是轻量统计特征。共有 5 个 concept：

```python
CONCEPT_NAMES = (
    "spike",
    "level_shift",
    "seasonal_break",
    "contextual_deviation",
    "correlation_break",
)
```

### 3.1 原始 evidence

给定一个窗口 `window`，形状为 `[win_size, n_vars]`，`extract_raw_evidence()` 输出 `[5]`：

| concept | 当前计算方式 |
| --- | --- |
| `spike` | 计算窗口值和一阶差分的 robust z-score 最大值，经过 sigmoid 映射 |
| `level_shift` | 将窗口分成左右两半，比较左右 median 的最大差异，并用全局 MAD scale 归一化 |
| `seasonal_break` | 对去均值信号做 FFT，结合频谱 entropy 和 high-frequency energy |
| `contextual_deviation` | 比较最后一个时间点与历史 5/50/95 分位范围的偏离程度 |
| `correlation_break` | 多变量且长度足够时，比较左右半窗口相关矩阵的平均绝对变化 |

输出范围经 sigmoid 后大致在 `[0, 1]`。

### 3.2 校准方式

训练开始时会在 train split 上拟合 `ConceptProfileExtractor`：

```python
extractor = ConceptProfileExtractor.fit(
    bundle.train.x,
    win_size,
    step,
    max_windows=profile_max_windows,
)
```

默认最多采样 `profile_max_windows=512` 个 train windows。拟合得到：

```text
median: [5]
mad:    [5]
```

窗口 profile 变换为：

```text
raw        = extract_raw_evidence(window)
calibrated = sigmoid((raw - median) / mad)
profile    = calibrated.astype(float32)
```

因此 policy 看到的是相对于正常训练窗口分布校准后的 profile，而不是未校准的原始统计值。

## 4. 弱监督 Action Label

代码位置：`code/sigla_exp/actions.py`

当前定义了 7 个 action：

```python
ACTION_NAMES = (
    "wait",
    "alarm",
    "suppress",
    "inspect",
    "request_evidence",
    "recalibrate",
    "escalate",
)
```

但当前弱标签生成只会实际产生 4 类：

```text
wait, alarm, request_evidence, escalate
```

生成逻辑：

1. 先从 point label 中找到连续异常事件，每个事件为 `[onset, end]`。
2. 对每个窗口使用窗口末端 `end_idx` 对齐未来事件。
3. 如果当前窗口已经包含异常点，即 `window_label=1`，标为 `escalate`。
4. 否则找到下一个未来事件，计算：

```text
lead_time = event.onset - end_idx
```

5. label 规则：

| 条件 | action |
| --- | --- |
| 没有未来事件 | `wait` |
| `l_min <= lead_time <= l_max` | `alarm` |
| `0 <= lead_time < l_min` | `request_evidence` |
| 其他正常区域 | `wait` |
| 当前窗口已覆盖异常 | `escalate` |

默认 `l_min=20`、`l_max=120`，表示希望在异常 onset 前 20 到 120 个时间点之间发出 alarm。

## 5. 模型架构

代码位置：`code/sigla_exp/models.py`

### 5.1 MLPAutoEncoder

用途：`--task autoencoder`

输入：

```text
signal: [B, win_size, n_vars]
```

内部 flatten：

```text
D = win_size * n_vars
x_flat: [B, D]
```

默认：

```text
win_size   = 50
latent_dim = 128
hidden_dim = 128
```

实际 hidden size：

```text
H = min(hidden_dim, max(32, D))
```

层结构：

```text
Encoder:
  Linear(D -> H)
  ReLU
  Linear(H -> latent_dim)
  ReLU

Decoder:
  Linear(latent_dim -> H)
  ReLU
  Linear(H -> D)

Output reshape:
  [B, D] -> [B, win_size, n_vars]
```

层数统计：

```text
Linear 层：4 层
ReLU：3 个
循环/卷积/注意力层：无
```

forward 输出：

```text
reconstruction: [B, win_size, n_vars]
```

异常分数：

```text
anomaly_score = mean((reconstruction - signal)^2, dim=(time, variable))
shape: [B]
```

### 5.2 SigLAPolicy

用途：`--task policy`

输入：

```text
signal:  [B, win_size, n_vars]
score:   [B, win_size, 1]
profile: [B, 5]
```

默认 hidden size：

```text
H = hidden_dim = 128
```

Signal encoder：

```text
seq = concat(signal, score, dim=-1)
shape: [B, win_size, n_vars + 1]

GRU(
  input_size = n_vars + 1,
  hidden_size = H,
  num_layers = 1,
  batch_first = True
)

signal_repr = final_hidden[-1]
shape: [B, H]
```

Profile encoder：

```text
Linear(5 -> H)
ReLU
Linear(H -> H)
ReLU

profile_repr: [B, H]
```

Risk state：

```text
risk_state = max(score over time)
shape: [B, 1]
```

Fusion：

```text
concat(signal_repr, profile_repr, risk_state)
shape: [B, 2H + 1]

Linear(2H + 1 -> H)
ReLU
Dropout(p=0.1)

fused: [B, H]
```

Heads：

```text
action_head: Linear(H -> 7)
arg_head:    Linear(H -> n_vars)
risk_head:   Linear(H -> 1)
```

输出：

| key | shape | 含义 |
| --- | --- | --- |
| `action_logits` | `[B, 7]` | 7 类 action logits |
| `arg_logits` | `[B, n_vars]` | 变量/argument logits |
| `risk_logit` | `[B]` | window anomaly risk logit |

层数统计：

```text
GRU：1 层
Profile MLP Linear：2 层
Fusion Linear：1 层
Head Linear：3 层
显式 Linear 总数：6 层
Dropout：1 个，p=0.1
```

## 6. 训练方式

代码位置：`code/sigla_exp/train.py`

### 6.1 通用训练流程

所有任务共享以下流程：

1. 解析 CLI 参数。
2. 设置随机种子：`random`、`numpy`、`torch`、`torch.cuda`。
3. 自动选择 device：有 CUDA 用 `cuda`，否则用 `cpu`。
4. 创建 run 目录：

```text
{output_dir}/{run_name}
```

默认 `output_dir=/u/ylin30/sigLA/code/runs`。

5. 加载数据并标准化。
6. 在 train split 上 fit `ConceptProfileExtractor`。
7. 写出 `config.json`。
8. 训练对应任务。
9. 写出 `metrics.json`。
10. 验证集 loss 最优时写出 `checkpoint_best.pt`。

checkpoint 内容：

```python
{
    "model": model.state_dict(),
    "args": vars(args),
    "n_vars": bundle.n_vars,
}
```

### 6.2 AutoEncoder 训练

命令示例：

```bash
cd /u/ylin30/sigLA/code
python -m sigla_exp.train \
  --task autoencoder \
  --dataset SMD_1-7 \
  --data_dir /u/ylin30/sigLA/data \
  --output_dir /u/ylin30/sigLA/code/runs \
  --run_name autoencoder_SMD_1-7_w50_s5_a100 \
  --win_size 50 \
  --step 5 \
  --train_ratio 0.8 \
  --batch_size 128 \
  --epochs 20 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --latent_dim 128 \
  --hidden_dim 128 \
  --profile_max_windows 512 \
  --seed 0 \
  --num_workers 0
```

训练数据：

```text
train_ds = WindowDataset(bundle.train, ...)
val_ds   = WindowDataset(bundle.val, ...)
```

优化器：

```text
AdamW(lr=1e-3, weight_decay=1e-4)
```

loss：

```text
loss = mse_loss(reconstruction, signal)
```

训练输出的 `metrics.json`：

```json
{
  "best_val_loss": 0.4414848731830716,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.6918506070971489,
      "val_loss": 0.5526036359369755
    }
  ]
}
```

当前已有一次 SMD_1-7 训练结果：

```text
run_dir: /u/ylin30/sigLA/code/runs/autoencoder_SMD_1-7_w50_s5_a100
best_val_loss: 0.4414848731830716
best epoch: 18
```

### 6.3 Policy 训练

命令示例：

```bash
cd /u/ylin30/sigLA/code
python -m sigla_exp.train \
  --task policy \
  --dataset SMD_1-7 \
  --policy_split test \
  --epochs 5
```

数据来源：

```text
split = bundle.train / bundle.val / bundle.test，由 --policy_split 控制
full_ds = WindowDataset(split, ...)
```

然后在该 split 的 windows 上做随机 80/20 切分：

```text
train_len = 80%
val_len   = 20%
```

注意：默认 `--policy_split test`，也就是用带异常标签的 test split 生成弱监督 action label，再随机切成 policy 的 train/val。这是为了早期实验能产生 precursor label，不是严格无监督评估设置。

优化器：

```text
AdamW(lr=1e-3, weight_decay=1e-4)
```

loss：

```text
action_loss = cross_entropy(action_logits, action)
arg_loss    = cross_entropy(arg_logits, arg)
risk_loss   = binary_cross_entropy_with_logits(risk_logit, label)

total_loss = action_loss + 0.1 * arg_loss + 0.2 * risk_loss
```

训练指标：

```text
train_loss
train_acc       # action prediction accuracy
val_loss
val_acc         # action prediction accuracy
val_pred_actions
```

best checkpoint 选择标准：

```text
val_loss 越低越好
```

当前没有单独的 policy eval 脚本；policy 的可见评估主要是训练过程中的 `val_acc`、`val_loss` 和 `val_pred_actions`。

## 7. Evaluation 方式

### 7.1 AutoEncoder 数值评估

代码位置：`code/eval/eval_autoencoder.py`

命令示例：

```bash
cd /u/ylin30/sigLA/code
python eval/eval_autoencoder.py \
  --run_dir /u/ylin30/sigLA/code/runs/autoencoder_SMD_1-7_w50_s5_a100 \
  --split test \
  --threshold_source val \
  --threshold_percentile 99.0 \
  --batch_size 128 \
  --num_workers 0
```

评估输入：

```text
checkpoint_best.pt
checkpoint 中保存的 args
dataset / data_dir / win_size / step / latent_dim / hidden_dim
```

可以用 CLI 覆盖 checkpoint 中的 dataset、data_dir、win_size、step 等参数。

评估 Dataset：

```text
SignalWindowDataset
```

它只输出：

| key | shape / type | 含义 |
| --- | --- | --- |
| `signal` | `[win_size, n_vars]` | 标准化窗口 |
| `label` | scalar long | 窗口内是否包含异常 |
| `start` | scalar long | 窗口起点 |
| `end` | scalar long | 窗口终点 |

窗口异常分数：

```text
score = mean((model(signal) - signal)^2 over time and variables)
shape: [num_windows]
```

阈值估计：

```text
threshold_source: train / val / test，默认 val
threshold_percentile: 默认 99.0
```

默认只用 threshold source 中的正常窗口估计阈值：

```text
threshold_scores = scores[labels == 0]
threshold = percentile(threshold_scores, threshold_percentile)
```

如果加上 `--threshold_all_windows`，则所有窗口都会参与阈值估计。

窗口级预测：

```text
window_pred = window_score > threshold
```

点级预测：

```text
point_score[t] = 覆盖 t 的所有窗口分数的最大值
point_pred[t]  = point_score[t] > threshold
```

未被任何窗口覆盖的点使用当前 split 的最小窗口分数填充。

输出文件：

```text
autoencoder_eval_{split}.json
autoencoder_window_scores_{split}.csv
```

JSON 指标：

```text
score_summary: min / median / mean / max
window_metrics:
  accuracy, precision, recall, f1, tp, fp, tn, fn,
  roc_auc, average_precision
point_metrics:
  accuracy, precision, recall, f1, tp, fp, tn, fn,
  roc_auc, average_precision
```

当前已有 SMD_1-7 test 评估：

```text
threshold: 17.199144344329913

window_metrics:
  accuracy: 0.8786469344608879
  precision: 1.0
  recall: 0.054365733113673806
  f1: 0.10312500000000001
  roc_auc: 0.9368136555450378
  average_precision: 0.7916393766614016

point_metrics:
  accuracy: 0.8980039667468456
  precision: 0.47246376811594204
  recall: 0.06797331109257715
  f1: 0.1188479766678819
  roc_auc: 0.950326277030787
  average_precision: 0.5734372376669473
```

### 7.2 AutoEncoder 可视化评估

代码位置：`code/eval/plot_autoencoder_eval.py`

命令示例：

```bash
cd /u/ylin30/sigLA/code
python eval/plot_autoencoder_eval.py \
  --run_dir /u/ylin30/sigLA/code/runs/autoencoder_SMD_1-7_w50_s5_a100 \
  --dataset SMD_1-7 \
  --data_dir /u/ylin30/sigLA/data
```

输入：

```text
autoencoder_eval_test.json
autoencoder_window_scores_test.csv
dataset 原始 test split
```

输出：

```text
autoencoder_eval_curves_test.svg
autoencoder_eval_zoom_test.svg
```

可视化内容：

1. full test split 的 point-level AE score 曲线。
2. threshold 虚线。
3. ground truth anomaly 区间。
4. predicted anomaly 区间。
5. zoom 图中还会画出局部绝对幅值最大的 3 个变量的标准化信号。

## 8. Delta 上的训练脚本

代码位置：`code/scripts/train_autoencoder_delta.sbatch`

该脚本用于 UIUC Delta GPU 节点：

```text
partition: gpuA100x4
gpus-per-node: 1
mem: 16G
time: 02:00:00
conda env: /projects/bflz/ylin30/conda_envs/sigla
```

执行内容：

1. 检查 PyTorch 和 CUDA。
2. 训练 `autoencoder`，run name 为 `autoencoder_SMD_1-7_w50_s5_a100`。
3. 训练结束后直接运行 `eval/eval_autoencoder.py`。

提交命令：

```bash
cd /u/ylin30/sigLA/code
sbatch scripts/train_autoencoder_delta.sbatch
```

## 9. CANDI 参考代码说明

`/u/ylin30/sigLA/CANDI` 是另一套较完整的 baseline / TTA 框架。它有自己的入口：

```bash
cd /u/ylin30/sigLA/CANDI
python main.py ...
```

主流程：

```text
main.py
  -> load_config()
  -> build_model(cfg)
  -> build_trainer(cfg, model)
  -> trainer.train() if cfg.TRAIN.ENABLE
  -> Predictor(cfg, model).predict() if cfg.TEST.ENABLE
```

默认模型配置里 `MODEL.NAME = 'MLP'`，也支持 `TIMESNET`。

CANDI 的 MLP autoencoder 与 `code/sigla_exp` 里的 minimal AE 不完全一样。CANDI MLP 结构为：

```text
D = win_size * n_vars
Z = cfg.MLP.Z_SIZE

Encoder:
  Linear(D -> D/2)
  ReLU
  Linear(D/2 -> D/4)
  ReLU
  Linear(D/4 -> Z)
  ReLU

Decoder:
  Linear(Z -> D/4)
  ReLU
  Linear(D/4 -> D/2)
  ReLU
  Linear(D/2 -> D)
```

CANDI MLP 训练 loss：

```text
mse_loss(reconstruction, input)
```

CANDI Predictor 评估：

1. 计算 train/val/test reconstruction scores。
2. 用 `Thresholder` 得到 threshold。
3. `pred = test_scores > threshold`。
4. 输出 AUROC、AUPRC、Precision、Recall、F1。
5. 可选 TTA；如果开启 CANDI TTA，会在测试过程中通过 adapter 动态适配。

因此，`CANDI/` 可以作为参考 baseline，但当前 `code/sigla_exp/train.py` 并不会调用 `CANDI/main.py`。

## 10. 当前实现边界

1. 当前 `policy` 是 behavior cloning，不是 RL，也没有 preference reward model。
2. 当前 `profile` 是统计特征，不是可学习的 profile encoder。
3. 当前 action label 是由 ground-truth event onset 生成的弱标签；实际发出的只有 `wait/alarm/request_evidence/escalate` 四类。
4. 当前只有 AutoEncoder 有独立 eval 脚本；Policy 尚无 test-time decision/eval pipeline。
5. `autoencoder` 训练只优化重构误差，不直接优化 F1、AUROC 或 early-warning lead time。
6. 默认 threshold 是 validation 正常窗口的 99 percentile；这会影响 precision/recall trade-off。
7. SMD/SWaT train label 默认全 0，适合无监督 reconstruction baseline；policy 若使用 `--policy_split train`，大多只会学到 `wait`。

## 11. 最小 smoke test

从 `/u/ylin30/sigLA/code` 运行：

```bash
python -m sigla_exp.train --dataset synthetic --task autoencoder --epochs 2 --limit_batches 3
python -m sigla_exp.train --dataset synthetic --task policy --epochs 2 --limit_batches 3
```

预期会在 `code/runs/` 下生成：

```text
config.json
metrics.json
checkpoint_best.pt
```

