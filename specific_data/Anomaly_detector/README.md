# Anomaly_detector —— SMD 数据准备

基于 [ServerMachineDataset (SMD)](../../data/ServerMachineDataset) 准备异常检测所需的数据。
本目录**只做数据加载与切窗**，不包含模型训练。

## 任务设定

- **输入**：一个时间窗口 `[win_size, n_vars]`（默认 `win_size=100`，`n_vars=38`）
- **输出**：该窗口是否存在异常（窗口内任一时刻被标为异常 → `1`）
- 训练好的检测器（如自编码器）还可用「与正常形态的偏离程度」（重构误差等）作为连续异常分数。

## 数据组织（已按需求确定）

| 维度 | 选择 |
|------|------|
| 粒度 | **全部 28 台机器合并**为一个训练集 / 测试集 |
| 窗口 | `win_size=100`，`stride=10` |
| 形式 | **不预先落盘**，提供 `Dataset` 类，运行时实时切窗 |

关键约定：
- **滑窗不跨机器边界**：每台机器内部独立切窗。
- **train 序列默认全部正常**（标签 0）；**test 用 `test_label` 真值**。
- 窗口标签 = 窗口内任一时刻异常即记 `1`（与 sigLA 现有 `code/eval/eval_autoencoder.py` 一致）。

数据规模（win=100, stride=10）：train ≈ 70,575 窗口，test ≈ 70,576 窗口，其中异常窗口 ≈ 8.72%。

## 文件

- `data.py` —— 加载 + 实时切窗 `Dataset`，核心入口 `build_smd_datasets()`
- `prepare_data.py` —— 自检脚本，打印形状与统计，验证数据可用

## 用法

```python
from data import build_smd_datasets
from torch.utils.data import DataLoader

bundle = build_smd_datasets(win_size=100, stride=10)   # 合并全部机器
train_loader = DataLoader(bundle.train, batch_size=256, shuffle=True)
test_loader  = DataLoader(bundle.test,  batch_size=256, shuffle=False)

batch = next(iter(train_loader))
batch["signal"]  # FloatTensor [B, 100, 38]
batch["label"]   # LongTensor  [B]   (train 恒为 0；test 为真实 0/1)
batch["series"]  # 窗口来自第几台机器
batch["start"]   # 窗口在该机器序列内的起始下标
```

自检：

```bash
python prepare_data.py --win_size 100 --stride 10
```

## 依赖说明

`data.py` 的 `Dataset` 依赖 **PyTorch**。当前环境（`/u/ylin30/.venv`）尚未安装 torch，
需先安装后才能运行 `prepare_data.py`：

```bash
pip install torch
```

纯数据加载（`load_all_machines` 等返回 numpy）逻辑已用 numpy 单独验证通过。
