#!/bin/bash
# 在 Delta 上用 1 块 A100 训练 SMD 异常检测器，并记录到 Weights & Biases。
#
# 提交：
#   sbatch scripts/train_anomaly_detector_delta.sh
#
# wandb 鉴权：优先用环境变量 WANDB_API_KEY；否则用登录节点上 `wandb login`
# 写入的 ~/.netrc（/u/ylin30/.netrc，与 Delta 共享 home，已配置）。
# 不要把 API key 硬编码进本脚本。

#SBATCH --job-name=sigla-anomaly
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-anomaly-%j.out
#SBATCH --error=sigla-anomaly-%j.err

set -euo pipefail

cd /u/ylin30/sigLA/code
mkdir -p /u/ylin30/sigLA/code/runs

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# wandb 设置（计算节点可能无外网；如离线则自动落本地，结束后可 `wandb sync` 上传）
export WANDB_PROJECT=sigla-anomaly-detector
# export WANDB_API_KEY=...   # 如需覆盖 ~/.netrc，在提交前 export，勿写死在脚本里
# export WANDB_MODE=offline  # 若计算节点无外网，取消注释改为离线模式

# GPU / 依赖检查（wandb 不存在不阻断训练）
"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，检查 Delta GPU 分配或环境。")
print("cuda_device_count", torch.cuda.device_count())
print("cuda_device_0", torch.cuda.get_device_name(0))
try:
    import wandb
    print("wandb", wandb.__version__)
except ImportError:
    print("wandb 未安装：将以无记录模式训练（pip install wandb 可启用）")
PY

# 单台机器训练（SMD 标准做法）。换机器改 MACHINE 即可，如 machine-1-7。
MACHINE=machine-1-1
RUN_NAME="anomaly_detector_${MACHINE}_w100_s10"

"${PYTHON}" train/anomaly_detector/train.py \
  --data_root /u/ylin30/sigLA/data/ServerMachineDataset \
  --output_dir /u/ylin30/sigLA/code/runs \
  --run_name "${RUN_NAME}" \
  --machines "${MACHINE}" \
  --win_size 100 \
  --stride 10 \
  --val_split 0.5 \
  --latent_dim 128 \
  --hidden_dim 128 \
  --batch_size 256 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 0 \
  --num_workers 8 \
  --device auto \
  --wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${RUN_NAME}"
