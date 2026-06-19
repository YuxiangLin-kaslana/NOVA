#!/bin/bash
# 在 Delta 上用 1 块 A100 训练 concept detector（多标签 / 合成监督），记录到 wandb。
#
# 提交：
#   sbatch scripts/train_concept_detector_delta.sh
#
# wandb 鉴权走 ~/.netrc（登录节点已 wandb login）；勿在脚本里硬编码 key。

#SBATCH --job-name=sigla-concept
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-concept-%j.out
#SBATCH --error=sigla-concept-%j.err

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
export WANDB_PROJECT=sigla-concept-detector
# export WANDB_MODE=offline   # 计算节点若无外网，取消注释

"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，检查 Delta GPU 分配或环境。")
print("device", torch.cuda.get_device_name(0))
try:
    import wandb; print("wandb", wandb.__version__)
except ImportError:
    print("wandb 未安装：将以无记录模式训练")
PY

# 单台机器 + CNN backbone（擅长局部形态：spike/oscillation/variance_burst）。
MACHINE=machine-1-1
MODEL=cnn
RUN_NAME="concept_detector_${MODEL}_${MACHINE}_w100_s10"

"${PYTHON}" train/concept_detector/train.py \
  --data_root /u/ylin30/sigLA/data/ServerMachineDataset \
  --output_dir /u/ylin30/sigLA/code/runs \
  --run_name "${RUN_NAME}" \
  --machines "${MACHINE}" \
  --model "${MODEL}" \
  --win_size 100 \
  --stride 10 \
  --kernel_size 7 \
  --p_normal 0.2 \
  --max_concepts 3 \
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
