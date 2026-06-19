#!/bin/bash
# 训练 SMD MLP 异常检测器 (sigla_exp.model.mlp.MLPAnomalyDetector)。
#
# 本地直接运行：
#   bash scripts/train_anomaly_detector.sh
# Delta 集群提交：
#   sbatch scripts/train_anomaly_detector.sh

#SBATCH --job-name=sigla-anomaly
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-anomaly-%j.out
#SBATCH --error=sigla-anomaly-%j.err

set -euo pipefail

cd /u/ylin30/sigLA/code
mkdir -p /u/ylin30/sigLA/code/runs

# 选择 Python：在 Delta 上用 conda 环境；否则回退到当前环境的 python。
if [ -f /sw/external/python/anaconda3/etc/profile.d/conda.sh ]; then
  source /sw/external/python/anaconda3/etc/profile.d/conda.sh
  export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
  conda activate /projects/bflz/ylin30/conda_envs/sigla
  PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python
else
  PYTHON="$(command -v python3)"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

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
  --epochs 20 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 0 \
  --num_workers 0 \
  --device auto
