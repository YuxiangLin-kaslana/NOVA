#!/bin/bash
# Submit on Delta with:
#   sbatch scripts/train_policy_delta.sh

#SBATCH --job-name=sigla-policy
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-policy-%j.out
#SBATCH --error=sigla-policy-%j.err

set -euo pipefail

cd /u/ylin30/sigLA/code
mkdir -p /u/ylin30/sigLA/code/runs

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

"${PYTHON}" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check Delta GPU allocation or Python environment.")
print("cuda_device_count", torch.cuda.device_count())
print("cuda_device_0", torch.cuda.get_device_name(0))
PY

"${PYTHON}" -m sigla_exp.train \
  --task policy \
  --dataset SMD_1-7 \
  --data_dir /u/ylin30/sigLA/data \
  --output_dir /u/ylin30/sigLA/code/runs \
  --run_name policy_SMD_1-7_test_w50_s5 \
  --win_size 50 \
  --step 5 \
  --train_ratio 0.8 \
  --batch_size 128 \
  --epochs 20 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --hidden_dim 128 \
  --profile_max_windows 512 \
  --l_min 20 \
  --l_max 120 \
  --seed 0 \
  --num_workers 0

"${PYTHON}" eval/eval_policy.py \
  --run_dir /u/ylin30/sigLA/code/runs/policy_SMD_1-7_test_w50_s5 \
  --split test \
  --batch_size 128 \
  --num_workers 0
