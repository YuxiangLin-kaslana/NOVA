#!/bin/bash
# 验证前兆窗口评测:在真实 SMD 上跑一次校准阈值检测(无 GPT),对比普通 AD F1 vs 早预警口径。
#SBATCH --job-name=sigla-prec
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-prec-%j.out
#SBATCH --error=sigla-prec-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs/precursor_test
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
M=machine-1-1
"/projects/bflz/ylin30/conda_envs/sigla/bin/python" scripts/run_online.py \
  --stream "${SIGLA}/specific_data/Online_training/streams/smd_${M}.npz" \
  --detector_ckpt "runs/anomaly_detector_${M}_w100_s10/checkpoint_best.pt" \
  --concept_ckpt  "runs/concept_detector_cnn_${M}_w100_s10/checkpoint_best.pt" \
  --concept_model cnn --win_size 100 --step 25 --hidden_dim 128 --latent_dim 128 \
  --cal_quantile 0.97 --cal_window 512 --cal_warmup 100 --cal_margin 1.0 \
  --decision calibrated_threshold --agent local --update_scope full --warmup_windows 0 \
  --l_min 25 --l_max 200 --device auto \
  --output runs/precursor_test/smd_${M}.json --predictions_csv runs/precursor_test/pred_${M}.csv
echo done
