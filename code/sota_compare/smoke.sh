#!/bin/bash
#SBATCH --job-name=sigla-sota-smoke
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-sota-smoke-%j.out
#SBATCH --error=sigla-sota-smoke-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
export CMP_SMOKE=1 CMP_NSEED=1
"/projects/bflz/ylin30/conda_envs/sigla/bin/python" sota_compare/run_detection_compare.py
echo smoke-done
