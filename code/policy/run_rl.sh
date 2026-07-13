#!/bin/bash
#SBATCH --job-name=sigla-rlft
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-rlft-%j.out
#SBATCH --error=sigla-rlft-%j.err
set -euo pipefail
cd /u/ylin30/sigLA/code; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
/projects/bflz/ylin30/conda_envs/sigla/bin/python policy/exp_rl_finetune.py
echo rl-done
