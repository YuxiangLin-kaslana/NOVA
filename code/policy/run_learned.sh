#!/bin/bash
#SBATCH --job-name=sigla-rl
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-rl-%j.out
#SBATCH --error=sigla-rl-%j.err
set -euo pipefail
cd /u/ylin30/sigLA/code; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
/projects/bflz/ylin30/conda_envs/sigla/bin/python policy/exp_learned_policy2.py
echo rl-done
