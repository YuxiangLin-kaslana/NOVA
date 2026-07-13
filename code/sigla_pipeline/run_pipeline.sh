#!/bin/bash
#SBATCH --job-name=sigla-pipe
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/u/ylin30/sigLA/code/runs/pipe-%j.out
#SBATCH --error=/u/ylin30/sigLA/code/runs/pipe-%j.err
set -euo pipefail
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
/projects/bflz/ylin30/conda_envs/sigla/bin/python /u/ylin30/sigLA/code/sigla_pipeline/run_pipeline.py
echo pipe-done
