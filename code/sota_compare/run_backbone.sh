#!/bin/bash
#SBATCH --job-name=sigla-bb
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/u/ylin30/sigLA/code/runs/bb-%j.out
set -euo pipefail
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
/projects/bflz/ylin30/conda_envs/sigla/bin/python /u/ylin30/sigLA/code/sota_compare/exp_backbone_openvocab.py
echo bb-done
