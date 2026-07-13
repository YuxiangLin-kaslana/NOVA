#!/bin/bash
#SBATCH --job-name=sigla-robust
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-robust-%j.out
#SBATCH --error=sigla-robust-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
if [ -f "${SIGLA}/.env" ]; then set -a; source "${SIGLA}/.env"; set +a; fi
PY="/projects/bflz/ylin30/conda_envs/sigla/bin/python"
echo "===== synthetic ====="; "${PY}" sota_compare/diag_robust.py
echo "===== real machine-1-1 ====="; REAL_MACHINE=1-1 "${PY}" sota_compare/diag_robust.py
echo robust-done
