#!/bin/bash
#SBATCH --job-name=sigla-nmulti
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-nmulti-%j.out
#SBATCH --error=sigla-nmulti-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
if [ -f "${SIGLA}/.env" ]; then set -a; source "${SIGLA}/.env"; set +a; fi
PY="/projects/bflz/ylin30/conda_envs/sigla/bin/python"
echo "##### synthetic #####"; "${PY}" sota_compare/exp_naming_baseline.py
for M in 1-1 2-1 3-1 1-6 2-5; do
  echo "##### machine-${M} #####"
  REAL_MACHINE="${M}" "${PY}" sota_compare/exp_naming_baseline.py
done
echo nmulti-done
