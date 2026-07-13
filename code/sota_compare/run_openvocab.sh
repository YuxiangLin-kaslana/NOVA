#!/bin/bash
#SBATCH --job-name=sigla-ovab
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-ovab-%j.out
#SBATCH --error=sigla-ovab-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
if [ -f "${SIGLA}/.env" ]; then set -a; source "${SIGLA}/.env"; set +a; fi
PY="/projects/bflz/ylin30/conda_envs/sigla/bin/python"
echo "##### synthetic #####"; "${PY}" sota_compare/exp_openvocab_namer.py
for M in 1-1 2-5; do
  echo "##### machine-${M} #####"
  REAL_MACHINE="${M}" "${PY}" sota_compare/exp_openvocab_namer.py
done
echo ovab-done
