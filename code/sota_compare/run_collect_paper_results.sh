#!/bin/bash
#SBATCH --job-name=sigla-collect
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-collect-%j.out
#SBATCH --error=sigla-collect-%j.err
set -euo pipefail

SIGLA=/u/ylin30/sigLA
cd "${SIGLA}"
export PYTHONUNBUFFERED=1

TAG="${COLLECT_TAG:-}"
OUT_PREFIX="${COLLECT_OUT_PREFIX:-code/runs/paper_results_summary_${TAG:-all}}"

python3 code/sota_compare/collect_paper_results.py \
  --tag "${TAG}" \
  --out-prefix "${OUT_PREFIX}"

echo done
