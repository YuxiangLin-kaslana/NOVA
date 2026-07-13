#!/bin/bash
#SBATCH --job-name=sigla-p3collect
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-p3collect-%j.out
#SBATCH --error=sigla-p3collect-%j.err
set -euo pipefail

SIGLA=/u/ylin30/sigLA
cd "${SIGLA}"
export PYTHONUNBUFFERED=1

TAG="${P3_COLLECT_TAG:-p3_20260708_132402}"
OUT_PREFIX="${P3_COLLECT_OUT_PREFIX:-code/runs/p3_results_summary_${TAG}}"

python3 code/sota_compare/collect_p3_results.py \
  --tag "${TAG}" \
  --out-prefix "${OUT_PREFIX}"

echo done
