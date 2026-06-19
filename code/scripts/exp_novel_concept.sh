#!/bin/bash
# 路线B 决定性实验:新异常类型识别 LLM zero-shot vs 参数化。详见 exp_novel_concept.py
#SBATCH --job-name=sigla-novel
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-novel-%j.out
#SBATCH --error=sigla-novel-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
if [ -f "${SIGLA}/.env" ]; then set -a; source "${SIGLA}/.env"; set +a; fi
NET_OK=$(/projects/bflz/ylin30/conda_envs/sigla/bin/python - <<'PY'
import os,urllib.request
try:
    urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/models",
        headers={"Authorization":f"Bearer {os.environ.get('OPENAI_API_KEY','')}"}),timeout=15).read();print("yes")
except Exception:print("no")
PY
)
echo "[preflight] OpenAI reachable: ${NET_OK}"
/projects/bflz/ylin30/conda_envs/sigla/bin/python scripts/exp_novel_concept.py
echo "done"
