#!/bin/bash
# B 臂单独重跑(校准提候选 + GPT decider + 在线 head_only),写**独立** predictions CSV,
# 用于分析 agent 在哪些窗否决/确认了候选。其余配置与 run_online_calibrated.sh 的 B 臂一致。
#SBATCH --job-name=sigla-B
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-B-%j.out
#SBATCH --error=sigla-B-%j.err
set -euo pipefail
SIGLA=/u/ylin30/sigLA; cd "${SIGLA}/code"; mkdir -p runs/online/calibrated
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python
export PYTHONUNBUFFERED=1
[ -f "${SIGLA}/.env" ] && { set -a; source "${SIGLA}/.env"; set +a; }
NET_OK=$("${PYTHON}" - <<'PY'
import os,urllib.request
try:
    urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/models",
        headers={"Authorization":f"Bearer {os.environ.get('OPENAI_API_KEY','')}"}),timeout=15).read();print("yes")
except Exception:print("no")
PY
)
echo "[preflight] OpenAI reachable: ${NET_OK}"
[ "${NET_OK}" = "yes" ] || { echo "[abort] B 臂需要 GPT,无外网,退出"; exit 1; }
MACHINE=machine-1-1
STREAM="${SIGLA}/specific_data/Online_training/streams/smd_${MACHINE}.npz"
"${PYTHON}" scripts/run_online.py \
  --stream "${STREAM}" \
  --detector_ckpt "runs/anomaly_detector_${MACHINE}_w100_s10/checkpoint_best.pt" \
  --concept_ckpt  "runs/concept_detector_cnn_${MACHINE}_w100_s10/checkpoint_best.pt" \
  --concept_model cnn --win_size 100 --step 25 --hidden_dim 128 --latent_dim 128 \
  --cal_quantile 0.95 --cal_window 512 --cal_warmup 100 --cal_margin 1.0 \
  --decision calibrated_agent --agent gpt --agent_model gpt-4o-mini \
  --anomaly_prior 0.095 --normal_sample_rate 0.05 \
  --update_scope head_only --retrain_every 25 --warmup_windows 0 --device auto \
  --output "runs/online/calibrated/smd_${MACHINE}_B_rerun.json" \
  --predictions_csv "runs/online/calibrated/pred_smd_${MACHINE}_B_rerun.csv"
echo "done -> runs/online/calibrated/pred_smd_${MACHINE}_B_rerun.csv"
