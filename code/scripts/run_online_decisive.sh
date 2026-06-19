#!/bin/bash
# ============================================================================
# 决定性实验:证明 LLM decider 的价值(论文核心主张)。
#
# 故意把校准阈值开松到高 recall 工作点 q=0.97 / m=1.0(纯阈值:P0.43 R0.97 F1=0.60,
# 178 个假阳性)。单标量阈值无法在不牺牲 recall 的前提下提 precision —— 这正是
# agent 的用武之地。改写后的 decider(怀疑者 prompt + 判别信号:score/threshold 比值、
# 最大概念概率、概念持续性)应当**否决假阳性候选、保住真阳性**,把 precision 救回来。
#
# 两臂(除 agent 外配置完全一致:同校准器、online full):
#   thr_q97   calibrated_threshold  q0.97/m1.0 + online full  (无 LLM,复现 sweep 基线)
#   agent_q97 calibrated_agent      q0.97/m1.0 + online full  (GPT 怀疑者 decider)
#
# 关键产出:agent 臂的 veto rate、否决里有多少真 FP / 误杀多少 TP、最终 P/R/F1。
# 提交:  sbatch scripts/run_online_decisive.sh
# ============================================================================
#SBATCH --job-name=sigla-dec
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-dec-%j.out
#SBATCH --error=sigla-dec-%j.err

set -euo pipefail

SIGLA=/u/ylin30/sigLA
cd "${SIGLA}/code"
mkdir -p runs/online/decisive

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python
export PYTHONUNBUFFERED=1
if [ -f "${SIGLA}/.env" ]; then set -a; source "${SIGLA}/.env"; set +a; fi

NET_OK=$("${PYTHON}" - <<'PY'
import os,urllib.request
try:
    urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/models",
        headers={"Authorization":f"Bearer {os.environ.get('OPENAI_API_KEY','')}"}),timeout=15).read();print("yes")
except Exception:print("no")
PY
)
echo "[preflight] OpenAI reachable: ${NET_OK}"
[ "${NET_OK}" = "yes" ] || { echo "[abort] decider 实验需要 GPT,无外网,退出"; exit 1; }

MACHINE=machine-1-1
STREAM="${SIGLA}/specific_data/Online_training/streams/smd_${MACHINE}.npz"
DET="runs/anomaly_detector_${MACHINE}_w100_s10/checkpoint_best.pt"
CON="runs/concept_detector_cnn_${MACHINE}_w100_s10/checkpoint_best.pt"

WIN=100; STEP=25
# 高 recall 工作点:q=0.97 / margin=1.0
CAL="--cal_quantile 0.97 --cal_window 512 --cal_warmup 100 --cal_margin 1.0"
COMMON="--stream ${STREAM} --detector_ckpt ${DET} --concept_ckpt ${CON} \
  --concept_model cnn --win_size ${WIN} --step ${STEP} \
  --hidden_dim 128 --latent_dim 128 ${CAL} --device auto \
  --update_scope full --detector_lr 1e-4 --retrain_every 25 --warmup_windows 0"

run () {  # $1=name  $2..=extra
  local name="$1"; shift
  local out="runs/online/decisive/${name}.json"
  local pred="runs/online/decisive/pred_${name}.csv"
  echo "==================== ${name} ===================="
  "${PYTHON}" scripts/run_online.py ${COMMON} "$@" --output "${out}" --predictions_csv "${pred}"
}

# ---- 基线臂:纯校准阈值(无 GPT) ---- #
run "thr_q97" --decision calibrated_threshold --agent local

# ---- 决定性臂:GPT 怀疑者 decider ---- #
run "agent_q97" --decision calibrated_agent --agent gpt --agent_model gpt-4o-mini \
    --anomaly_prior 0.095 --normal_sample_rate 0.05

echo "done -> runs/online/decisive/"
