#!/bin/bash
# ============================================================================
# SigLA 重设计实验:把异常**决策**还给「校准后的检测分数」(proposer→decider)。
#
# 起因:上一轮在线实验两次都塌成平凡分类器(全报警 / 全不报),因为让 LLM 对
# 未校准的裸重建误差凭感觉判 0/1。诊断显示信号其实都在(detector AUC≈0.95),
# 是流水线把它丢了。本脚本用 ScoreCalibrator 把分数→分位阈值→候选异常,
# agent 退为「确认候选 + 概念标注」。
#
# 三臂主图(抗漂移):
#   A. calibrated_threshold  纯校准阈值决策(无 LLM)+ 在线适应  ← 即得 F1≈0.7
#   B. calibrated_agent      校准提候选 + GPT 仅在候选/采样正常上确认 + 在线 head_only
#   C. frozen                校准阈值 + 冻结不在线(基线,展示漂移退化)
#
# 提交:  sbatch scripts/run_online_calibrated.sh
# ============================================================================

#SBATCH --job-name=sigla-cal
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-cal-%j.out
#SBATCH --error=sigla-cal-%j.err

set -euo pipefail

SIGLA=/u/ylin30/sigLA
cd "${SIGLA}/code"
mkdir -p runs/online/calibrated

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python
export PYTHONUNBUFFERED=1

if [ -f "${SIGLA}/.env" ]; then set -a; source "${SIGLA}/.env"; set +a; fi

# ---- 外网预检(决定 GPT decider 能否工作) ---- #
NET_OK=$("${PYTHON}" - <<'PY'
import os, urllib.request
key=os.environ.get("OPENAI_API_KEY","")
try:
    req=urllib.request.Request("https://api.openai.com/v1/models",headers={"Authorization":f"Bearer {key}"})
    urllib.request.urlopen(req,timeout=15).read(); print("yes")
except Exception: print("no")
PY
)
echo "[preflight] OpenAI reachable: ${NET_OK}"
AGENT=gpt
[ "${NET_OK}" = "yes" ] || { echo "[warn] 无外网，B 臂回退 local agent"; AGENT=local; }

MACHINE=machine-1-1
STREAM="${SIGLA}/specific_data/Online_training/streams/smd_${MACHINE}.npz"
DET="runs/anomaly_detector_${MACHINE}_w100_s10/checkpoint_best.pt"
CON="runs/concept_detector_cnn_${MACHINE}_w100_s10/checkpoint_best.pt"

WIN=100; STEP=25
CAL="--cal_quantile 0.95 --cal_window 512 --cal_warmup 100 --cal_margin 1.0"
COMMON="--stream ${STREAM} --detector_ckpt ${DET} --concept_ckpt ${CON} \
  --concept_model cnn --win_size ${WIN} --step ${STEP} \
  --hidden_dim 128 --latent_dim 128 ${CAL} --device auto"

run () {  # $1=name $2..=extra args
  local name="$1"; shift
  local out="runs/online/calibrated/smd_${MACHINE}_${name}.json"
  local pred="runs/online/calibrated/pred_smd_${MACHINE}_${name}.csv"
  echo "==================== ${name} ===================="
  "${PYTHON}" scripts/run_online.py ${COMMON} "$@" --output "${out}" --predictions_csv "${pred}"
  "${PYTHON}" - "${out}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
o=m["overall"]; c=m["candidate_only"]
print(f"  decision={m['decision']} online={m['online']} scope={m['update_scope']}")
print(f"  overall  P/R/F1 = {o['precision']:.3f}/{o['recall']:.3f}/{o['f1']:.3f}")
print(f"  candidate-only F1 = {c['f1']:.3f} | agent_calls={m['agent_calls']}/{m['n_windows']} ({m['agent_call_rate']:.1%})")
print(f"  per-regime F1 = "+", ".join(f"{r}:{v['f1']:.2f}" for r,v in m['per_regime'].items()))
print(f"  online_stats = {m['online_stats']}")
PY
}

# ---- A: 纯校准阈值 + 在线检测器适应(无 LLM) ---- #
run "A_calibrated_threshold" --decision calibrated_threshold --agent local \
    --update_scope full --detector_lr 1e-4 --retrain_every 25 --warmup_windows 0

# ---- B: 校准提候选 + GPT decider(仅候选+5%正常) + 在线 head_only ---- #
run "B_calibrated_agent" --decision calibrated_agent --agent "${AGENT}" \
    --agent_model gpt-4o-mini --anomaly_prior 0.095 --normal_sample_rate 0.05 \
    --update_scope head_only --retrain_every 25 --warmup_windows 0

# ---- C: 校准阈值 + 冻结不在线(基线) ---- #
run "C_frozen" --decision calibrated_threshold --agent local --no_online

echo "done -> runs/online/calibrated/"
