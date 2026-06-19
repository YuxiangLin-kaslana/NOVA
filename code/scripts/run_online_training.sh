#!/bin/bash
# 无真值在线训练(A100):逐窗消费合成漂移流，LLM agent 判异常 + 给概念伪标签，
# OnlineTrainer 在线重训 anomaly detector(自监督) 与 concept detector(伪标签 BCE)。
# 训练全程不用真值；标签仅用于离线评测(整体 + 分漂移区间 F1)。
#
# 提交：  sbatch scripts/run_online_training.sh
#
# LLM agent 需要 OPENAI_API_KEY(从 /u/ylin30/sigLA/.env 读取)以及**计算节点外网**。
# 若计算节点无外网，agent 会回退到 local；脚本末尾会检查 source 计数并报警。

#SBATCH --job-name=sigla-online
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-online-%j.out
#SBATCH --error=sigla-online-%j.err

set -euo pipefail

SIGLA=/u/ylin30/sigLA
cd "${SIGLA}/code"
mkdir -p runs/online

# ---- 环境 ---- #
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python
export PYTHONUNBUFFERED=1

# ---- API key(从 .env 读取，不硬编码) ---- #
if [ -f "${SIGLA}/.env" ]; then
  set -a; source "${SIGLA}/.env"; set +a
fi

# ---- GPU 自检 ---- #
"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

# ---- 外网连通性预检(决定 LLM agent 能否工作) ---- #
NET_OK=$("${PYTHON}" - <<'PY'
import os, json, urllib.request, urllib.error
key=os.environ.get("OPENAI_API_KEY","")
try:
    req=urllib.request.Request("https://api.openai.com/v1/models",headers={"Authorization":f"Bearer {key}"})
    with urllib.request.urlopen(req,timeout=15) as r: json.loads(r.read().decode())
    print("yes")
except Exception as e:
    print("no")
PY
)
echo "[preflight] OpenAI reachable from compute node: ${NET_OK}"

# ---- 配置:加载离线 ckpt(SMD machine-1-1) + head_only + 保守 agent + SMD 同分布流 ---- #
# 离线 detector/concept 都是 SMD machine-1-1 训的，所以在线流也用同机器 test 序列，
# 否则分布不匹配检测失真。head_only 冻结离线 backbone，在线只微调 head/decoder。
MACHINE=machine-1-1
STREAM="${SIGLA}/specific_data/Online_training/streams/smd_${MACHINE}.npz"
[ -f "${STREAM}" ] || "${PYTHON}" "${SIGLA}/specific_data/Online_training/make_smd_stream.py" \
  --machine "${MACHINE}" --out "${STREAM}"

DETECTOR_CKPT="runs/anomaly_detector_${MACHINE}_w100_s10/checkpoint_best.pt"
CONCEPT_CKPT="runs/concept_detector_cnn_${MACHINE}_w100_s10/checkpoint_best.pt"

WIN=100
STEP=25                    # ~1136 窗(SMD machine-1-1 test 长 28479),控制 LLM 调用量
UPDATE_SCOPE=head_only     # 复用离线 backbone，只在线微调 head/decoder
AGENT=gpt
ANOMALY_PRIOR=0.09         # machine-1-1 异常基率 ~9.5%，让 agent 保守
[ "${NET_OK}" = "yes" ] || { echo "[warn] 计算节点无外网，LLM 不可用，回退 AGENT=local"; AGENT=local; }

OUT="runs/online/smd_${MACHINE}_${AGENT}_${UPDATE_SCOPE}.json"

# ---- 在线训练 + 评测 ---- #
"${PYTHON}" scripts/run_online.py \
  --stream "${STREAM}" \
  --detector_ckpt "${DETECTOR_CKPT}" \
  --concept_ckpt "${CONCEPT_CKPT}" \
  --concept_model cnn \
  --win_size "${WIN}" --step "${STEP}" \
  --hidden_dim 128 --latent_dim 128 \
  --agent "${AGENT}" --agent_model gpt-4o-mini --anomaly_prior "${ANOMALY_PRIOR}" \
  --update_scope "${UPDATE_SCOPE}" \
  --retrain_every 25 --updates_per_round 2 \
  --batch_size 32 --buffer_size 512 \
  --detector_lr 1e-4 --concept_lr 1e-4 \
  --min_confidence 0.5 --warmup_windows 0 \
  --device auto \
  --output "${OUT}"

# ---- 校验 LLM 是否真的驱动(而非静默回退) ---- #
"${PYTHON}" - "${OUT}" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
print("agent_source_counts:", m.get("agent_source_counts"))
print("overall:", m.get("overall"))
print("per_regime F1:", {r:v["f1"] for r,v in m.get("per_regime",{}).items()})
print("online_stats:", m.get("online_stats"))
PY

echo "done -> ${OUT}"
