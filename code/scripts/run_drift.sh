#!/bin/bash
# ============================================================================
# 抗漂移主实验:在线适应 vs 冻结(合成漂移流 drift_gradual)。
#
# 动机(由 6-16 决定性实验确定):高 recall 下 precision 天花板来自**漂移诱发的 FP**
# ——漂移正常段里 detector 重建误差因良性漂移升高,误报为异常。本实验直接验证:
#   持续在线适应能压低漂移段的重建误差 → FP 下降 → precision 回升;
#   冻结的 detector 在漂移后退化(重建误差/FP 随区间上升)。
#
# 受控对照(关键):两臂都从 SMD 预训练 detector 起步,在 regime 0(窗 0..116)上
# **完全相同**地适应(warmup=117);在第一个漂移点(窗 117)处:
#   online  freeze_after=0    继续适应全部漂移区间
#   frozen  freeze_after=117  就此冻结,不再更新
# 于是两臂在漂移开始时状态完全一致,唯一变量 = 漂移后是否继续适应。
# 决策一律用校准阈值(无 LLM):本实验只测 detector 适应,不涉及 decider。
#
# 提交:  sbatch scripts/run_drift.sh
# ============================================================================
#SBATCH --job-name=sigla-drift
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-drift-%j.out
#SBATCH --error=sigla-drift-%j.err

set -euo pipefail
SIGLA=/u/ylin30/sigLA
cd "${SIGLA}/code"
mkdir -p runs/online/drift

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
PYTHON=/projects/bflz/ylin30/conda_envs/sigla/bin/python
export PYTHONUNBUFFERED=1

# 漂移流:gradual(连续漂移,在线总滞后)或 abrupt(分段平稳,在线可追上)。可用环境变量覆盖。
STEM="${STEM:-drift_gradual}"
STREAM="${SIGLA}/specific_data/Online_training/streams/${STEM}.npz"
OUTDIR="runs/online/drift_${STEM}"
CON="runs/concept_detector_cnn_machine-1-1_w100_s10/checkpoint_best.pt"
mkdir -p "${OUTDIR}"

# ---- step 0: 在 regime-0 正常窗上训练共同起点 detector(SMD detector 不迁移到合成流) ---- #
DET="runs/drift_detector_${STEM}_regime0/checkpoint_best.pt"
if [ ! -f "${DET}" ]; then
  echo "==================== train regime-0 detector (${STEM}) ===================="
  "${PYTHON}" scripts/train_drift_detector.py "${STEM}"
else
  echo "[skip] regime-0 detector 已存在: ${DET}"
fi

WIN=100; STEP=5   # 密采样 -> 每区间 ~595 窗,给在线适应足够时间在区间内收敛
CAL="--cal_quantile 0.95 --cal_window 512 --cal_warmup 32 --cal_margin 1.0"
COMMON="--stream ${STREAM} --detector_ckpt ${DET} --concept_ckpt ${CON} \
  --concept_model cnn --win_size ${WIN} --step ${STEP} --hidden_dim 128 --latent_dim 128 \
  ${CAL} --device auto --decision calibrated_threshold --agent local"

run () {  # $1=name $2..=extra
  local name="$1"; shift
  echo "==================== ${name} ===================="
  "${PYTHON}" scripts/run_online.py ${COMMON} "$@" \
    --output "${OUTDIR}/${name}.json" \
    --predictions_csv "${OUTDIR}/pred_${name}.csv"
}

# 三臂从同一个 regime-0 detector 起步:
#   frozen        部署后完全冻结(--no_online)
#   online_naive  在线适应但只学校准器判正常的窗(漂移后被饿死,预期不恢复)
#   online_track  漂移跟踪:学近期所有窗(小缓冲快速遗忘旧区间)→ 跟住漂移正常流形
# 突发重训(burst refit):周期性对当前缓冲做大批量、多步的近似全量重拟合,
# 匹配离线探针(150ep→误差降到阈值下)。连续 mini-batch SGD 会停在远高于阈值的平台。
ADAPT="--update_scope full --detector_lr 1e-3 --retrain_every 50 --updates_per_round 200 --batch_size 256 --warmup_windows 0"
run "frozen" --no_online
run "online_naive" ${ADAPT}
# 稀疏覆盖缓冲:buffer 200 × stride 3 ≈ 600 窗 = 一个区间的跨度(step=5),
# 让 burst-refit 看到整个近期区间(逼近离线探针),而非重叠窄切片。
run "online_track" ${ADAPT} --detector_track_all --buffer_size 200 --detector_buffer_stride 3

echo "done -> ${OUTDIR}/  (用 ragenv2 的 python 跑 make_drift_figure.py ${STEM} 出图)"
