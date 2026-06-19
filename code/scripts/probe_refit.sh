#!/bin/bash
# 诊断:regime-0 训练的 detector 能否在 regime-1 正常窗上重训压低重建误差?
#   能 -> online 失败是适应预算问题(可调);不能 -> AE 根本拟合不了漂移信号(需换机制)。
# 在 A40 上跑(轻量诊断)。
#SBATCH --job-name=sigla-probe
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --output=sigla-probe-%j.out
#SBATCH --error=sigla-probe-%j.err
set -euo pipefail
cd /u/ylin30/sigLA/code
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/projects/bflz/ylin30/conda_pkgs
conda activate /projects/bflz/ylin30/conda_envs/sigla
export PYTHONUNBUFFERED=1
/projects/bflz/ylin30/conda_envs/sigla/bin/python - <<'PY'
import torch, numpy as np, torch.nn.functional as F, sys
sys.path.insert(0,'.')
from sigla_exp.model import MLPAnomalyDetector
dev=torch.device('cuda'); print("device",torch.cuda.get_device_name(0))
m=MLPAnomalyDetector(100,38,latent_dim=128,hidden_dim=128).to(dev)
print("submodules:", [n for n,_ in m.named_children()])
print("norm layers:", [type(mm).__name__ for mm in m.modules() if 'Norm' in type(mm).__name__] or "NONE")
d=np.load('../specific_data/Online_training/streams/drift_gradual.npz')
x,y=d['x'].astype('float32'),d['y']
def wins(lo,hi,step=10):
    s=np.arange(lo,hi-100+1,step); return np.asarray([x[i:i+100] for i in s if not np.any(y[i:i+100]==1)],'float32')
ck=torch.load('runs/drift_detector_regime0/checkpoint_best.pt',map_location='cpu',weights_only=False)
m.load_state_dict(ck['detector']); m.eval()
def err(W):
    with torch.no_grad():
        b=torch.from_numpy(W).to(dev); return float(torch.mean((m(b)-b)**2))
THR=0.0139
for reg,(lo,hi) in [("regime1",(3000,6000)),("regime3",(9000,12000))]:
    m.load_state_dict(ck['detector']); m.eval()
    W=wins(lo,hi); print(f"\n=== {reg}: {len(W)} normal windows | fixed thr={THR} ===")
    print("  BEFORE adapt recon err: %.5f  (>thr -> 全误报)"%err(W))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3); Xt=torch.from_numpy(W).to(dev)
    for e in range(150):
        m.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),64):
            bb=Xt[perm[i:i+64]]; loss=F.mse_loss(m(bb),bb); opt.zero_grad(); loss.backward(); opt.step()
        if (e+1)%50==0: m.eval(); print("    ep%3d: %.5f"%(e+1,err(W)))
    m.eval(); print("  AFTER 150ep refit: %.5f  -> %s"%(err(W), "可学(调参问题)" if err(W)<THR else "学不动(根本限制)"))
PY
echo "probe done"
