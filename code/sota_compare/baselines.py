"""前人 SOTA 异常检测方法的**核心机制忠实再实现**(faithful compact),统一接口。

每个 baseline 都是**无监督正常性建模 + 标量异常分数**的检测器(在已知数据上学"正常",
对新窗给一个 anomaly score)。关键点:它们**只输出标量分数,没有类型概念**——
所以能(对强信号)把 novel 窗判为"异常",但**永远命名不出新类型**,也做不了类型化早预警。
这正是 SigLA 闭环要补的空白。

统一接口:
    det.fit(normal_windows)            # 在"正常"窗口上无监督预训练(可含已知异常?见各注释)
    det.score_stream(windows) -> ndarray   # 逐窗 anomaly score(MemStream 在此在线更新记忆→抗漂移)

threshold 由调用方在留出正常集上按 FAR≈5%(q95)标定,与 exp_early_warning 口径一致。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ #
#  MemStream (Bhatia et al., WWW 2022) —— 记忆 + 在线更新,显式抗 concept drift
#  机制:去噪自编码器把记录编码到嵌入;一块固定大小的"正常嵌入记忆库"M;
#       新窗 score = 嵌入到记忆库的最近邻 L2 距离;若 score<β(看起来正常)则把该嵌入
#       FIFO 写入记忆(吸收漂移),否则不污染记忆(挡住异常)。→ 适应分布漂移,但只有标量分数。
# ============================================================================ #
class _DenoisingAE(nn.Module):
    def __init__(self, in_dim, emb_dim=32, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, emb_dim))
        self.dec = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, in_dim))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class MemStream:
    name = "MemStream"

    def __init__(self, win, nvars, device, emb_dim=32, mem_size=256,
                 beta_q=0.95, epochs=40, lr=1e-3, seed=0):
        self.win, self.nvars, self.device = win, nvars, device
        self.in_dim = win * nvars
        self.emb_dim, self.mem_size = emb_dim, mem_size
        self.beta_q, self.epochs, self.lr, self.seed = beta_q, epochs, lr, seed
        self.ae = _DenoisingAE(self.in_dim, emb_dim).to(device)

    def _flat(self, X):
        return torch.tensor(np.stack(X).reshape(len(X), -1), dtype=torch.float32, device=self.device)

    def fit(self, normal_windows):
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        X = self._flat(normal_windows)
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
        self.mu, self.sd = mu, sd                                   # 标准化(数值稳定)
        Xn = (X - mu) / sd
        opt = torch.optim.AdamW(self.ae.parameters(), lr=self.lr, weight_decay=1e-5)
        self.ae.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xn), generator=g).to(self.device)
            for i in range(0, len(Xn), 128):
                idx = perm[i:i + 128]
                noisy = Xn[idx] + 0.1 * torch.randn_like(Xn[idx])    # 去噪
                rec, _ = self.ae(noisy)
                loss = F.mse_loss(rec, Xn[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.ae.eval()
        # 异常分 = AE 重构误差(标准化输入空间)。注:原始"嵌入到记忆最近邻距离"在本合成
        # benchmark 对注入异常非判别(已诊断,见 README),故采用重构误差的 faithful AE 变体;
        # MemStream 的特色——memory + β 门控的在线更新(抗漂移)——保留如下。
        with torch.no_grad():
            rec, _ = self.ae(Xn)
            r0 = ((rec - Xn) ** 2).mean(1)
        self.beta = float(torch.quantile(r0, self.beta_q))         # 记忆/在线更新的"看起来正常"门
        self.mem = [w for w in normal_windows[: self.mem_size]]    # 正常窗记忆库(供在线再训)
        self._ptr = 0
        self.opt = torch.optim.AdamW(self.ae.parameters(), lr=self.lr * 0.3, weight_decay=1e-5)
        return self

    @torch.no_grad()
    def _recon_err(self, x):
        xf = (self._flat([x]) - self.mu) / self.sd
        rec, _ = self.ae(xf)
        return float(((rec - xf) ** 2).mean())

    def _online_update(self):
        """在记忆库(近期正常窗)上轻量再训 AE → 吸收分布漂移(MemStream 的在线更新)。"""
        self.ae.train()
        X = (self._flat(self.mem) - self.mu) / self.sd
        for _ in range(2):
            noisy = X + 0.1 * torch.randn_like(X)
            rec, _ = self.ae(noisy)
            loss = F.mse_loss(rec, X)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.ae.eval()

    def score_stream(self, windows, update=True):
        """逐窗 anomaly score = 重构误差;update 且 score<β(看起来正常)时把窗写入记忆(FIFO)
        并周期性在记忆上轻量再训 AE → 在线吸收漂移。标定阈值时用 update=False(不更新)。"""
        self.ae.eval()
        out = []
        since = 0
        for x in windows:
            r = self._recon_err(x)
            out.append(r)
            if update and r < self.beta:                           # 看起来正常 → 进记忆(抗漂移)
                self.mem[self._ptr] = x
                self._ptr = (self._ptr + 1) % len(self.mem)
                since += 1
                if since >= 64:                                    # 周期性在线再训
                    self._online_update(); since = 0
        return np.asarray(out, np.float32)


# ============================================================================ #
#  Anomaly Transformer (Xu et al., ICLR 2022) —— 关联差异(association discrepancy)
#  机制:每层算 series-association S(学到的注意力)与 prior-association P(按时间距离的
#       可学习高斯先验);两者的对称 KL = 关联差异;minimax 训练放大异常处的差异。
#       逐点异常分 = softmax(-AssDis) · 重构误差;窗级 = 逐点均值。冻结后无在线更新(闭集 SOTA)。
# ============================================================================ #
class _AnomalyAttention(nn.Module):
    def __init__(self, d_model, n_heads, win):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.win = win
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.sigma = nn.Linear(d_model, n_heads)                    # 每位置每头的高斯尺度
        idx = torch.arange(win)
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs().float())  # |i-j|

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # [B,h,L,dk]
        # series-association(学到的)
        attn = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)
        S = torch.softmax(attn, dim=-1)                            # [B,h,L,L]
        # prior-association(可学习高斯,按 |i-j|)
        sigma = self.sigma(x).transpose(1, 2)                      # [B,h,L]
        sigma = torch.clamp(F.softplus(sigma), min=1e-2, max=50.0).unsqueeze(-1)  # [B,h,L,1]
        d = self.dist.to(x.device)[None, None]                     # [1,1,L,L]
        P = torch.exp(-(d ** 2) / (2 * sigma ** 2))
        P = P / (P.sum(-1, keepdim=True) + 1e-8)                    # 行归一化 [B,h,L,L]
        ctx = (S @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out(ctx), P, S


class _ATLayer(nn.Module):
    def __init__(self, d_model, n_heads, win, ff=128):
        super().__init__()
        self.att = _AnomalyAttention(d_model, n_heads, win)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model))

    def forward(self, x):
        a, P, S = self.att(x)
        x = self.n1(x + a)
        x = self.n2(x + self.ff(x))
        return x, P, S


class _ATNet(nn.Module):
    def __init__(self, win, nvars, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.inp = nn.Linear(nvars, d_model)
        self.pos = nn.Parameter(torch.randn(1, win, d_model) * 0.02)
        self.layers = nn.ModuleList([_ATLayer(d_model, n_heads, win) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, nvars)

    def forward(self, x):                                          # x: [B,L,nvars]
        h = self.inp(x) + self.pos
        Ps, Ss = [], []
        for lyr in self.layers:
            h, P, S = lyr(h)
            Ps.append(P); Ss.append(S)
        return self.out(h), Ps, Ss                                 # 重构, 先验/序列关联列表


def _assoc_dis(P, S):
    """对称 KL(逐 query 位置) → [B,h,L];P,S 行归一化。"""
    eps = 1e-8
    kl_ps = (P * ((P + eps).log() - (S + eps).log())).sum(-1)
    kl_sp = (S * ((S + eps).log() - (P + eps).log())).sum(-1)
    return 0.5 * (kl_ps + kl_sp)


class AnomalyTransformer:
    name = "AnomalyTransformer"

    def __init__(self, win, nvars, device, d_model=64, n_heads=4, n_layers=3,
                 epochs=40, lr=1e-3, lam=3.0, seed=0):
        self.win, self.nvars, self.device = win, nvars, device
        self.epochs, self.lr, self.lam, self.seed = epochs, lr, lam, seed
        self.net = _ATNet(win, nvars, d_model, n_heads, n_layers).to(device)

    def _t(self, X):
        return torch.tensor(np.stack(X), dtype=torch.float32, device=self.device)

    def _mean_ad(self, Ps, Ss, detach_p=False, detach_s=False):
        ads = []
        for P, S in zip(Ps, Ss):
            ads.append(_assoc_dis(P.detach() if detach_p else P,
                                   S.detach() if detach_s else S))
        return torch.stack(ads, 0).mean(0)                         # [B,h,L] 跨层均值

    def fit(self, normal_windows):
        torch.manual_seed(self.seed)
        X = self._t(normal_windows)
        mu, sd = X.mean((0, 1), keepdim=True), X.std((0, 1), keepdim=True) + 1e-6
        self.mu, self.sd = mu, sd
        Xn = (X - mu) / sd
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.net.train()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xn), generator=g).to(self.device)
            for i in range(0, len(Xn), 64):
                xb = Xn[perm[i:i + 64]]
                rec, Ps, Ss = self.net(xb)
                rl = F.mse_loss(rec, xb)
                # minimax:min-phase 让先验贴近序列;max-phase 推开序列(放大异常差异)
                ad_min = self._mean_ad(Ps, Ss, detach_s=True).mean()
                opt.zero_grad(); (rl + self.lam * ad_min).backward(retain_graph=True); opt.step()
                rec2, Ps2, Ss2 = self.net(xb)
                rl2 = F.mse_loss(rec2, xb)
                ad_max = self._mean_ad(Ps2, Ss2, detach_p=True).mean()
                opt.zero_grad(); (rl2 - self.lam * ad_max).backward(); opt.step()
        self.net.eval()

    @torch.no_grad()
    def score_stream(self, windows, update=True):
        """窗级异常分 = mean_t[ softmax_t(-AssDis_t) · 重构误差_t ](论文 AnomalyScore 的窗级聚合)。
        冻结 SOTA,无在线更新;update 仅为接口统一,忽略。"""
        self.net.eval()
        out = []
        for i in range(0, len(windows), 128):
            xb = (self._t(windows[i:i + 128]) - self.mu) / self.sd
            rec, Ps, Ss = self.net(xb)
            rerr = ((xb - rec) ** 2).mean(-1)                      # [B,L] 逐点重构误差
            ad = self._mean_ad(Ps, Ss).mean(1)                     # [B,L] 跨头均值
            w = torch.softmax(-ad, dim=-1)                         # [B,L]
            score = (w * rerr).mean(-1)                            # [B] 窗级
            out.append(score.cpu().numpy())
        return np.concatenate(out).astype(np.float32)
