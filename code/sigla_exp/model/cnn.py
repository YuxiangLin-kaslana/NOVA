from __future__ import annotations

import torch
from torch import nn


class CNNConceptDetector(nn.Module):
    """1D-CNN 多标签 concept detector（沿时间轴卷积）。

    输入窗口 [B, win, n_vars]：把 n_vars 当作输入通道、win 当作序列长度，
    用 Conv1d 沿时间卷积。卷积核**平移不变**，天生擅长检测「不管出现在
    窗口哪个位置」的局部形态（尖峰 / 高频振荡 / 方差爆发），补上扁平 MLP
    对位置随机的局部模式抓不住的短板；同时每个卷积核在每个时间步会混合
    所有维度，因此也能学到维间结构（correlation_break）。

    结构：
        Conv1d(n_vars -> C1, k) - BN - ReLU
        Conv1d(C1 -> C2, k)     - BN - ReLU
        全局池化(avg ⊕ max over time) -> 2*C2
        Linear - ReLU - Dropout - Linear -> n_concepts logits（多标签）
    """

    def __init__(
        self,
        win_size: int,
        n_vars: int,
        n_concepts: int = 6,
        channels: tuple[int, ...] = (64, 128),
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.n_vars = n_vars
        self.n_concepts = n_concepts

        layers: list[nn.Module] = []
        in_c = n_vars
        pad = kernel_size // 2                      # 'same' 长度
        for out_c in channels:
            layers += [
                nn.Conv1d(in_c, out_c, kernel_size, padding=pad),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
            ]
            in_c = out_c
        self.conv = nn.Sequential(*layers)

        feat_dim = 2 * in_c                          # avg ⊕ max 拼接
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, n_concepts),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = signal.transpose(1, 2)                   # [B, n_vars, win]
        h = self.conv(x)                             # [B, C, win]
        avg = h.mean(dim=-1)                          # [B, C]
        mx = h.max(dim=-1).values                     # [B, C]
        z = torch.cat([avg, mx], dim=-1)             # [B, 2C]
        return self.head(z)                          # logits [B, n_concepts]

    @torch.no_grad()
    def predict_proba(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(signal))
