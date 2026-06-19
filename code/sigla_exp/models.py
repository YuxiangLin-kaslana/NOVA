from __future__ import annotations

import torch
from torch import nn

from .actions import ACTION_NAMES
from .profiles import CONCEPT_NAMES


class MLPAutoEncoder(nn.Module):
    def __init__(self, win_size: int, n_vars: int, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = win_size * n_vars
        hidden_dim = min(hidden_dim, max(32, in_dim))
        self.win_size = win_size
        self.n_vars = n_vars
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        batch = signal.shape[0]
        z = self.encoder(signal.reshape(batch, -1))
        return self.decoder(z).reshape(batch, self.win_size, self.n_vars)

    @torch.no_grad()
    def anomaly_score(self, signal: torch.Tensor) -> torch.Tensor:
        recon = self.forward(signal)
        return torch.mean((recon - signal) ** 2, dim=(1, 2))


class SigLAPolicy(nn.Module):
    """Small Signal-Profile-Action policy for behavior-cloning experiments."""

    def __init__(
        self,
        n_vars: int,
        hidden_dim: int = 128,
        profile_dim: int = len(CONCEPT_NAMES),
        n_actions: int = len(ACTION_NAMES),
    ) -> None:
        super().__init__()
        self.signal_encoder = nn.GRU(
            input_size=n_vars + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.arg_head = nn.Linear(hidden_dim, n_vars)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, signal: torch.Tensor, score: torch.Tensor, profile: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = torch.cat([signal, score], dim=-1)
        _, hidden = self.signal_encoder(seq)
        signal_repr = hidden[-1]
        profile_repr = self.profile_encoder(profile)
        risk_state = torch.max(score, dim=1).values
        fused = self.fusion(torch.cat([signal_repr, profile_repr, risk_state], dim=-1))
        return {
            "action_logits": self.action_head(fused),
            "arg_logits": self.arg_head(fused),
            "risk_logit": self.risk_head(fused).squeeze(-1),
        }

