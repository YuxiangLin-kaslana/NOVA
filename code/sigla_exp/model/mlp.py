from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..actions import ACTION_NAMES
from ..profiles import CONCEPT_NAMES


def _mlp(dims: list[int], dropout: float = 0.0, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        is_last = idx == len(dims) - 2
        if final_activation or not is_last:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MLPAnomalyDetector(nn.Module):
    """Trainable bottom detector: reconstruct windows and use MSE as anomaly score."""

    def __init__(self, win_size: int, n_vars: int, latent_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        in_dim = win_size * n_vars
        hidden_dim = min(hidden_dim, max(32, in_dim))
        self.win_size = win_size
        self.n_vars = n_vars
        self.encoder = _mlp([in_dim, hidden_dim, latent_dim], final_activation=True)
        self.decoder = _mlp([latent_dim, hidden_dim, in_dim], final_activation=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        batch = signal.shape[0]
        z = self.encoder(signal.reshape(batch, -1))
        return self.decoder(z).reshape(batch, self.win_size, self.n_vars)

    @torch.no_grad()
    def anomaly_score(self, signal: torch.Tensor) -> torch.Tensor:
        recon = self.forward(signal)
        return torch.mean((recon - signal) ** 2, dim=(1, 2))


class MLPConceptExtractor(nn.Module):
    """Trainable concept extractor from evidence vector to five concept logits."""

    def __init__(
        self,
        evidence_dim: int = len(CONCEPT_NAMES),
        hidden_dim: int = 128,
        concept_dim: int = len(CONCEPT_NAMES),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = _mlp([evidence_dim, hidden_dim, hidden_dim // 2, concept_dim], dropout=dropout)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class MLPConceptDetector(nn.Module):
    """Multi-label concept detector over raw windows.

    Maps a window [B, win, n_vars] to per-concept logits [B, n_concepts].
    Trained on synthetically-injected concepts (spike / level_shift /
    oscillation / variance_burst / trend / correlation_break) with BCE,
    so a single window can carry multiple concepts at once (multi-label).
    Use sigmoid(logits) > threshold for per-concept decisions.
    """

    def __init__(
        self,
        win_size: int,
        n_vars: int,
        n_concepts: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.n_vars = n_vars
        self.n_concepts = n_concepts
        in_dim = win_size * n_vars
        self.backbone = _mlp([in_dim, hidden_dim, hidden_dim], dropout=dropout, final_activation=True)
        self.head = nn.Linear(hidden_dim, n_concepts)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        batch = signal.shape[0]
        hidden = self.backbone(signal.reshape(batch, -1))
        return self.head(hidden)               # logits [B, n_concepts]

    @torch.no_grad()
    def predict_proba(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(signal))


class MLPActionPolicy(nn.Module):
    """Trainable flat MLP policy for action type, action argument, and risk."""

    def __init__(
        self,
        win_size: int,
        n_vars: int,
        score_dim: int = 1,
        profile_dim: int = len(CONCEPT_NAMES),
        hidden_dim: int = 256,
        n_actions: int = len(ACTION_NAMES),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.n_vars = n_vars
        in_dim = win_size * n_vars + win_size * score_dim + profile_dim
        self.backbone = _mlp([in_dim, hidden_dim, hidden_dim], dropout=dropout, final_activation=True)
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.arg_head = nn.Linear(hidden_dim, n_vars)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, signal: torch.Tensor, score: torch.Tensor, profile: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = signal.shape[0]
        x = torch.cat(
            [
                signal.reshape(batch, -1),
                score.reshape(batch, -1),
                profile.reshape(batch, -1),
            ],
            dim=-1,
        )
        hidden = self.backbone(x)
        return {
            "action_logits": self.action_head(hidden),
            "arg_logits": self.arg_head(hidden),
            "risk_logit": self.risk_head(hidden).squeeze(-1),
        }


@dataclass
class TrainableModelBundle:
    detector: MLPAnomalyDetector
    concept_extractor: MLPConceptExtractor
    policy: MLPActionPolicy
