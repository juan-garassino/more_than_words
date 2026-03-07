import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnergyMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.4):
        super().__init__()
        self.margin = margin

    def forward(self, e_pos, e_neg):
        return F.relu(self.margin - (e_neg - e_pos)).mean()


class AttractorConvergenceLoss(nn.Module):
    def __init__(self, threshold: float = 0.75):
        super().__init__()
        self.threshold = threshold

    def forward(self, cumulative_dims):
        deficit = F.relu(self.threshold - cumulative_dims)
        return deficit.mean()


class LyapunovRegularization(nn.Module):
    def forward(self, path_energies):
        if path_energies.size(1) < 2:
            return torch.tensor(0.0, device=path_energies.device)
        deltas = path_energies[:, 1:] - path_energies[:, :-1]
        increases = F.relu(deltas)
        return increases.mean()


class CombinedMysteryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.margin_loss = EnergyMarginLoss(margin=0.4)
        self.attractor_loss = AttractorConvergenceLoss(threshold=0.75)
        self.lyapunov_reg = LyapunovRegularization()
        self.convergence_loss = nn.MSELoss()
        self.retrieval_loss = HopfieldRetrievalLoss()

    def forward(self, batch: dict) -> dict:
        l_margin = self.margin_loss(batch["energy_positive"], batch["energy_negative"])
        l_attractor = self.attractor_loss(batch["cumulative_dimensions"])
        l_lyapunov = self.lyapunov_reg(batch["path_energies"])
        l_convergence = self.convergence_loss(
            batch["predicted_delta"], batch["target_delta"]
        )

        # Retrieval loss (optional — only present when invariant targets are provided)
        l_retrieval = torch.tensor(0.0, device=l_margin.device)
        if "retrieval_logits" in batch and "invariant_indices" in batch:
            l_retrieval = self.retrieval_loss(
                batch["retrieval_logits"], batch["invariant_indices"]
            )

        total = (
            1.0 * l_margin
            + 0.8 * l_attractor
            + 0.4 * l_lyapunov
            + 0.5 * l_convergence
            + 0.3 * l_retrieval
        )

        return {
            "total": total,
            "margin": l_margin,
            "attractor": l_attractor,
            "lyapunov": l_lyapunov,
            "convergence": l_convergence,
            "retrieval": l_retrieval,
        }


class HopfieldRetrievalLoss(nn.Module):
    """
    Cross-entropy loss for multi-dim invariant token retrieval.

    logits:  (B, n_dims, V)  — one score per vocabulary token per dimension
    targets: (B, n_dims)     — ground-truth invariant token index per dimension
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        B, n_dims, V = logits.shape
        return F.cross_entropy(logits.reshape(B * n_dims, V), targets.reshape(B * n_dims))


class EdgeCoherenceLoss(nn.Module):
    """BCE loss for edge coherence prediction (connection model)."""

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        # predicted: (B, 1), target: (B,)
        return F.binary_cross_entropy(predicted.squeeze(-1), target)


class CombinedConnectionLoss(nn.Module):
    """
    Composite loss for the connection model.
    weights: coherence=1.0, lyapunov=0.4
    """

    def __init__(self):
        super().__init__()
        self.coherence_loss = EdgeCoherenceLoss()
        self.lyapunov_reg = LyapunovRegularization()

    def forward(self, batch: dict) -> dict:
        l_coherence = self.coherence_loss(
            batch["predicted_coherence"], batch["target_coherence"]
        )
        if "path_energies" in batch:
            l_lyapunov = self.lyapunov_reg(batch["path_energies"])
        else:
            l_lyapunov = torch.tensor(0.0, device=l_coherence.device)

        total = 1.0 * l_coherence + 0.4 * l_lyapunov

        return {"total": total, "coherence": l_coherence, "lyapunov": l_lyapunov}
