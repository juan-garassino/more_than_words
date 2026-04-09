import torch
import torch.nn as nn
import torch.nn.functional as F

from .energy_model import TokenEmbedding


class TamagotchiStateModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        n_needs: int = 6,
        n_personality_dims: int = 8,
    ):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embedding_dim)

        self.time_enc = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16))
        self.need_enc = nn.Sequential(nn.Linear(n_needs, 32), nn.GELU(), nn.Linear(32, 32))
        self.personality_enc = nn.Sequential(
            nn.Linear(n_personality_dims, 32), nn.GELU(), nn.Linear(32, 32)
        )

        self.coherence = nn.Sequential(
            nn.Linear(embedding_dim + 16 + 32 + 32, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.response = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 32, 128),
            nn.GELU(),
            nn.Linear(128, n_needs + n_personality_dims),
        )

    def encode_time(self, hour: torch.Tensor) -> torch.Tensor:
        angle = (hour.float() / 24.0) * 2 * 3.14159
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def coherence_energy(
        self,
        state_token_ids,
        state_class_ids,
        state_phase_ids,
        hour,
        need_levels,
        personality_state,
    ):
        state_emb = self.token_emb(state_token_ids, state_class_ids, state_phase_ids).mean(
            dim=1
        )

        time = self.time_enc(self.encode_time(hour))
        needs = self.need_enc(need_levels)
        personality = self.personality_enc(personality_state)

        combined = torch.cat([state_emb, time, needs, personality], dim=-1)
        return self.coherence(combined)

    def response_delta(
        self,
        state_emb,
        offering_token_ids,
        offering_class_ids,
        offering_phase_ids,
        personality_state,
    ):
        offering_emb = self.token_emb(
            offering_token_ids.unsqueeze(1),
            offering_class_ids.unsqueeze(1),
            offering_phase_ids.unsqueeze(1),
        ).squeeze(1)

        personality = self.personality_enc(personality_state)
        combined = torch.cat([state_emb, offering_emb, personality], dim=-1)
        return self.response(combined)


class TamagotchiCoherenceLoss(nn.Module):
    def __init__(self, margin: float = 0.4):
        super().__init__()
        self.margin = margin

    def forward(self, e_alive, e_incoherent, e_dissolved):
        l1 = F.relu(self.margin - (e_incoherent - e_alive))
        l2 = F.relu(self.margin - (e_dissolved - e_incoherent))
        return (l1 + l2).mean()
