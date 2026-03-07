from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from core.hopfield import TokenGraph
except Exception:  # pragma: no cover - optional dependency for compute_energy
    TokenGraph = None  # type: ignore


class TokenEmbedding(nn.Module):
    # Streams: EVIDENCE, ATMOSPHERE, OPENING, INVARIANT = 4
    N_STREAMS = 4
    # Agencies: PLAYER, ENGINE, SHARED = 3
    N_AGENCIES = 3

    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.class_emb = nn.Embedding(14, 16)
        self.phase_emb = nn.Embedding(8, 8)
        self.stream_emb = nn.Embedding(self.N_STREAMS, 8)
        self.agency_emb = nn.Embedding(self.N_AGENCIES, 8)
        self.proj = nn.Linear(embedding_dim + 16 + 8 + 8 + 8, embedding_dim)

    def forward(self, token_ids, class_ids, phase_ids, stream_ids=None, agency_ids=None):
        t = self.token_emb(token_ids)
        c = self.class_emb(class_ids)
        p = self.phase_emb(phase_ids)
        parts = [t, c, p]
        if stream_ids is not None:
            parts.append(self.stream_emb(stream_ids))
        else:
            parts.append(t.new_zeros(*t.shape[:-1], 8))
        if agency_ids is not None:
            parts.append(self.agency_emb(agency_ids))
        else:
            parts.append(t.new_zeros(*t.shape[:-1], 8))
        return self.proj(torch.cat(parts, dim=-1))


class CasebookEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64, context_dim: int = 128):
        super().__init__()
        self.spatial_enc = nn.Linear(2, 16)
        self.proj = nn.Linear(embedding_dim + 16, context_dim)
        self.attn = nn.MultiheadAttention(context_dim, num_heads=4, batch_first=True)
        self.pool = nn.Linear(context_dim, context_dim)

    def forward(self, placed_emb, positions, mask=None):
        spatial = self.spatial_enc(positions.float())
        combined = self.proj(torch.cat([placed_emb, spatial], dim=-1))
        if mask is not None:
            empty = mask.all(dim=1)
            pooled = combined.new_zeros((combined.size(0), combined.size(2)))
            if (~empty).any():
                idx = (~empty).nonzero(as_tuple=True)[0]
                attended, _ = self.attn(
                    combined[idx],
                    combined[idx],
                    combined[idx],
                    key_padding_mask=mask[idx],
                )
                attended = attended.masked_fill(mask[idx].unsqueeze(-1), 0)
                n = (~mask[idx]).float().sum(1, keepdim=True).clamp_min(1.0)
                pooled[idx] = attended.sum(1) / n
        else:
            attended, _ = self.attn(combined, combined, combined, key_padding_mask=None)
            pooled = attended.mean(1)
        return self.pool(pooled)


class TriadEnergyHead(nn.Module):
    def __init__(self, embedding_dim: int = 64, context_dim: int = 128):
        super().__init__()
        self.joint = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.combiner = nn.Sequential(
            nn.Linear(128 + context_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, triad_emb, context):
        flat = triad_emb.reshape(triad_emb.size(0), -1)
        joint = self.joint(flat)
        combined = torch.cat([joint, context], dim=-1)
        return self.combiner(combined)


class ConvergenceHead(nn.Module):
    def __init__(self, embedding_dim: int = 64, context_dim: int = 128, n_dims: int = 3):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(embedding_dim * 3 + context_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_dims),
            nn.Sigmoid(),
        )

    def forward(self, triad_emb, context):
        flat = triad_emb.reshape(triad_emb.size(0), -1)
        return self.pred(torch.cat([flat, context], dim=-1))


class TokenResonanceHead(nn.Module):
    """
    Vocabulary projection: given a context vector, surface which tokens the
    field is 'calling for' next. This is the core game mechanic — after each
    player move, the engine reveals the top-k resonating tokens from the deck.

    Output: (B, vocab_size) logits (pre-softmax).
    """

    def __init__(self, context_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.scale = embedding_dim ** -0.5
        # Small bottleneck: project context into embedding space for dot-product.
        self.ctx_proj = nn.Linear(context_dim, embedding_dim, bias=False)

    def forward(self, context: torch.Tensor, all_token_embs: torch.Tensor) -> torch.Tensor:
        """
        context:        (B, context_dim)
        all_token_embs: (V, embedding_dim)  — embedding matrix for the full vocabulary
        returns:        (B, V) logits
        """
        q = self.ctx_proj(context)  # (B, D)
        return q @ all_token_embs.t() * self.scale  # (B, V)


class MysteryEnergyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        context_dim: int = 128,
        n_attractor_dims: int = 3,
        token_graph: Optional["TokenGraph"] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.casebook_encoder = CasebookEncoder(embedding_dim, context_dim)
        self.energy_head = TriadEnergyHead(embedding_dim, context_dim)
        self.convergence_head = ConvergenceHead(embedding_dim, context_dim, n_attractor_dims)
        self.resonance_head = TokenResonanceHead(context_dim, embedding_dim)
        self.token_graph = token_graph

    def forward(
        self,
        placed_token_ids,
        placed_class_ids,
        placed_phase_ids,
        placed_positions,
        placed_mask,
        candidate_token_ids,
        candidate_class_ids,
        candidate_phase_ids,
        placed_stream_ids=None,
        placed_agency_ids=None,
        candidate_stream_ids=None,
        candidate_agency_ids=None,
    ) -> dict:
        placed_emb = self.token_embedding(
            placed_token_ids, placed_class_ids, placed_phase_ids,
            placed_stream_ids, placed_agency_ids,
        )
        context = self.casebook_encoder(placed_emb, placed_positions, placed_mask)
        candidate_emb = self.token_embedding(
            candidate_token_ids, candidate_class_ids, candidate_phase_ids,
            candidate_stream_ids, candidate_agency_ids,
        )

        # Full vocabulary embedding matrix for resonance (no stream/agency bias).
        all_ids = torch.arange(self.vocab_size, device=placed_token_ids.device)
        all_token_embs = self.token_embedding.token_emb(all_ids)

        return {
            "energy": self.energy_head(candidate_emb, context),
            "convergence_delta": self.convergence_head(candidate_emb, context),
            "resonance_logits": self.resonance_head(context, all_token_embs),
        }

    def compute_energy(self, triad) -> float:
        """
        Deterministic energy estimate for analysis tools (Lyapunov checks).
        Falls back to 0.0 if no token graph is available.
        """
        if self.token_graph is None:
            return 0.0

        tokens = None
        if hasattr(triad, "tokens"):
            tokens = triad.tokens
        elif isinstance(triad, (list, tuple)):
            tokens = triad

        if tokens is None:
            return 0.0

        token_ids = [t.id for t in tokens]
        return float(self.token_graph.subgraph_energy(token_ids))
