from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from core.hopfield import TokenGraph
except Exception:  # pragma: no cover
    TokenGraph = None  # type: ignore

from trainer.energy_model import TokenEmbedding, CasebookEncoder, TokenResonanceHead
from core.connection import N_RELATIONS


class EdgeEmbedding(nn.Module):
    """
    Combines (emb_a, relation_emb, emb_b) → a single edge vector.
    relation_emb: nn.Embedding(N_RELATIONS, relation_dim)
    proj: Linear(embedding_dim + relation_dim + embedding_dim, embedding_dim)
    """

    def __init__(self, embedding_dim: int = 64, relation_dim: int = 16):
        super().__init__()
        self.relation_emb = nn.Embedding(N_RELATIONS, relation_dim)
        self.proj = nn.Linear(embedding_dim + relation_dim + embedding_dim, embedding_dim)

    def forward(
        self,
        emb_a: torch.Tensor,    # (..., embedding_dim)
        rel_ids: torch.Tensor,  # (...,)
        emb_b: torch.Tensor,    # (..., embedding_dim)
    ) -> torch.Tensor:          # (..., embedding_dim)
        rel = self.relation_emb(rel_ids)
        return self.proj(torch.cat([emb_a, rel, emb_b], dim=-1))


class TheoryEncoder(nn.Module):
    """
    Set attention over variable-length edge embeddings → fixed-size theory context.
    Mirrors the pattern used by CasebookEncoder.
    """

    def __init__(self, embedding_dim: int = 64, context_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, context_dim)
        self.attn = nn.MultiheadAttention(context_dim, num_heads=4, batch_first=True)
        self.pool = nn.Linear(context_dim, context_dim)

    def forward(
        self,
        edge_embs: torch.Tensor,                 # (B, max_edges, embedding_dim)
        mask: Optional[torch.Tensor] = None,     # (B, max_edges) True = padding
    ) -> torch.Tensor:                           # (B, context_dim)
        x = self.proj(edge_embs)                 # (B, E, context_dim)

        if mask is not None:
            empty = mask.all(dim=1)
            pooled = x.new_zeros(x.size(0), x.size(2))
            if (~empty).any():
                idx = (~empty).nonzero(as_tuple=True)[0]
                attended, _ = self.attn(
                    x[idx], x[idx], x[idx], key_padding_mask=mask[idx]
                )
                attended = attended.masked_fill(mask[idx].unsqueeze(-1), 0)
                n = (~mask[idx]).float().sum(1, keepdim=True).clamp_min(1.0)
                pooled[idx] = attended.sum(1) / n
        else:
            attended, _ = self.attn(x, x, x)
            pooled = attended.mean(1)

        return self.pool(pooled)


class CrossStreamAttention(nn.Module):
    """
    Fuses theory_context (B, C) and scene_context (B, C) into a single vector.
    Simple gated Linear + GELU.
    """

    def __init__(self, context_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
        )

    def forward(self, theory: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        return self.gate(torch.cat([theory, scene], dim=-1))


class EdgeCoherenceHead(nn.Module):
    """
    (edge_emb (B,64), combined_context (B,128)) → (B, 1) coherence score in [0,1].
    """

    def __init__(self, embedding_dim: int = 64, context_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([edge_emb, context], dim=-1))


class MysteryConnectionModel(nn.Module):
    """
    Connection-based Mystery model.

    Takes a theory (a set of typed token-pair edges), atmospheric context,
    and a candidate edge, and outputs:
      - edge_coherence:   (B, 1) probability the candidate fits the theory
      - resonance_logits: (B, vocab_size) next-token resonance scores
    """

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

        # Shared token embedding (same architecture as energy_model.py)
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)

        # Theory stream
        self.edge_embedding = EdgeEmbedding(embedding_dim)
        self.theory_encoder = TheoryEncoder(embedding_dim, context_dim)

        # Atmosphere stream — reuse CasebookEncoder
        self.atmosphere_encoder = CasebookEncoder(embedding_dim, context_dim)

        # Cross-stream fusion
        self.cross_attn = CrossStreamAttention(context_dim)

        # Output heads
        self.coherence_head = EdgeCoherenceHead(embedding_dim, context_dim)
        self.resonance_head = TokenResonanceHead(context_dim, embedding_dim)

        self.token_graph = token_graph

    def forward(
        self,
        # Theory: placed edges (variable length, padded)
        edge_a_ids, edge_a_class, edge_a_stream, edge_a_agency,  # (B, max_edges)
        edge_rel_ids,                                              # (B, max_edges)
        edge_b_ids, edge_b_class, edge_b_stream, edge_b_agency,  # (B, max_edges)
        edge_mask,                                                 # (B, max_edges) bool
        # Atmosphere: engine tokens (variable length, padded)
        atm_token_ids, atm_class, atm_stream,                    # (B, max_atm)
        atm_positions,                                             # (B, max_atm, 2)
        atm_mask,                                                  # (B, max_atm) bool
        # Candidate edge
        cand_a_ids, cand_a_class, cand_a_stream, cand_a_agency,  # (B,)
        cand_rel_ids,                                              # (B,)
        cand_b_ids, cand_b_class, cand_b_stream, cand_b_agency,  # (B,)
    ) -> Dict[str, torch.Tensor]:

        B, E = edge_a_ids.shape

        # ---- Theory edges ------------------------------------------------
        # Phase is not tracked for connections; use zeros.
        edge_a_phase = torch.zeros_like(edge_a_ids)
        edge_b_phase = torch.zeros_like(edge_b_ids)

        emb_a = self.token_embedding(
            edge_a_ids, edge_a_class, edge_a_phase, edge_a_stream, edge_a_agency
        )  # (B, E, D)
        emb_b = self.token_embedding(
            edge_b_ids, edge_b_class, edge_b_phase, edge_b_stream, edge_b_agency
        )  # (B, E, D)

        # Flatten batch × edges for EdgeEmbedding, then reshape back
        emb_a_flat = emb_a.reshape(B * E, -1)
        rel_flat = edge_rel_ids.reshape(B * E)
        emb_b_flat = emb_b.reshape(B * E, -1)
        edge_embs = self.edge_embedding(emb_a_flat, rel_flat, emb_b_flat)
        edge_embs = edge_embs.reshape(B, E, -1)  # (B, E, D)

        theory_ctx = self.theory_encoder(edge_embs, edge_mask)   # (B, context_dim)

        # ---- Atmosphere --------------------------------------------------
        atm_phase = torch.zeros_like(atm_token_ids)
        atm_agency_ids = torch.zeros_like(atm_token_ids)
        atm_emb = self.token_embedding(
            atm_token_ids, atm_class, atm_phase, atm_stream, atm_agency_ids
        )  # (B, max_atm, D)
        scene_ctx = self.atmosphere_encoder(atm_emb, atm_positions, atm_mask)  # (B, C)

        # ---- Cross-stream fusion -----------------------------------------
        combined = self.cross_attn(theory_ctx, scene_ctx)  # (B, context_dim)

        # ---- Candidate edge ----------------------------------------------
        cand_a_phase = torch.zeros_like(cand_a_ids)
        cand_b_phase = torch.zeros_like(cand_b_ids)

        cand_emb_a = self.token_embedding(
            cand_a_ids, cand_a_class, cand_a_phase, cand_a_stream, cand_a_agency
        )  # (B, D)
        cand_emb_b = self.token_embedding(
            cand_b_ids, cand_b_class, cand_b_phase, cand_b_stream, cand_b_agency
        )  # (B, D)
        cand_edge_emb = self.edge_embedding(cand_emb_a, cand_rel_ids, cand_emb_b)  # (B, D)

        # ---- Heads -------------------------------------------------------
        coherence = self.coherence_head(cand_edge_emb, combined)  # (B, 1)

        all_ids = torch.arange(self.vocab_size, device=cand_a_ids.device)
        all_token_embs = self.token_embedding.token_emb(all_ids)   # (V, D)
        resonance_logits = self.resonance_head(combined, all_token_embs)  # (B, V)

        return {
            "edge_coherence": coherence,
            "resonance_logits": resonance_logits,
        }
