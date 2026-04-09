"""
Dialogue Transformer for Living Tales mystery cases.

Autoregressive next-token prediction model for interleaved player/engine
dialogue. Intentionally small (~200K params) to overfit on a single case.

Reuses TokenEmbedding and TokenResonanceHead from energy_model.py.
Adds sequential position encoding and causal masking for autoregressive
generation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.energy_model import TokenEmbedding, TokenResonanceHead


class SequentialPositionEncoding(nn.Module):
    """Sinusoidal position encoding for dialogue sequences."""

    def __init__(self, max_len: int = 64, embedding_dim: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]


class DialogueTransformer(nn.Module):
    """
    Autoregressive transformer for mystery dialogue.

    Takes a sequence of token IDs (with class/phase/stream/agency metadata)
    and predicts the next token at each position via causal self-attention.

    Architecture:
        TokenEmbedding (64-dim) + PositionEncoding
        -> project up to context_dim (128)
        -> TransformerEncoder (causal mask, 2 layers, 4 heads)
        -> TokenResonanceHead (dot-product with vocab embeddings -> logits)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        context_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
        dropout: float = 0.0,  # no dropout — we want overfitting
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_encoding = SequentialPositionEncoding(max_seq_len, embedding_dim)
        self.proj_up = nn.Linear(embedding_dim, context_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=n_heads,
            dim_feedforward=context_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.resonance_head = TokenResonanceHead(context_dim, embedding_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        token_ids: torch.Tensor,       # (B, S)
        class_ids: torch.Tensor,       # (B, S)
        phase_ids: torch.Tensor,       # (B, S)
        stream_ids: torch.Tensor,      # (B, S)
        agency_ids: torch.Tensor,      # (B, S)
        padding_mask: torch.Tensor,    # (B, S) bool, True = padded
    ) -> torch.Tensor:
        """
        Returns (B, S, V) logits for next-token prediction at each position.
        """
        B, S = token_ids.shape

        # Embed tokens with metadata
        emb = self.token_embedding(
            token_ids, class_ids, phase_ids, stream_ids, agency_ids,
        )  # (B, S, embedding_dim)

        # Add positional encoding
        emb = emb + self.position_encoding(S)

        # Project to transformer dimension
        h = self.proj_up(emb)  # (B, S, context_dim)

        # Causal self-attention
        causal = self._causal_mask(S, h.device)
        h = self.transformer(
            h, mask=causal, src_key_padding_mask=padding_mask,
        )  # (B, S, context_dim)

        # Project to vocab logits via resonance head
        # Need full vocab embeddings (no metadata — just base token embeddings)
        all_ids = torch.arange(self.vocab_size, device=token_ids.device)
        all_token_embs = self.token_embedding.token_emb(all_ids)  # (V, embedding_dim)

        # Apply resonance head at each position
        # Reshape: (B*S, context_dim) -> (B*S, V) -> (B, S, V)
        h_flat = h.reshape(B * S, self.context_dim)
        logits_flat = self.resonance_head(h_flat, all_token_embs)  # (B*S, V)
        logits = logits_flat.reshape(B, S, self.vocab_size)

        return logits

    @torch.no_grad()
    def predict_next(
        self,
        token_ids: torch.Tensor,       # (1, S)
        class_ids: torch.Tensor,       # (1, S)
        phase_ids: torch.Tensor,       # (1, S)
        stream_ids: torch.Tensor,      # (1, S)
        agency_ids: torch.Tensor,      # (1, S)
        valid_mask: Optional[torch.Tensor] = None,  # (V,) bool — True = valid token
        temperature: float = 1.0,
    ) -> Tuple[int, torch.Tensor]:
        """
        Predict the next token given dialogue history.

        Args:
            valid_mask: Boolean mask over vocabulary. True = token is eligible.
                        Used to enforce phase gating + agency + no repeats.
            temperature: Sampling temperature. 0 = argmax.

        Returns:
            (chosen_idx, probabilities) — index into vocab and full prob vector.
        """
        self.eval()
        padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        logits = self.forward(
            token_ids, class_ids, phase_ids, stream_ids, agency_ids, padding_mask,
        )  # (1, S, V)

        # Take logits at the last position
        last_logits = logits[0, -1, :]  # (V,)

        # Mask invalid tokens
        if valid_mask is not None:
            last_logits = last_logits.masked_fill(~valid_mask, float("-inf"))

        if temperature <= 0:
            chosen = last_logits.argmax().item()
            probs = F.softmax(last_logits, dim=-1)
        else:
            scaled = last_logits / temperature
            probs = F.softmax(scaled, dim=-1)
            chosen = torch.multinomial(probs, 1).item()

        return chosen, probs
