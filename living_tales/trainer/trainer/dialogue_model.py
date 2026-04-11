"""
Dialogue Transformer for Living Tales.

Two model variants:
  DialogueTransformer — single-token autoregressive (legacy)
  SceneTransformer    — multi-head parallel prediction (new)

SceneTransformer predicts N tokens in one forward pass, one per attractor
dimension. Player plays one token → model responds with a full scene
(mood + location + time + evidence + ...) simultaneously.
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

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_inference_inputs(
        self,
        token_ids: torch.Tensor,
        class_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        stream_ids: torch.Tensor,
        agency_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = self._model_device()
        token_ids = token_ids.to(device=device, dtype=torch.long)
        class_ids = class_ids.to(device=device, dtype=torch.long)
        phase_ids = phase_ids.to(device=device, dtype=torch.long)
        stream_ids = stream_ids.to(device=device, dtype=torch.long)
        agency_ids = agency_ids.to(device=device, dtype=torch.long)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        return token_ids, class_ids, phase_ids, stream_ids, agency_ids, valid_mask

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
        token_ids, class_ids, phase_ids, stream_ids, agency_ids, valid_mask = self._prepare_inference_inputs(
            token_ids, class_ids, phase_ids, stream_ids, agency_ids, valid_mask,
        )
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


# ---------------------------------------------------------------------------
# SceneTransformer — multi-head parallel prediction
# ---------------------------------------------------------------------------

class SceneTransformer(nn.Module):
    """
    Multi-head scene transformer for Living Tales.

    One forward pass predicts N tokens in parallel — one per attractor
    dimension. Each head has its own vocabulary mask (tokens relevant to
    that dimension) and its own linear projection.

    Architecture:
        TokenEmbedding + PositionEncoding
        → project up to context_dim
        → TransformerEncoder (causal mask)
        → N parallel Linear heads, each masked to dimension-relevant tokens
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 96,
        context_dim: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        n_output_heads: int = 3,
        head_vocab_masks: Optional[torch.Tensor] = None,  # (N, V) bool
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.n_output_heads = n_output_heads
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

        # N parallel output heads — one per attractor dimension
        self.output_heads = nn.ModuleList([
            nn.Linear(context_dim, vocab_size)
            for _ in range(n_output_heads)
        ])

        # Per-head vocabulary masks: (N, V) bool, True = token is valid for this head
        if head_vocab_masks is not None:
            self.register_buffer("head_vocab_masks", head_vocab_masks)
        else:
            # Default: all tokens valid for all heads (override before training)
            self.register_buffer(
                "head_vocab_masks",
                torch.ones(n_output_heads, vocab_size, dtype=torch.bool),
            )

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode(
        self,
        token_ids: torch.Tensor,       # (B, S)
        class_ids: torch.Tensor,       # (B, S)
        phase_ids: torch.Tensor,       # (B, S)
        stream_ids: torch.Tensor,      # (B, S)
        agency_ids: torch.Tensor,      # (B, S)
        padding_mask: torch.Tensor,    # (B, S) bool
    ) -> torch.Tensor:
        """Encode sequence → (B, S, context_dim)."""
        B, S = token_ids.shape
        emb = self.token_embedding(
            token_ids, class_ids, phase_ids, stream_ids, agency_ids,
        )
        emb = emb + self.position_encoding(S)
        h = self.proj_up(emb)
        causal = self._causal_mask(S, h.device)
        h = self.transformer(h, mask=causal, src_key_padding_mask=padding_mask)
        return h

    def forward(
        self,
        token_ids: torch.Tensor,
        class_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        stream_ids: torch.Tensor,
        agency_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Returns list of N tensors, each (B, S, V) — logits per head per position.
        Each head's logits are masked to its valid vocabulary.
        """
        h = self.encode(
            token_ids, class_ids, phase_ids, stream_ids, agency_ids, padding_mask,
        )  # (B, S, C)
        B, S, C = h.shape

        all_logits = []
        for i, head in enumerate(self.output_heads):
            logits = head(h)  # (B, S, V)
            mask = self.head_vocab_masks[i].unsqueeze(0).unsqueeze(0)  # (1, 1, V)
            logits = logits.masked_fill(~mask, float("-inf"))
            all_logits.append(logits)

        return all_logits

    @torch.no_grad()
    def predict_scene(
        self,
        token_ids: torch.Tensor,       # (1, S)
        class_ids: torch.Tensor,       # (1, S)
        phase_ids: torch.Tensor,       # (1, S)
        stream_ids: torch.Tensor,      # (1, S)
        agency_ids: torch.Tensor,      # (1, S)
        per_head_valid: Optional[List[torch.Tensor]] = None,  # list of N (V,) bool
        temperature: float = 0.8,
    ) -> List[Tuple[int, torch.Tensor]]:
        """
        Predict a full scene — N tokens in parallel, one per head.

        Args:
            per_head_valid: optional extra validity masks per head (e.g. phase gating,
                           no-repeat constraints). Applied on top of head_vocab_masks.

        Returns:
            List of (chosen_idx, probs) tuples, one per head.
        """
        self.eval()
        device = self._model_device()
        token_ids = token_ids.to(device, dtype=torch.long)
        class_ids = class_ids.to(device, dtype=torch.long)
        phase_ids = phase_ids.to(device, dtype=torch.long)
        stream_ids = stream_ids.to(device, dtype=torch.long)
        agency_ids = agency_ids.to(device, dtype=torch.long)

        padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        all_logits = self.forward(
            token_ids, class_ids, phase_ids, stream_ids, agency_ids, padding_mask,
        )

        results = []
        # Track tokens already chosen by earlier heads for deduplication
        used_indices: List[int] = []

        for i, logits in enumerate(all_logits):
            last = logits[0, -1, :].clone()  # (V,)

            # Apply extra validity mask if provided
            if per_head_valid is not None and per_head_valid[i] is not None:
                extra_mask = per_head_valid[i].to(device, dtype=torch.bool)
                last = last.masked_fill(~extra_mask, float("-inf"))

            # Mask out tokens already chosen by previous heads
            for idx in used_indices:
                last[idx] = float("-inf")

            if temperature <= 0:
                chosen = last.argmax().item()
                probs = F.softmax(last, dim=-1)
            else:
                scaled = last / temperature
                probs = F.softmax(scaled, dim=-1)
                # Handle case where all logits are -inf (no valid tokens for this head)
                if probs.sum() < 1e-8:
                    chosen = -1  # no valid token
                    probs = torch.zeros_like(probs)
                else:
                    chosen = torch.multinomial(probs, 1).item()

            if chosen >= 0:
                used_indices.append(chosen)
            results.append((chosen, probs))

        return results
