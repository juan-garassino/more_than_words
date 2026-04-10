"""
Training profile — auto-derived from case spec.

Computes model architecture, anti-collapse parameters, and learning rate
based on case characteristics (engine pool size, game mode, token count).
Supports size overrides for scaling experiments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from core.cartridge import CartridgeSpec
from core.token import TokenAgency, TokenStream


# Model size presets
MODEL_SIZES = {
    "S": {"n_layers": 6, "n_heads": 6, "embedding_dim": 96, "context_dim": 192},    # ~1.85M
    "M": {"n_layers": 8, "n_heads": 8, "embedding_dim": 128, "context_dim": 256},   # ~4M
    "L": {"n_layers": 10, "n_heads": 10, "embedding_dim": 160, "context_dim": 320},  # ~8M
}


@dataclass
class TrainingProfile:
    """Training hyperparameters derived from case characteristics."""
    case_id: str
    game_type: str             # "mystery" or "creature"
    model_size: str            # "S", "M", or "L"
    n_engine_tokens: int
    n_dims: int
    max_entropy: float
    collapse_threshold: float
    warmup_epochs: int         # skip anti-collapse for this many epochs
    entropy_coef: float
    diversity_coef: float
    focal_gamma: float
    kd_alpha: float
    lr: float
    # Model architecture
    n_layers: int
    n_heads: int
    embedding_dim: int
    context_dim: int

    @classmethod
    def from_spec(cls, spec: CartridgeSpec, model_size_override: str | None = None) -> "TrainingProfile":
        is_creature = getattr(spec, "mode", "converging") == "oscillating"
        game_type = "creature" if is_creature else "mystery"

        # Count engine pool
        n_engine = sum(
            1 for t in spec.tokens
            if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        )
        n_engine = max(n_engine, 2)
        max_ent = math.log(n_engine)

        # Collapse threshold: 40% of max entropy
        collapse_threshold = 0.40 * max_ent

        if is_creature:
            entropy_coef = 0.20
            diversity_coef = 0.4
            focal_gamma = 2.0
            kd_alpha = 0.15
        else:
            entropy_coef = 0.10
            diversity_coef = 0.2
            focal_gamma = 1.5
            kd_alpha = 0.20

        # Model size: override > auto-detect
        if model_size_override:
            size = model_size_override.upper()
        elif n_engine >= 80:
            size = "M"  # creatures + medium mysteries
        else:
            size = "S"  # small mysteries

        arch = MODEL_SIZES[size]

        # LR scales with model size
        lr = {"S": 1e-3, "M": 8e-4, "L": 5e-4}[size]

        return cls(
            case_id=spec.case_id,
            game_type=game_type,
            model_size=size,
            n_engine_tokens=n_engine,
            n_dims=spec.n_attractor_dims,
            max_entropy=max_ent,
            collapse_threshold=collapse_threshold,
            warmup_epochs=30,
            entropy_coef=entropy_coef,
            diversity_coef=diversity_coef,
            focal_gamma=focal_gamma,
            kd_alpha=kd_alpha,
            lr=lr,
            n_layers=arch["n_layers"],
            n_heads=arch["n_heads"],
            embedding_dim=arch["embedding_dim"],
            context_dim=arch["context_dim"],
        )

    def log_summary(self) -> str:
        return (
            f"[{self.game_type}] {self.case_id}: "
            f"model={self.model_size} ({self.n_layers}L/{self.embedding_dim}E/{self.context_dim}C) "
            f"eng={self.n_engine_tokens} dims={self.n_dims} heads={self.n_dims} "
            f"collapse<{self.collapse_threshold:.2f} warmup={self.warmup_epochs} "
            f"focal_γ={self.focal_gamma} kd_α={self.kd_alpha} lr={self.lr:.0e}"
        )


def build_head_vocab_masks(
    spec: CartridgeSpec,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Build per-head vocabulary masks from attractor weights.

    Returns (n_dims, vocab_size) bool tensor where mask[d][t] = True if token t
    has significant weight on dimension d (abs(weight) > threshold).

    Tokens with weight on a dimension are candidates for that head. Invariant
    tokens and opening tokens are excluded from all heads.
    """
    n_dims = spec.n_attractor_dims
    V = spec.vocab_size
    masks = torch.zeros(n_dims, V, dtype=torch.bool)

    for tok_idx, tok in enumerate(spec.tokens):
        if tok.is_invariant or tok.stream == TokenStream.OPENING:
            continue
        for dim in range(n_dims):
            if abs(tok.attractor_weights[dim]) > threshold:
                masks[dim, tok_idx] = True

    # Ensure every head has at least some candidates
    for dim in range(n_dims):
        if masks[dim].sum() < 3:
            # Fallback: include top tokens by abs weight on this dimension
            weights = torch.tensor([
                abs(tok.attractor_weights[dim]) if not tok.is_invariant else 0.0
                for tok in spec.tokens
            ])
            topk = min(10, V)
            top_indices = weights.topk(topk).indices
            masks[dim, top_indices] = True

    return masks
