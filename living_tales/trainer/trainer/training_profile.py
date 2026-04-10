"""
Training profile — auto-derived from case spec.

Computes anti-collapse parameters (entropy bonus, diversity loss, focal gamma,
KD alpha, learning rate) based on the case's engine pool size and game mode.
This ensures small mysteries aren't pushed to uniform by settings tuned for
large creature cases, and vice versa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.cartridge import CartridgeSpec
from core.token import TokenAgency, TokenStream


@dataclass
class TrainingProfile:
    """Training hyperparameters derived from case characteristics."""
    case_id: str
    game_type: str             # "mystery" or "creature"
    n_engine_tokens: int       # engine + shared non-invariant non-opening
    n_dims: int                # attractor dimensions
    max_entropy: float         # ln(n_engine_tokens)
    collapse_threshold: float  # entropy below this → anti-collapse kicks in
    entropy_coef: float        # strength of entropy bonus when collapsing
    diversity_coef: float      # strength of batch diversity loss when collapsing
    focal_gamma: float         # focal loss exponent
    kd_alpha: float            # KD soft target weight (rest is hard CE)
    lr: float                  # learning rate

    @classmethod
    def from_spec(cls, spec: CartridgeSpec) -> TrainingProfile:
        is_creature = getattr(spec, "mode", "converging") == "oscillating"
        game_type = "creature" if is_creature else "mystery"

        # Count engine pool
        n_engine = sum(
            1 for t in spec.tokens
            if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        )
        n_engine = max(n_engine, 2)  # safety
        max_ent = math.log(n_engine)

        # Collapse threshold: 40% of max entropy
        # Small mystery (42 eng): 1.5 nats ≈ collapsing to ~4 tokens
        # XL creature (176 eng): 2.1 nats ≈ collapsing to ~8 tokens
        collapse_threshold = 0.40 * max_ent

        if is_creature:
            # Creatures: decay tokens dominate, need stronger anti-collapse
            entropy_coef = 0.20
            diversity_coef = 0.4
            focal_gamma = 2.0
            kd_alpha = 0.15       # hard targets dominant
        else:
            # Mysteries: gentler settings, let conditional learning proceed
            entropy_coef = 0.10
            diversity_coef = 0.2
            focal_gamma = 1.5     # softer focal loss
            kd_alpha = 0.20       # slightly more KD (graph structure matters)

        # LR: slightly lower for larger models
        lr = 1e-3 if n_engine < 100 else 8e-4

        return cls(
            case_id=spec.case_id,
            game_type=game_type,
            n_engine_tokens=n_engine,
            n_dims=spec.n_attractor_dims,
            max_entropy=max_ent,
            collapse_threshold=collapse_threshold,
            entropy_coef=entropy_coef,
            diversity_coef=diversity_coef,
            focal_gamma=focal_gamma,
            kd_alpha=kd_alpha,
            lr=lr,
        )

    def log_summary(self) -> str:
        return (
            f"[{self.game_type}] {self.case_id}: "
            f"eng={self.n_engine_tokens} dims={self.n_dims} "
            f"max_ent={self.max_entropy:.1f} collapse<{self.collapse_threshold:.2f} "
            f"focal_γ={self.focal_gamma} kd_α={self.kd_alpha} lr={self.lr:.0e}"
        )
