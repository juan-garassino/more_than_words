from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .token import Token


@dataclass
class Triad:
    tokens: List[Token]
    joint_energy: float = 0.0
    convergence_delta: np.ndarray | None = None
    suggested_position: Tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if len(self.tokens) != 3:
            raise ValueError("Triad must contain exactly 3 tokens")

    @property
    def is_class_diverse(self) -> bool:
        return len({t.token_class for t in self.tokens}) == 3

    @property
    def attractor_contribution(self) -> np.ndarray:
        weights = np.stack([t.attractor_weights for t in self.tokens], axis=0)
        return weights.mean(axis=0)
