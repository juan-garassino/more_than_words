from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .token import Token


GridPosition = Tuple[int, int]


@dataclass
class CasebookState:
    convergence_dimensions: np.ndarray
    convergence_rate: float = 0.25
    placed_triads: Dict[GridPosition, List[Token]] = field(default_factory=dict)
    turn_count: int = 0

    @property
    def convergence_score(self) -> float:
        if self.convergence_dimensions.size == 0:
            return 0.0
        return float(self.convergence_dimensions.min())

    def place_triad(self, tokens: List[Token], position: GridPosition) -> None:
        if len(tokens) != 3:
            raise ValueError("Triad must have exactly three tokens")
        self.placed_triads[position] = tokens
        self.turn_count += 1

        contribution = np.stack([t.attractor_weights for t in tokens]).mean(axis=0)
        self.convergence_dimensions = np.minimum(
            1.0, self.convergence_dimensions + contribution * self.convergence_rate
        )

    def all_placed_tokens(self) -> List[Token]:
        tokens: List[Token] = []
        for triad in self.placed_triads.values():
            tokens.extend(triad)
        return tokens

    def placed_token_ids(self) -> set[str]:
        return {t.id for t in self.all_placed_tokens()}

    def active_affinity_tags(self) -> set[str]:
        tags: set[str] = set()
        for token in self.all_placed_tokens():
            tags.update(token.affinity_tags)
        return tags
