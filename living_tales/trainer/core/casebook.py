from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .game_mode import (
    MODE_CONVERGING,
    default_dimension_bounds,
    default_initial_dimension_value,
    score_dimensions,
    update_dimensions,
)
from .token import Token


GridPosition = Tuple[int, int]


@dataclass
class CasebookState:
    convergence_dimensions: np.ndarray
    mode: str = MODE_CONVERGING
    convergence_rate: float = 0.25
    dimension_lower_bound: float = 0.0
    dimension_upper_bound: float = 1.0
    placed_triads: Dict[GridPosition, List[Token]] = field(default_factory=dict)
    turn_count: int = 0

    def __post_init__(self) -> None:
        if self.convergence_dimensions.size == 0:
            return
        if self.mode == MODE_CONVERGING:
            self.dimension_lower_bound, self.dimension_upper_bound = default_dimension_bounds(self.mode)
        self.convergence_dimensions = np.clip(
            self.convergence_dimensions,
            self.dimension_lower_bound,
            self.dimension_upper_bound,
        )

    @property
    def convergence_score(self) -> float:
        return score_dimensions(self.convergence_dimensions, self.mode)

    @classmethod
    def create(cls, n_dims: int, mode: str, convergence_rate: float) -> "CasebookState":
        lower_bound, upper_bound = default_dimension_bounds(mode)
        initial = default_initial_dimension_value(mode)
        return cls(
            convergence_dimensions=np.full(n_dims, initial, dtype=np.float32),
            mode=mode,
            convergence_rate=convergence_rate,
            dimension_lower_bound=lower_bound,
            dimension_upper_bound=upper_bound,
        )

    def apply_delta(self, delta: np.ndarray) -> None:
        self.convergence_dimensions = update_dimensions(
            self.convergence_dimensions,
            delta,
            mode=self.mode,
            lower_bound=self.dimension_lower_bound,
            upper_bound=self.dimension_upper_bound,
        )

    def place_triad(self, tokens: List[Token], position: GridPosition) -> None:
        if len(tokens) != 3:
            raise ValueError("Triad must have exactly three tokens")
        self.placed_triads[position] = tokens
        self.turn_count += 1

        contribution = np.stack([t.attractor_weights for t in tokens]).mean(axis=0)
        self.apply_delta(contribution * self.convergence_rate)

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
