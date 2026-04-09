from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


MODE_CONVERGING = "converging"
MODE_OSCILLATING = "oscillating"
SUPPORTED_GAME_MODES = {MODE_CONVERGING, MODE_OSCILLATING}


def normalize_game_mode(mode: str | None) -> str:
    candidate = (mode or MODE_CONVERGING).strip().lower()
    if candidate not in SUPPORTED_GAME_MODES:
        return MODE_CONVERGING
    return candidate


def infer_game_mode(weight_vectors: Iterable[Iterable[float]], mode: str | None = None) -> str:
    normalized = normalize_game_mode(mode)
    if mode is not None:
        return normalized
    for weights in weight_vectors:
        if any(value < 0.0 for value in weights):
            return MODE_OSCILLATING
    return MODE_CONVERGING


def default_dimension_bounds(mode: str) -> Tuple[float, float]:
    normalized = normalize_game_mode(mode)
    if normalized == MODE_OSCILLATING:
        return 0.0, 1.0
    return 0.0, 1.0


def default_initial_dimension_value(mode: str) -> float:
    normalized = normalize_game_mode(mode)
    if normalized == MODE_OSCILLATING:
        return 0.15  # Creature starts needy — player sees progress from first action
    return 0.0


def score_dimensions(values: np.ndarray, mode: str) -> float:
    if values.size == 0:
        return 0.0
    # Both modes use min — the creature/detective is only as well as their weakest dimension.
    # For oscillating: if hunger is at 0.1, the creature is suffering regardless of other dims.
    # For converging: if one mystery dimension is unsolved, the case isn't closed.
    return float(values.min())


def update_dimensions(
    current: np.ndarray,
    delta: np.ndarray,
    *,
    mode: str,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    normalized = normalize_game_mode(mode)
    next_values = current + delta
    if normalized == MODE_CONVERGING:
        next_values = np.maximum(current, next_values)
    return np.clip(next_values, lower_bound, upper_bound)
