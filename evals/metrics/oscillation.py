"""Oscillating-mode metrics for creature balancing."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from evals.datasets._schema import GameResult
except ImportError:  # pragma: no cover
    from datasets._schema import GameResult


def compute_decay_recovery_closure_rate(results: List[GameResult]) -> float:
    closed = total = 0
    for result in results:
        pending = 0
        for turn in result.turns:
            role = turn.token_role
            if role in ("decay", "decline"):
                pending += 1
                total += 1
            elif role == "recovery" and pending > 0:
                pending -= 1
                closed += 1
    return closed / max(total, 1)


def compute_combo_frequency(results: List[GameResult]) -> float:
    total_turns = sum(len(result.turns) for result in results)
    combo_turns = sum(
        1
        for result in results
        for turn in result.turns
        if turn.token_role == "combo"
    )
    return combo_turns / max(total_turns, 1)


def compute_repetition_rate(results: List[GameResult]) -> float:
    repeats = total = 0
    for result in results:
        for prev, curr in zip(result.turns, result.turns[1:]):
            total += 1
            if prev.token_role == curr.token_role:
                repeats += 1
    return repeats / max(total, 1)


def compute_dead_turn_rate(results: List[GameResult], delta_threshold: float = 0.015) -> float:
    dead = total = 0
    for result in results:
        for prev, curr in zip(result.turns, result.turns[1:]):
            total += 1
            if abs(curr.convergence_at_step - prev.convergence_at_step) < delta_threshold:
                dead += 1
    return dead / max(total, 1)


def compute_arc_diversity(results: List[GameResult], window: int = 4) -> float:
    arcs: set[tuple[str, ...]] = set()
    for result in results:
        roles = [turn.token_role or "unknown" for turn in result.turns]
        for idx in range(max(0, len(roles) - window + 1)):
            arcs.add(tuple(roles[idx: idx + window]))
    return float(len(arcs))


def compute_dimension_volatility(results: List[GameResult]) -> Dict[str, float]:
    per_dim: Dict[int, List[float]] = {}
    for result in results:
        snapshots = [turn.dimension_snapshot for turn in result.turns if turn.dimension_snapshot]
        for prev, curr in zip(snapshots, snapshots[1:]):
            for idx, (a, b) in enumerate(zip(prev, curr)):
                per_dim.setdefault(idx, []).append(abs(b - a))
    return {
        str(idx): float(np.mean(values)) if values else 0.0
        for idx, values in sorted(per_dim.items())
    }
