"""Convergence metrics: rate, invariant accuracy, Lyapunov monotonicity."""
from __future__ import annotations

from typing import List

from datasets._schema import GameResult


def compute_convergence_rate(
    results: List[GameResult], threshold: float = 0.75,
) -> float:
    """Fraction of games reaching convergence_score >= threshold."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.converged) / len(results)


def compute_invariant_accuracy(
    results: List[GameResult], expected_invariants: List[str],
) -> float:
    """
    Fraction of converged games that ended with the correct invariant tokens
    present in the final placed token set.
    """
    converged = [r for r in results if r.converged]
    if not converged:
        return 0.0

    correct = 0
    for r in converged:
        placed = set(r.final_token_ids)
        if all(inv in placed for inv in expected_invariants):
            correct += 1
    return correct / len(converged)


def compute_lyapunov_monotonicity(
    results: List[GameResult], tolerance: float = 0.01,
) -> float:
    """Fraction of turns where energy decreased (or stayed flat within tolerance)."""
    monotone = 0
    total = 0

    for r in results:
        for i in range(1, len(r.turns)):
            prev_e = r.turns[i - 1].energy_at_step
            curr_e = r.turns[i].energy_at_step
            total += 1
            if curr_e <= prev_e + tolerance:
                monotone += 1

    return monotone / max(total, 1)
