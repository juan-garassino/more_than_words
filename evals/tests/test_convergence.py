"""Eval Pass B: Multi-path convergence — rate, invariants, Lyapunov."""
from metrics.convergence import (
    compute_convergence_rate,
    compute_invariant_accuracy,
    compute_lyapunov_monotonicity,
)


def test_convergence_rate(game_results):
    rate = compute_convergence_rate(game_results)
    assert rate >= 0.8, f"Only {rate:.1%} converged (need >= 80%)"


def test_invariant_accuracy(game_results, spec):
    acc = compute_invariant_accuracy(game_results, spec.invariant_token_ids)
    assert acc >= 0.8, f"Invariant accuracy {acc:.1%} (need >= 80%)"


def test_lyapunov_monotonicity(game_results):
    rate = compute_lyapunov_monotonicity(game_results)
    assert rate >= 0.80, f"Lyapunov monotonicity {rate:.1%} (need >= 80%)"
