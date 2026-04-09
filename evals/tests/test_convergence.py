"""Eval Pass B: Multi-path convergence relative to baselines."""
from metrics.convergence import (
    compute_convergence_rate,
    compute_invariant_accuracy,
    compute_lyapunov_monotonicity,
)


def test_convergence_rate(game_results, baselines):
    rate = compute_convergence_rate(game_results)
    threshold = max(0.8, baselines["random_convergence_rate"] * 1.5)
    assert rate >= threshold, (
        f"Convergence {rate:.1%} below threshold {threshold:.1%} "
        f"(baseline random={baselines['random_convergence_rate']:.1%})"
    )


def test_invariant_accuracy(game_results, spec):
    acc = compute_invariant_accuracy(game_results, spec.invariant_token_ids)
    assert acc >= 0.8, f"Invariant accuracy {acc:.1%} (need >= 80%)"


def test_lyapunov_monotonicity(game_results, baselines):
    rate = compute_lyapunov_monotonicity(game_results)
    threshold = max(0.80, baselines["random_lyapunov_monotonicity"] * 1.1)
    assert rate >= threshold, (
        f"Lyapunov monotonicity {rate:.1%} below threshold {threshold:.1%} "
        f"(baseline random={baselines['random_lyapunov_monotonicity']:.1%})"
    )
