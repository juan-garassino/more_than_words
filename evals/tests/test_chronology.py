"""Eval Pass D: Chronology compliance relative to baselines."""
from metrics.chronology import compute_phase_compliance


def test_phase_compliance(game_results, baselines):
    rate = compute_phase_compliance(game_results)
    threshold = max(0.90, baselines["random_phase_compliance"] * 1.05)
    assert rate >= threshold, (
        f"Phase compliance {rate:.1%} below threshold {threshold:.1%} "
        f"(baseline random={baselines['random_phase_compliance']:.1%})"
    )
