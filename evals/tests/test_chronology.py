"""Eval Pass D: Chronology compliance — phase ordering."""
from metrics.chronology import compute_phase_compliance


def test_phase_compliance(game_results):
    rate = compute_phase_compliance(game_results)
    assert rate >= 0.90, f"Phase compliance {rate:.1%} (need >= 90%)"
