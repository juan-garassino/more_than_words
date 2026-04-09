import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.datasets._schema import DialogueTurn, GameResult
from evals.metrics.oscillation import (
    compute_arc_diversity,
    compute_combo_frequency,
    compute_dead_turn_rate,
    compute_decay_recovery_closure_rate,
    compute_dimension_volatility,
    compute_repetition_rate,
)


def _creature_result(turns):
    return GameResult(
        seed=0,
        turns=turns,
        converged=False,
        final_convergence=turns[-1].convergence_at_step,
        final_token_ids=[turn.token_id for turn in turns],
        total_energy=-1.0,
        mode="oscillating",
    )


def test_decay_recovery_closure_rate_counts_closed_events():
    result = _creature_result([
        DialogueTurn("opening", "engine", "EVENT", "EARLY", 0.0, 0.50, token_role="context"),
        DialogueTurn("decay:hunger", "engine", "EVENT", "MID", -0.1, 0.40, token_role="decay"),
        DialogueTurn("action:fill_bowl", "player", "ACTION", "MID", -0.2, 0.48, token_role="action"),
        DialogueTurn("recovery:lick", "engine", "EMOTION", "MID", -0.3, 0.56, token_role="recovery"),
    ])
    assert compute_decay_recovery_closure_rate([result]) == 1.0


def test_combo_and_repetition_metrics():
    result = _creature_result([
        DialogueTurn("opening", "engine", "EVENT", "EARLY", 0.0, 0.50, token_role="context"),
        DialogueTurn("combo:zoomies", "engine", "EVENT", "MID", -0.1, 0.55, token_role="combo"),
        DialogueTurn("combo:stole_sock", "engine", "EVENT", "MID", -0.2, 0.53, token_role="combo"),
        DialogueTurn("recovery:nap", "player", "ACTION", "MID", -0.3, 0.62, token_role="recovery"),
    ])
    assert compute_combo_frequency([result]) > 0.0
    assert compute_repetition_rate([result]) > 0.0


def test_dead_turn_arc_diversity_and_volatility():
    turns = [
        DialogueTurn("opening", "engine", "EVENT", "EARLY", 0.0, 0.50, token_role="context", dimension_snapshot=[0.5, 0.5]),
        DialogueTurn("context:rain", "engine", "EVENT", "MID", -0.1, 0.505, token_role="context", dimension_snapshot=[0.5, 0.51]),
        DialogueTurn("decay:lonely", "engine", "EVENT", "MID", -0.2, 0.41, token_role="decay", dimension_snapshot=[0.32, 0.50]),
        DialogueTurn("recovery:wag", "player", "ACTION", "MID", -0.3, 0.58, token_role="recovery", dimension_snapshot=[0.62, 0.54]),
    ]
    result = _creature_result(turns)
    assert compute_dead_turn_rate([result]) > 0.0
    assert compute_arc_diversity([result]) >= 1.0
    volatility = compute_dimension_volatility([result])
    assert "0" in volatility and volatility["0"] > 0.0
