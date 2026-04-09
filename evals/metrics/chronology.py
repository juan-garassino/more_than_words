"""Chronology compliance metrics: phase-order validation."""
from __future__ import annotations

from typing import List

from datasets._schema import GameResult

# Phase ordering: EARLY < MID < LATE
_PHASE_ORDER = {"EARLY": 0, "MID": 1, "LATE": 2, "INVARIANT": 3, "ANY": -1}


def compute_phase_compliance(results: List[GameResult]) -> float:
    """
    Fraction of tokens placed in valid chronological order.
    A token is compliant if its phase is >= all previously placed phases
    (ignoring ANY and OPENING tokens).
    """
    compliant = 0
    total = 0

    for r in results:
        max_phase_seen = -1
        for t in r.turns:
            phase_val = _PHASE_ORDER.get(t.phase, -1)
            if phase_val < 0:  # ANY or unknown — always valid
                compliant += 1
                total += 1
                continue

            total += 1
            if phase_val >= max_phase_seen:
                compliant += 1

            max_phase_seen = max(max_phase_seen, phase_val)

    return compliant / max(total, 1)
