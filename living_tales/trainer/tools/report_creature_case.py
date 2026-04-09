#!/usr/bin/env python3
"""Generate a balancing report for an oscillating creature case."""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evals"))

from core.cartridge import CartridgeSpec
from core.creature_case import classify_creature_token_role
from evals.metrics.oscillation import (
    compute_arc_diversity,
    compute_combo_frequency,
    compute_dead_turn_rate,
    compute_decay_recovery_closure_rate,
    compute_dimension_volatility,
    compute_repetition_rate,
)
from evals.utils.creature_runner import CreatureGameRunner
from tools.pack_case import pack_case as pack_case_json


def _load_spec(case_id: str) -> CartridgeSpec:
    spec_path = _ROOT / "living_tales" / "trainer" / "cases" / case_id / "spec.json"
    if spec_path.exists():
        return CartridgeSpec.load(str(spec_path))

    case_json = _ROOT / "cases" / f"{case_id}.json"
    if not case_json.exists():
        raise FileNotFoundError(f"Packed case not found and source JSON missing: {case_id}")

    with tempfile.TemporaryDirectory(prefix=f"{case_id}_packed_") as tmpdir:
        out_dir = Path(tmpdir) / case_id
        pack_case_json(case_json, out_dir)
        return CartridgeSpec.load(str(out_dir / "spec.json"))


def report(case_id: str, runs: int) -> None:
    spec = _load_spec(case_id)
    if spec.mode != "oscillating":
        raise ValueError(f"{case_id} is mode={spec.mode}; use an oscillating creature case")

    results = CreatureGameRunner(spec).run_batch(runs, seeds=list(range(runs)))
    token_counts = Counter(
        turn.token_id
        for result in results
        for turn in result.turns
        if turn.token_role not in ("context", "state")
    )
    role_counts = Counter(
        turn.token_role
        for result in results
        for turn in result.turns
        if turn.token_role is not None
    )
    unresolved = Counter()
    for result in results:
        pending = []
        for turn in result.turns:
            if turn.token_role in ("decay", "decline"):
                pending.append(turn.token_id)
            elif turn.token_role == "recovery" and pending:
                pending.pop(0)
        unresolved.update(pending)

    print("=" * 68)
    print(f"  CREATURE REPORT — {spec.title}")
    print("=" * 68)
    print(f"  case_id                 : {spec.case_id}")
    print(f"  runs                    : {runs}")
    print(f"  final score mean        : {sum(r.final_convergence for r in results) / max(len(results), 1):.3f}")
    print(f"  decay->recovery closure : {compute_decay_recovery_closure_rate(results):.1%}")
    print(f"  combo frequency         : {compute_combo_frequency(results):.1%}")
    print(f"  repetition rate         : {compute_repetition_rate(results):.1%}")
    print(f"  dead-turn rate          : {compute_dead_turn_rate(results):.1%}")
    print(f"  arc diversity           : {compute_arc_diversity(results):.0f}")
    print(f"  dimension volatility    : {compute_dimension_volatility(results)}")

    print("\n  Role mix:")
    for role, count in role_counts.most_common():
        print(f"    {role:>8s}: {count}")

    print("\n  Most frequent non-context turns:")
    for token_id, count in token_counts.most_common(10):
        print(f"    {token_id:32s} {count:4d}  [{classify_creature_token_role(token_id)}]")

    print("\n  Unclosed decay/decline tokens:")
    for token_id, count in unresolved.most_common(10):
        print(f"    {token_id:32s} {count:4d}")

    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a balancing report for a creature case.")
    parser.add_argument("case_id", default="little_creature_M", nargs="?", help="Packed case ID")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    report(args.case_id, args.runs)


if __name__ == "__main__":
    main()
