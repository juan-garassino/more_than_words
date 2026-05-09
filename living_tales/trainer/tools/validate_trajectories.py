#!/usr/bin/env python3
"""
Validate Trajectories
=====================
Walks one or all hand-authored trajectories for a case, derives the
per-turn game state from the trajectory itself, and checks every scene
tuple against the constraints in `cases/<case>/constraints.json`.

Usage
-----
    python tools/validate_trajectories.py amber_cipher --traj voss_via_cufflink
    python tools/validate_trajectories.py amber_cipher --all
    python tools/validate_trajectories.py amber_cipher --all --coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Path setup: tools/ is a sibling of generator/ inside trainer/.
_HERE = Path(__file__).resolve().parent
_TRAINER_DIR = _HERE.parent
_PROJECT_ROOT = _TRAINER_DIR.parent.parent  # .../010-more-than-words
sys.path.insert(0, str(_TRAINER_DIR))

from generator.constraints_compiler import ConstraintMask  # noqa: E402
from generator.trajectory_loader import (  # noqa: E402
    Trajectory,
    TrajectoryLoader,
    Turn,
)


# ─── Case asset loaders ──────────────────────────────────────────────────────
def _load_case_assets(case_id: str) -> Tuple[Dict, Dict, Dict[str, str]]:
    case_dir = _TRAINER_DIR / "cases" / case_id
    with open(case_dir / "dimensions.json") as f:
        dims = json.load(f)
    with open(case_dir / "constraints.json") as f:
        constraints = json.load(f)
    token_class_map: Dict[str, str] = {}
    tokens_path = case_dir / "tokens.json"
    if tokens_path.exists():
        with open(tokens_path) as f:
            tokens = json.load(f)
        if isinstance(tokens, list):
            iterable = tokens
        elif isinstance(tokens, dict):
            iterable = tokens.get("tokens") or list(tokens.values())
        else:
            iterable = []
        for tok in iterable:
            tid = tok.get("id")
            cls = tok.get("token_class") or tok.get("class")
            if tid and cls:
                token_class_map[tid] = cls
    return dims, constraints, token_class_map


def _dim_vocab(dimensions_json: Dict) -> Dict[str, List[str]]:
    return {
        d["name"]: list(d.get("vocab", []))
        for d in dimensions_json.get("dimensions", [])
    }


# ─── Game-state derivation ───────────────────────────────────────────────────
def _initial_state(
    opening_tokens: List[str],
    initial_convergence: Optional[List[float]] = None,
) -> Dict[str, Any]:
    visited: Set[str] = set()
    prev_locs: List[str] = []
    for tok in opening_tokens:
        if isinstance(tok, str) and tok.startswith("location:") \
                and tok != "location:none":
            visited.add(tok)
            prev_locs.append(tok)
    return {
        "previous_locations": prev_locs,
        "visited_locations": visited,
        "scene_index": 0,
        "convergence_dims": list(initial_convergence or [0.0, 0.0, 0.0]),
        "game_turn": 0,
        "last_player_card": None,
    }


def _state_for_turn(
    base_state: Dict[str, Any],
    turn: Turn,
    turn_index: int,
) -> Dict[str, Any]:
    """Build the state seen by the validator at the moment `turn` is
    being emitted. Convergence reflects the BEFORE value (i.e. the
    convergence available when deciding this turn's tuple) — for the
    first turn, that's the initial convergence; for later turns, it's
    the previous turn's `convergence_after`."""
    state = dict(base_state)
    state["scene_index"] = turn_index
    state["game_turn"] = turn.turn
    state["last_player_card"] = turn.player_card
    return state


def _advance_state_after(
    state: Dict[str, Any],
    turn: Turn,
) -> Dict[str, Any]:
    """Update state with the turn's emitted scene + convergence_after."""
    new_state = dict(state)
    new_state["previous_locations"] = list(state.get("previous_locations", []))
    new_state["visited_locations"] = set(state.get("visited_locations", set()))
    loc = turn.scene.get("LOCATION")
    if loc and loc != "location:none":
        if not new_state["previous_locations"] \
                or new_state["previous_locations"][-1] != loc:
            new_state["previous_locations"].append(loc)
        new_state["visited_locations"].add(loc)
    if turn.convergence_after:
        new_state["convergence_dims"] = list(turn.convergence_after)
    return new_state


# ─── Validation ──────────────────────────────────────────────────────────────
def validate_trajectory(
    traj: Trajectory,
    mask: ConstraintMask,
    dim_vocab: Dict[str, List[str]],
) -> Dict[str, Any]:
    violations: List[Tuple[int, str, str]] = []  # (turn_idx, rule_id, info)
    vocab_violations: List[Tuple[int, str, str]] = []  # (turn_idx, dim, val)

    state = _initial_state(traj.opening)
    for i, turn in enumerate(traj.turns):
        # Vocab-membership check first (cheap sanity).
        for dim, vocab in dim_vocab.items():
            val = turn.scene.get(dim)
            if val is None:
                vocab_violations.append((turn.turn, dim, "<missing>"))
            elif val not in vocab:
                vocab_violations.append((turn.turn, dim, val))

        per_turn_state = _state_for_turn(state, turn, i)
        ok, violated = mask.is_valid_tuple(turn.scene, per_turn_state)
        if not ok:
            for rid in violated:
                violations.append((turn.turn, rid, ""))

        state = _advance_state_after(per_turn_state, turn)

    return {
        "trajectory_id": traj.trajectory_id,
        "turn_count": len(traj.turns),
        "violations": violations,
        "vocab_violations": vocab_violations,
        "passed": len(violations) == 0 and len(vocab_violations) == 0,
    }


# ─── Coverage report ─────────────────────────────────────────────────────────
COVERAGE_DIMS = [
    "LOCATION", "PRESENCE", "OBJECT_FOCUS",
    "CAUSE", "REVELATION", "TRANSITION", "BEAT",
]


def coverage_report(
    trajs: List[Trajectory],
    dim_vocab: Dict[str, List[str]],
) -> None:
    print("\n=== Coverage report ===")
    # Per-dim coverage with min-trajectory thresholds per AUTHORING_TRAJECTORIES.md
    thresholds = {
        "LOCATION": 2,
        "PRESENCE": 2,
        "TRANSITION": 2,
        "OBJECT_FOCUS": 1,
        "CAUSE": 1,
        "REVELATION": 1,
        "BEAT": 1,
    }
    for dim in COVERAGE_DIMS:
        token_in_trajs: Dict[str, Set[str]] = defaultdict(set)
        for traj in trajs:
            for turn in traj.turns:
                tok = turn.scene.get(dim)
                if tok:
                    token_in_trajs[tok].add(traj.trajectory_id)
        thresh = thresholds.get(dim, 1)
        print(f"\n{dim} (need each token in >= {thresh} trajectories)")
        missing: List[str] = []
        for tok in dim_vocab.get(dim, []):
            if tok in {"presence:alone", "transition:none", "cause:none",
                       "stance:none", "tell:none", "object_focus:none",
                       "atmosphere:none", "revelation:none", "location:none"}:
                continue
            count = len(token_in_trajs.get(tok, set()))
            mark = "ok" if count >= thresh else "MISSING"
            if count < thresh:
                missing.append(tok)
            print(f"  [{mark}] {tok}: {count} trajectory(ies)")
        if missing:
            print(f"  -> {len(missing)} token(s) below threshold")

    # BEAT-phase coverage
    beats_seen: Set[str] = set()
    for traj in trajs:
        for turn in traj.turns:
            b = turn.scene.get("BEAT")
            if b:
                beats_seen.add(b)
    print("\nBEAT phases covered:", sorted(beats_seen))

    # Outcomes per suspect
    outcomes = Counter(t.outcome for t in trajs)
    print("\nOutcome distribution:")
    for k, v in outcomes.most_common():
        print(f"  {k}: {v}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate hand-authored trajectories against constraints."
    )
    parser.add_argument("case_id")
    parser.add_argument("--traj", help="Single trajectory id to validate.")
    parser.add_argument("--all", action="store_true",
                        help="Validate all trajectories in the manifest.")
    parser.add_argument("--coverage", action="store_true",
                        help="Print coverage report (requires --all).")
    args = parser.parse_args()

    dimensions, constraints, token_class_map = _load_case_assets(args.case_id)
    dim_vocab = _dim_vocab(dimensions)
    mask = ConstraintMask(constraints, dim_vocab, token_class_map)
    loader = TrajectoryLoader(args.case_id, project_root=_PROJECT_ROOT)

    targets: List[Trajectory] = []
    if args.traj:
        targets = [loader.load(args.traj)]
    elif args.all:
        targets = loader.load_all()
    else:
        # Default: every trajectory listed in manifest.
        targets = loader.load_all()

    total_violations = 0
    total_vocab = 0
    for traj in targets:
        result = validate_trajectory(traj, mask, dim_vocab)
        print(
            f"\n--- {result['trajectory_id']}  "
            f"({result['turn_count']} turns) ---"
        )
        if result["passed"]:
            print("  PASS: no violations")
        else:
            print(
                f"  FAIL: {len(result['violations'])} rule violation(s), "
                f"{len(result['vocab_violations'])} vocab violation(s)"
            )
            if result["vocab_violations"]:
                print("  Vocab issues:")
                for tn, dim, val in result["vocab_violations"]:
                    print(f"    turn {tn:>3}  [{dim}] illegal value: {val}")
            if result["violations"]:
                print("  Rule violations:")
                for tn, rid, info in result["violations"]:
                    extra = f"  ({info})" if info else ""
                    print(f"    turn {tn:>3}  rule={rid}{extra}")
        total_violations += len(result["violations"])
        total_vocab += len(result["vocab_violations"])

    if args.coverage and len(targets) > 1:
        coverage_report(targets, dim_vocab)

    print(
        f"\n=== Summary: {len(targets)} trajectory(ies), "
        f"{total_violations} rule violation(s), "
        f"{total_vocab} vocab violation(s) ==="
    )
    return 0 if (total_violations == 0 and total_vocab == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
