"""
Audit a case JSON for dialogue training viability.

Checks token pool sizes, phase distribution, affinity tag overlap,
and convergence feasibility with single-token dialogue sampling.

Usage:
    cd living_tales/trainer
    python3 tools/audit_case_dialogue.py amber_cipher
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from core.cartridge import CartridgeSpec
from core.token import TokenAgency, TokenClass, TokenPhase, TokenStream
from generator.dialogue_sampler import DialogueSampler


def audit(spec: CartridgeSpec, n_simulations: int = 100) -> dict:
    """Run a full dialogue viability audit on a case."""
    results = {}

    # --- Token pool sizes ---
    player = [t for t in spec.tokens if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
              and not t.is_invariant and t.stream != TokenStream.OPENING]
    engine = [t for t in spec.tokens if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
              and not t.is_invariant and t.stream != TokenStream.OPENING]

    results["player_pool"] = len(player)
    results["engine_pool"] = len(engine)
    results["shared_count"] = sum(1 for t in spec.tokens if t.agency == TokenAgency.SHARED and not t.is_invariant)

    # --- Phase distribution per pool ---
    def phase_counts(pool):
        c = Counter(t.phase.value for t in pool)
        return {p: c.get(p, 0) for p in ["EARLY", "MID", "LATE", "ANY"]}

    results["player_phases"] = phase_counts(player)
    results["engine_phases"] = phase_counts(engine)

    # --- Affinity tag overlap ---
    player_tags = set()
    engine_tags = set()
    for t in player:
        player_tags.update(t.affinity_tags)
    for t in engine:
        engine_tags.update(t.affinity_tags)

    union = len(player_tags | engine_tags)
    inter = len(player_tags & engine_tags)
    results["tag_overlap_jaccard"] = inter / union if union > 0 else 0.0
    results["player_tags"] = len(player_tags)
    results["engine_tags"] = len(engine_tags)
    results["shared_tags"] = inter

    # --- Class distribution ---
    results["player_classes"] = dict(Counter(t.token_class.value for t in player))
    results["engine_classes"] = dict(Counter(t.token_class.value for t in engine))

    # --- Convergence feasibility at different rates ---
    feasibility = {}
    for rate in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sampler = DialogueSampler(
            spec, strategy="random", convergence_rate=rate, allow_partial=True,
        )
        paths = sampler.sample_batch(n_simulations, verbose=False)
        converged = sum(1 for p in paths if p.converged)
        feasibility[str(rate)] = converged / max(len(paths), 1)
    results["convergence_feasibility"] = feasibility

    # --- Suggestions ---
    suggestions = []
    if results["engine_pool"] < 15:
        suggestions.append(f"Engine pool too small ({results['engine_pool']}). Need >= 15 for dialogue.")
    if results["player_pool"] < 15:
        suggestions.append(f"Player pool too small ({results['player_pool']}). Need >= 15 for dialogue.")
    if results["tag_overlap_jaccard"] < 0.1:
        suggestions.append("Very low tag overlap between player/engine pools. Responsiveness reward won't work.")

    for pool_name, phases in [("player", results["player_phases"]), ("engine", results["engine_phases"])]:
        for phase in ["EARLY", "MID", "LATE"]:
            if phases.get(phase, 0) < 3:
                suggestions.append(f"{pool_name} pool has only {phases.get(phase, 0)} {phase} tokens (need >= 3).")

    # Find best convergence rate
    best_rate = max(feasibility, key=feasibility.get)
    if feasibility[best_rate] < 0.5:
        suggestions.append(f"Low convergence even at rate={best_rate}. Case may need graph redesign.")
    elif float(best_rate) != spec.convergence_rate:
        suggestions.append(f"Consider convergence_rate={best_rate} for dialogue (currently {spec.convergence_rate}).")

    results["suggestions"] = suggestions
    return results


def print_report(spec: CartridgeSpec, results: dict):
    """Print a formatted audit report."""
    w = 60
    print("=" * w)
    print(f"  DIALOGUE AUDIT — {spec.title}")
    print("=" * w)

    print(f"\n  Token Pools:")
    print(f"    Player:  {results['player_pool']} tokens")
    print(f"    Engine:  {results['engine_pool']} tokens")
    print(f"    Shared:  {results['shared_count']} tokens")

    print(f"\n  Phase Distribution:")
    for pool_name in ["player", "engine"]:
        phases = results[f"{pool_name}_phases"]
        parts = "  ".join(f"{k}:{v}" for k, v in phases.items() if v > 0)
        print(f"    {pool_name:>7s}: {parts}")

    print(f"\n  Affinity Tags:")
    print(f"    Player tags: {results['player_tags']}")
    print(f"    Engine tags: {results['engine_tags']}")
    print(f"    Shared tags: {results['shared_tags']}")
    print(f"    Overlap (Jaccard): {results['tag_overlap_jaccard']:.2f}")

    print(f"\n  Convergence Feasibility (random play):")
    for rate, conv in results["convergence_feasibility"].items():
        bar = "█" * int(conv * 20) + "░" * (20 - int(conv * 20))
        marker = " ← current" if float(rate) == spec.convergence_rate else ""
        print(f"    rate={rate}: [{bar}] {conv:.0%}{marker}")

    if results["suggestions"]:
        print(f"\n  Suggestions:")
        for s in results["suggestions"]:
            print(f"    ⚠ {s}")
    else:
        print(f"\n  ✓ No issues found. Case is dialogue-ready.")

    print("=" * w)


def main():
    parser = argparse.ArgumentParser(description="Audit a case for dialogue viability.")
    parser.add_argument("case_id", help="Case ID (e.g. amber_cipher)")
    parser.add_argument("--simulations", type=int, default=100)
    args = parser.parse_args()

    case_dir = _HERE.parent / "cases" / args.case_id
    spec_path = case_dir / "spec.json"
    if not spec_path.exists():
        print(f"Error: {spec_path} not found. Pack the case first.")
        sys.exit(1)

    spec = CartridgeSpec.load(str(spec_path))
    results = audit(spec, args.simulations)
    print_report(spec, results)


if __name__ == "__main__":
    main()
