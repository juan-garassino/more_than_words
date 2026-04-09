"""
Baseline runner: compute eval metrics on random play (no model).

Establishes "what random looks like" so eval thresholds can be set
relative to baselines rather than guessed.

Usage:
    cd evals
    python3 -m utils.baseline_runner amber_cipher
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
_EVALS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "thornfield" / "trainer"))

from core.cartridge import CartridgeSpec
from core.token import Token, TokenAgency, TokenStream
from dataclasses import dataclass, field as dc_field


@dataclass
class _DialogueTurn:
    token_id: str
    role: str
    token_class: str
    phase: str
    energy_at_step: float
    convergence_at_step: float


@dataclass
class _GameResult:
    seed: int
    turns: list
    converged: bool
    final_convergence: float
    final_token_ids: list
    total_energy: float


# Phase ordering for chronology check
_PHASE_ORDER = {"EARLY": 0, "MID": 1, "LATE": 2, "INVARIANT": 3, "ANY": -1}


def _compute_convergence_rate(results):
    if not results:
        return 0.0
    return sum(1 for r in results if r.converged) / len(results)


def _compute_lyapunov_monotonicity(results, tolerance=0.01):
    monotone = total = 0
    for r in results:
        for i in range(1, len(r.turns)):
            total += 1
            if r.turns[i].energy_at_step <= r.turns[i-1].energy_at_step + tolerance:
                monotone += 1
    return monotone / max(total, 1)


def _compute_jaccard_distance(results):
    if len(results) < 2:
        return 0.0
    sets = [set(t.token_id for t in r.turns) for r in results]
    dists = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            u = len(sets[i] | sets[j])
            dists.append(1.0 - len(sets[i] & sets[j]) / u if u else 0.0)
    return float(np.mean(dists))


def _compute_class_entropy(results):
    counts = {}
    for r in results:
        for t in r.turns:
            counts[t.token_class] = counts.get(t.token_class, 0) + 1
    total = sum(counts.values())
    return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)


def _compute_phase_compliance(results):
    compliant = total = 0
    for r in results:
        max_phase = -1
        for t in r.turns:
            pv = _PHASE_ORDER.get(t.phase, -1)
            total += 1
            if pv < 0 or pv >= max_phase:
                compliant += 1
            if pv >= 0:
                max_phase = max(max_phase, pv)
    return compliant / max(total, 1)


def _run_random_game(spec: CartridgeSpec, seed: int, max_turns: int = 60) -> _GameResult:
    """Play one dialogue game with uniform random token selection."""
    rng = np.random.RandomState(seed)

    player_pool = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]
    engine_pool = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]

    convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
    placed_ids: set = set()
    context_ids: list = []
    turns: list = []

    # Opening
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        placed_ids.add(tok.id)
        context_ids.append(tok.id)
        convergence_dims = np.minimum(
            1.0, convergence_dims + tok.attractor_weights * spec.convergence_rate,
        )
        turns.append(_DialogueTurn(
            token_id=tok.id, role="engine", token_class=tok.token_class.value,
            phase=tok.phase.value,
            energy_at_step=spec.token_graph.subgraph_energy(context_ids),
            convergence_at_step=float(convergence_dims.min()),
        ))

    is_player = True
    for step in range(len(turns), max_turns):
        game_turn = step // 2
        conv_score = float(convergence_dims.min())
        if conv_score >= spec.convergence_threshold and game_turn >= spec.min_turns:
            break

        pool = player_pool if is_player else engine_pool
        candidates = [
            t for t in pool
            if t.id not in placed_ids and t.is_available_at_turn(game_turn)
        ]
        if not candidates:
            # Try the other pool
            pool = engine_pool if is_player else player_pool
            candidates = [
                t for t in pool
                if t.id not in placed_ids and t.is_available_at_turn(game_turn)
            ]
        if not candidates:
            break

        chosen = candidates[rng.randint(len(candidates))]
        placed_ids.add(chosen.id)
        context_ids.append(chosen.id)
        convergence_dims = np.minimum(
            1.0, convergence_dims + chosen.attractor_weights * spec.convergence_rate,
        )
        turns.append(_DialogueTurn(
            token_id=chosen.id,
            role="player" if is_player else "engine",
            token_class=chosen.token_class.value,
            phase=chosen.phase.value,
            energy_at_step=spec.token_graph.subgraph_energy(context_ids),
            convergence_at_step=float(convergence_dims.min()),
        ))
        is_player = not is_player

    final_conv = float(convergence_dims.min())
    return _GameResult(
        seed=seed,
        turns=turns,
        converged=final_conv >= spec.convergence_threshold,
        final_convergence=final_conv,
        final_token_ids=list(context_ids),
        total_energy=spec.token_graph.subgraph_energy(context_ids) if context_ids else 0.0,
    )


def compute_baselines(spec: CartridgeSpec, n_games: int = 200) -> dict:
    """Run random games and compute all metric baselines."""
    print(f"Running {n_games} random dialogue games...", flush=True)
    results = [_run_random_game(spec, seed=i) for i in range(n_games)]

    converged = sum(1 for r in results if r.converged)
    print(f"  {converged}/{n_games} converged ({converged/n_games:.1%})", flush=True)

    baselines = {
        "case_id": spec.case_id,
        "n_games": n_games,
        "random_convergence_rate": _compute_convergence_rate(results),
        "random_lyapunov_monotonicity": _compute_lyapunov_monotonicity(results),
        "random_jaccard_distance": _compute_jaccard_distance(results),
        "random_phase_compliance": _compute_phase_compliance(results),
        "random_class_entropy": _compute_class_entropy(results),
        "random_perplexity": float(spec.vocab_size),
        "random_accuracy": 1.0 / spec.vocab_size,
        "random_mean_length": float(np.mean([len(r.turns) for r in results])),
        "timestamp": datetime.now().isoformat(),
    }

    return baselines


def main():
    parser = argparse.ArgumentParser(description="Compute eval baselines from random play.")
    parser.add_argument("case_id", help="Case ID (e.g. amber_cipher)")
    parser.add_argument("--n-games", type=int, default=200)
    args = parser.parse_args()

    case_dir = _ROOT / "thornfield" / "trainer" / "cases" / args.case_id
    spec_path = case_dir / "spec.json"
    if not spec_path.exists():
        print(f"Error: {spec_path} not found. Pack the case first.")
        sys.exit(1)

    spec = CartridgeSpec.load(str(spec_path))
    baselines = compute_baselines(spec, args.n_games)

    # Save
    out_dir = _ROOT / "evals" / "datasets_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}_baselines.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nBaselines saved to {out_path}")
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
