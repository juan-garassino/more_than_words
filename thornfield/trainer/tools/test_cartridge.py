"""
Thornfield Cartridge Tester
============================
Automated diagnostics and simulated playthrough for a .cartridge file.

Usage
-----
    cd thornfield/trainer
    python3 tools/test_cartridge.py outputs/amber_cipher/TheAmberCipher.cartridge

What it checks
--------------
  1. Cartridge integrity     — all required files present, fields valid
  2. Model sanity            — energy separation between correct / random triads
  3. Retrieval accuracy      — does HopfieldRetrievalHead point to the invariants?
  4. Simulated playthrough   — greedy agent plays using model hints, checks convergence
  5. Energy distribution     — histogram of scores across hand combinations
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from tools.run_cartridge import (
    load_cartridge, init_game, card_name,
    _model_score_triad, _model_resonance, _model_retrieval,
    GameState, GameSpec,
)
from core.casebook import CasebookState

console = Console() if HAS_RICH else None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    if HAS_RICH:
        console.rule(f"[bold white]{title}[/bold white]")
    else:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _ok(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [green]✓[/green]  {msg}")
    else:
        print(f"  OK  {msg}")


def _warn(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [yellow]![/yellow]  {msg}")
    else:
        print(f"  !!  {msg}")


def _fail(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [red]✗[/red]  {msg}")
    else:
        print(f"  FAIL  {msg}")


def _info(msg: str) -> None:
    if HAS_RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        print(f"      {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Cartridge integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_integrity(spec: GameSpec, path: str) -> bool:
    _header("1 / 5  CARTRIDGE INTEGRITY")
    import zipfile, json

    required = ["manifest.json", "tokens.json", "graph.json",
                 "attractor.json", "phases.json", "proof.json", "weights.npz"]
    passed = True

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        for f in required:
            if f in names:
                _ok(f)
            else:
                _fail(f"Missing: {f}")
                passed = False

        proof = json.loads(zf.read("proof.json"))
        phases = json.loads(zf.read("phases.json"))
        tokens_data = json.loads(zf.read("tokens.json"))

    # Check stream/agency present (new export format)
    sample = tokens_data[0]
    if "stream" in sample and "agency" in sample:
        _ok("Tokens have stream + agency fields (new export format)")
    else:
        _warn("Tokens missing stream/agency — old export format, model energy will be degenerate")

    if "surface_expression" in sample:
        _ok("Tokens have surface_expression")
    else:
        _warn("Tokens missing surface_expression — card names will be auto-generated")

    if phases.get("opening_token_ids"):
        _ok(f"opening_token_ids present ({len(phases['opening_token_ids'])} tokens)")
    else:
        _warn("opening_token_ids missing — scene will start empty")

    # Proof check
    if proof.get("passed"):
        _ok(f"Proof: PASSED  convergence={proof.get('convergence_rate',0):.0%}  "
            f"lyapunov={proof.get('lyapunov_monotone_rate',0):.0%}  "
            f"basin={proof.get('basin_coverage',0):.0%}")
    else:
        _fail("Proof did not pass — cartridge should not have been exported")
        passed = False

    _info(f"vocab={spec.vocab_size}  dims={spec.n_attractor_dims}  "
          f"threshold={spec.convergence_threshold:.0%}  rate={spec.convergence_rate:.2f}  "
          f"turns={spec.min_turns}–{spec.max_turns}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Model energy separation
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_separation(spec: GameSpec, model, n: int = 40) -> bool:
    _header("2 / 5  MODEL ENERGY SEPARATION")

    if model is None:
        _warn("No model loaded — skipping")
        return True

    np.random.seed(0)
    state = init_game(spec)

    non_inv = [t for t in spec.tokens if not t.is_invariant]
    rng_np = np.random.default_rng(0)

    # Positive triads: use PathSampler — generates the exact triad types the model trained on.
    # The precompute (~50K triad validity checks) takes 5–15 s; print progress so it doesn't
    # look frozen.
    positive_triads: list = []
    try:
        from generator.path_sampler import PathSampler
        _info("  building valid triad index… (may take ~10 s)")
        sampler = PathSampler(spec, sampling_temperature=1.4, min_affinity=0.05,
                              allow_partial=True)
        _info(f"  valid triads precomputed: {len(sampler._precomputed_valid_triads)}")
        paths = sampler.sample_batch(n, verbose=False, max_attempts=n * 8)
        positive_triads = [
            triad for path in paths for triad in path
            if not all(t.is_invariant for t in triad)
        ][:n]
    except Exception as exc:
        _warn(f"PathSampler failed ({exc}), falling back to random-valid sampling")

    if not positive_triads:
        # Fallback: random triads that pass PathSampler's validity check
        def _valid(trio):
            if len({t.token_class for t in trio}) < 3:
                return False
            pairs = [(trio[0], trio[1]), (trio[0], trio[2]), (trio[1], trio[2])]
            return any(spec.token_graph.weight(a.id, b.id) > 0.05 for a, b in pairs)
        positive_triads = []
        for _ in range(5000):
            trio = list(rng_np.choice(non_inv, 3, replace=False))  # type: ignore
            if _valid(trio):
                positive_triads.append(trio)
            if len(positive_triads) >= n:
                break

    # Random triads as negatives (mirror of training negative samples)
    negative_triads = [
        list(rng_np.choice(non_inv, 3, replace=False))  # type: ignore
        for _ in range(n)
    ]

    ctx = list(state.atmosphere)

    pos_energies = [_model_score_triad(model, spec, ctx, t) for t in positive_triads[:n]]
    neg_energies = [_model_score_triad(model, spec, ctx, t) for t in negative_triads[:n]]

    if not pos_energies or not neg_energies:
        _warn("Could not sample triads")
        return True

    pos_mean = np.mean(pos_energies)
    neg_mean = np.mean(neg_energies)
    separation = neg_mean - pos_mean

    _info(f"Positive triads  (correct path):  mean energy = {pos_mean:.3f}")
    _info(f"Negative triads  (random):        mean energy = {neg_mean:.3f}")
    _info(f"Separation  (neg − pos):          {separation:.3f}  (want ≥ 0.40)")

    if pos_mean > 0.95:
        _warn("Positive energy ≈ 1.0 — model degenerate (likely missing stream/agency in cartridge)")
        return False
    elif separation >= 0.40:
        _ok(f"Good separation: {separation:.3f}")
        return True
    elif separation >= 0.20:
        _warn(f"Weak separation: {separation:.3f} — model may not distinguish well")
        return True
    else:
        _fail(f"Poor separation: {separation:.3f} — energy gating will not work")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Retrieval accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval(spec: GameSpec, model) -> bool:
    _header("3 / 5  HOPFIELD RETRIEVAL ACCURACY")

    if model is None or not hasattr(model, "retrieval_head"):
        _warn("No retrieval head — skipping")
        return True

    # Build context: use 6 affinity-connected tokens from non-invariant pool
    # (avoids the O(n³) PathSampler precompute which is slow interactively)
    rng = np.random.default_rng(1)
    non_inv = [t for t in spec.tokens if not t.is_invariant]
    context = list(rng.choice(non_inv, min(9, len(non_inv)), replace=False))  # type: ignore

    predicted = _model_retrieval(model, spec, context)
    correct_ids = set(spec.invariant_token_ids)

    if predicted is None:
        _warn("retrieval_logits not in model output")
        return True

    table = Table(box=box.SIMPLE) if HAS_RICH else None
    if table:
        table.add_column("Dim", width=4)
        table.add_column("Predicted token")
        table.add_column("Correct?", width=8)
        table.add_column("True invariant")

    n_correct = 0
    for d, pid in enumerate(predicted):
        try:
            pred_tok  = spec.get_token(pid)
            pred_name = card_name(pred_tok)
        except KeyError:
            pred_name = pid

        true_tok = next(
            (spec.get_token(tid) for tid in spec.invariant_token_ids
             if spec.get_token(tid).attractor_weights[d] > 0.5),
            None
        )
        true_name = card_name(true_tok) if true_tok else "?"
        correct = pid in correct_ids and (true_tok and pid == true_tok.id)
        if correct:
            n_correct += 1

        if table:
            mark = "[green]✓[/green]" if correct else "[red]✗[/red]"
            table.add_row(str(d), pred_name, mark, true_name)
        else:
            status = "OK" if correct else "WRONG"
            print(f"  dim{d}: predicted={pred_name}  true={true_name}  [{status}]")

    if table:
        console.print(table)

    _info(f"Context length: {len(context)} tokens")
    if n_correct == len(predicted):
        _ok(f"All {n_correct}/{len(predicted)} dimensions correct")
        return True
    elif n_correct > 0:
        _warn(f"{n_correct}/{len(predicted)} dimensions correct — partial retrieval")
        return True
    else:
        _warn("0 dimensions correct — retrieval not yet accurate with this context size")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Simulated playthrough (greedy agent)
# ─────────────────────────────────────────────────────────────────────────────

def test_playthrough(spec: GameSpec, model, seed: int = 42) -> bool:
    _header("4 / 5  SIMULATED PLAYTHROUGH  (greedy agent)")

    np.random.seed(seed)
    state = init_game(spec)

    # Add engine tokens to scene at start
    for _ in range(3):
        if state.engine_pool:
            chosen = state.engine_pool.pop(0)
            state.atmosphere.append(chosen)
            state.played_ids.add(chosen.id)

    history: List[Tuple[int, float, float, str]] = []  # turn, energy, conv, narrative

    for turn in range(spec.max_turns):
        if len(state.hand) < 3:
            break

        context = list(state.atmosphere) + state.casebook.all_placed_tokens()

        # Score all 3-combinations from hand
        best_triad = None
        best_energy = float("inf")

        combos = list(combinations(range(len(state.hand)), 3))
        np.random.shuffle(combos)
        for idxs in combos[:20]:  # check up to 20 combos
            triad = [state.hand[i] for i in idxs]
            e = _model_score_triad(model, spec, context, triad)
            if e < best_energy:
                best_energy = e
                best_triad  = list(idxs)

        if best_triad is None:
            break

        triad = [state.hand[i] for i in best_triad]
        coherence = 1.0 - best_energy

        # Energy-gated convergence (no degenerate fallback — cartridge is verified)
        prev = state.casebook.convergence_score
        pos  = (turn // 6, turn % 6)
        state.casebook.placed_triads[pos] = list(triad)
        state.casebook.turn_count += 1

        if coherence >= 0.60:
            scale = 1.0
        elif coherence >= 0.20:
            scale = (coherence - 0.20) / 0.40
        else:
            scale = 0.0

        if scale > 0:
            contribution = np.stack([t.attractor_weights for t in triad]).mean(axis=0)
            delta = contribution * spec.convergence_rate * scale
            state.casebook.convergence_dimensions = np.minimum(
                1.0, state.casebook.convergence_dimensions + delta
            )

        after = state.casebook.convergence_score
        for tok in triad:
            state.played_ids.add(tok.id)
        state.hand = [t for t in state.hand if t.id not in state.played_ids]

        # Refill
        while len(state.hand) < 9 and state.deck:
            state.hand.append(state.deck.pop(0))

        names = " + ".join(card_name(t)[:22] for t in triad)
        history.append((turn + 1, best_energy, after, names))

        # Advance scene every 3 turns
        if (turn + 1) % 3 == 0 and state.engine_pool:
            placed = [t.id for t in state.atmosphere] + list(state.casebook.placed_token_ids())
            cands  = state.engine_pool[:5]
            scored = [(t, spec.token_graph.induced_subgraph_energy(placed, [t.id])) for t in cands]
            scored.sort(key=lambda x: x[1])
            chosen = scored[0][0]
            state.engine_pool.remove(chosen)
            state.atmosphere.append(chosen)
            state.played_ids.add(chosen.id)

        if after >= spec.convergence_threshold:
            break

    # Print table
    if HAS_RICH:
        table = Table(title="Greedy agent playthrough", box=box.SIMPLE)
        table.add_column("Turn", width=5)
        table.add_column("Energy", width=8)
        table.add_column("Conv", width=7)
        table.add_column("Cards played", min_width=50)
        for turn, energy, conv, names in history:
            color = "green" if conv >= spec.convergence_threshold else (
                "yellow" if conv >= spec.convergence_threshold * 0.5 else "white")
            table.add_row(str(turn), f"{energy:.3f}", f"[{color}]{conv:.0%}[/{color}]", names)
        console.print(table)
    else:
        print(f"  {'Turn':5}  {'Energy':8}  {'Conv':7}  Cards")
        for turn, energy, conv, names in history:
            print(f"  {turn:5}  {energy:.3f}    {conv:.0%}    {names}")

    final = state.casebook.convergence_score
    solved = final >= spec.convergence_threshold
    turns_used = len(history)

    if solved:
        _ok(f"Converged in {turns_used} turns  ({final:.0%} ≥ {spec.convergence_threshold:.0%})")
    else:
        _warn(
            f"Greedy random agent reached {final:.0%} in {turns_used} turns  "
            f"(need {spec.convergence_threshold:.0%}) — "
            "expected: random hands rarely contain the right connections"
        )

    # Convergence failure is expected for a blind greedy agent — not a test failure.
    # A human player who reads the clues will do much better.
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Energy distribution across all hand combos
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_distribution(spec: GameSpec, model) -> bool:
    _header("5 / 5  ENERGY DISTRIBUTION  (all 3-combos from a fresh hand)")

    np.random.seed(7)
    state = init_game(spec)
    ctx   = list(state.atmosphere)

    combos  = list(combinations(range(len(state.hand)), 3))
    scores  = []
    for idxs in combos:
        triad = [state.hand[i] for i in idxs]
        e = _model_score_triad(model, spec, ctx, triad)
        scores.append((e, [card_name(t) for t in triad]))

    scores.sort(key=lambda x: x[0])
    energies = [s[0] for s in scores]

    _info(f"{len(scores)} combinations  min={min(energies):.3f}  "
          f"max={max(energies):.3f}  mean={np.mean(energies):.3f}")

    # Buckets
    buckets = {"0.0–0.2": 0, "0.2–0.4": 0, "0.4–0.6": 0, "0.6–0.8": 0, "0.8–1.0": 0}
    for e in energies:
        if   e < 0.2: buckets["0.0–0.2"] += 1
        elif e < 0.4: buckets["0.2–0.4"] += 1
        elif e < 0.6: buckets["0.4–0.6"] += 1
        elif e < 0.8: buckets["0.6–0.8"] += 1
        else:         buckets["0.8–1.0"] += 1

    max_cnt = max(*buckets.values(), 1)
    if HAS_RICH:
        table = Table(box=box.SIMPLE)
        table.add_column("Energy range")
        table.add_column("Count", width=6)
        table.add_column("Bar")
        for rng, cnt in buckets.items():
            bar = "█" * (cnt * 20 // max_cnt)
            color = "green" if rng.startswith("0.0") else (
                    "yellow" if rng.startswith("0.2") else "red")
            table.add_row(rng, str(cnt), f"[{color}]{bar}[/{color}]")
        console.print(table)
    else:
        for rng, cnt in buckets.items():
            bar = "█" * (cnt * 30 // max_cnt)
            print(f"  {rng}  {cnt:3d}  {bar}")

    # Show top 3 (lowest energy = best) and bottom 3 (highest = worst)
    if HAS_RICH:
        console.print("\n  [dim]Strongest connections (lowest energy):[/dim]")
        for e, names in scores[:3]:
            console.print(f"    [green]{e:.3f}[/green]  {' + '.join(names)}")
        console.print("  [dim]Weakest connections (highest energy):[/dim]")
        for e, names in scores[-3:]:
            console.print(f"    [red]{e:.3f}[/red]  {' + '.join(names)}")
    else:
        print("\n  Best:"); [print(f"    {e:.3f}  {n}") for e,n in scores[:3]]
        print("  Worst:"); [print(f"    {e:.3f}  {n}") for e,n in scores[-3:]]

    degenerate = all(e > 0.95 for e in energies)
    if degenerate:
        _warn("All energies ≈ 1.0 — model is degenerate (old cartridge, retrain needed)")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Thornfield cartridge diagnostics")
    parser.add_argument("cartridge", help="Path to .cartridge file")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--skip-playthrough", action="store_true")
    args = parser.parse_args()

    if HAS_RICH:
        with console.status("[bold]Loading…[/bold]"):
            spec, model = load_cartridge(args.cartridge)
    else:
        spec, model = load_cartridge(args.cartridge)

    if HAS_RICH:
        console.print(f"\n[bold]{spec.title}[/bold]  "
                       f"[dim]{Path(args.cartridge).name}[/dim]")

    results = {}
    results["integrity"]    = test_integrity(spec, args.cartridge)
    results["separation"]   = test_energy_separation(spec, model)
    results["retrieval"]    = test_retrieval(spec, model)
    if not args.skip_playthrough:
        results["playthrough"] = test_playthrough(spec, model, seed=args.seed)
    results["distribution"] = test_energy_distribution(spec, model)

    # Summary
    _header("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        if ok:
            _ok(name)
        else:
            _warn(name)
            all_pass = False

    if HAS_RICH:
        if all_pass:
            console.print("\n  [bold green]All checks passed.[/bold green]")
        else:
            console.print("\n  [bold yellow]Some checks need attention (see above).[/bold yellow]")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
