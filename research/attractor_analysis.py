#!/usr/bin/env python3
"""
research/attractor_analysis.py
================================
Two-track validation of Thornfield's theoretical foundations.

TRACK 1 — [HOPFIELD-LIKE ENERGY GEOMETRY]
  Measures properties derived from the TokenGraph weight matrix.
  subgraph_energy(S) = -sum(w_ij for i,j in S) is a genuine Lyapunov function:
  for non-negative weights, energy is guaranteed monotone decreasing as tokens
  are added.  These checks validate Hopfield-like properties of the *graph structure*.

TRACK 2 — [MODEL-GUIDED RETRIEVAL DYNAMICS]
  Measures convergence_score, which is an *additive accumulation counter*:
    convergence_dimensions += mean(attractor_weights) * convergence_rate
    convergence_score = min(convergence_dimensions)
  This is NOT classical Hopfield relaxation.  These checks validate associative
  recall by measuring whether greedy evidence selection reaches threshold within
  the turn budget.

Usage
-----
  # Run against a packed case directory:
  cd /path/to/more_than_words
  python3 research/attractor_analysis.py thornfield/trainer/cases/amber_cipher
  python3 research/attractor_analysis.py thornfield/trainer/cases/amber_cipher_L

  # Save plots to disk:
  python3 research/attractor_analysis.py thornfield/trainer/cases/amber_cipher --plots
"""
from __future__ import annotations

import argparse
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINER = _REPO_ROOT / "thornfield" / "trainer"
if str(_TRAINER) not in sys.path:
    sys.path.insert(0, str(_TRAINER))

from core.cartridge import CartridgeSpec       # noqa: E402
from core.casebook import CasebookState        # noqa: E402
from core.token import Token                   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _energy(spec: CartridgeSpec, token_ids: List[str]) -> float:
    return spec.token_graph.subgraph_energy(token_ids)


def _induced_energy(spec: CartridgeSpec, candidates: List[str], context: List[str]) -> float:
    return spec.token_graph.induced_subgraph_energy(candidates, context)


def _convergence_score(spec: CartridgeSpec, token_seqs: List[List[Token]]) -> float:
    """Simulate placing token_seqs (each a list of 3 tokens) and return final score."""
    state = CasebookState(
        convergence_dimensions=np.zeros(spec.n_attractor_dims, dtype=np.float32),
        convergence_rate=spec.convergence_rate,
    )
    for i, triad in enumerate(token_seqs):
        state.place_triad(triad, position=(0, i))
    return state.convergence_score


def _greedy_triads(
    spec: CartridgeSpec,
    pool: List[Token],
    context_ids: List[str],
    n_triads: int,
    n_candidates: int = 150,
) -> Tuple[List[List[Token]], List[str]]:
    """
    Greedy triad selection by -induced_subgraph_energy.
    Returns (selected_triads, updated_context_ids).
    NOTE: purely graph-energy scoring — no neural model involved.
    Samples n_candidates triads randomly rather than enumerating all C(n,3).
    """
    remaining = list(pool)
    context = list(context_ids)
    selected: List[List[Token]] = []

    for _ in range(n_triads):
        n = len(remaining)
        if n < 3:
            break
        best_score = float("-inf")
        best_triad: Optional[List[Token]] = None
        # Sample without enumerating all C(n,3)
        seen: set = set()
        for _ in range(min(n_candidates, max(1, n * (n - 1) * (n - 2) // 6))):
            idx = tuple(sorted(random.sample(range(n), 3)))
            if idx in seen:
                continue
            seen.add(idx)
            triple = [remaining[i] for i in idx]
            ids = [t.id for t in triple]
            score = -_induced_energy(spec, ids, context)
            if score > best_score:
                best_score = score
                best_triad = triple
        if best_triad is None:
            break
        selected.append(best_triad)
        best_ids = {t.id for t in best_triad}
        for t in best_triad:
            context.append(t.id)
        remaining = [r for r in remaining if r.id not in best_ids]

    return selected, context


def _random_triads(pool: List[Token], n_triads: int) -> List[List[Token]]:
    shuffled = list(pool)
    random.shuffle(shuffled)
    result = []
    for i in range(0, min(n_triads * 3, len(shuffled) - 2), 3):
        result.append(shuffled[i:i + 3])
    return result[:n_triads]


def _n_trials(spec: CartridgeSpec, base: int = 50) -> int:
    """Scale down trial count for larger cases to keep runtime reasonable."""
    if spec.vocab_size > 100:
        return max(20, base // 2)
    return base


def _header(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print('=' * 60)


def _subheader(label: str) -> None:
    print(f"\n  -- {label}")


# ===========================================================================
# TRACK 1: HOPFIELD-LIKE ENERGY GEOMETRY
# ===========================================================================

def track1_energy_geometry(spec: CartridgeSpec, rng: random.Random) -> Dict:
    """
    [HOPFIELD-LIKE ENERGY GEOMETRY]

    Examines the TokenGraph weight matrix as a Hopfield energy landscape.
    All measurements use subgraph_energy(), the genuine Lyapunov function.
    """
    _header("[HOPFIELD-LIKE ENERGY GEOMETRY]")

    non_inv = [t for t in spec.tokens if not t.is_invariant]
    inv_tokens = spec.invariant_tokens
    inv_ids = [t.id for t in inv_tokens]
    n_inv = len(inv_ids)

    results: Dict = {}

    # ------------------------------------------------------------------
    # 1A. Energy distribution: random vs partial-correct vs solution
    # ------------------------------------------------------------------
    _subheader("1A. Energy distribution (random / partial-correct / solution-core)")

    set_size = min(12, len(non_inv) // 2)

    random_energies = []
    for _ in range(200):
        sample = rng.sample(non_inv, set_size)
        random_energies.append(_energy(spec, [t.id for t in sample]))

    # Partial-correct: 50% tokens that are neighbours of invariants
    inv_neighbors: List[str] = []
    for iid in inv_ids:
        for (a, b) in spec.token_graph.edges:
            if a == iid and b not in inv_ids:
                inv_neighbors.append(b)
            elif b == iid and a not in inv_ids:
                inv_neighbors.append(a)
    inv_neighbors = list(set(inv_neighbors))
    inv_neighbor_tokens = [t for t in non_inv if t.id in inv_neighbors]

    partial_energies = []
    if len(inv_neighbor_tokens) >= set_size // 2:
        for _ in range(200):
            n_correct = set_size // 2
            correct_sample = rng.sample(inv_neighbor_tokens, min(n_correct, len(inv_neighbor_tokens)))
            fill_pool = [t for t in non_inv if t not in correct_sample]
            fill = rng.sample(fill_pool, set_size - len(correct_sample))
            sample = correct_sample + fill
            partial_energies.append(_energy(spec, [t.id for t in sample]))

    # Solution-core: invariant-neighbour heavy set
    solution_energies = []
    if len(inv_neighbor_tokens) >= set_size:
        for _ in range(200):
            sample = rng.sample(inv_neighbor_tokens, set_size)
            solution_energies.append(_energy(spec, [t.id for t in sample]))

    r_mean = np.mean(random_energies)
    r_std = np.std(random_energies)
    p_mean = np.mean(partial_energies) if partial_energies else float("nan")
    s_mean = np.mean(solution_energies) if solution_energies else float("nan")

    print(f"    Set size: {set_size}  |  n_inv_neighbours: {len(inv_neighbors)}")
    print(f"    Random         mean={r_mean:.4f}  std={r_std:.4f}")
    print(f"    Partial-correct mean={p_mean:.4f}")
    print(f"    Solution-core  mean={s_mean:.4f}")
    if not np.isnan(s_mean):
        gap = s_mean - r_mean
        print(f"    Energy gap (solution - random): {gap:.4f}  "
              f"({'negative = solution IS lower energy ✓' if gap < 0 else 'positive = solution NOT lower energy ✗'})")

    results["energy_distribution"] = {
        "set_size": set_size,
        "random_mean": float(r_mean),
        "partial_mean": float(p_mean),
        "solution_mean": float(s_mean),
        "gap_solution_minus_random": float(s_mean - r_mean) if not np.isnan(s_mean) else None,
    }

    # ------------------------------------------------------------------
    # 1B. Energy monotonicity under greedy evidence accumulation
    # ------------------------------------------------------------------
    _subheader("1B. Energy monotonicity (greedy-by-energy ordering)")
    print("    NOTE: For non-negative edge weights, subgraph_energy is")
    print("    GUARANTEED to decrease at each step — this is a structural property.")

    n_turns = spec.max_turns
    context: List[str] = list(spec.opening_token_ids)
    pool = [t for t in non_inv if t.id not in context]

    triads, _ = _greedy_triads(spec, pool, context, n_turns)

    energy_trace = []
    cumulative_ids = list(context)
    for triad in triads:
        cumulative_ids.extend(t.id for t in triad)
        energy_trace.append(_energy(spec, cumulative_ids))

    monotone_steps = sum(
        1 for i in range(1, len(energy_trace))
        if energy_trace[i] <= energy_trace[i - 1] + 1e-9
    )
    total_steps = max(len(energy_trace) - 1, 1)
    monotone_rate = monotone_steps / total_steps

    print(f"    Turns simulated: {len(energy_trace)}")
    print(f"    Monotone steps: {monotone_steps}/{total_steps}  "
          f"({monotone_rate * 100:.1f}%)")
    if energy_trace:
        print(f"    Energy range: [{min(energy_trace):.3f}, {max(energy_trace):.3f}]")
        print(f"    Final energy: {energy_trace[-1]:.4f}")

    results["energy_monotonicity"] = {
        "n_turns": len(energy_trace),
        "monotone_rate": monotone_rate,
        "energy_start": float(energy_trace[0]) if energy_trace else None,
        "energy_end": float(energy_trace[-1]) if energy_trace else None,
    }

    # ------------------------------------------------------------------
    # 1C. Invariant neighbourhood energy gap
    # ------------------------------------------------------------------
    _subheader("1C. Invariant neighbourhood energy gap")
    print("    Invariants are graph-isolated by design (no edges between them,")
    print("    energy of invariant-only set is always 0).  We measure the energy")
    print("    of each invariant's immediate graph neighbourhood (attractor basin core).")

    # Collect per-invariant neighbourhoods (non-invariant neighbours only)
    inv_neigh_map: Dict[str, List[str]] = {}
    for iid in inv_ids:
        neighbours = []
        for (a, b) in spec.token_graph.edges:
            if a == iid and b not in inv_ids:
                neighbours.append(b)
            elif b == iid and a not in inv_ids:
                neighbours.append(a)
        inv_neigh_map[iid] = neighbours
        n_top = min(5, len(neighbours))
        if n_top >= 2:
            # Sort by weight to get strongest neighbours
            sorted_n = sorted(neighbours,
                              key=lambda tid: spec.token_graph.weight(iid, tid),
                              reverse=True)
            basin_ids = [iid] + sorted_n[:n_top]
            basin_energy = _energy(spec, basin_ids)
            print(f"    {iid}: {n_top} neighbours, basin energy = {basin_energy:.4f}")

    # Measure combined neighbourhood of all invariants
    combined_basin = list(inv_ids)
    for neighbours in inv_neigh_map.values():
        # top-3 per invariant
        sorted_n = sorted(neighbours,
                          key=lambda tid: sum(spec.token_graph.weight(iid2, tid) for iid2 in inv_ids),
                          reverse=True)
        combined_basin.extend(sorted_n[:3])
    combined_basin = list(dict.fromkeys(combined_basin))  # dedup, preserve order
    basin_size_n = len(combined_basin)
    combined_energy = _energy(spec, combined_basin)

    # Compare to random same-size non-invariant sets
    sample_size = min(basin_size_n, len(non_inv))
    random_n_energies = [
        _energy(spec, [t.id for t in rng.sample(non_inv, sample_size)])
        for _ in range(500)
    ]
    rn_mean = np.mean(random_n_energies)
    rn_std = np.std(random_n_energies)
    percentile = sum(1 for e in random_n_energies if e > combined_energy) / len(random_n_energies)

    print(f"    Combined basin ({basin_size_n} tokens) energy: {combined_energy:.4f}")
    print(f"    Random {basin_size_n}-token sets: mean={rn_mean:.4f}  std={rn_std:.4f}")
    print(f"    Basin energy below {percentile * 100:.0f}% of random sets  "
          f"({'lower = solution basin more stable ✓' if percentile > 0.6 else 'not clearly lower ✗'})")

    results["invariant_energy_gap"] = {
        "basin_size": basin_size_n,
        "basin_energy": float(combined_energy),
        "random_mean": float(rn_mean),
        "random_std": float(rn_std),
        "percentile_above_random": float(percentile),
    }

    # ------------------------------------------------------------------
    # 1D. Edge weight distribution
    # ------------------------------------------------------------------
    _subheader("1D. Edge weight distribution")
    weights = list(spec.token_graph.edges.values())
    w_arr = np.array(weights)
    neg_count = int(np.sum(w_arr < 0))
    print(f"    Total edges: {len(weights)}")
    print(f"    Weight range: [{w_arr.min():.3f}, {w_arr.max():.3f}]")
    print(f"    Mean: {w_arr.mean():.4f}  Std: {w_arr.std():.4f}")
    print(f"    Negative weights: {neg_count}  "
          f"({'none — Lyapunov guarantee holds ✓' if neg_count == 0 else 'present — Lyapunov not guaranteed ✗'})")

    results["edge_weights"] = {
        "n_edges": len(weights),
        "min": float(w_arr.min()),
        "max": float(w_arr.max()),
        "mean": float(w_arr.mean()),
        "n_negative": neg_count,
        "lyapunov_guaranteed": neg_count == 0,
    }

    return results


# ===========================================================================
# TRACK 2: MODEL-GUIDED RETRIEVAL DYNAMICS
# ===========================================================================

def track2_retrieval_dynamics(spec: CartridgeSpec, rng: random.Random) -> Dict:
    """
    [MODEL-GUIDED RETRIEVAL DYNAMICS]

    Measures associative recall using greedy evidence selection.
    Convergence_score is an ADDITIVE COUNTER (not classical Hopfield relaxation):
      convergence_dimensions += mean(attractor_weights) * convergence_rate
      convergence_score = min(convergence_dimensions)
    Greedy policy: -induced_subgraph_energy (graph-energy based, no neural model).
    """
    _header("[MODEL-GUIDED RETRIEVAL DYNAMICS]")
    print("  NOTE: convergence_score = additive counter (sum of mean attractor weights * rate)")
    print("  This is NOT classical Hopfield energy relaxation.")
    print("  Greedy policy = -induced_subgraph_energy (pure graph scoring, no neural model)")

    non_inv = [t for t in spec.tokens if not t.is_invariant]
    threshold = spec.convergence_threshold
    max_turns = spec.max_turns

    results: Dict = {}

    # ------------------------------------------------------------------
    # 2A. Cue completion: k cues → greedy play → convergence_score
    # ------------------------------------------------------------------
    _subheader("2A. Cue completion (k tokens → greedy extension)")
    print(f"    Threshold: {threshold}  max_turns: {max_turns}")

    # Define "high-signal" tokens: top-20% by sum(attractor_weights)
    sorted_by_weight = sorted(non_inv, key=lambda t: float(t.attractor_weights.sum()), reverse=True)
    top_n = max(5, len(sorted_by_weight) // 5)
    high_signal = sorted_by_weight[:top_n]

    cue_results = {}
    n_tr = _n_trials(spec)
    for k in [1, 3, 5, 10]:
        if k > len(high_signal):
            continue
        scores = []
        for trial in range(n_tr):
            cue_tokens = rng.sample(high_signal, k)
            cue_ids = [t.id for t in cue_tokens]
            pool = [t for t in non_inv if t.id not in cue_ids]
            # Greedily fill to max_turns triads
            n_more = max_turns - (k // 3 + 1)
            triads, _ = _greedy_triads(spec, pool, cue_ids, n_more)
            # Build triad sequences for convergence sim
            cue_triad = cue_tokens[:3] if len(cue_tokens) >= 3 else cue_tokens + [pool[0]] * (3 - len(cue_tokens))
            all_triads = [cue_triad] + triads[:max_turns - 1]
            score = _convergence_score(spec, all_triads)
            scores.append(score)

        mean_score = np.mean(scores)
        success_rate = np.mean([s >= threshold for s in scores])
        print(f"    k={k:2d} cues: mean_score={mean_score:.4f}  "
              f"success_rate={success_rate * 100:.1f}%  (n={n_tr} trials)")
        cue_results[k] = {"mean_score": float(mean_score), "success_rate": float(success_rate)}

    results["cue_completion"] = cue_results

    # ------------------------------------------------------------------
    # 2B. Noise tolerance: inject wrong tokens, measure recovery
    # ------------------------------------------------------------------
    _subheader("2B. Noise tolerance (inject wrong tokens)")

    # Baseline: pure greedy from opening
    opening = list(spec.opening_token_ids)
    pool_base = [t for t in non_inv if t.id not in opening]
    greedy_triads, _ = _greedy_triads(spec, pool_base, opening, max_turns)
    baseline_score = _convergence_score(spec, greedy_triads)
    print(f"    Baseline greedy score (no noise): {baseline_score:.4f}")

    noise_results = {}
    n_tr_noise = _n_trials(spec, base=30)
    for noise_frac in [0.1, 0.2, 0.3, 0.5]:
        scores = []
        for trial in range(n_tr_noise):
            pool = [t for t in non_inv if t.id not in opening]
            rng.shuffle(pool)
            n_noise = int(len(pool) * noise_frac)
            # Move noise tokens to front so greedy starts from them
            noise_pool = pool[:n_noise]
            good_pool = pool[n_noise:]
            noisy_triads = _random_triads(noise_pool, n_noise // 3)
            # Then greedily extend
            noise_ids = [t.id for triple in noisy_triads for t in triple]
            remaining = [t for t in good_pool if t.id not in noise_ids]
            n_more = max_turns - len(noisy_triads)
            recovery_triads, _ = _greedy_triads(spec, remaining, opening + noise_ids, n_more)
            all_triads = noisy_triads + recovery_triads
            score = _convergence_score(spec, all_triads[:max_turns])
            scores.append(score)

        mean_score = np.mean(scores)
        degradation = baseline_score - mean_score
        print(f"    noise={int(noise_frac * 100):2d}%: mean_score={mean_score:.4f}  "
              f"degradation={degradation:.4f}  (n={n_tr_noise})")
        noise_results[noise_frac] = {
            "mean_score": float(mean_score),
            "degradation": float(degradation),
        }

    results["noise_tolerance"] = {
        "baseline_score": float(baseline_score),
        "by_noise_fraction": noise_results,
    }

    # ------------------------------------------------------------------
    # 2C. Basin proxy: convergence success rate vs pre-placed high-signal triads
    # ------------------------------------------------------------------
    _subheader("2C. Basin proxy (success rate vs pre-placed high-signal turns)")
    print("    'Correct' turns = triads built from top-20% attractor-weight tokens.")
    print("    We pre-place f * max_turns correct triads, then greedily fill the rest.")
    print("    Measures how much correct signal is needed to reach threshold.")

    basin_results = {}
    n_tr_basin = _n_trials(spec, base=30)
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        successes = 0
        n_pre = int(frac * max_turns)  # pre-placed correct triads

        for trial in range(n_tr_basin):
            # Build pre-placed triads from high_signal pool
            hs_shuffled = list(high_signal)
            rng.shuffle(hs_shuffled)
            pre_triads: List[List[Token]] = []
            pre_ids: List[str] = []
            for i in range(0, min(n_pre * 3, len(hs_shuffled) - 2), 3):
                triple = hs_shuffled[i:i + 3]
                if len(triple) == 3:
                    pre_triads.append(triple)
                    pre_ids.extend(t.id for t in triple)
                if len(pre_triads) >= n_pre:
                    break

            # Greedily fill remaining turns
            remain_pool = [t for t in non_inv if t.id not in pre_ids]
            n_more = max_turns - len(pre_triads)
            greedy_fill, _ = _greedy_triads(spec, remain_pool,
                                            list(opening) + pre_ids, n_more)
            all_triads = pre_triads + greedy_fill
            score = _convergence_score(spec, all_triads[:max_turns])
            if score >= threshold:
                successes += 1

        success_rate = successes / n_tr_basin
        print(f"    pre_placed={frac:.1f}×max_turns ({n_pre} turns): "
              f"success_rate={success_rate * 100:.1f}%  (n={n_tr_basin})")
        basin_results[frac] = float(success_rate)

    results["basin_proxy"] = basin_results

    # ------------------------------------------------------------------
    # 2D. Convergence score dissection
    # ------------------------------------------------------------------
    _subheader("2D. Convergence score dissection (additive counter analysis)")
    print("    Showing per-dimension accumulation for the greedy path.")

    state = CasebookState(
        convergence_dimensions=np.zeros(spec.n_attractor_dims, dtype=np.float32),
        convergence_rate=spec.convergence_rate,
    )
    pool_d = [t for t in non_inv if t.id not in opening]
    triads_d, _ = _greedy_triads(spec, pool_d, opening, min(10, max_turns))

    print(f"    convergence_rate={spec.convergence_rate}  threshold={threshold}")
    print(f"    {'Turn':>4}  {'Score':>7}  {'Dims':}")
    for i, triad in enumerate(triads_d):
        state.place_triad(triad, position=(0, i))
        dims_str = "  ".join(f"{d:.3f}" for d in state.convergence_dimensions)
        print(f"    {i + 1:>4}  {state.convergence_score:>7.4f}  [{dims_str}]")

    results["convergence_dissection"] = {
        "convergence_rate": spec.convergence_rate,
        "threshold": threshold,
        "n_attractor_dims": spec.n_attractor_dims,
        "final_score": float(state.convergence_score),
        "final_dims": [float(d) for d in state.convergence_dimensions],
    }

    return results


# ===========================================================================
# SUMMARY
# ===========================================================================

def print_summary(spec: CartridgeSpec, r1: Dict, r2: Dict) -> None:
    _header("SUMMARY")
    print(f"  Case: {spec.case_id}  ({spec.n_attractor_dims}-dim, {spec.vocab_size} tokens)")
    print()
    print("  [HOPFIELD-LIKE ENERGY GEOMETRY]")
    eg = r1.get("energy_distribution", {})
    gap = eg.get("gap_solution_minus_random")
    if gap is not None:
        status = "✓ solution-core lower energy" if gap < 0 else "✗ solution not lower energy"
        print(f"    Energy gap (solution-random): {gap:.4f}  {status}")

    em = r1.get("energy_monotonicity", {})
    mr = em.get("monotone_rate", 0)
    print(f"    Monotone rate: {mr * 100:.1f}%  "
          f"({'✓' if mr >= 0.99 else '✗' if mr < 0.90 else '~'})")

    ew = r1.get("edge_weights", {})
    print(f"    Negative edges: {ew.get('n_negative', '?')}  "
          f"({'Lyapunov guaranteed ✓' if ew.get('lyapunov_guaranteed') else 'not guaranteed ✗'})")

    iag = r1.get("invariant_energy_gap", {})
    pct = iag.get("percentile_above_random", 0)
    print(f"    Basin energy below {pct * 100:.0f}% of random sets  "
          f"({'✓' if pct > 0.6 else '~' if pct > 0.4 else '✗'})")

    print()
    print("  [MODEL-GUIDED RETRIEVAL DYNAMICS]")
    nt = r2.get("noise_tolerance", {})
    bs = nt.get("baseline_score", 0)
    print(f"    Greedy baseline score: {bs:.4f}  "
          f"({'✓ converges' if bs >= spec.convergence_threshold else '✗ below threshold'})")

    bp = r2.get("basin_proxy", {})
    if bp:
        high_frac_success = bp.get(1.0, 0)
        low_frac_success = bp.get(0.0, 0)
        print(f"    Basin proxy  all-high: {high_frac_success * 100:.1f}%  "
              f"all-low: {low_frac_success * 100:.1f}%")

    cd = r2.get("convergence_dissection", {})
    print(f"    Convergence score after 10 turns: {cd.get('final_score', '?'):.4f}  "
          f"(rate={cd.get('convergence_rate', '?')}, dims={cd.get('n_attractor_dims', '?')})")

    print()
    print("  ARCHITECTURE NOTE")
    print("    subgraph_energy  = Σ -w_ij  (Hopfield Lyapunov, monotone by construction)")
    print("    convergence_score = min(Σ mean(w_attractor) * rate)  (additive counter)")
    print("    These measure DIFFERENT things.  Track 1 confirms graph structure is")
    print("    Hopfield-like.  Track 2 confirms additive counter reaches threshold.")


# ===========================================================================
# Optional plots
# ===========================================================================

def save_plots(spec: CartridgeSpec, r1: Dict, r2: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Attractor Analysis: {spec.case_id}", fontsize=13)

    # Plot 1: Energy distribution (conceptual — show gap)
    ax = axes[0]
    eg = r1.get("energy_distribution", {})
    labels = ["random", "partial", "solution"]
    values = [eg.get("random_mean"), eg.get("partial_mean"), eg.get("solution_mean")]
    colors = ["#aaaaaa", "#f0a040", "#4080f0"]
    bars = [v for v in values if v is not None and not np.isnan(v)]
    bar_labels = [l for l, v in zip(labels, values) if v is not None and not np.isnan(v)]
    bar_colors = [c for c, v in zip(colors, values) if v is not None and not np.isnan(v)]
    ax.bar(bar_labels, bars, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_title("[Hopfield] Mean subgraph_energy\nby token set type")
    ax.set_ylabel("energy (lower = more stable)")
    ax.axhline(0, color="black", linewidth=0.5)

    # Plot 2: Basin proxy curve
    ax = axes[1]
    bp = r2.get("basin_proxy", {})
    if bp:
        fracs = sorted(bp.keys())
        rates = [bp[f] for f in fracs]
        ax.plot(fracs, rates, marker="o", color="#4080f0")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("fraction of high-signal tokens")
        ax.set_ylabel("convergence success rate")
        ax.set_title("[Retrieval] Basin proxy curve\n(additive counter success)")

    # Plot 3: Cue completion curve
    ax = axes[2]
    cc = r2.get("cue_completion", {})
    if cc:
        ks = sorted(cc.keys())
        scores = [cc[k]["mean_score"] for k in ks]
        ax.plot(ks, scores, marker="o", color="#40a080")
        ax.axhline(spec.convergence_threshold, color="red", linestyle="--", linewidth=0.8,
                   label=f"threshold={spec.convergence_threshold}")
        ax.set_xlabel("number of high-signal cue tokens")
        ax.set_ylabel("mean convergence_score")
        ax.set_title("[Retrieval] Cue completion\n(k cues → greedy play)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"attractor_analysis_{spec.case_id}.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n  Plots saved to: {out_path}")
    plt.close()


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-track Thornfield attractor analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "case_dir",
        help="Path to packed case directory (must contain spec.json)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save matplotlib plots to research/plots/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    spec_path = case_dir / "spec.json"
    if not spec_path.exists():
        print(f"ERROR: spec.json not found in {case_dir}")
        print("Run: python3 thornfield/trainer/tools/pack_case.py <case.json>")
        sys.exit(1)

    print(f"Loading case from: {case_dir}")
    spec = CartridgeSpec.load(str(spec_path))
    print(f"  {spec.case_id}: {spec.vocab_size} tokens, {len(spec.token_graph.edges)} edges, "
          f"{spec.n_attractor_dims} dims")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    r1 = track1_energy_geometry(spec, rng)
    r2 = track2_retrieval_dynamics(spec, rng)
    print_summary(spec, r1, r2)

    if args.plots:
        plots_dir = _REPO_ROOT / "research" / "plots"
        save_plots(spec, r1, r2, plots_dir)


if __name__ == "__main__":
    main()
