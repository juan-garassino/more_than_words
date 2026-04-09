from __future__ import annotations

import math


def compute_reward(
    graph,
    ids_before: list[str],
    ids_after: list[str],
    turn: int,
    max_turns: int,
    new_affinity_tags: set[str],
    prev_affinity_tags: set[str],
) -> float:
    """
    Compute per-step reward for the RL policy.

    energy_reward  = drop in total system energy (positive when well-connected
                     tokens are added, since subgraph_energy is negative for
                     positive-weight edges and becomes more negative over time)
    speed_penalty  = exp(-turn/max_turns): fades from 1.0 → ~0.4 by turn 12,
                     penalising triads placed very early (discourages rushing)
    diversity_bonus= fraction of new affinity tags introduced (encourages
                     broad exploration of the graph)
    """
    energy_reward = graph.subgraph_energy(ids_before) - graph.subgraph_energy(ids_after)
    speed_penalty = math.exp(-turn / max_turns)
    new_tags = new_affinity_tags - prev_affinity_tags
    diversity_bonus = min(1.0, len(new_tags) / 3.0)
    return energy_reward - 0.5 * speed_penalty + 0.1 * diversity_bonus
