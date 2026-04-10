"""
Configurable reward system for dialogue REINFORCE training.

Replaces the hard-coded inline reward in train_dialogue.py with a
dataclass-based config and three new reward signals: responsiveness,
pacing, and arc shaping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from core.hopfield import TokenGraph
from core.token import Token
from core.creature_case import classify_creature_token_role


@dataclass
class DialogueRewardConfig:
    energy_weight: float = 1.0
    chronology_weight: float = 0.2
    diversity_weight: float = 0.1
    speed_penalty_weight: float = 0.3
    responsiveness_weight: float = 0.15
    pacing_weight: float = 0.1
    arc_shape: str = "linear"       # "linear" | "late_break" | "early_burst"
    arc_weight: float = 0.2
    unresolved_need_penalty_weight: float = 0.0
    repeat_role_penalty_weight: float = 0.0
    recovery_bonus_weight: float = 0.0


class DialogueRewardComputer:
    """Compute per-turn reward for dialogue RL episodes."""

    def __init__(
        self,
        config: DialogueRewardConfig,
        graph: TokenGraph,
        max_turns: int,
        convergence_threshold: float = 0.75,
    ):
        self.config = config
        self.graph = graph
        self.max_turns = max_turns
        self.convergence_threshold = convergence_threshold

    def compute_turn_reward(
        self,
        context_ids_before: List[str],
        context_ids_after: List[str],
        token: Token,
        turn: int,
        prev_affinity_tags: set,
        recent_player_tokens: List[Token],
        signal_history: List[float],
        convergence_at_step: float,
    ) -> float:
        c = self.config
        reward = 0.0

        reward += c.energy_weight * self._energy_reward(context_ids_before, context_ids_after)
        reward += c.chronology_weight * self._chronology_bonus(token, turn)
        reward += c.diversity_weight * self._diversity_bonus(token, prev_affinity_tags)
        reward -= c.speed_penalty_weight * self._speed_penalty(turn)
        reward += c.responsiveness_weight * self._responsiveness(token, recent_player_tokens)
        reward += c.pacing_weight * self._pacing(token, signal_history)
        reward += c.arc_weight * self._arc_shaping(convergence_at_step, turn)
        reward += c.unresolved_need_penalty_weight * self._creature_need_pressure(context_ids_after)
        reward += c.repeat_role_penalty_weight * self._repeat_role_penalty(context_ids_after)
        reward += c.recovery_bonus_weight * self._recovery_bonus(context_ids_before, token.id)

        return reward

    def _energy_reward(self, ids_before: List[str], ids_after: List[str]) -> float:
        e_before = self.graph.subgraph_energy(ids_before) if ids_before else 0.0
        e_after = self.graph.subgraph_energy(ids_after)
        return e_before - e_after

    def _chronology_bonus(self, token: Token, turn: int) -> float:
        game_turn = turn // 2
        return 1.0 if token.is_available_at_turn(game_turn) else 0.0

    def _diversity_bonus(self, token: Token, prev_tags: set) -> float:
        new_tags = set(token.affinity_tags) - prev_tags
        return min(1.0, len(new_tags) / 3.0)

    def _speed_penalty(self, turn: int) -> float:
        return math.exp(-turn / max(self.max_turns, 1))

    def _responsiveness(self, token: Token, recent_player_tokens: List[Token]) -> float:
        """Reward when engine token's tags overlap with recent player tokens."""
        if not recent_player_tokens or not token.affinity_tags:
            return 0.0
        player_tags = set()
        for pt in recent_player_tokens:
            player_tags.update(pt.affinity_tags)
        if not player_tags:
            return 0.0
        overlap = set(token.affinity_tags) & player_tags
        union = set(token.affinity_tags) | player_tags
        return len(overlap) / len(union) if union else 0.0

    def _pacing(self, token: Token, signal_history: List[float]) -> float:
        """Reward even spacing of strong-signal tokens. Penalize clustering."""
        signal = float(np.linalg.norm(token.attractor_weights))
        history = signal_history + [signal]
        if len(history) < 3:
            return 0.0
        # Compute gaps between consecutive signals
        arr = np.array(history[-10:])  # last 10 turns
        std = float(np.std(arr))
        # Higher std = more even distribution = better pacing
        # (counter-intuitive but: we want variation, not all-high or all-low clusters)
        return min(1.0, std)

    def _arc_shaping(self, convergence: float, turn: int) -> float:
        """Reward matching a target convergence curve shape."""
        progress = turn / max(self.max_turns, 1)
        target = self._arc_target(progress)
        # Reward = negative absolute deviation from target
        return -abs(convergence - target)

    def _arc_target(self, progress: float) -> float:
        """Target convergence value at a given progress fraction."""
        shape = self.config.arc_shape
        if shape == "late_break":
            if progress < 0.7:
                return progress * 0.3  # slow buildup
            else:
                return 0.3 + (progress - 0.7) / 0.3 * 0.7  # rapid convergence
        elif shape == "early_burst":
            if progress < 0.3:
                return progress / 0.3 * 0.5  # fast to 50%
            else:
                return 0.5 + (progress - 0.3) / 0.7 * 0.5  # slow climb
        else:  # "linear"
            return progress * self.convergence_threshold

    def _creature_need_pressure(self, context_ids_after: List[str]) -> float:
        recent_roles = [
            classify_creature_token_role(token_id)
            for token_id in context_ids_after[-6:]
        ]
        pending = sum(1 for role in recent_roles if role in {"decay", "decline", "need", "mood", "combo"})
        return -float(pending)

    def _repeat_role_penalty(self, context_ids_after: List[str]) -> float:
        recent_roles = [
            classify_creature_token_role(token_id)
            for token_id in context_ids_after[-4:]
        ]
        if len(recent_roles) < 2:
            return 0.0
        last_role = recent_roles[-1]
        repeats = sum(1 for role in recent_roles[:-1] if role == last_role)
        return -float(repeats)

    def _recovery_bonus(self, context_ids_before: List[str], token_id: str) -> float:
        if classify_creature_token_role(token_id) != "recovery":
            return 0.0
        prior_roles = [
            classify_creature_token_role(prev_id)
            for prev_id in context_ids_before[-4:]
        ]
        if any(role in {"decay", "decline", "need", "mood", "combo"} for role in prior_roles):
            return 1.0
        return 0.0
