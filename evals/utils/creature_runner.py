"""
Run oscillating creature sessions programmatically for evaluation and balancing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent / "living_tales" / "trainer"))

from core.cartridge import CartridgeSpec
from core.casebook import CasebookState
from core.creature_case import classify_creature_token_role
from core.token import Token, TokenAgency, TokenStream
try:
    from evals.datasets._schema import DialogueTurn, GameResult
except ImportError:  # pragma: no cover
    from datasets._schema import DialogueTurn, GameResult


class CreatureGameRunner:
    """Heuristic runner for oscillating creature cartridges."""

    def __init__(self, spec: CartridgeSpec):
        self.spec = spec
        self.player_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
            and not t.is_invariant and t.stream != TokenStream.OPENING
        ]
        self.engine_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
            and not t.is_invariant and t.stream != TokenStream.OPENING
        ]

    def _score_candidate(
        self,
        token: Token,
        current_score: float,
        min_dim: float,
        min_dim_idx: int,
        recent_roles: List[str],
        rng: np.random.RandomState,
    ) -> float:
        role = classify_creature_token_role(token.id)
        w = token.attractor_weights
        score = rng.uniform(0.9, 1.1)

        # Atmospheric filler: heavily penalize to reduce dead turns
        if role in ("context", "state", "stage"):
            score *= 0.15
        # Objects/companions/events: moderate — they set scenes but don't drive arcs
        if role in ("object", "companion", "event", "trait", "memory"):
            score *= 0.4
        # Moods are meaningful state — moderate weight
        if role == "mood":
            score *= 0.7

        # === DISTRESS-DRIVEN SELECTION ===
        # Use min_dim as the real distress signal
        if min_dim < 0.25:
            # Creature is suffering — URGENT care needed
            if role in ("action", "recovery"):
                score *= 4.0
            elif role == "need":
                score *= 2.5  # needs communicate the problem
        elif min_dim < 0.4:
            # Creature is uncomfortable — care helps
            if role in ("action", "recovery"):
                score *= 2.5
            elif role == "need":
                score *= 1.8

        # === DECAY PRESSURE ===
        # When creature is healthy, push decay HARD
        if min_dim > 0.55:
            if role in ("decay", "combo"):
                score *= 4.0
            elif role == "decline":
                score *= 2.5
        elif min_dim > 0.45:
            if role in ("decay", "decline"):
                score *= 2.0

        # === ARC COMPLETION ===
        # After decay/decline/need, favor action/recovery (close the arc)
        if recent_roles:
            last = recent_roles[-1]
            if last in ("decay", "decline", "need") and role in ("action", "recovery"):
                score *= 3.0
            # After action, favor mood/recovery (show the result)
            if last == "action" and role in ("mood", "recovery"):
                score *= 2.0
            # After recovery, DON'T chain more recovery
            if last == "recovery" and role == "recovery":
                score *= 0.2
            # Penalize repeating same role
            if last == role:
                score *= 0.4
        # Triple repeat = almost impossible
        if len(recent_roles) >= 2 and recent_roles[-1] == recent_roles[-2] == role:
            score *= 0.1

        # === DIMENSION TARGETING ===
        # Boost tokens that address the weakest dimension
        if min_dim_idx < len(w):
            dim_contribution = w[min_dim_idx]
            if dim_contribution > 0.03 and min_dim < 0.4:
                # This token helps the weakest dimension — boost it
                score *= 2.0
            elif dim_contribution < -0.03 and min_dim > 0.5:
                # This token decays a strong dimension (not the weakest) — interesting
                score *= 1.5

        return score

    def _pick_token(
        self,
        pool: List[Token],
        used_ids: set[str],
        turn_index: int,
        current_score: float,
        min_dim: float,
        min_dim_idx: int,
        recent_roles: List[str],
        rng: np.random.RandomState,
    ) -> Optional[Token]:
        candidates = [
            token for token in pool
            if token.id not in used_ids and token.is_available_at_turn(turn_index)
        ]
        if not candidates:
            return None
        weights = np.array([
            self._score_candidate(token, current_score, min_dim, min_dim_idx, recent_roles, rng)
            for token in candidates
        ], dtype=np.float64)
        weights = np.maximum(weights, 1e-6)
        weights /= weights.sum()
        idx = rng.choice(len(candidates), p=weights)
        return candidates[idx]

    def run_game(self, seed: int, max_turns: Optional[int] = None) -> GameResult:
        rng = np.random.RandomState(seed)
        state = CasebookState.create(
            n_dims=self.spec.n_attractor_dims,
            mode=self.spec.mode,
            convergence_rate=self.spec.convergence_rate,
        )
        used_ids: set[str] = set()
        context_ids: list[str] = []
        turns: list[DialogueTurn] = []
        recent_roles: list[str] = []
        limit = max_turns or self.spec.max_turns

        for tid in self.spec.opening_token_ids:
            token = self.spec.get_token(tid)
            state.apply_delta(token.attractor_weights * self.spec.convergence_rate)
            used_ids.add(token.id)
            context_ids.append(token.id)
            role = classify_creature_token_role(token.id)
            turns.append(DialogueTurn(
                token_id=token.id,
                role="engine",
                token_class=token.token_class.value,
                phase=token.phase.value,
                energy_at_step=self.spec.token_graph.subgraph_energy(context_ids),
                convergence_at_step=state.convergence_score,
                token_role=role,
                dimension_snapshot=state.convergence_dimensions.tolist(),
            ))
            recent_roles.append(role)

        is_player_turn = True
        for turn_index in range(len(turns), limit):
            pool = self.player_tokens if is_player_turn else self.engine_tokens
            dims = state.convergence_dimensions
            min_dim = float(dims.min()) if dims.size > 0 else 0.35
            min_dim_idx = int(dims.argmin()) if dims.size > 0 else 0
            token = self._pick_token(
                pool,
                used_ids,
                turn_index=turn_index,
                current_score=state.convergence_score,
                min_dim=min_dim,
                min_dim_idx=min_dim_idx,
                recent_roles=recent_roles[-3:],
                rng=rng,
            )
            if token is None:
                other_pool = self.engine_tokens if is_player_turn else self.player_tokens
                token = self._pick_token(
                    other_pool,
                    used_ids,
                    turn_index=turn_index,
                    current_score=state.convergence_score,
                    min_dim=min_dim,
                    min_dim_idx=min_dim_idx,
                    recent_roles=recent_roles[-3:],
                    rng=rng,
                )
            if token is None:
                break

            state.apply_delta(token.attractor_weights * self.spec.convergence_rate)
            used_ids.add(token.id)
            context_ids.append(token.id)
            role = classify_creature_token_role(token.id)
            turns.append(DialogueTurn(
                token_id=token.id,
                role="player" if is_player_turn else "engine",
                token_class=token.token_class.value,
                phase=token.phase.value,
                energy_at_step=self.spec.token_graph.subgraph_energy(context_ids),
                convergence_at_step=state.convergence_score,
                token_role=role,
                dimension_snapshot=state.convergence_dimensions.tolist(),
            ))
            recent_roles.append(role)
            is_player_turn = not is_player_turn

        final_score = state.convergence_score
        return GameResult(
            seed=seed,
            turns=turns,
            converged=final_score >= self.spec.convergence_threshold,
            final_convergence=final_score,
            final_token_ids=context_ids,
            total_energy=self.spec.token_graph.subgraph_energy(context_ids) if context_ids else 0.0,
            mode=self.spec.mode,
        )

    def run_batch(self, n_games: int, seeds: Optional[List[int]] = None) -> List[GameResult]:
        if seeds is None:
            seeds = list(range(n_games))
        return [self.run_game(seed) for seed in seeds]
