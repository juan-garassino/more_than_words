"""
Dialogue trajectory sampler for Thornfield mystery cases.

Generates interleaved (player_token, engine_token, player_token, ...)
sequences using the Hopfield graph as scoring oracle. Respects chronological
phase ordering (EARLY -> MID -> LATE) and always starts from the story origin.

Each dialogue also produces per-step Hopfield soft targets for knowledge
distillation into the dialogue transformer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.cartridge import CartridgeSpec
from core.token import Token, TokenAgency, TokenPhase, TokenStream


ROLE_PLAYER = "player"
ROLE_ENGINE = "engine"


@dataclass
class DialogueTurn:
    token: Token
    role: str  # ROLE_PLAYER or ROLE_ENGINE
    turn_index: int
    energy_at_step: float
    convergence_at_step: float


@dataclass
class DialoguePath:
    turns: List[DialogueTurn]
    soft_targets: List[np.ndarray]  # per-turn: shape (vocab_size,) softmax over attractor weights
    converged: bool
    final_convergence: float


class DialogueSampler:
    """
    Monte Carlo sampler for interleaved player/engine dialogue trajectories.
    """

    def __init__(
        self,
        spec: CartridgeSpec,
        player_temperature: float = 1.2,
        engine_temperature: float = 0.8,
        max_turns: int | None = None,
        min_turns: int | None = None,
        convergence_rate: float | None = None,
        soft_target_temperature: float = 2.0,
        allow_partial: bool = True,
    ):
        self.spec = spec
        self.graph = spec.token_graph
        self.player_temperature = player_temperature
        self.engine_temperature = engine_temperature
        self.max_turns = max_turns if max_turns is not None else spec.max_turns * 2
        self.min_turns = min_turns if min_turns is not None else spec.min_turns
        self.convergence_rate = convergence_rate if convergence_rate is not None else spec.convergence_rate
        self.soft_target_temperature = soft_target_temperature
        self.allow_partial = allow_partial

        # Partition tokens by agency
        self._player_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        ]
        self._engine_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        ]

        # Build attractor weight matrix for soft targets: shape (vocab_size, n_dims)
        self._attractor_matrix = np.stack(
            [t.attractor_weights for t in spec.tokens], axis=0
        )  # (V, D)

    def _game_turn(self, dialogue_position: int) -> int:
        """Map dialogue position to game turn for phase gating."""
        return dialogue_position // 2

    def _phase_valid(self, token: Token, dialogue_position: int) -> bool:
        """Check if token's phase is valid at current dialogue position."""
        game_turn = self._game_turn(dialogue_position)
        return token.is_available_at_turn(game_turn)

    def _get_candidates(
        self, pool: List[Token], placed_ids: set, dialogue_position: int
    ) -> List[Token]:
        """Get phase-valid, unplayed tokens from the given pool."""
        return [
            t for t in pool
            if t.id not in placed_ids and self._phase_valid(t, dialogue_position)
        ]

    def _score_token(self, token: Token, context_ids: List[str]) -> float:
        """Energy-based scoring: lower energy + higher narrative gradient = better."""
        energy = self.graph.induced_subgraph_energy([token.id], context_ids)
        return -energy + token.narrative_gradient * 0.2

    def _sample_from_pool(
        self,
        candidates: List[Token],
        context_ids: List[str],
        temperature: float,
    ) -> Optional[Token]:
        """Softmax sample a token from scored candidates."""
        if not candidates:
            return None

        scores = np.array([self._score_token(t, context_ids) for t in candidates])
        scores = scores / max(temperature, 1e-8)
        scores -= scores.max()
        weights = np.exp(scores)
        total = weights.sum()
        if total < 1e-12:
            return candidates[np.random.randint(len(candidates))]
        weights /= total

        idx = np.random.choice(len(candidates), p=weights)
        return candidates[idx]

    def _compute_soft_target(self) -> np.ndarray:
        """
        Compute vocabulary-wide soft target from Hopfield attractor weights.
        Averages softmax across attractor dimensions.
        """
        T = self.soft_target_temperature
        # For each dimension, softmax over vocab
        # shape: (V, D) -> per-dim softmax -> average -> (V,)
        logits = self._attractor_matrix / max(T, 1e-8)  # (V, D)
        # Softmax per dimension (column-wise)
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        soft_per_dim = exp_logits / exp_logits.sum(axis=0, keepdims=True)  # (V, D)
        # Average across dimensions for a single vocab-wide distribution
        soft_target = soft_per_dim.mean(axis=1)  # (V,)
        # Re-normalize
        soft_target /= soft_target.sum()
        return soft_target.astype(np.float32)

    def sample_dialogue(self) -> Optional[DialoguePath]:
        """
        Sample one interleaved dialogue trajectory.

        Always starts from opening tokens (the story origin).
        Alternates player/engine turns with phase-gated token selection.
        Returns None if the dialogue cannot produce any valid moves.
        """
        convergence_dims = np.zeros(self.spec.n_attractor_dims, dtype=np.float32)
        placed_ids: set = set()
        context_ids: List[str] = []
        turns: List[DialogueTurn] = []
        soft_targets: List[np.ndarray] = []
        soft_target = self._compute_soft_target()

        # --- Opening: place scene-setting tokens ---
        for tid in self.spec.opening_token_ids:
            token = self.spec.get_token(tid)
            placed_ids.add(token.id)
            context_ids.append(token.id)
            energy = self.graph.subgraph_energy(context_ids)
            convergence_dims = np.minimum(
                1.0, convergence_dims + token.attractor_weights * self.convergence_rate
            )
            turns.append(DialogueTurn(
                token=token,
                role=ROLE_ENGINE,  # opening is engine-placed
                turn_index=len(turns),
                energy_at_step=energy,
                convergence_at_step=float(convergence_dims.min()),
            ))
            soft_targets.append(soft_target.copy())

        # --- Alternating dialogue ---
        dialogue_pos = len(turns)
        is_player_turn = True

        for _ in range(self.max_turns - len(turns)):
            if is_player_turn:
                pool = self._player_tokens
                temp = self.player_temperature
            else:
                pool = self._engine_tokens
                temp = self.engine_temperature

            candidates = self._get_candidates(pool, placed_ids, dialogue_pos)

            # Fallback: try the other pool if this one is empty
            if not candidates:
                other_pool = self._engine_tokens if is_player_turn else self._player_tokens
                candidates = self._get_candidates(other_pool, placed_ids, dialogue_pos)

            if not candidates:
                break

            chosen = self._sample_from_pool(candidates, context_ids, temp)
            if chosen is None:
                break

            placed_ids.add(chosen.id)
            context_ids.append(chosen.id)
            energy = self.graph.subgraph_energy(context_ids)

            convergence_dims = np.minimum(
                1.0, convergence_dims + chosen.attractor_weights * self.convergence_rate
            )
            conv_score = float(convergence_dims.min())

            turns.append(DialogueTurn(
                token=chosen,
                role=ROLE_PLAYER if is_player_turn else ROLE_ENGINE,
                turn_index=len(turns),
                energy_at_step=energy,
                convergence_at_step=conv_score,
            ))
            soft_targets.append(soft_target.copy())

            dialogue_pos += 1
            is_player_turn = not is_player_turn

            # Check convergence
            game_turn = self._game_turn(dialogue_pos)
            if conv_score >= self.spec.convergence_threshold and game_turn >= self.min_turns:
                return DialoguePath(
                    turns=turns,
                    soft_targets=soft_targets,
                    converged=True,
                    final_convergence=conv_score,
                )

        if not turns:
            return None

        final_conv = float(convergence_dims.min())
        if self.allow_partial or final_conv >= self.spec.convergence_threshold:
            return DialoguePath(
                turns=turns,
                soft_targets=soft_targets,
                converged=final_conv >= self.spec.convergence_threshold,
                final_convergence=final_conv,
            )
        return None

    def sample_batch(
        self,
        n: int,
        verbose: bool = True,
        max_attempts: int | None = None,
    ) -> List[DialoguePath]:
        """Sample n dialogue trajectories."""
        paths: List[DialoguePath] = []
        attempts = 0
        cap = max_attempts if max_attempts is not None else n * 6

        while len(paths) < n and attempts < cap:
            path = self.sample_dialogue()
            if path is not None:
                paths.append(path)
            attempts += 1

            if verbose and len(paths) % 50 == 0 and len(paths) > 0:
                rate = len(paths) / attempts
                converged = sum(1 for p in paths if p.converged)
                print(
                    f"  {len(paths)}/{n} dialogues | "
                    f"success: {rate:.1%} | converged: {converged}/{len(paths)}",
                    flush=True,
                )

        if verbose:
            converged = sum(1 for p in paths if p.converged)
            print(
                f"  Complete: {len(paths)} dialogues from {attempts} attempts "
                f"({converged} converged)",
                flush=True,
            )
        return paths
