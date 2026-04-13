"""
Dialogue trajectory sampler for Living Tales mystery cases.

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
from core.token import Token, TokenAgency, TokenClass, TokenPhase, TokenStream


ROLE_PLAYER = "player"
ROLE_ENGINE = "engine"

STRATEGIES = ("energy", "random", "red_herring_first", "location_first", "suspect_first", "object_first",
              "follow_suspect_0", "follow_suspect_1", "follow_suspect_2")

_CLASS_BIAS_MAP = {
    "location_first": TokenClass.LOCATION,
    "suspect_first": TokenClass.SUSPECT,
    "object_first": TokenClass.OBJECT,
}


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


@dataclass
class SceneTurn:
    """One engine scene: N tokens predicted in parallel (one per dimension)."""
    tokens: List[Token]         # N tokens, one per head/dimension
    dim_indices: List[int]      # which dimension each token belongs to
    role: str                   # always ROLE_ENGINE for scene turns
    turn_index: int
    energy_at_step: float
    convergence_at_step: float


@dataclass
class SceneDialoguePath:
    """A dialogue where engine turns produce N-token scenes."""
    player_turns: List[DialogueTurn]    # single-token player turns
    scene_turns: List[SceneTurn]        # multi-token engine scenes
    all_tokens: List[Token]             # flat sequence of all tokens in order
    all_roles: List[str]                # role for each token in flat sequence
    soft_targets: List[np.ndarray]      # per-position in flat sequence
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
        strategy: str = "energy",
        temperature_jitter: float = 0.0,
    ):
        self.spec = spec
        self.graph = spec.token_graph
        self.player_temperature = player_temperature
        self.engine_temperature = engine_temperature
        self._strategy = strategy
        self._temperature_jitter = temperature_jitter
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

    def _score_token(
        self, token: Token, context_ids: List[str], dialogue_pos: int,
    ) -> float:
        """Score a token based on the current strategy."""
        strategy = self._strategy

        if strategy == "random":
            return 0.0  # uniform random

        energy = self.graph.induced_subgraph_energy([token.id], context_ids)
        base_score = -energy + token.narrative_gradient * 0.2

        if strategy == "red_herring_first":
            progress = dialogue_pos / max(self.max_turns, 1)
            if progress < 0.6:
                # Invert: prefer low-signal tokens early
                return -base_score
            return base_score

        if strategy in _CLASS_BIAS_MAP:
            target_class = _CLASS_BIAS_MAP[strategy]
            progress = dialogue_pos / max(self.max_turns, 1)
            if progress < 0.4 and token.token_class == target_class:
                base_score += 2.0
            return base_score

        # Default: "energy" strategy
        return base_score

    def _sample_from_pool(
        self,
        candidates: List[Token],
        context_ids: List[str],
        temperature: float,
        dialogue_pos: int = 0,
        role_counts: Optional[Dict[str, int]] = None,
    ) -> Optional[Token]:
        """Softmax sample a token from scored candidates with role balancing."""
        if not candidates:
            return None

        # Apply temperature jitter
        temp = temperature
        if self._temperature_jitter > 0:
            temp += np.random.uniform(-self._temperature_jitter, self._temperature_jitter)
            temp = max(temp, 0.1)

        if self._strategy == "random":
            return candidates[np.random.randint(len(candidates))]

        scores = np.array([
            self._score_token(t, context_ids, dialogue_pos) for t in candidates
        ])

        # --- Role balancing: even distribution, with boost for creature-reactive roles ---
        if role_counts is not None:
            total_placed = sum(role_counts.values()) or 1
            # Creature-reactive roles should be at least as frequent as atmospheric ones
            reactive_roles = {'mood', 'need', 'decay', 'decline', 'recovery', 'combo', 'mischief'}

            for i, candidate in enumerate(candidates):
                role = candidate.id.split(':')[0]
                role_freq = role_counts.get(role, 0) / total_placed

                if role in reactive_roles:
                    # Reactive tokens: boost if underrepresented (target ~8% each)
                    if role_freq < 0.06:
                        scores[i] *= 3.0
                    elif role_freq < 0.10:
                        scores[i] *= 1.5
                else:
                    # Atmospheric tokens: penalize if overrepresented (target ~8% each)
                    if role_freq > 0.15:
                        scores[i] *= 0.2
                    elif role_freq > 0.10:
                        scores[i] *= 0.5

        # Suspect-branch strategies: boost candidates connected to a specific suspect
        if self._strategy.startswith("follow_suspect_"):
            suspect_idx = int(self._strategy.split("_")[-1])
            suspects = [t for t in self.spec.tokens if t.token_class == TokenClass.SUSPECT and not t.is_invariant]
            if suspect_idx < len(suspects):
                target_id = suspects[suspect_idx].id
                # Boost candidates connected to this suspect in the graph
                for i, candidate in enumerate(candidates):
                    affinity = self.graph.weight(candidate.id, target_id)
                    scores[i] += affinity * 4.0

        scores = scores / max(temp, 1e-8)
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
        Used as fallback when no context is available.
        """
        T = self.soft_target_temperature
        logits = self._attractor_matrix / max(T, 1e-8)
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        soft_per_dim = exp_logits / exp_logits.sum(axis=0, keepdims=True)
        soft_target = soft_per_dim.mean(axis=1)
        soft_target /= soft_target.sum()
        return soft_target.astype(np.float32)

    def _compute_soft_target_conditional(self, last_token_id: str, context_ids: List[str]) -> np.ndarray:
        """
        Compute soft target conditioned on graph edges to the last token and recent context.

        After toss_ball, tokens connected to toss_ball by strong edges (like eager_bounce)
        get high probability. This teaches the transformer to follow graph structure.
        """
        T = max(self.soft_target_temperature, 1e-8)
        scores = np.zeros(len(self.spec.tokens), dtype=np.float32)

        # Recent context for broader affinity (last 3 tokens)
        recent = context_ids[-3:] if len(context_ids) >= 3 else context_ids

        for i, tok in enumerate(self.spec.tokens):
            # Direct edge weight to the last token placed (strongest signal)
            edge_to_last = self.graph.weight(tok.id, last_token_id)

            # Average edge weight to recent context (weaker background signal)
            context_affinity = 0.0
            if recent:
                context_affinity = sum(self.graph.weight(tok.id, cid) for cid in recent) / len(recent)

            # Attractor weight norm as base prior (very weak)
            attractor_norm = float(np.linalg.norm(tok.attractor_weights))

            # Combine: edges dominate, attractor is a tiebreaker
            scores[i] = edge_to_last * 5.0 + context_affinity * 2.0 + attractor_norm * 0.3

        # Softmax with temperature
        scores = scores / T
        scores -= scores.max()
        probs = np.exp(scores)
        total = probs.sum()
        if total < 1e-12:
            # Fallback to uniform
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= total

        return probs.astype(np.float32)

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
        fallback_soft_target = self._compute_soft_target()

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
            # Opening tokens use fallback (no last-action context yet)
            soft_targets.append(fallback_soft_target.copy())

        # --- Track role distribution for balanced sampling ---
        role_counts: Dict[str, int] = {}
        for t in turns:
            role = t.token.id.split(':')[0]
            role_counts[role] = role_counts.get(role, 0) + 1

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

            chosen = self._sample_from_pool(candidates, context_ids, temp, dialogue_pos, role_counts)
            if chosen is None:
                break

            placed_ids.add(chosen.id)
            context_ids.append(chosen.id)
            # Update role counts for balancing
            chosen_role = chosen.id.split(':')[0]
            role_counts[chosen_role] = role_counts.get(chosen_role, 0) + 1
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
            # Context-conditioned soft target: what should come AFTER this token?
            soft_targets.append(
                self._compute_soft_target_conditional(chosen.id, context_ids)
            )

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

    def _build_dim_pools(self, threshold: float = 0.05) -> List[List[Token]]:
        """Build per-dimension token pools from attractor weights."""
        n_dims = self.spec.n_attractor_dims
        pools: List[List[Token]] = [[] for _ in range(n_dims)]
        for tok in self._engine_tokens:
            for d in range(n_dims):
                if abs(tok.attractor_weights[d]) > threshold:
                    pools[d].append(tok)
        # Ensure every dim has candidates
        for d in range(n_dims):
            if len(pools[d]) < 3:
                sorted_by_dim = sorted(
                    self._engine_tokens,
                    key=lambda t: abs(t.attractor_weights[d]),
                    reverse=True,
                )
                pools[d] = sorted_by_dim[:10]
        return pools

    def sample_scene_dialogue(self) -> Optional[SceneDialoguePath]:
        """
        Sample a dialogue where engine turns produce N-token scenes.

        Player plays 1 token → engine responds with N tokens (one per dimension).
        The flat sequence is: [opening..., player, eng_d0, eng_d1, ..., eng_dN, player, ...]
        """
        n_dims = self.spec.n_attractor_dims
        dim_pools = self._build_dim_pools()
        convergence_dims = np.zeros(n_dims, dtype=np.float32)
        placed_ids: set = set()
        context_ids: List[str] = []
        all_tokens: List[Token] = []
        all_roles: List[str] = []
        soft_targets: List[np.ndarray] = []
        player_turns: List[DialogueTurn] = []
        scene_turns: List[SceneTurn] = []
        fallback_soft = self._compute_soft_target()

        # --- Opening ---
        for tid in self.spec.opening_token_ids:
            token = self.spec.get_token(tid)
            placed_ids.add(token.id)
            context_ids.append(token.id)
            convergence_dims = np.minimum(
                1.0, convergence_dims + token.attractor_weights * self.convergence_rate
            )
            all_tokens.append(token)
            all_roles.append(ROLE_ENGINE)
            soft_targets.append(fallback_soft.copy())

        dialogue_pos = len(all_tokens)
        role_counts: Dict[str, int] = {}

        # Token recycling interval (reset placed_ids to allow replaying tokens)
        recycle_interval = 15
        # After this many steps, relax phase gating (all tokens available)
        relax_after = (self.min_turns // 2) * 2 if self.min_turns > 10 else 999

        for step in range(self.max_turns):
            game_turn = dialogue_pos // 2

            # Recycle: reset placed_ids periodically (keep opening + invariant)
            if step > 0 and step % recycle_interval == 0:
                keep = set(self.spec.opening_token_ids) | set(self.spec.invariant_token_ids)
                placed_ids = keep

            # --- Player turn: 1 token ---
            relaxed = dialogue_pos >= relax_after
            if relaxed:
                candidates = [
                    t for t in self._player_tokens
                    if t.id not in placed_ids
                ]
            else:
                candidates = self._get_candidates(self._player_tokens, placed_ids, dialogue_pos)
            if not candidates:
                # Fallback: reset and try again
                placed_ids = set(self.spec.opening_token_ids) | set(self.spec.invariant_token_ids)
                candidates = [t for t in self._player_tokens if t.id not in placed_ids]
            if not candidates:
                break
            chosen = self._sample_from_pool(
                candidates, context_ids, self.player_temperature, dialogue_pos, role_counts,
            )
            if chosen is None:
                break

            placed_ids.add(chosen.id)
            context_ids.append(chosen.id)
            convergence_dims = np.minimum(
                1.0, convergence_dims + chosen.attractor_weights * self.convergence_rate
            )
            energy = self.graph.subgraph_energy(context_ids)
            conv_score = float(convergence_dims.min())

            all_tokens.append(chosen)
            all_roles.append(ROLE_PLAYER)
            soft_targets.append(
                self._compute_soft_target_conditional(chosen.id, context_ids)
            )
            player_turns.append(DialogueTurn(
                token=chosen, role=ROLE_PLAYER, turn_index=len(all_tokens) - 1,
                energy_at_step=energy, convergence_at_step=conv_score,
            ))
            dialogue_pos += 1

            # --- Engine scene: N tokens (one per dimension) ---
            scene_tokens = []
            scene_dims = []
            for d in range(n_dims):
                if relaxed:
                    dim_candidates = [
                        t for t in dim_pools[d]
                        if t.id not in placed_ids
                    ]
                else:
                    dim_candidates = [
                        t for t in dim_pools[d]
                        if t.id not in placed_ids and self._phase_valid(t, dialogue_pos)
                    ]
                if not dim_candidates:
                    # Allow replaying tokens for this dimension
                    dim_candidates = [
                        t for t in dim_pools[d]
                        if self._phase_valid(t, dialogue_pos) or relaxed
                    ]
                if not dim_candidates:
                    continue

                # Score and sample from dimension pool
                scores = np.array([
                    self._score_token(t, context_ids, dialogue_pos) +
                    abs(t.attractor_weights[d]) * 2.0  # boost tokens strong on this dim
                    for t in dim_candidates
                ])

                # Graph-aware boost: reward candidates connected to recent player tokens
                recent_player_ids = [
                    tok.id for tok, role in zip(all_tokens, all_roles)
                    if role == ROLE_PLAYER
                ][-5:]
                if recent_player_ids:
                    for idx, candidate in enumerate(dim_candidates):
                        affinity = sum(
                            self.graph.weight(candidate.id, pid)
                            for pid in recent_player_ids
                        )
                        scores[idx] += affinity * 3.0  # graph-aware boost

                # Dimension-aware: boost tokens that help underdeveloped dimensions
                dim_deficit = 1.0 - float(convergence_dims[d])
                for idx, candidate in enumerate(dim_candidates):
                    dim_weight = abs(candidate.attractor_weights[d])
                    scores[idx] += dim_weight * dim_deficit * 2.0

                # Red herring injection: 15% chance to force atmospheric/misleading token
                RED_HERRING_TAGS = {'surface', 'plausible', 'dramatic'}
                if np.random.random() < 0.15:
                    rh_mask = np.array([
                        bool(set(getattr(t, 'affinity_tags', [])) & RED_HERRING_TAGS)
                        for t in dim_candidates
                    ])
                    if rh_mask.any():
                        scores[~rh_mask] = -1e9  # force red herring selection this turn

                temp = self.engine_temperature
                if self._temperature_jitter > 0:
                    temp += np.random.uniform(-self._temperature_jitter, self._temperature_jitter)
                    temp = max(temp, 0.1)
                scores = scores / max(temp, 1e-8)
                scores -= scores.max()
                weights = np.exp(scores)
                total = weights.sum()
                if total < 1e-12:
                    chosen_tok = dim_candidates[np.random.randint(len(dim_candidates))]
                else:
                    weights /= total
                    idx = np.random.choice(len(dim_candidates), p=weights)
                    chosen_tok = dim_candidates[idx]

                placed_ids.add(chosen_tok.id)
                context_ids.append(chosen_tok.id)
                convergence_dims = np.minimum(
                    1.0, convergence_dims + chosen_tok.attractor_weights * self.convergence_rate
                )
                scene_tokens.append(chosen_tok)
                scene_dims.append(d)

                all_tokens.append(chosen_tok)
                all_roles.append(ROLE_ENGINE)
                soft_targets.append(
                    self._compute_soft_target_conditional(chosen_tok.id, context_ids)
                )
                dialogue_pos += 1

            if not scene_tokens:
                break

            energy = self.graph.subgraph_energy(context_ids)
            conv_score = float(convergence_dims.min())
            scene_turns.append(SceneTurn(
                tokens=scene_tokens, dim_indices=scene_dims, role=ROLE_ENGINE,
                turn_index=len(all_tokens) - len(scene_tokens),
                energy_at_step=energy, convergence_at_step=conv_score,
            ))

            # Update role counts
            for tok in scene_tokens:
                r = tok.id.split(':')[0]
                role_counts[r] = role_counts.get(r, 0) + 1

            # Check convergence
            if conv_score >= self.spec.convergence_threshold and game_turn >= self.min_turns:
                return SceneDialoguePath(
                    player_turns=player_turns, scene_turns=scene_turns,
                    all_tokens=all_tokens, all_roles=all_roles,
                    soft_targets=soft_targets, converged=True,
                    final_convergence=conv_score,
                )

        if not all_tokens:
            return None

        final_conv = float(convergence_dims.min())
        if self.allow_partial or final_conv >= self.spec.convergence_threshold:
            return SceneDialoguePath(
                player_turns=player_turns, scene_turns=scene_turns,
                all_tokens=all_tokens, all_roles=all_roles,
                soft_targets=soft_targets,
                converged=final_conv >= self.spec.convergence_threshold,
                final_convergence=final_conv,
            )
        return None

    def sample_scene_batch(
        self,
        n: int,
        verbose: bool = True,
        max_attempts: int | None = None,
    ) -> List[SceneDialoguePath]:
        """Sample n scene dialogue trajectories."""
        paths: List[SceneDialoguePath] = []
        attempts = 0
        cap = max_attempts if max_attempts is not None else n * 6

        while len(paths) < n and attempts < cap:
            path = self.sample_scene_dialogue()
            if path is not None:
                paths.append(path)
            attempts += 1

            if verbose and len(paths) % 500 == 0 and len(paths) > 0:
                rate = len(paths) / attempts
                converged = sum(1 for p in paths if p.converged)
                print(
                    f"  {len(paths)}/{n} scene dialogues | "
                    f"success: {rate:.1%} | converged: {converged}/{len(paths)}",
                    flush=True,
                )

        if verbose:
            converged = sum(1 for p in paths if p.converged)
            avg_len = np.mean([len(p.all_tokens) for p in paths]) if paths else 0
            print(
                f"  Complete: {len(paths)} scene dialogues from {attempts} attempts "
                f"({converged} converged, avg {avg_len:.0f} tokens)",
                flush=True,
            )
        return paths

    def sample_batch(
        self,
        n: int,
        verbose: bool = True,
        max_attempts: int | None = None,
        strategy_mix: Dict[str, float] | None = None,
    ) -> List[DialoguePath]:
        """
        Sample n dialogue trajectories.

        If strategy_mix is provided (e.g. {"energy": 0.5, "random": 0.3, "red_herring_first": 0.2}),
        each sample draws a strategy proportionally.
        """
        mix_strategies: Optional[List[str]] = None
        mix_weights: Optional[np.ndarray] = None
        if strategy_mix:
            mix_strategies = list(strategy_mix.keys())
            mix_weights = np.array(list(strategy_mix.values()), dtype=np.float64)
            mix_weights /= mix_weights.sum()

        paths: List[DialoguePath] = []
        attempts = 0
        cap = max_attempts if max_attempts is not None else n * 6

        while len(paths) < n and attempts < cap:
            # Set strategy for this sample
            if mix_strategies is not None:
                self._strategy = np.random.choice(mix_strategies, p=mix_weights)

            path = self.sample_dialogue()
            if path is not None:
                paths.append(path)
            attempts += 1

            if verbose and len(paths) % 500 == 0 and len(paths) > 0:
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
