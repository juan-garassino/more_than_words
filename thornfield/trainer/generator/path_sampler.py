from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.casebook import CasebookState
from core.token import Token
from core.cartridge import CartridgeSpec


class PathSampler:
    """
    Monte Carlo sampler for Mystery cartridge training paths.
    """

    def __init__(
        self,
        spec: CartridgeSpec,
        sampling_temperature: float = 1.2,
        max_turns: int | None = None,
        min_turns: int | None = None,
        enforce_monotone: bool = False,
        monotone_tolerance: float = 0.01,
    ):
        self.spec = spec
        self.graph = spec.token_graph
        self.sampling_temperature = sampling_temperature
        self.max_turns = max_turns if max_turns is not None else spec.max_turns
        self.min_turns = min_turns if min_turns is not None else spec.min_turns
        self.enforce_monotone = enforce_monotone
        self.monotone_tolerance = monotone_tolerance
        self._precomputed_valid_triads = self._precompute_valid_triads()
        self._tag_index = self._build_tag_index()

    def _precompute_valid_triads(self) -> List[List[Token]]:
        tokens = [t for t in self.spec.tokens if not t.is_invariant]
        valid = []

        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                for k in range(j + 1, len(tokens)):
                    trio = [tokens[i], tokens[j], tokens[k]]
                    if self._is_valid_triad(trio):
                        valid.append(trio)

        return valid

    def _is_valid_triad(self, tokens: List[Token]) -> bool:
        if len({t.token_class for t in tokens}) < 3:
            return False

        pairs = [(tokens[0], tokens[1]), (tokens[0], tokens[2]), (tokens[1], tokens[2])]
        has_edge = any(self.graph.weight(a.id, b.id) > 0.1 for a, b in pairs)
        if not has_edge:
            return False

        for a, b in pairs:
            a_repels_b = bool(set(a.repulsion_tags) & set(b.affinity_tags))
            b_repels_a = bool(set(b.repulsion_tags) & set(a.affinity_tags))
            if a_repels_b or b_repels_a:
                return False

        return True

    def _build_tag_index(self) -> Dict[str, List[List[Token]]]:
        index: Dict[str, List[List[Token]]] = {}
        for triad in self._precomputed_valid_triads:
            all_tags = set()
            for token in triad:
                all_tags.update(token.affinity_tags)
            for tag in all_tags:
                index.setdefault(tag, []).append(triad)
        return index

    def _get_candidates(self, casebook: CasebookState) -> List[List[Token]]:
        active_tags = casebook.active_affinity_tags()
        placed_ids = casebook.placed_token_ids()
        seen = set()
        candidates = []

        for tag in active_tags:
            for triad in self._tag_index.get(tag, []):
                key = tuple(sorted(t.id for t in triad))
                if key not in seen:
                    if not any(t.id in placed_ids for t in triad):
                        if all(t.is_available_at_turn(casebook.turn_count) for t in triad):
                            candidates.append(triad)
                            seen.add(key)

        if candidates:
            return candidates

        # Fallback: allow any valid triad if tag-based lookup yields none.
        for triad in self._precomputed_valid_triads:
            key = tuple(sorted(t.id for t in triad))
            if key in seen:
                continue
            if any(t.id in placed_ids for t in triad):
                continue
            if not all(t.is_available_at_turn(casebook.turn_count) for t in triad):
                continue
            candidates.append(triad)
            seen.add(key)

        return candidates

    def _triad_energy(self, triad: List[Token], casebook: CasebookState) -> float:
        candidate_ids = [t.id for t in triad]
        context_ids = [t.id for t in casebook.all_placed_tokens()]
        return self.graph.induced_subgraph_energy(candidate_ids, context_ids)

    def _score_triad(self, triad: List[Token], casebook: CasebookState) -> float:
        candidate_ids = [t.id for t in triad]
        context_ids = [t.id for t in casebook.all_placed_tokens()]

        energy = self.graph.induced_subgraph_energy(candidate_ids, context_ids)
        gradient_bonus = sum(t.narrative_gradient for t in triad) / 3.0

        return -energy + gradient_bonus * 0.2

    def sample_path(self) -> Optional[List[List[Token]]]:
        casebook = CasebookState(
            convergence_dimensions=np.zeros(self.spec.n_attractor_dims, dtype=np.float32)
        )
        path: List[List[Token]] = []

        opening = [self.spec.get_token(tid) for tid in self.spec.opening_token_ids]
        casebook.place_triad(opening, position=(0, 0))
        path.append(opening)

        for turn in range(1, self.max_turns):
            candidates = self._get_candidates(casebook)
            if len(candidates) < 1:
                return None

            if self.enforce_monotone:
                prev_energy = self._triad_energy(path[-1], casebook) if path else float("inf")
                filtered = []
                for c in candidates:
                    e = self._triad_energy(c, casebook)
                    if e <= prev_energy + self.monotone_tolerance:
                        filtered.append(c)
                if filtered:
                    candidates = filtered

            scores = np.array([self._score_triad(c, casebook) for c in candidates])
            scores = scores / self.sampling_temperature
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()

            idx = np.random.choice(len(candidates), p=weights)
            chosen = candidates[idx]

            row = min(turn, 7)
            col = np.random.randint(0, 4)
            casebook.place_triad(chosen, position=(row, col))
            path.append(chosen)

            if casebook.convergence_score >= 0.75 and turn >= self.min_turns:
                invariants = self.spec.invariant_tokens
                casebook.place_triad(invariants, position=(7, 2))
                path.append(invariants)
                return path

        return None

    def sample_batch(self, n: int, verbose: bool = True) -> List[List[List[Token]]]:
        paths: List[List[List[Token]]] = []
        attempts = 0

        while len(paths) < n and attempts < n * 6:
            path = self.sample_path()
            if path is not None:
                paths.append(path)
            attempts += 1

            if verbose and len(paths) % 100 == 0 and len(paths) > 0:
                rate = len(paths) / attempts
                print(f"  {len(paths)}/{n} paths | success rate: {rate:.1%}")

        if verbose:
            print(f"  Complete: {len(paths)} paths from {attempts} attempts")

        return paths
