from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.token import Token, TokenAgency
from core.cartridge import CartridgeSpec
from core.connection import (
    Connection, auto_relation, valid_relations, RelationType, N_RELATIONS,
)
from generator.path_sampler import PathSampler


@dataclass
class ConnectionExample:
    theory_edges: List[Connection]      # edges placed so far
    atmosphere_tokens: List[Token]      # engine tokens placed so far
    candidate_edge: Connection          # the proposed next connection
    target_coherence: float             # 1.0 if positive, 0.0 if negative


class ConnectionSampler:
    """
    Converts triad paths from PathSampler into supervised ConnectionExamples.

    Each triad is decomposed into adjacent token pairs; a RelationType is
    auto-derived from the class pair.  Atmosphere tokens are interleaved
    every two player connections.
    """

    def __init__(self, spec: CartridgeSpec) -> None:
        self.spec = spec
        self._path_sampler = PathSampler(
            spec,
            sampling_temperature=1.4,
            min_affinity=0.05,
            allow_partial=True,
        )

        # Prefer ENGINE tokens for atmosphere; fall back to ENGINE+SHARED.
        engine_only = [
            t for t in spec.tokens
            if t.agency == TokenAgency.ENGINE and not t.is_invariant
        ]
        self._engine_tokens: List[Token] = engine_only if engine_only else [
            t for t in spec.tokens
            if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED) and not t.is_invariant
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _triads_to_connections(self, triads: List[List[Token]]) -> List[Connection]:
        """
        For every non-invariant triad, emit one Connection per adjacent pair
        (i,j) with i < j, deriving the relation from the class pair.
        """
        connections: List[Connection] = []
        for triad in triads:
            if all(t.is_invariant for t in triad):
                continue
            for i in range(len(triad)):
                for j in range(i + 1, len(triad)):
                    a, b = triad[i], triad[j]
                    rel = auto_relation(a, b)
                    if rel is None:
                        rel = RelationType.BELONGS_TO
                    connections.append(
                        Connection(token_a=a, relation=rel, token_b=b, coherence=1.0)
                    )
        return connections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_path(self) -> Optional[List[ConnectionExample]]:
        """
        1. Run PathSampler → triad path.
        2. Decompose triads into a flat connection sequence.
        3. Interleave ENGINE atmosphere tokens every 2 player connections.
        4. Emit ConnectionExample(edges[:t], atm[:t], edges[t]) for each t.
        """
        triad_path = self._path_sampler.sample_path()
        if triad_path is None:
            return None

        connections = self._triads_to_connections(triad_path)
        if not connections:
            return None

        # Build atmosphere pool from tokens not already placed in triads.
        placed_ids = {t.id for triad in triad_path for t in triad}
        atm_pool = [t for t in self._engine_tokens if t.id not in placed_ids]
        np.random.shuffle(atm_pool)

        examples: List[ConnectionExample] = []
        placed_edges: List[Connection] = []
        placed_atm: List[Token] = []
        atm_idx = 0

        for i, conn in enumerate(connections):
            # Interleave one atmosphere token every 2 player connections.
            if i > 0 and i % 2 == 0 and atm_idx < len(atm_pool):
                placed_atm.append(atm_pool[atm_idx])
                atm_idx += 1

            examples.append(
                ConnectionExample(
                    theory_edges=list(placed_edges),
                    atmosphere_tokens=list(placed_atm),
                    candidate_edge=conn,
                    target_coherence=1.0,
                )
            )
            placed_edges.append(conn)

        return examples

    def sample_negative(self, positive: ConnectionExample) -> ConnectionExample:
        """
        Same theory + atmosphere as `positive`, but with an incoherent candidate.
        Option A (50 %): replace token_b with an unrelated token.
        Option B (50 %): keep the token pair, use an invalid relation type.
        """
        orig = positive.candidate_edge
        all_non_inv = [t for t in self.spec.tokens if not t.is_invariant]

        # Collect all IDs already in the positive example.
        used_ids = {orig.token_a.id, orig.token_b.id}
        for e in positive.theory_edges:
            used_ids.add(e.token_a.id)
            used_ids.add(e.token_b.id)

        unrelated = [t for t in all_non_inv if t.id not in used_ids]

        if unrelated and np.random.random() < 0.5:
            # Option A: random unrelated token_b
            new_b = unrelated[np.random.randint(len(unrelated))]
            rel = auto_relation(orig.token_a, new_b)
            if rel is None:
                rel = RelationType.BELONGS_TO
            neg_edge = Connection(
                token_a=orig.token_a, relation=rel, token_b=new_b, coherence=0.0
            )
        else:
            # Option B: invalid relation type
            valid = valid_relations(orig.token_a, orig.token_b)
            invalid_rels = [r for r in RelationType if r not in valid]
            if invalid_rels:
                neg_rel = invalid_rels[np.random.randint(len(invalid_rels))]
            else:
                neg_rel = RelationType((int(orig.relation) + 1) % N_RELATIONS)
            neg_edge = Connection(
                token_a=orig.token_a, relation=neg_rel, token_b=orig.token_b, coherence=0.0
            )

        return ConnectionExample(
            theory_edges=positive.theory_edges,
            atmosphere_tokens=positive.atmosphere_tokens,
            candidate_edge=neg_edge,
            target_coherence=0.0,
        )

    def sample_batch(self, n: int) -> List[ConnectionExample]:
        """Sample n positive ConnectionExamples."""
        examples: List[ConnectionExample] = []
        attempts = 0
        cap = n * 8

        while len(examples) < n and attempts < cap:
            path = self.sample_path()
            if path:
                for ex in path:
                    examples.append(ex)
                    if len(examples) >= n:
                        break
            attempts += 1

        return examples[:n]
