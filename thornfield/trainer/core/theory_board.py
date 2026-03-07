from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .token import Token
from .hopfield import TokenGraph
from .connection import Connection, auto_relation, RelationType


@dataclass
class AccusationResult:
    solved: bool
    wrong_dimensions: List[int]  # indices of the 3 dims that don't match


@dataclass
class TheoryBoard:
    """
    Runtime state for the connection-based mystery engine.
    Replaces CasebookState for game play; training pipeline is unchanged.
    """

    token_graph: TokenGraph
    convergence_rate: float
    n_attractor_dims: int
    convergence_threshold: float = 0.75
    invariant_token_ids: List[str] = field(default_factory=list)
    edges: List[Connection] = field(default_factory=list)
    atmosphere_tokens: List[Token] = field(default_factory=list)
    # Initialised in __post_init__ using n_attractor_dims
    convergence_dimensions: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.convergence_dimensions = np.zeros(self.n_attractor_dims, dtype=np.float32)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def convergence_score(self) -> float:
        return float(self.convergence_dimensions.min())

    @property
    def theory_token_ids(self) -> List[str]:
        """Deduplicated IDs of all tokens participating in theory edges."""
        seen: set = set()
        result: List[str] = []
        for conn in self.edges:
            for tok in (conn.token_a, conn.token_b):
                if tok.id not in seen:
                    seen.add(tok.id)
                    result.append(tok.id)
        return result

    @property
    def atmosphere_ids(self) -> List[str]:
        return [t.id for t in self.atmosphere_tokens]

    # ------------------------------------------------------------------
    # Energy helpers
    # ------------------------------------------------------------------

    def _energy(self, theory_ids: List[str], atm_ids: List[str]) -> float:
        return self.token_graph.induced_subgraph_energy(theory_ids, atm_ids)

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def is_contradiction(self, token_a_id: str, token_b_id: str) -> bool:
        """
        Returns True if adding these two tokens to the theory raises Hopfield energy
        by more than the contradiction threshold (0.05).
        """
        current_ids = self.theory_token_ids
        atm_ids = self.atmosphere_ids
        old_energy = self._energy(current_ids, atm_ids)

        new_ids = list(current_ids)
        for tid in (token_a_id, token_b_id):
            if tid not in new_ids:
                new_ids.append(tid)
        new_energy = self._energy(new_ids, atm_ids)

        return new_energy > old_energy + 0.05

    def add_edge(self, conn: Connection) -> bool:
        """
        Add a connection to the theory.
        Returns True if the connection is a contradiction (energy increased).
        Updates convergence_dimensions if not a contradiction.
        """
        contradiction = self.is_contradiction(conn.token_a.id, conn.token_b.id)
        conn.is_contradiction = contradiction
        self.edges.append(conn)

        if not contradiction:
            contribution = (
                conn.token_a.attractor_weights + conn.token_b.attractor_weights
            ) / 2.0
            self.convergence_dimensions = np.minimum(
                1.0,
                self.convergence_dimensions + contribution * self.convergence_rate,
            )

        return contradiction

    def add_atmosphere(self, token: Token) -> List[Connection]:
        """
        Place an atmosphere token.
        Returns the list of previously non-contradicting edges now invalidated
        (i.e., edges whose energy increased after this atmospheric token was added).
        """
        self.atmosphere_tokens.append(token)
        new_atm_ids = self.atmosphere_ids

        invalidated: List[Connection] = []
        for conn in self.edges:
            if conn.is_contradiction:
                continue
            edge_ids = [conn.token_a.id, conn.token_b.id]
            old_e = self.token_graph.induced_subgraph_energy(edge_ids, new_atm_ids[:-1])
            new_e = self.token_graph.induced_subgraph_energy(edge_ids, new_atm_ids)
            if new_e > old_e + 0.05:
                conn.is_contradiction = True
                invalidated.append(conn)

        return invalidated

    def can_accuse(self) -> bool:
        return self.convergence_score >= self.convergence_threshold

    def check_accusation(
        self,
        suspect_id: str,
        mechanism_id: str,
        motive_id: str,
        invariant_ids: List[str],
    ) -> AccusationResult:
        """
        Check an accusation against the case invariants.
        invariant_ids must be ordered [suspect_inv, mechanism_inv, motive_inv].
        Returns wrong_dimensions without revealing which token is correct.
        """
        guesses = [suspect_id, mechanism_id, motive_id]
        wrong = [
            i for i, (g, inv) in enumerate(zip(guesses, invariant_ids)) if g != inv
        ]
        return AccusationResult(solved=len(wrong) == 0, wrong_dimensions=wrong)
