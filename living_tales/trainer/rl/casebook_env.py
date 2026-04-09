from __future__ import annotations

from typing import Tuple

from core.casebook import CasebookState
from core.cartridge import CartridgeSpec
from core.hopfield import TokenGraph
from rl.rewards import compute_reward


class CasebookEnv:
    """
    Gym-style environment wrapping CasebookState + CartridgeSpec for RL training.

    Observation:
        {
            "placed_token_ids": list[str],  # ordered list of all placed token IDs
            "turn": int,
        }

    Action:  list[str] — exactly 3 token IDs forming a valid triad

    Reward:  compute_reward() from rl/rewards.py

    Termination:
        - convergence_score >= threshold AND turn >= min_turns  → converged
        - turn >= max_turns                                      → timeout
    """

    def __init__(self, spec: CartridgeSpec, graph: TokenGraph):
        self.spec = spec
        self.graph = graph
        self._casebook: CasebookState | None = None
        self._turn: int = 0

    def reset(self) -> dict:
        """Place opening triad in a fresh CasebookState and return initial obs."""
        self._casebook = CasebookState.create(
            n_dims=self.spec.n_attractor_dims,
            mode=self.spec.mode,
            convergence_rate=self.spec.convergence_rate,
        )
        self._turn = 0

        opening = [self.spec.get_token(tid) for tid in self.spec.opening_token_ids]
        self._casebook.place_triad(opening, position=(0, 0))
        self._turn = 1

        return {
            "placed_token_ids": [t.id for t in self._casebook.all_placed_tokens()],
            "turn": self._turn,
        }

    def step(self, token_ids: list[str]) -> Tuple[dict, float, bool, dict]:
        """
        Place a triad defined by token_ids.

        Returns (obs, reward, done, info).
            obs  = {"placed_token_ids": [...], "turn": int}
            done = True when converged OR timed out
            info = {"converged": bool, "energy": float, "turn": int,
                    "correct_invariants": bool, "convergence_score": float}
        """
        if self._casebook is None:
            raise RuntimeError("Call reset() before step()")

        ids_before = [t.id for t in self._casebook.all_placed_tokens()]
        prev_affinity_tags = self._casebook.active_affinity_tags()

        tokens = [self.spec.get_token(tid) for tid in token_ids]
        # Use (turn, 0) — unique per turn, no collision
        self._casebook.place_triad(tokens, position=(self._turn, 0))
        self._turn += 1

        ids_after = [t.id for t in self._casebook.all_placed_tokens()]
        new_affinity_tags = self._casebook.active_affinity_tags()

        reward = compute_reward(
            graph=self.graph,
            ids_before=ids_before,
            ids_after=ids_after,
            turn=self._turn,
            max_turns=self.spec.max_turns,
            new_affinity_tags=new_affinity_tags,
            prev_affinity_tags=prev_affinity_tags,
        )

        score = self._casebook.convergence_score
        converged = (
            score >= self.spec.convergence_threshold
            and self._turn >= self.spec.min_turns
        )
        timed_out = self._turn >= self.spec.max_turns
        done = converged or timed_out

        placed_ids = set(ids_after)
        inv_ids = set(self.spec.invariant_token_ids)
        correct_invariants = inv_ids.issubset(placed_ids)

        obs = {
            "placed_token_ids": ids_after,
            "turn": self._turn,
        }
        info = {
            "converged": converged,
            "energy": self.graph.subgraph_energy(ids_after),
            "turn": self._turn,
            "correct_invariants": correct_invariants,
            "convergence_score": float(score),
        }

        return obs, reward, done, info
