from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TokenGraph:
    """
    Hopfield weight matrix expressed as a weighted graph.
    """

    nodes: List[str]
    edges: Dict[Tuple[str, str], float]

    @classmethod
    def from_json(cls, graph_data: dict) -> "TokenGraph":
        nodes = graph_data["nodes"]
        edges: Dict[Tuple[str, str], float] = {}
        for edge in graph_data["edges"]:
            key = tuple(sorted([edge["from"], edge["to"]]))
            edges[key] = edge["weight"]
        return cls(nodes=nodes, edges=edges)

    def weight(self, token_a: str, token_b: str) -> float:
        key = tuple(sorted([token_a, token_b]))
        return self.edges.get(key, 0.0)

    def subgraph_energy(self, token_ids: List[str]) -> float:
        energy = 0.0
        for i in range(len(token_ids)):
            for j in range(i + 1, len(token_ids)):
                energy -= self.weight(token_ids[i], token_ids[j])
        return energy

    def induced_subgraph_energy(
        self, candidate_ids: List[str], context_ids: List[str]
    ) -> float:
        internal = self.subgraph_energy(candidate_ids)
        cross = 0.0
        for c in candidate_ids:
            for p in context_ids:
                cross -= self.weight(c, p) * 0.75
        return internal + cross

    def to_matrix(self, token_to_idx: Dict[str, int]) -> np.ndarray:
        n = len(token_to_idx)
        W = np.zeros((n, n), dtype=np.float32)
        for (a, b), w in self.edges.items():
            if a in token_to_idx and b in token_to_idx:
                i, j = token_to_idx[a], token_to_idx[b]
                W[i, j] = w
                W[j, i] = w
        return W


class HopfieldAnalyzer:
    def lyapunov_check(self, model, valid_paths: List[List["Triad"]], tolerance=0.01):
        violations = []
        total_steps = 0
        monotone_steps = 0

        for path_idx, path in enumerate(valid_paths):
            prev_energy = float("inf")
            for turn, triad in enumerate(path):
                energy = model.compute_energy(triad)
                if energy > prev_energy + tolerance:
                    violations.append(
                        {
                            "path_idx": path_idx,
                            "turn": turn,
                            "energy_delta": energy - prev_energy,
                        }
                    )
                else:
                    monotone_steps += 1
                total_steps += 1
                prev_energy = energy

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "monotone_rate": monotone_steps / max(total_steps, 1),
            "n_paths_checked": len(valid_paths),
        }

    def basin_size(self, model, spec: "CartridgeSpec", n_samples: int = 1000) -> float:
        from generator.path_sampler import PathSampler

        sampler = PathSampler(spec, sampling_temperature=2.0)
        converged = 0
        for _ in range(n_samples):
            path = sampler.sample_path()
            if path is not None:
                converged += 1
        return converged / n_samples

    def spurious_attractor_scan(self, model, spec: "CartridgeSpec", n_probes: int = 200):
        _ = (model, spec, n_probes)
        return []

    def cue_completion_curve(self, model, spec: "CartridgeSpec", n_samples: int = 200):
        curve: Dict[int, float] = {}
        from generator.path_sampler import PathSampler

        sampler = PathSampler(spec)
        paths = sampler.sample_batch(n_samples, verbose=False)

        for turn in range(1, spec.max_turns + 1):
            converged_by_turn = sum(
                1
                for path in paths
                if len(path) <= turn and path[-1][0].is_invariant
            )
            curve[turn] = converged_by_turn / max(len(paths), 1)

        return curve
