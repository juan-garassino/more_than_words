from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency, compute_narrative_gradient
from .hopfield import TokenGraph


@dataclass
class CartridgeSpec:
    case_id: str
    title: str
    vocab_size: int
    embedding_dim: int
    context_dim: int
    n_attractor_dims: int
    convergence_threshold: float
    convergence_rate: float
    min_turns: int
    max_turns: int
    tokens: List[Token]
    token_graph: TokenGraph
    opening_token_ids: List[str]
    invariant_token_ids: List[str]

    @property
    def invariant_tokens(self) -> List[Token]:
        return [self.get_token(tid) for tid in self.invariant_token_ids]

    def get_token(self, token_id: str) -> Token:
        for token in self.tokens:
            if token.id == token_id:
                return token
        raise KeyError(f"Unknown token id: {token_id}")

    @classmethod
    def load(cls, spec_path: str) -> "CartridgeSpec":
        base = Path(spec_path).parent
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        with open(base / "tokens.json", "r", encoding="utf-8") as f:
            tokens_data = json.load(f)

        with open(base / "graph.json", "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        tokens: List[Token] = []
        for item in tokens_data:
            weights = np.array(item["attractor_weights"], dtype=np.float32)
            token = Token(
                id=item["id"],
                token_class=TokenClass(item["token_class"]),
                phase=TokenPhase(item["phase"]),
                attractor_weights=weights,
                affinity_tags=item.get("affinity_tags", []),
                repulsion_tags=item.get("repulsion_tags", []),
                temperature=float(item.get("temperature", 0.5)),
                narrative_gradient=0.0,
                is_invariant=bool(item.get("is_invariant", False)),
                surface_expression=item.get("surface_expression", ""),
                stream=TokenStream(item.get("stream", "EVIDENCE")),
                agency=TokenAgency(item.get("agency", "SHARED")),
            )
            token.narrative_gradient = compute_narrative_gradient(
                token, spec["n_attractor_dims"]
            )
            tokens.append(token)

        graph = TokenGraph.from_json(graph_data)

        return cls(
            case_id=spec["case_id"],
            title=spec["title"],
            vocab_size=spec["vocab_size"],
            embedding_dim=spec["embedding_dim"],
            context_dim=spec["context_dim"],
            n_attractor_dims=spec["n_attractor_dims"],
            convergence_threshold=spec["convergence_threshold"],
            convergence_rate=spec["convergence_rate"],
            min_turns=spec["min_turns"],
            max_turns=spec["max_turns"],
            tokens=tokens,
            token_graph=graph,
            opening_token_ids=spec["opening_token_ids"],
            invariant_token_ids=spec["invariant_token_ids"],
        )


@dataclass
class TamagotchiSpec:
    case_id: str
    title: str
    vocab_size: int
    embedding_dim: int
    n_needs: int
    n_personality_dims: int
    tokens: List[Token]

    @classmethod
    def load(cls, spec_path: str) -> "TamagotchiSpec":
        base = Path(spec_path).parent
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        with open(base / "tokens.json", "r", encoding="utf-8") as f:
            tokens_data = json.load(f)

        tokens: List[Token] = []
        for item in tokens_data:
            token = Token(
                id=item["id"],
                token_class=TokenClass(item["token_class"]),
                phase=TokenPhase(item["phase"]),
                attractor_weights=np.array(item["attractor_weights"], dtype=np.float32),
                affinity_tags=item.get("affinity_tags", []),
                repulsion_tags=item.get("repulsion_tags", []),
                temperature=float(item.get("temperature", 0.5)),
                narrative_gradient=0.0,
                is_invariant=bool(item.get("is_invariant", False)),
                surface_expression=item.get("surface_expression", ""),
            )
            tokens.append(token)

        return cls(
            case_id=spec["case_id"],
            title=spec["title"],
            vocab_size=spec["vocab_size"],
            embedding_dim=spec["embedding_dim"],
            n_needs=spec["n_needs"],
            n_personality_dims=spec["n_personality_dims"],
            tokens=tokens,
        )
