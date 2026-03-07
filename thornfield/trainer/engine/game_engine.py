from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.cartridge import CartridgeSpec
from core.token import Token, TokenAgency, TokenClass, TokenStream
from core.connection import Connection, RelationType, auto_relation
from core.theory_board import TheoryBoard, AccusationResult

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@dataclass
class ConnectResult:
    coherence: float                          # model output 0–1
    is_contradiction: bool
    contradiction_tokens: List[str]           # token IDs that conflict
    resonance_topk: List[Tuple[str, float]]  # top-5 (token_id, score) from hand
    invalidated_edges: List[Connection]       # edges broken by atmosphere advance


class MysteryGameEngine:
    """
    Stateful game engine for the connection-based mystery mechanic.

    The player draws typed connections between tokens in their hand / atmosphere;
    the engine periodically advances the scene with atmosphere tokens.
    The Hopfield energy defines contradiction; no new neural heads are needed.
    """

    def __init__(
        self,
        spec: CartridgeSpec,
        model,                # MysteryConnectionModel | None
        hand_size: int = 10,
    ) -> None:
        self.spec = spec
        self.model = model
        self.hand_size = hand_size

        self.board = TheoryBoard(
            token_graph=spec.token_graph,
            convergence_rate=spec.convergence_rate,
            n_attractor_dims=spec.n_attractor_dims,
            convergence_threshold=spec.convergence_threshold,
            invariant_token_ids=list(spec.invariant_token_ids),
        )

        # Partition vocab by agency
        player_tokens = [
            t for t in spec.tokens
            if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        ]
        engine_tokens = [
            t for t in spec.tokens
            if t.agency == TokenAgency.ENGINE
            and not t.is_invariant
            and t.stream != TokenStream.OPENING
        ]

        np.random.shuffle(player_tokens)
        self.hand: List[Token] = player_tokens[:hand_size]
        self._remaining_player = player_tokens[hand_size:]

        self.atm_pool: List[Token] = list(engine_tokens)
        np.random.shuffle(self.atm_pool)

        # Fast lookup maps
        self._id_to_token = {t.id: t for t in spec.tokens}
        self._id_to_idx = {t.id: i for i, t in enumerate(spec.tokens)}

    # ------------------------------------------------------------------
    # Setup actions
    # ------------------------------------------------------------------

    def deal(self) -> List[Token]:
        """Return the current hand (already populated in __init__)."""
        return list(self.hand)

    def opening_scene(self) -> List[Token]:
        """Place the case opening tokens into the atmosphere."""
        opening = [self.spec.get_token(tid) for tid in self.spec.opening_token_ids]
        for tok in opening:
            self.board.add_atmosphere(tok)
        return opening

    # ------------------------------------------------------------------
    # Player action
    # ------------------------------------------------------------------

    def player_connect(
        self,
        token_a_id: str,
        relation: RelationType,
        token_b_id: str,
    ) -> ConnectResult:
        """
        Player draws a typed connection between two tokens.
        Both tokens must be in hand or in the atmosphere.
        """
        hand_ids = {t.id for t in self.hand}
        atm_ids = set(self.board.atmosphere_ids)
        allowed = hand_ids | atm_ids

        if token_a_id not in allowed or token_b_id not in allowed:
            missing = [tid for tid in (token_a_id, token_b_id) if tid not in allowed]
            return ConnectResult(
                coherence=0.0,
                is_contradiction=True,
                contradiction_tokens=missing,
                resonance_topk=[],
                invalidated_edges=[],
            )

        token_a = self._id_to_token[token_a_id]
        token_b = self._id_to_token[token_b_id]

        coherence_val = 0.5
        resonance_logits = None
        if HAS_TORCH and self.model is not None:
            coherence_val, resonance_logits = self._model_forward(
                token_a, relation, token_b
            )

        conn = Connection(
            token_a=token_a, relation=relation, token_b=token_b, coherence=coherence_val
        )
        is_contradiction = self.board.add_edge(conn)

        return ConnectResult(
            coherence=coherence_val,
            is_contradiction=is_contradiction,
            contradiction_tokens=[token_a_id, token_b_id] if is_contradiction else [],
            resonance_topk=self._resonance_topk(resonance_logits),
            invalidated_edges=[],
        )

    # ------------------------------------------------------------------
    # Engine action
    # ------------------------------------------------------------------

    def engine_advance(self) -> ConnectResult:
        """
        Engine places the next atmosphere token (energy-guided selection).
        Returns any edges that were invalidated by the new token.
        """
        if not self.atm_pool:
            return ConnectResult(
                coherence=0.0, is_contradiction=False,
                contradiction_tokens=[], resonance_topk=[], invalidated_edges=[],
            )

        # Prefer the atmosphere token that minimises Hopfield energy.
        theory_ids = self.board.theory_token_ids
        atm_ids = self.board.atmosphere_ids
        candidates = self.atm_pool[:20]

        scored = [
            (tok, self.spec.token_graph.induced_subgraph_energy(
                theory_ids, atm_ids + [tok.id]
            ))
            for tok in candidates
        ]
        scored.sort(key=lambda x: x[1])
        chosen = scored[0][0]
        self.atm_pool.remove(chosen)

        invalidated = self.board.add_atmosphere(chosen)

        return ConnectResult(
            coherence=0.0,
            is_contradiction=False,
            contradiction_tokens=[],
            resonance_topk=[],
            invalidated_edges=invalidated,
        )

    # ------------------------------------------------------------------
    # Accusation
    # ------------------------------------------------------------------

    def accuse(
        self, suspect_id: str, mechanism_id: str, motive_id: str
    ) -> AccusationResult:
        return self.board.check_accusation(
            suspect_id, mechanism_id, motive_id, self.spec.invariant_token_ids
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resonance_topk(self, logits, k: int = 5) -> List[Tuple[str, float]]:
        if not HAS_TORCH or logits is None:
            return []
        probs = F.softmax(logits[0], dim=-1)  # (vocab_size,)
        hand_indices = [
            self._id_to_idx[t.id] for t in self.hand if t.id in self._id_to_idx
        ]
        if not hand_indices:
            return []
        scores = [(self.spec.tokens[idx].id, float(probs[idx])) for idx in hand_indices]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def _model_forward(
        self, token_a: Token, relation: RelationType, token_b: Token
    ) -> Tuple[float, object]:
        """Run MysteryConnectionModel for a single candidate edge (B=1)."""
        import torch

        class_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
        stream_to_idx = {s.value: i for i, s in enumerate(TokenStream)}
        agency_to_idx = {a.value: i for i, a in enumerate(TokenAgency)}

        def _t(lst, dtype=torch.long):
            return torch.tensor(lst, dtype=dtype)

        def _tok_feats(tok):
            return (
                self._id_to_idx.get(tok.id, 0),
                class_to_idx.get(tok.token_class.value, 0),
                stream_to_idx.get(tok.stream.value, 0),
                agency_to_idx.get(tok.agency.value, 0),
            )

        # ---- Theory edges (B=1, E=max_edges) ----------------------------
        edges = self.board.edges
        if edges:
            ea = [[self._id_to_idx.get(e.token_a.id, 0) for e in edges]]
            ec = [[class_to_idx.get(e.token_a.token_class.value, 0) for e in edges]]
            es = [[stream_to_idx.get(e.token_a.stream.value, 0) for e in edges]]
            eg = [[agency_to_idx.get(e.token_a.agency.value, 0) for e in edges]]
            er = [[int(e.relation) for e in edges]]
            eb = [[self._id_to_idx.get(e.token_b.id, 0) for e in edges]]
            ebc = [[class_to_idx.get(e.token_b.token_class.value, 0) for e in edges]]
            ebs = [[stream_to_idx.get(e.token_b.stream.value, 0) for e in edges]]
            ebg = [[agency_to_idx.get(e.token_b.agency.value, 0) for e in edges]]
            emask = torch.zeros(1, len(edges), dtype=torch.bool)
        else:
            ea = ec = es = eg = er = eb = ebc = ebs = ebg = [[0]]
            emask = torch.ones(1, 1, dtype=torch.bool)

        # ---- Atmosphere (B=1, max_atm) -----------------------------------
        atm = self.board.atmosphere_tokens
        if atm:
            ai = [[self._id_to_idx.get(t.id, 0) for t in atm]]
            ac = [[class_to_idx.get(t.token_class.value, 0) for t in atm]]
            as_ = [[stream_to_idx.get(t.stream.value, 0) for t in atm]]
            apos = torch.zeros(1, len(atm), 2)
            amask = torch.zeros(1, len(atm), dtype=torch.bool)
        else:
            ai = ac = as_ = [[0]]
            apos = torch.zeros(1, 1, 2)
            amask = torch.ones(1, 1, dtype=torch.bool)

        # ---- Candidate (B=1) --------------------------------------------
        a_idx, a_cls, a_str, a_agt = _tok_feats(token_a)
        b_idx, b_cls, b_str, b_agt = _tok_feats(token_b)

        self.model.eval()
        with torch.no_grad():
            out = self.model(
                _t(ea),  _t(ec),  _t(es),  _t(eg),
                _t(er),
                _t(eb),  _t(ebc), _t(ebs), _t(ebg),
                emask,
                _t(ai),  _t(ac),  _t(as_),
                apos,    amask,
                _t([[a_idx]]).squeeze(-1),  _t([[a_cls]]).squeeze(-1),
                _t([[a_str]]).squeeze(-1),  _t([[a_agt]]).squeeze(-1),
                _t([int(relation)]),
                _t([[b_idx]]).squeeze(-1),  _t([[b_cls]]).squeeze(-1),
                _t([[b_str]]).squeeze(-1),  _t([[b_agt]]).squeeze(-1),
            )

        coherence = float(out["edge_coherence"][0, 0])
        return coherence, out["resonance_logits"]
