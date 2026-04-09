"""
Run N programmatic dialogue games for evaluation (no TUI).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "living_tales" / "trainer"))

from core.cartridge import CartridgeSpec
from core.token import Token, TokenAgency, TokenStream
try:
    from evals.datasets._schema import DialogueTurn, GameResult
except ImportError:  # pragma: no cover
    from datasets._schema import DialogueTurn, GameResult


class DialogueGameRunner:
    """Play dialogue games programmatically with a trained model."""

    def __init__(self, model, spec: CartridgeSpec, mappings: dict, device: str = "cpu"):
        self.model = model
        self.spec = spec
        self.mappings = mappings
        self.device = device

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

    def _encode_token(self, tok: Token) -> Tuple[int, int, int, int, int]:
        m = self.mappings
        return (
            m["id_to_idx"][tok.id],
            m["class_to_idx"][tok.token_class.value],
            m["phase_to_idx"][tok.phase.value],
            m["stream_to_idx"][tok.stream.value],
            m["agency_to_idx"][tok.agency.value],
        )

    def run_game(self, seed: int, max_turns: int = 60) -> GameResult:
        """Play one dialogue game with the given random seed."""
        import torch

        rng = np.random.RandomState(seed)
        spec = self.spec
        convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
        placed_ids: set = set()
        context_ids: list = []
        turns: list = []

        seq_t, seq_c, seq_p, seq_s, seq_a = [], [], [], [], []
        id_to_idx = self.mappings["id_to_idx"]
        idx_to_id = {v: k for k, v in id_to_idx.items()}

        # Opening
        for tid in spec.opening_token_ids:
            tok = spec.get_token(tid)
            placed_ids.add(tok.id)
            context_ids.append(tok.id)
            convergence_dims = np.minimum(
                1.0, convergence_dims + tok.attractor_weights * spec.convergence_rate,
            )
            enc = self._encode_token(tok)
            seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
            seq_s.append(enc[3]); seq_a.append(enc[4])
            turns.append(DialogueTurn(
                token_id=tok.id, role="engine", token_class=tok.token_class.value,
                phase=tok.phase.value,
                energy_at_step=spec.token_graph.subgraph_energy(context_ids),
                convergence_at_step=float(convergence_dims.min()),
            ))

        player_pool = list(self.player_tokens)
        rng.shuffle(player_pool)
        is_player = True

        for step in range(len(turns), max_turns):
            game_turn = step // 2
            conv_score = float(convergence_dims.min())

            if conv_score >= spec.convergence_threshold and game_turn >= spec.min_turns:
                break

            if is_player:
                # Random valid player token
                candidates = [
                    t for t in player_pool
                    if t.id not in placed_ids and t.is_available_at_turn(game_turn)
                ]
                if not candidates:
                    break
                chosen = candidates[rng.randint(len(candidates))]
            else:
                # Model-driven engine token
                inp_t = torch.tensor([seq_t], dtype=torch.long)
                inp_c = torch.tensor([seq_c], dtype=torch.long)
                inp_p = torch.tensor([seq_p], dtype=torch.long)
                inp_s = torch.tensor([seq_s], dtype=torch.long)
                inp_a = torch.tensor([seq_a], dtype=torch.long)

                valid_mask = torch.zeros(spec.vocab_size, dtype=torch.bool)
                for t in self.engine_tokens:
                    if t.id not in placed_ids and t.is_available_at_turn(game_turn):
                        valid_mask[id_to_idx[t.id]] = True

                if not valid_mask.any():
                    break

                chosen_idx, _ = self.model.predict_next(
                    inp_t, inp_c, inp_p, inp_s, inp_a,
                    valid_mask=valid_mask, temperature=0.8,
                )
                chosen_id = idx_to_id[chosen_idx]
                chosen = spec.get_token(chosen_id)

            placed_ids.add(chosen.id)
            context_ids.append(chosen.id)
            convergence_dims = np.minimum(
                1.0, convergence_dims + chosen.attractor_weights * spec.convergence_rate,
            )
            enc = self._encode_token(chosen)
            seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
            seq_s.append(enc[3]); seq_a.append(enc[4])

            energy = spec.token_graph.subgraph_energy(context_ids)
            turns.append(DialogueTurn(
                token_id=chosen.id,
                role="player" if is_player else "engine",
                token_class=chosen.token_class.value,
                phase=chosen.phase.value,
                energy_at_step=energy,
                convergence_at_step=float(convergence_dims.min()),
            ))
            is_player = not is_player

        final_conv = float(convergence_dims.min())
        return GameResult(
            seed=seed,
            turns=turns,
            converged=final_conv >= spec.convergence_threshold,
            final_convergence=final_conv,
            final_token_ids=list(context_ids),
            total_energy=spec.token_graph.subgraph_energy(context_ids) if context_ids else 0.0,
        )

    def run_batch(self, n_games: int, seeds: Optional[List[int]] = None) -> List[GameResult]:
        if seeds is None:
            seeds = list(range(n_games))
        return [self.run_game(s) for s in seeds]
