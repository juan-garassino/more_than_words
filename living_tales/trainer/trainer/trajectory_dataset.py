"""
TrajectoryDataset
=================
PyTorch Dataset wrapping hand-authored trajectories for the Living Tales
StructuredSceneTransformer.

Each example corresponds to ONE turn of ONE trajectory:

    history       — flat sequence of token IDs from prior turns
                    (opening tokens, then alternating
                     [player_card, scene_dim_1, ..., scene_dim_K]
                     for each prior turn).
    history_dims  — parallel sequence of dim IDs (one per token in history)
                    so the model knows which slot each history token came from.
    player_card   — global vocab index of the player's card this turn.
    target_scene  — dict[dim_name -> local-vocab index] for the supervised
                    scene tuple this turn.

The full vocab is the union of every token appearing in any dim's vocab
(plus the player_card vocab, which is folded into the global index via
`full_vocab_to_idx`). Each dim also has its own LOCAL vocab indexing so the
model's per-dim heads only project over their relevant tokens.

`history_dims` uses an integer code per dim:
    0..len(DIM_ORDER)-1   — scene-dim slot
    PLAYER_CARD_DIM_ID    — the just-played player card
    OPENING_DIM_ID        — opening / pre-game tokens
    PAD_DIM_ID            — padding (after collation)

Padding strategy
----------------
Histories are padded RIGHT to `max_history`. A `padding_mask` is returned
(True at padded positions) so the model can ignore them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

# ── Dim ordering shared with the model ──────────────────────────────────────
# This MUST match StructuredSceneTransformer.DIM_ORDER.
DIM_ORDER: List[str] = [
    "LOCATION",
    "TRANSITION",
    "CAUSE",
    "PRESENCE",
    "STANCE",
    "ACTION",
    "OBJECT_FOCUS",
    "TELL",
    "ATMOSPHERE",
    "REVELATION",
    "BEAT",
]

# Special dim-ID codes.
#
# StructuredSceneTransformer's `dim_tag_embedding` has size
# `len(DIM_ORDER) + 1`, with `len(DIM_ORDER)` reserved as the padding/pad-id
# slot. We collapse player-card and opening tokens onto that same slot
# (the actual token embedding still distinguishes them; the model's
# padding_mask differentiates real positions from padding).
PLAYER_CARD_DIM_ID: int = len(DIM_ORDER)       # 11 — same as model dim_pad_id
OPENING_DIM_ID: int = len(DIM_ORDER)           # 11
PAD_DIM_ID: int = len(DIM_ORDER)               # 11
NUM_DIM_CODES: int = len(DIM_ORDER) + 1        # 12


# ── Helpers ─────────────────────────────────────────────────────────────────
def _strip_accuse(token: str) -> str:
    """Player accusation cards arrive as `ACCUSE:suspect:...`. Map back to
    the underlying suspect token for vocab lookup."""
    if isinstance(token, str) and token.startswith("ACCUSE:"):
        return token[len("ACCUSE:"):]
    return token


# ── Dataset ─────────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    """One example per turn across all trajectories."""

    def __init__(
        self,
        trajectories: Sequence[Any],
        dim_vocab: Dict[str, List[str]],
        full_vocab_to_idx: Dict[str, int],
        dim_vocab_to_idx: Dict[str, Dict[str, int]],
        max_history: int = 80,
    ):
        self.dim_vocab = dim_vocab
        self.full_vocab_to_idx = full_vocab_to_idx
        self.dim_vocab_to_idx = dim_vocab_to_idx
        self.max_history = max_history

        # Index of an "unknown" / "none-like" token per dim. Used as a fallback
        # when an authored token isn't present in that dim's vocab. Prefers
        # any "*:none" token in the dim, else local index 0.
        self._dim_fallback_local: Dict[str, int] = {}
        for d, toks in dim_vocab.items():
            fallback = 0
            for i, t in enumerate(toks):
                if t.endswith(":none"):
                    fallback = i
                    break
            self._dim_fallback_local[d] = fallback

        self.examples: List[Dict[str, Any]] = []
        self._build(trajectories)

    # ── Construction ──
    def _vocab_idx(self, token: str) -> int:
        token = _strip_accuse(token)
        if token in self.full_vocab_to_idx:
            return self.full_vocab_to_idx[token]
        # Unknown player-card / token: map to first vocab entry as a safe
        # fallback. (Player cards like `travel:*` and ACCUSE-stripped suspect
        # tokens may not all live in dim vocabs; we still need a stable id.)
        return 0

    def _local_idx(self, dim: str, token: str) -> int:
        idx_map = self.dim_vocab_to_idx.get(dim, {})
        if token in idx_map:
            return idx_map[token]
        return self._dim_fallback_local.get(dim, 0)

    def _encode_scene_to_history(
        self, scene: Dict[str, str]
    ) -> List[tuple]:
        """Returns [(token_id, dim_code), ...] for one scene's K dim tokens."""
        out = []
        for slot_idx, d in enumerate(DIM_ORDER):
            tok = scene.get(d, "")
            if not tok:
                continue
            out.append((self._vocab_idx(tok), slot_idx))
        return out

    def _build(self, trajectories: Sequence[Any]):
        for traj in trajectories:
            # Opening tokens — pre-game state.
            opening_seq: List[tuple] = []
            for tok in getattr(traj, "opening", []) or []:
                opening_seq.append((self._vocab_idx(tok), OPENING_DIM_ID))

            history: List[tuple] = list(opening_seq)
            turns = list(getattr(traj, "turns", []) or [])

            for turn in turns:
                player_card = getattr(turn, "player_card", "") or ""
                scene = getattr(turn, "scene", {}) or {}

                # Skip terminal accusation-only turns where scene may be
                # underspecified — only if no scene present.
                if not scene:
                    continue

                # Build target_scene dict[dim -> local idx].
                target_scene = {
                    d: self._local_idx(d, scene.get(d, ""))
                    for d in DIM_ORDER
                }

                # Truncate history to max_history (keep most-recent tokens).
                hist = history
                if len(hist) > self.max_history:
                    hist = hist[-self.max_history:]

                hist_tokens = [h[0] for h in hist]
                hist_dims = [h[1] for h in hist]

                self.examples.append({
                    "history_tokens": torch.tensor(hist_tokens, dtype=torch.long),
                    "history_dims": torch.tensor(hist_dims, dtype=torch.long),
                    "player_card": torch.tensor(
                        self._vocab_idx(player_card), dtype=torch.long
                    ),
                    "target_scene": {
                        d: torch.tensor(target_scene[d], dtype=torch.long)
                        for d in DIM_ORDER
                    },
                })

                # Append the just-completed turn to history for the next
                # example: [player_card, scene_dim_1, ..., scene_dim_K].
                history.append(
                    (self._vocab_idx(player_card), PLAYER_CARD_DIM_ID)
                )
                history.extend(self._encode_scene_to_history(scene))

    # ── Dataset protocol ──
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.examples[i]

    # ── Collate ──
    @staticmethod
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_len = max(int(ex["history_tokens"].numel()) for ex in batch)
        max_len = max(max_len, 1)  # avoid zero-length sequences

        history_tokens = torch.zeros((B, max_len), dtype=torch.long)
        history_dims = torch.full(
            (B, max_len), fill_value=PAD_DIM_ID, dtype=torch.long,
        )
        padding_mask = torch.ones((B, max_len), dtype=torch.bool)

        for i, ex in enumerate(batch):
            t = ex["history_tokens"]
            d = ex["history_dims"]
            n = int(t.numel())
            if n > 0:
                history_tokens[i, :n] = t
                history_dims[i, :n] = d
                padding_mask[i, :n] = False

        player_card = torch.stack([ex["player_card"] for ex in batch], dim=0)

        target_scene = {
            dim: torch.stack(
                [ex["target_scene"][dim] for ex in batch], dim=0
            )
            for dim in DIM_ORDER
        }

        return {
            "history_tokens": history_tokens,       # (B, S)
            "history_dims": history_dims,           # (B, S)
            "padding_mask": padding_mask,           # (B, S) True = pad
            "player_card": player_card,             # (B,)
            "target_scene": target_scene,           # dict[dim] -> (B,)
        }
