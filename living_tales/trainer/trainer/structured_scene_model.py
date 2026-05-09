"""
Structured Scene Transformer for Living Tales (multidimensional schema v1).
==========================================================================

Per-case overfitter (~300-500K params). Predicts a complete 11-dim scene
tuple given dialogue history + the player's current card.

Design points
-------------
- Token embedding shared across all dims; the vocab is the union of all
  per-dim vocabularies.
- Sequential dim conditioning: dims are emitted in `DIM_ORDER`. Each dim's
  head sees (a) the encoded history summary, (b) the player card embedding,
  and (c) the embeddings of all dims emitted earlier in this scene (zero
  vectors for dims not yet emitted).
- Hard masks live OUTSIDE the model. They are applied at inference only
  (via `predict_scene`); training data is already constraint-clean.
- Per-dim heads are small linear projections from hidden_dim to the size of
  that dim's local vocab, so the model never has to choose across the full
  union vocab when emitting a single slot.

This module deliberately does NOT import the legacy energy_model — it stays
self-contained so the new pipeline doesn't drag the legacy graph along.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Position encoding ───────────────────────────────────────────────────────
class _SinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len, :]


# ─── Model ───────────────────────────────────────────────────────────────────
class StructuredSceneTransformer(nn.Module):
    """Tiny per-case overfitter. ~300-500K params total.

    Sequential dim-conditioning: each head sees prior emitted dims.
    Hard-mask layer applied at inference (not training).
    """

    DIM_ORDER: List[str] = [
        "LOCATION", "TRANSITION", "PRESENCE", "STANCE",
        "CAUSE", "ACTION", "OBJECT_FOCUS", "TELL",
        "ATMOSPHERE", "REVELATION", "BEAT",
    ]

    def __init__(
        self,
        dim_vocab: Dict[str, List[str]],   # dim_name -> list of token IDs
        full_vocab: List[str],             # complete token universe
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_history: int = 80,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,              # overfitting-friendly
    ):
        super().__init__()

        # Validate that DIM_ORDER is covered.
        missing = [d for d in self.DIM_ORDER if d not in dim_vocab]
        if missing:
            raise ValueError(
                f"dim_vocab missing required dimensions: {missing}"
            )

        self.dim_vocab: Dict[str, List[str]] = {
            d: list(dim_vocab[d]) for d in self.DIM_ORDER
        }
        self.full_vocab: List[str] = list(full_vocab)
        self.hidden_dim = hidden_dim
        self.max_history = max_history

        # ── Token <-> id maps (full vocab is the shared embedding space) ──
        self.token_to_id: Dict[str, int] = {
            t: i for i, t in enumerate(self.full_vocab)
        }
        # PAD id is the last slot — append if not already present.
        if "<pad>" not in self.token_to_id:
            self.token_to_id["<pad>"] = len(self.full_vocab)
            self.full_vocab.append("<pad>")
        self.pad_id: int = self.token_to_id["<pad>"]
        vocab_size = len(self.full_vocab)
        self.vocab_size = vocab_size

        # ── Per-dim local vocab tables ──
        # token_id -> local index, for each dim. Useful for training targets.
        self.dim_token_to_local: Dict[str, Dict[str, int]] = {}
        self.dim_local_to_global: Dict[str, List[int]] = {}
        for dim, toks in self.dim_vocab.items():
            self.dim_token_to_local[dim] = {t: i for i, t in enumerate(toks)}
            self.dim_local_to_global[dim] = [
                self.token_to_id[t] for t in toks if t in self.token_to_id
            ]

        # ── Embedding + positional ──
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=self.pad_id
        )
        self.pos_encoding = _SinusoidalPositionEncoding(max_history, hidden_dim)

        # Dim-tag embedding lets the encoder know which dim each history
        # token belongs to (LOCATION, ACTION, ...). +1 for "unknown/pad".
        self.dim_tag_embedding = nn.Embedding(
            len(self.DIM_ORDER) + 1, hidden_dim,
            padding_idx=len(self.DIM_ORDER),
        )
        self.dim_pad_id = len(self.DIM_ORDER)
        self.dim_to_tag_id: Dict[str, int] = {
            d: i for i, d in enumerate(self.DIM_ORDER)
        }

        # ── Encoder ──
        ffn_dim = ffn_dim or (hidden_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Per-dim conditioning ──
        # For each dim N we build an input vector:
        #   [history_summary | player_card_embed | prior_dim_embeds...]
        # where prior_dim_embeds is a fixed-length concat: for every dim in
        # DIM_ORDER[:N] we use the actual (or teacher-forced) embedding,
        # and zero vectors for dims not yet emitted.
        # The head then projects this back down to hidden_dim and produces
        # local-vocab logits.
        # Per-dim "slot" embedding so each emitted dim contributes a
        # distinguishable signal regardless of token (helps when two dims
        # share the same token id).
        self.dim_slot_embedding = nn.Embedding(len(self.DIM_ORDER), hidden_dim)

        # Shared context fuser: collapses (history_summary, player_card,
        # sum-of-prior-dim-embeds) into a single hidden_dim vector before
        # the per-dim head. This keeps total params well under the budget
        # because each head is just hidden_dim -> |dim_vocab|.
        # We sum the prior-dim embeddings rather than concatenate; the
        # dim_slot_embedding gives each contribution a distinct signature
        # so order/identity is preserved.
        self.context_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.heads = nn.ModuleDict()
        for dim in self.DIM_ORDER:
            local_size = len(self.dim_vocab[dim])
            self.heads[dim] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, local_size),
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _encode_history(
        self,
        history_tokens: torch.Tensor,   # (B, T)
        history_dims: torch.Tensor,     # (B, T)
    ) -> torch.Tensor:
        """Return (B, hidden_dim) summary of history."""
        B, T = history_tokens.shape
        T = min(T, self.max_history)
        history_tokens = history_tokens[:, :T]
        history_dims = history_dims[:, :T]

        tok_emb = self.token_embedding(history_tokens)         # (B, T, D)
        dim_emb = self.dim_tag_embedding(history_dims)         # (B, T, D)
        pos_emb = self.pos_encoding(T).to(tok_emb.device)      # (1, T, D)

        x = tok_emb + dim_emb + pos_emb                        # (B, T, D)

        pad_mask = history_tokens == self.pad_id               # (B, T)
        # If a row is fully padded the mask collapses encoder; guard with
        # a "all-true row" -> mark first as not-pad to keep softmax stable.
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 0] = False

        encoded = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, T, D)

        # Masked mean pool over non-pad positions.
        mask_f = (~pad_mask).float().unsqueeze(-1)             # (B, T, 1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        summary = (encoded * mask_f).sum(dim=1) / denom        # (B, D)
        return self.layer_norm(summary)

    def _build_prior_dims_block(
        self,
        prior_token_ids: Dict[str, torch.Tensor],   # dim -> (B,) global ids or None
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sum (token_emb + slot_emb) over all already-emitted dims.

        For each dim in DIM_ORDER that has been emitted, contribute
        `token_embedding(id) + dim_slot_embedding(dim_idx)`. Dims not yet
        emitted contribute nothing. Returns (B, hidden_dim).
        """
        out = torch.zeros(batch_size, self.hidden_dim, device=device)
        for i, dim in enumerate(self.DIM_ORDER):
            ids = prior_token_ids.get(dim)
            if ids is None:
                continue
            ids = ids.to(device)
            tok_emb = self.token_embedding(ids)                    # (B, D)
            slot_emb = self.dim_slot_embedding(
                torch.full((batch_size,), i, dtype=torch.long, device=device)
            )                                                      # (B, D)
            out = out + tok_emb + slot_emb
        return out                                                 # (B, D)

    def _local_to_global(self, dim: str, local_idx: int) -> int:
        return self.dim_local_to_global[dim][local_idx]

    # ─── Training forward (teacher-forced, parallel) ─────────────────────
    def forward(
        self,
        history_tokens: torch.Tensor,
        history_dims: torch.Tensor,
        player_card_idx: torch.Tensor,
        target_scene: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training pass.

        Args
        ----
        history_tokens : (B, T) global token ids for prior turns' tokens.
        history_dims   : (B, T) dim-tag ids for those tokens (use
                         `self.dim_pad_id` for pad/unknown).
        player_card_idx: (B,) global token id of the current player card.
        target_scene   : dict[dim_name -> (B,)] of GLOBAL token ids for
                         each dim of this scene. Required for training.

        Returns
        -------
        dict[dim_name -> (B, |dim_vocab[dim]|)] logits over the dim's local
        vocab.
        """
        B = history_tokens.shape[0]
        device = history_tokens.device

        history_summary = self._encode_history(history_tokens, history_dims)  # (B, D)
        player_emb = self.token_embedding(player_card_idx.to(device))         # (B, D)

        if target_scene is None:
            raise ValueError(
                "forward() requires target_scene for teacher-forced training. "
                "Use predict_scene() for inference."
            )

        # Build prior_token_ids dict that, for each dim N, carries the
        # ground-truth ids for DIM_ORDER[:N] and None thereafter.
        # We compute one head pass per dim (cheap; each is a tiny MLP) so
        # the conditioning is exactly causal in dim order.
        out: Dict[str, torch.Tensor] = {}
        prior_ids: Dict[str, torch.Tensor] = {}
        for dim in self.DIM_ORDER:
            prior_block = self._build_prior_dims_block(
                prior_ids, batch_size=B, device=device,
            )                                                              # (B, D)
            fused = self.context_fuser(
                torch.cat([history_summary, player_emb, prior_block], dim=1)
            )                                                              # (B, D)
            logits = self.heads[dim](fused)                                # (B, |dim_vocab|)
            out[dim] = logits

            # Teacher-force: feed the ground-truth token id forward.
            if dim in target_scene:
                prior_ids[dim] = target_scene[dim].to(device)
            else:
                # Should not happen during full-scene training; guard anyway.
                prior_ids[dim] = torch.full(
                    (B,), self.pad_id, dtype=torch.long, device=device,
                )

        return out

    # ─── Inference (sequential, masked) ──────────────────────────────────
    @torch.no_grad()
    def predict_scene(
        self,
        history: Dict[str, torch.Tensor],   # {"tokens": (T,), "dims": (T,)}
        player_card: int,                   # global token id
        constraint_mask: Any,               # ConstraintMask instance (or None)
        game_state: Dict[str, Any],
        temperature: float = 0.1,
    ) -> Dict[str, str]:
        """Inference. Returns Dict[dim_name -> token_id_str].

        Sequential conditioning: each emitted dim feeds back into the next
        dim's head input. After raw logits per dim:
          1. Restrict to that dim's vocab (already implicit in head shape).
          2. Apply hard-mask from `constraint_mask.applicable_for_dim(...)`.
          3. Sample with temperature.
        """
        device = next(self.parameters()).device

        tokens = history.get("tokens")
        dims = history.get("dims")
        if tokens is None or dims is None:
            tokens = torch.tensor([self.pad_id], dtype=torch.long, device=device)
            dims = torch.tensor([self.dim_pad_id], dtype=torch.long, device=device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            dims = dims.unsqueeze(0)
        tokens = tokens.to(device)
        dims = dims.to(device)

        history_summary = self._encode_history(tokens, dims)           # (1, D)
        player_emb = self.token_embedding(
            torch.tensor([player_card], dtype=torch.long, device=device)
        )                                                              # (1, D)

        scene_so_far: Dict[str, str] = {}
        prior_ids: Dict[str, torch.Tensor] = {}

        for dim in self.DIM_ORDER:
            prior_block = self._build_prior_dims_block(
                prior_ids, batch_size=1, device=device,
            )
            fused = self.context_fuser(
                torch.cat([history_summary, player_emb, prior_block], dim=1)
            )
            logits = self.heads[dim](fused).squeeze(0)                 # (|dim_vocab|,)

            # ── Hard-mask: restrict to constraint-allowed token ids ──
            allowed_token_ids: Optional[set] = None
            if constraint_mask is not None and hasattr(
                constraint_mask, "applicable_for_dim"
            ):
                try:
                    allowed_token_ids = constraint_mask.applicable_for_dim(
                        dim, scene_so_far, game_state,
                    )
                except Exception:
                    allowed_token_ids = None

            mask = torch.ones_like(logits, dtype=torch.bool)
            if allowed_token_ids is not None:
                # Map dim's local indices to "allowed?" booleans.
                vocab_list = self.dim_vocab[dim]
                allowed_mask = torch.tensor(
                    [tok in allowed_token_ids for tok in vocab_list],
                    dtype=torch.bool, device=device,
                )
                if allowed_mask.any():
                    mask &= allowed_mask
                # If the mask zeros everything (impossible state), fall
                # back to full vocab to avoid NaNs — the scene will then
                # fail validation downstream.

            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))

            # ── Sample ──
            if temperature <= 0:
                local_idx = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                if torch.isnan(probs).any() or probs.sum() <= 0:
                    local_idx = int(torch.argmax(logits).item())
                else:
                    local_idx = int(torch.multinomial(probs, num_samples=1).item())

            chosen_token = self.dim_vocab[dim][local_idx]
            scene_so_far[dim] = chosen_token
            global_id = self.token_to_id.get(chosen_token, self.pad_id)
            prior_ids[dim] = torch.tensor(
                [global_id], dtype=torch.long, device=device,
            )

        return scene_so_far

    # ─── Diagnostics ─────────────────────────────────────────────────────
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            out[top] = out.get(top, 0) + p.numel()
        return out


# ─── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    HERE = Path(__file__).resolve()
    PROJECT_ROOT = HERE.parents[3]
    DIMS_PATH = (
        PROJECT_ROOT / "living_tales" / "trainer" / "cases"
        / "amber_cipher" / "dimensions.json"
    )

    with open(DIMS_PATH) as f:
        dims_json = json.load(f)

    dim_vocab: Dict[str, List[str]] = {
        d["name"]: list(d["vocab"]) for d in dims_json["dimensions"]
    }
    # full vocab = union of all per-dim vocabs (preserve order, de-dup).
    seen = set()
    full_vocab: List[str] = []
    for d in StructuredSceneTransformer.DIM_ORDER:
        for tok in dim_vocab[d]:
            if tok not in seen:
                seen.add(tok)
                full_vocab.append(tok)

    model = StructuredSceneTransformer(
        dim_vocab=dim_vocab,
        full_vocab=full_vocab,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        max_history=80,
    )
    model.eval()

    n_params = model.num_parameters()
    print(f"[OK] StructuredSceneTransformer built. Total params: {n_params:,}")
    print("[OK] Param breakdown:")
    for k, v in sorted(model.parameter_breakdown().items(), key=lambda x: -x[1]):
        print(f"     {k:24s} {v:>10,}")
    assert n_params < 1_000_000, f"param count {n_params} exceeds 1M budget"

    # Dummy forward pass.
    B = 2
    T = 16
    history_tokens = torch.randint(0, len(full_vocab), (B, T))
    history_dims = torch.randint(0, len(StructuredSceneTransformer.DIM_ORDER), (B, T))
    player_card_idx = torch.randint(0, len(full_vocab), (B,))
    target_scene = {
        dim: torch.tensor(
            [model.token_to_id[dim_vocab[dim][0]]] * B, dtype=torch.long,
        )
        for dim in StructuredSceneTransformer.DIM_ORDER
    }
    out = model(history_tokens, history_dims, player_card_idx, target_scene)

    print("[OK] Forward pass shapes:")
    for dim in StructuredSceneTransformer.DIM_ORDER:
        expected = len(dim_vocab[dim])
        actual = tuple(out[dim].shape)
        ok = actual == (B, expected)
        print(f"     {dim:14s} -> {actual}  (expected (B={B}, {expected})) {'OK' if ok else 'FAIL'}")
        assert ok, f"shape mismatch on {dim}"

    # Sample two dims explicitly.
    print(f"[OK] LOCATION logits sample (B=0): shape={tuple(out['LOCATION'][0].shape)}, "
          f"vocab_size={len(dim_vocab['LOCATION'])}")
    print(f"[OK] BEAT logits sample (B=0): shape={tuple(out['BEAT'][0].shape)}, "
          f"vocab_size={len(dim_vocab['BEAT'])}")

    # Inference dry-run (no constraint mask).
    scene = model.predict_scene(
        history={"tokens": history_tokens[0], "dims": history_dims[0]},
        player_card=int(player_card_idx[0].item()),
        constraint_mask=None,
        game_state={"previous_locations": [], "visited_locations": set(),
                    "scene_index": 0, "convergence_dims": [0.0, 0.0, 0.0],
                    "game_turn": 1, "last_player_card": None},
        temperature=0.5,
    )
    print(f"[OK] predict_scene returned {len(scene)} dims:")
    for dim, tok in scene.items():
        print(f"     {dim:14s} -> {tok}")
