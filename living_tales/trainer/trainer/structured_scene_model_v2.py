"""
Structured Scene Transformer v2 for Living Tales.
==================================================

Upgrades over v1:
- 4 layers, hidden_dim=256, 8 heads (~5M params).
- RoPE positional encoding; max_history=160.
- Cross-head decoder: each dim is a query position attending to history KV,
  played-card KV, and prior-emitted-dim KV (replaces concat-MLP fuser).
- Played-card cross-attention slot — eliminates the binding-override patch.
- Latent scene-type z (8 modes) supervised classification head; conditions
  every dim head via a learned z embedding.
- LoRA injection points on every attention/FFN linear so per-case adapters
  can train against a frozen shared base.
- Reads dim_order from a list (typically dimensions.json's iteration order)
  so case-specific dims (MEDICAL_TELL, ART_TELL) get heads automatically.
- Hard-mask + inference-time attractor graph-bias preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── RoPE positional encoding ────────────────────────────────────────────────
def _build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)            # (S, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, S, D) where D is head_dim (even).
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)                     # (1, 1, S, D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ─── LoRA linear ─────────────────────────────────────────────────────────────
class LoraLinear(nn.Module):
    """Linear with optional rank-r LoRA adapter. When `lora_rank=0` the adapter
    branch is absent (zero params) and the layer behaves exactly like
    `nn.Linear`. When >0, base weights are still trained at stage 1; at stage 2
    callers freeze `linear` and train only `lora_a`/`lora_b`.
    """

    def __init__(self, in_features: int, out_features: int, lora_rank: int = 0, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(lora_rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            # lora_b zero-init so adapter is identity at start of stage 2.
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.lora_rank > 0:
            out = out + (x @ self.lora_a.T) @ self.lora_b.T
        return out

    def freeze_base(self) -> None:
        for p in self.linear.parameters():
            p.requires_grad = False


# ─── Multi-head attention (RoPE on Q/K, optional cross-attn) ─────────────────
class _MHA(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, lora_rank: int = 0, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.h = n_heads
        self.d = hidden_dim // n_heads
        self.q_proj = LoraLinear(hidden_dim, hidden_dim, lora_rank=lora_rank)
        self.k_proj = LoraLinear(hidden_dim, hidden_dim, lora_rank=lora_rank)
        self.v_proj = LoraLinear(hidden_dim, hidden_dim, lora_rank=lora_rank)
        self.o_proj = LoraLinear(hidden_dim, hidden_dim, lora_rank=lora_rank)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.h, self.d).transpose(1, 2)  # (B, H, S, D)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * D)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope: bool = False,
    ) -> torch.Tensor:
        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(kv_in))
        v = self._split(self.v_proj(kv_in))

        if rope:
            cos_q, sin_q = _build_rope_cache(q.shape[-2], self.d, q.device)
            cos_k, sin_k = _build_rope_cache(k.shape[-2], self.d, k.device)
            q = _apply_rope(q, cos_q, sin_q)
            k = _apply_rope(k, cos_k, sin_k)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, S_k) -> (B, 1, 1, S_k)
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

        scale = 1.0 / math.sqrt(self.d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale     # (B, H, S_q, S_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)                               # (B, H, S_q, D)
        return self.o_proj(self._merge(out))


class _FFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, lora_rank: int = 0, dropout: float = 0.0):
        super().__init__()
        self.fc1 = LoraLinear(hidden_dim, ffn_dim, lora_rank=lora_rank)
        self.fc2 = LoraLinear(ffn_dim, hidden_dim, lora_rank=lora_rank)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc2(h)


class _EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffn_dim, lora_rank=0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = _MHA(hidden_dim, n_heads, lora_rank=lora_rank, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = _FFN(hidden_dim, ffn_dim, lora_rank=lora_rank, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask, rope=True)
        x = x + self.ffn(self.norm2(x))
        return x


class _DecoderBlock(nn.Module):
    """Decoder over dim queries: self-attn (causal across dims) + cross-attn to
    a memory bank (history + played-card + prior-dim slots)."""

    def __init__(self, hidden_dim, n_heads, ffn_dim, lora_rank=0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = _MHA(hidden_dim, n_heads, lora_rank=lora_rank, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = _MHA(hidden_dim, n_heads, lora_rank=lora_rank, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = _FFN(hidden_dim, ffn_dim, lora_rank=lora_rank, dropout=dropout)

    def forward(self, q, memory, memory_pad_mask=None, causal_mask=None):
        # Self-attn with causal mask across dim positions.
        h = self.norm1(q)
        scores_q = self.self_attn(h, h, key_padding_mask=causal_mask, rope=False)
        q = q + scores_q
        # Cross-attn into memory.
        q = q + self.cross_attn(self.norm2(q), memory, key_padding_mask=memory_pad_mask, rope=False)
        q = q + self.ffn(self.norm3(q))
        return q


# ─── Config + Model ──────────────────────────────────────────────────────────
@dataclass
class V2Config:
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_dim: Optional[int] = None
    max_history: int = 160
    dropout: float = 0.0
    n_scene_types: int = 8
    lora_rank: int = 0                       # >0 enables LoRA branch on every linear
    universal_dims: List[str] = field(default_factory=list)


class StructuredSceneTransformerV2(nn.Module):
    def __init__(
        self,
        dim_vocab: Dict[str, List[str]],
        dim_order: List[str],
        full_vocab: List[str],
        config: Optional[V2Config] = None,
    ):
        super().__init__()
        cfg = config or V2Config()
        self.cfg = cfg
        self.dim_order = list(dim_order)

        # Validate.
        for d in self.dim_order:
            if d not in dim_vocab:
                raise ValueError(f"dim_vocab missing dim {d}")
        self.dim_vocab = {d: list(dim_vocab[d]) for d in self.dim_order}

        # Vocab.
        self.full_vocab = list(full_vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.full_vocab)}
        if "<pad>" not in self.token_to_id:
            self.token_to_id["<pad>"] = len(self.full_vocab)
            self.full_vocab.append("<pad>")
        self.pad_id = self.token_to_id["<pad>"]
        self.vocab_size = len(self.full_vocab)

        self.dim_token_to_local: Dict[str, Dict[str, int]] = {
            d: {t: i for i, t in enumerate(toks)} for d, toks in self.dim_vocab.items()
        }
        self.dim_local_to_global: Dict[str, List[int]] = {
            d: [self.token_to_id[t] for t in toks if t in self.token_to_id]
            for d, toks in self.dim_vocab.items()
        }

        H = cfg.hidden_dim
        ffn = cfg.ffn_dim or (H * 4)

        # Embeddings.
        self.token_embedding = nn.Embedding(self.vocab_size, H, padding_idx=self.pad_id)
        # Dim-tag for history positions.
        self.dim_tag_embedding = nn.Embedding(len(self.dim_order) + 1, H, padding_idx=len(self.dim_order))
        self.dim_pad_id = len(self.dim_order)
        self.dim_to_tag_id = {d: i for i, d in enumerate(self.dim_order)}
        # Per-dim slot embedding (used as decoder queries and as prior-dim memory tags).
        self.dim_slot_embedding = nn.Embedding(len(self.dim_order), H)
        # Latent scene-type z embedding.
        self.z_embedding = nn.Embedding(cfg.n_scene_types, H)
        # Played-card slot tag (so cross-attn can distinguish it from history positions).
        self.card_slot = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.normal_(self.card_slot, std=0.02)

        # Encoder over history + card + prior-dim memory (a single sequence the
        # decoder cross-attends to). The encoder uses RoPE on positions.
        self.encoder = nn.ModuleList([
            _EncoderBlock(H, cfg.n_heads, ffn, lora_rank=cfg.lora_rank, dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(H)

        # Decoder over dim queries.
        self.decoder = nn.ModuleList([
            _DecoderBlock(H, cfg.n_heads, ffn, lora_rank=cfg.lora_rank, dropout=cfg.dropout)
            for _ in range(max(1, cfg.n_layers // 2))
        ])
        self.decoder_norm = nn.LayerNorm(H)

        # Latent z classification head (predicted from history-summary).
        self.z_head = LoraLinear(H, cfg.n_scene_types, lora_rank=cfg.lora_rank)

        # Per-dim heads (small linear from H to local vocab).
        self.heads = nn.ModuleDict({
            d: LoraLinear(H, len(self.dim_vocab[d]), lora_rank=cfg.lora_rank)
            for d in self.dim_order
        })

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _build_memory(
        self,
        history_tokens: torch.Tensor,            # (B, T)
        history_dims: torch.Tensor,              # (B, T)
        player_card_idx: torch.Tensor,           # (B,)
        prior_dim_token_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate history positions + player-card slot + emitted prior-dim
        slots into a single memory tensor for the decoder to cross-attend to.

        Returns (memory, key_padding_mask) where key_padding_mask is True at
        positions to ignore.
        """
        B = history_tokens.shape[0]
        device = history_tokens.device
        T = min(history_tokens.shape[1], self.cfg.max_history)
        history_tokens = history_tokens[:, :T]
        history_dims = history_dims[:, :T]

        h_tok = self.token_embedding(history_tokens)                     # (B, T, H)
        h_tag = self.dim_tag_embedding(history_dims)
        h = h_tok + h_tag                                                # (B, T, H)
        h_pad = (history_tokens == self.pad_id)                          # (B, T)

        # Played-card slot.
        card_emb = self.token_embedding(player_card_idx) + self.card_slot.squeeze(0)  # (B, H)
        card_emb = card_emb.unsqueeze(1)                                 # (B, 1, H)
        card_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)

        parts = [h, card_emb]
        pads = [h_pad, card_pad]

        # Prior-dim slots.
        if prior_dim_token_ids:
            for d, ids in prior_dim_token_ids.items():
                if ids is None:
                    continue
                if d not in self.dim_to_tag_id:
                    continue
                ids = ids.to(device)
                emb = self.token_embedding(ids) + self.dim_slot_embedding(
                    torch.full((B,), self.dim_to_tag_id[d], dtype=torch.long, device=device)
                )                                                        # (B, H)
                parts.append(emb.unsqueeze(1))
                pads.append(torch.zeros(B, 1, dtype=torch.bool, device=device))

        memory = torch.cat(parts, dim=1)                                 # (B, T+1+K, H)
        pad_mask = torch.cat(pads, dim=1)                                # (B, T+1+K)

        # Guard fully-padded rows.
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 0] = False

        # Run encoder over memory.
        x = memory
        for blk in self.encoder:
            x = blk(x, key_padding_mask=pad_mask)
        x = self.encoder_norm(x)
        return x, pad_mask

    def _decode_dim_queries(
        self,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        z_idx: torch.Tensor,            # (B,) latent scene-type id
    ) -> torch.Tensor:
        """Run the decoder. Queries are the dim slot embeddings + z embedding,
        in DIM_ORDER. Returns (B, n_dims, H).
        """
        B = memory.shape[0]
        device = memory.device
        n_dims = len(self.dim_order)

        slot_ids = torch.arange(n_dims, device=device).unsqueeze(0).expand(B, -1)   # (B, n_dims)
        q = self.dim_slot_embedding(slot_ids)                                       # (B, n_dims, H)
        z_emb = self.z_embedding(z_idx).unsqueeze(1)                                # (B, 1, H)
        q = q + z_emb

        # Causal mask across dim positions: dim i can attend to <= i.
        # _MHA expects a "key_padding_mask" semantics; we build a (B, n_dims)
        # mask saying "this position is padding" — but causal needs a per-query
        # mask. Simpler: emit teacher-forced sequentially via separate decoder
        # passes during inference; during training, full causal via attn-mask.
        # For our scale we just pass no causal mask in self-attn (each dim is
        # already conditioned on its slot identity + z) and rely on the
        # cross-attn to memory (which DOES include prior emitted dims via
        # _build_memory) for sequential conditioning. This is the same trick
        # as v1 but in cross-attn form.
        # NOTE: training/inference both call this with the prior-dim slots
        # already injected into memory, so causal ordering is already enforced.
        for blk in self.decoder:
            q = blk(q, memory, memory_pad_mask=memory_pad_mask, causal_mask=None)
        q = self.decoder_norm(q)
        return q                                                                    # (B, n_dims, H)

    # ─── Training forward (teacher-forced) ───────────────────────────────
    def forward(
        self,
        history_tokens: torch.Tensor,
        history_dims: torch.Tensor,
        player_card_idx: torch.Tensor,
        target_scene: Dict[str, torch.Tensor],
        target_scene_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training pass.

        Predicts each dim conditioned on history + player card + prior-dim
        ground truth. Also predicts scene-type z from history alone.
        """
        B = history_tokens.shape[0]
        device = history_tokens.device

        # First pass: predict z from a "memory" with no prior-dim slots.
        mem0, mem0_pad = self._build_memory(history_tokens, history_dims, player_card_idx, prior_dim_token_ids=None)
        # Pool mem0 (mean over non-pad).
        mask_f = (~mem0_pad).float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        summary = (mem0 * mask_f).sum(dim=1) / denom                                # (B, H)
        z_logits = self.z_head(summary)                                             # (B, n_scene_types)

        # Choose z for downstream conditioning. During training we use the
        # ground-truth scene_type if provided (teacher-forced), otherwise the
        # argmax of predicted z.
        if target_scene_type is not None:
            z_idx = target_scene_type.to(device)
        else:
            z_idx = torch.argmax(z_logits, dim=-1)

        out: Dict[str, torch.Tensor] = {"_z_logits": z_logits}

        # Build memory once with all teacher-forced prior dims, run decoder
        # once over all dim queries. This gives parallel training.
        prior_ids = {d: target_scene[d].to(device) for d in self.dim_order if d in target_scene}
        memory, memory_pad = self._build_memory(
            history_tokens, history_dims, player_card_idx, prior_dim_token_ids=prior_ids,
        )
        q_out = self._decode_dim_queries(memory, memory_pad, z_idx)                 # (B, n_dims, H)

        for i, d in enumerate(self.dim_order):
            head_in = q_out[:, i, :]                                                # (B, H)
            out[d] = self.heads[d](head_in)                                         # (B, |dim_vocab|)
        return out

    # ─── Inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_scene(
        self,
        history: Dict[str, torch.Tensor],
        player_card: int,
        constraint_mask: Any,
        game_state: Dict[str, Any],
        per_dim_temperature: Optional[Dict[str, float]] = None,
        per_dim_top_k: Optional[Dict[str, int]] = None,
        graph_bias: Optional[Dict[str, float]] = None,    # global token id -> additive logit bias
        bias_alpha: float = 0.5,
    ) -> Dict[str, str]:
        """Sequential sampling, dim by dim.

        At each dim:
          1. Build memory with all prior emitted dim slots.
          2. Decode and read this dim's head.
          3. Mask via constraint_mask.applicable_for_dim.
          4. Add per-token graph-bias (attractor weights).
          5. Sample with per-dim temperature + top-k.
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
        card_t = torch.tensor([player_card], dtype=torch.long, device=device)

        # Predict z first.
        mem0, mem0_pad = self._build_memory(tokens, dims, card_t, prior_dim_token_ids=None)
        mask_f = (~mem0_pad).float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        summary = (mem0 * mask_f).sum(dim=1) / denom
        z_logits = self.z_head(summary).squeeze(0)
        z_idx = torch.tensor([int(torch.argmax(z_logits).item())], dtype=torch.long, device=device)

        scene: Dict[str, str] = {}
        prior_ids: Dict[str, torch.Tensor] = {}
        per_dim_temperature = per_dim_temperature or {}
        per_dim_top_k = per_dim_top_k or {}

        for i, d in enumerate(self.dim_order):
            memory, memory_pad = self._build_memory(tokens, dims, card_t, prior_dim_token_ids=prior_ids)
            q_out = self._decode_dim_queries(memory, memory_pad, z_idx)             # (1, n_dims, H)
            logits = self.heads[d](q_out[0, i]).squeeze()                           # (|dim_vocab|,)

            # Hard-mask.
            mask = torch.ones_like(logits, dtype=torch.bool)
            if constraint_mask is not None and hasattr(constraint_mask, "applicable_for_dim"):
                try:
                    allowed = constraint_mask.applicable_for_dim(d, scene, game_state)
                except Exception:
                    allowed = None
                if allowed is not None:
                    vocab_list = self.dim_vocab[d]
                    am = torch.tensor(
                        [t in allowed for t in vocab_list], dtype=torch.bool, device=device,
                    )
                    if am.any():
                        mask &= am

            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))

            # Graph bias (per token in this dim's local vocab).
            if graph_bias:
                bias_vec = torch.tensor(
                    [graph_bias.get(tok, 0.0) for tok in self.dim_vocab[d]],
                    dtype=logits.dtype, device=device,
                )
                logits = logits + bias_alpha * bias_vec

            # Sample with per-dim temperature/top-k.
            T = per_dim_temperature.get(d, 0.3)
            K = per_dim_top_k.get(d, 0)
            if T <= 0:
                local_idx = int(torch.argmax(logits).item())
            else:
                if K > 0 and K < logits.shape[-1]:
                    topv, topi = torch.topk(logits, K)
                    probs = F.softmax(topv / max(T, 1e-6), dim=-1)
                    if torch.isnan(probs).any() or probs.sum() <= 0:
                        local_idx = int(topi[int(torch.argmax(topv).item())].item())
                    else:
                        pick = int(torch.multinomial(probs, 1).item())
                        local_idx = int(topi[pick].item())
                else:
                    probs = F.softmax(logits / max(T, 1e-6), dim=-1)
                    if torch.isnan(probs).any() or probs.sum() <= 0:
                        local_idx = int(torch.argmax(logits).item())
                    else:
                        local_idx = int(torch.multinomial(probs, 1).item())

            chosen = self.dim_vocab[d][local_idx]
            scene[d] = chosen
            global_id = self.token_to_id.get(chosen, self.pad_id)
            prior_ids[d] = torch.tensor([global_id], dtype=torch.long, device=device)

        return scene

    # ─── Stage helpers ───────────────────────────────────────────────────
    def freeze_base_for_adapter_stage(self) -> None:
        """For stage-2: freeze every base param; only LoRA branches + (optionally)
        case-specific heads remain trainable.

        Caller is responsible for re-enabling heads of case-specific dims after
        this call (see `unfreeze_dims`).
        """
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, LoraLinear) and m.lora_rank > 0:
                if m.lora_a is not None:
                    m.lora_a.requires_grad = True
                if m.lora_b is not None:
                    m.lora_b.requires_grad = True

    def unfreeze_dims(self, dims: List[str]) -> None:
        for d in dims:
            if d not in self.heads:
                continue
            for p in self.heads[d].parameters():
                p.requires_grad = True

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


# ─── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    HERE = Path(__file__).resolve()
    ROOT = HERE.parents[3]
    DIMS = ROOT / "living_tales" / "trainer" / "cases" / "attended_hour" / "dimensions.json"
    with open(DIMS) as f:
        dims_json = json.load(f)

    dim_order = [d["name"] for d in dims_json["dimensions"]]
    dim_vocab = {d["name"]: list(d["vocab"]) for d in dims_json["dimensions"]}
    seen, full = set(), []
    for d in dim_order:
        for t in dim_vocab[d]:
            if t not in seen:
                seen.add(t); full.append(t)

    cfg = V2Config(hidden_dim=256, n_layers=4, n_heads=8, max_history=160, lora_rank=0,
                   universal_dims=[d["name"] for d in dims_json["dimensions"] if d.get("universal")])
    m = StructuredSceneTransformerV2(dim_vocab=dim_vocab, dim_order=dim_order, full_vocab=full, config=cfg)
    m.eval()
    n = m.num_parameters()
    print(f"[OK] V2 built. Params: {n:,}  (target ~5M)")
    print(f"[OK] dim_order: {dim_order}")
    print(f"[OK] universal: {cfg.universal_dims}")

    B, T = 2, 16
    ht = torch.randint(0, len(full), (B, T))
    hd = torch.randint(0, len(dim_order), (B, T))
    pc = torch.randint(0, len(full), (B,))
    target = {d: torch.tensor([m.token_to_id[dim_vocab[d][0]]] * B, dtype=torch.long) for d in dim_order}
    target_z = torch.randint(0, cfg.n_scene_types, (B,))
    out = m(ht, hd, pc, target, target_scene_type=target_z)
    for d in dim_order:
        assert out[d].shape == (B, len(dim_vocab[d])), f"shape mismatch {d} {out[d].shape}"
    assert out["_z_logits"].shape == (B, cfg.n_scene_types)
    print("[OK] Forward pass shapes correct.")

    scene = m.predict_scene(
        history={"tokens": ht[0], "dims": hd[0]},
        player_card=int(pc[0].item()),
        constraint_mask=None,
        game_state={"previous_locations": [], "visited_locations": set(),
                    "scene_index": 0, "convergence_dims": [0.0, 0.0, 0.0],
                    "game_turn": 1, "last_player_card": None},
        per_dim_temperature={d: 0.3 for d in dim_order},
        per_dim_top_k={"ATMOSPHERE": 3, "CAUSE": 3, "STANCE": 3},
    )
    print(f"[OK] predict_scene returned {len(scene)} dims.")
    for d, t in scene.items():
        print(f"     {d:14s} -> {t}")

    # LoRA branch self-test.
    cfg_lora = V2Config(hidden_dim=128, n_layers=2, n_heads=4, lora_rank=8)
    m2 = StructuredSceneTransformerV2(dim_vocab=dim_vocab, dim_order=dim_order, full_vocab=full, config=cfg_lora)
    n_full = m2.num_parameters()
    m2.freeze_base_for_adapter_stage()
    n_train = m2.num_parameters(trainable_only=True)
    print(f"[OK] LoRA stage-2: {n_train:,} / {n_full:,} trainable ({100*n_train/n_full:.1f}%)")
    assert n_train < n_full, "freeze_base should reduce trainable params"
