"""
TrajectoryDatasetV2
====================

Drop-in replacement for v1 with:
- Configurable `dim_order` (read from dimensions.json) — supports per-case
  extension dims (MEDICAL_TELL, ART_TELL).
- Optional `universal_only` mode (stage-1 base pretrain): drops case-specific
  dim targets, leaving only universal-core supervision.
- History-truncation augmentation (30% of training pairs see history truncated
  to last 8 turns).
- Counterfactual-branch sampling: when present on a trajectory, branches are
  treated as additional training examples sharing the parent's prefix.
- Per-example `scene_type` label for the latent z head, plus `forbidden_dims`
  for contrastive training.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

# Sentinels — must match V2 model.
PAD_DIM_ID_OFFSET: int = 0  # callers pass len(dim_order); we treat that as pad.

# Default scene-type modes (must match V2Config.n_scene_types).
SCENE_TYPES: List[str] = [
    "transition", "cold_examination", "hot_confrontation",
    "revelation", "dead_end", "evasion", "breakthrough", "recap",
]
SCENE_TYPE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(SCENE_TYPES)}


def _strip_accuse(token: str) -> str:
    if isinstance(token, str) and token.startswith("ACCUSE:"):
        return token[len("ACCUSE:"):]
    return token


class TrajectoryDatasetV2(Dataset):
    """One example per turn across all trajectories.

    Args
    ----
    trajectories       : list of Trajectory dataclasses from trajectory_loader.
    dim_order          : ordered list of dim names (e.g. read from dimensions.json).
    dim_vocab          : dict[dim_name -> list of token ids].
    full_vocab_to_idx  : token id -> global vocab index.
    dim_vocab_to_idx   : dict[dim_name -> dict[token id -> local idx]].
    max_history        : history pad length.
    universal_dims     : if `universal_only=True`, supervision is restricted
                         to these dims. Required when `universal_only=True`.
    universal_only     : stage-1 base pretrain mode.
    truncate_history_p : fraction of examples that get history truncated to
                         last `truncate_to_turns` turns of tokens.
    truncate_to_turns  : how many recent turns to keep when truncating.
    include_branches   : whether to expand counterfactual branches into extra
                         training examples.
    rng                : numpy.random.Generator-like or None.
    """

    def __init__(
        self,
        trajectories: Sequence[Any],
        dim_order: List[str],
        dim_vocab: Dict[str, List[str]],
        full_vocab_to_idx: Dict[str, int],
        dim_vocab_to_idx: Dict[str, Dict[str, int]],
        max_history: int = 160,
        universal_dims: Optional[List[str]] = None,
        universal_only: bool = False,
        truncate_history_p: float = 0.3,
        truncate_to_turns: int = 8,
        include_branches: bool = True,
        seed: int = 0,
    ):
        if universal_only and not universal_dims:
            raise ValueError("universal_only=True requires universal_dims to be set")

        self.dim_order = list(dim_order)
        self.supervised_dims = (
            list(universal_dims) if universal_only else list(dim_order)
        )
        self.dim_vocab = dim_vocab
        self.full_vocab_to_idx = full_vocab_to_idx
        self.dim_vocab_to_idx = dim_vocab_to_idx
        self.max_history = max_history
        self.truncate_history_p = float(truncate_history_p)
        self.truncate_to_turns = int(truncate_to_turns)
        self.universal_only = universal_only
        self.rng = random.Random(seed)

        self.dim_pad_id = len(self.dim_order)
        self.player_card_dim_id = self.dim_pad_id
        self.opening_dim_id = self.dim_pad_id
        self.pad_dim_id = self.dim_pad_id

        # Per-dim fallback (a `*:none` token if present, else local 0).
        self._dim_fallback_local: Dict[str, int] = {}
        for d, toks in dim_vocab.items():
            fb = 0
            for i, t in enumerate(toks):
                if t.endswith(":none"):
                    fb = i
                    break
            self._dim_fallback_local[d] = fb

        # Approximate token count per turn — used to estimate "last N turns"
        # for history truncation. With dim_order length K plus a player-card
        # token, each turn contributes ~K+1 tokens.
        self._tokens_per_turn = len(self.dim_order) + 1

        self.examples: List[Dict[str, Any]] = []
        self._build(trajectories, include_branches=include_branches)

    # ── Lookups ──────────────────────────────────────────────────────────
    def _vocab_idx(self, token: str) -> int:
        token = _strip_accuse(token)
        return self.full_vocab_to_idx.get(token, 0)

    def _local_idx(self, dim: str, token: str) -> int:
        idx_map = self.dim_vocab_to_idx.get(dim, {})
        if token in idx_map:
            return idx_map[token]
        return self._dim_fallback_local.get(dim, 0)

    def _scene_type_idx(self, scene_type: Optional[str]) -> int:
        if scene_type and scene_type in SCENE_TYPE_TO_IDX:
            return SCENE_TYPE_TO_IDX[scene_type]
        # Default: "cold_examination" — a neutral mode. Trainer can mask the
        # z-loss for unannotated turns; until then this still trains heads.
        return SCENE_TYPE_TO_IDX["cold_examination"]

    # ── Construction ─────────────────────────────────────────────────────
    def _encode_scene_to_history(self, scene: Dict[str, str]) -> List[tuple]:
        out = []
        for slot_idx, d in enumerate(self.dim_order):
            tok = scene.get(d, "")
            if not tok:
                continue
            out.append((self._vocab_idx(tok), slot_idx))
        return out

    def _make_example(
        self,
        history: List[tuple],
        player_card: str,
        scene: Dict[str, str],
        scene_type: Optional[str],
        forbidden_dims: Optional[Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        # Optional augmentation: truncate history to last N turns ~30% of the time.
        hist = history
        if self.rng.random() < self.truncate_history_p and len(hist) > self.truncate_to_turns * self._tokens_per_turn:
            keep = self.truncate_to_turns * self._tokens_per_turn
            hist = hist[-keep:]
        if len(hist) > self.max_history:
            hist = hist[-self.max_history:]

        hist_tokens = [h[0] for h in hist]
        hist_dims = [h[1] for h in hist]

        # Target scene: only supervised_dims contribute training loss; for
        # unsupervised dims (universal_only stage-1), set target to fallback
        # AND record an `active_mask` so trainer can zero those losses.
        target_scene: Dict[str, torch.Tensor] = {}
        active_mask: Dict[str, torch.Tensor] = {}
        for d in self.dim_order:
            tok = scene.get(d, "")
            local = self._local_idx(d, tok) if tok else self._dim_fallback_local.get(d, 0)
            target_scene[d] = torch.tensor(local, dtype=torch.long)
            active_mask[d] = torch.tensor(
                1.0 if (d in self.supervised_dims and tok) else 0.0,
                dtype=torch.float,
            )

        # Forbidden tokens per dim → boolean mask in local-vocab space.
        forbidden_local: Dict[str, torch.Tensor] = {}
        if forbidden_dims:
            for d, toks in forbidden_dims.items():
                if d not in self.dim_vocab:
                    continue
                vec = torch.zeros(len(self.dim_vocab[d]), dtype=torch.bool)
                for t in toks:
                    if t in self.dim_vocab_to_idx.get(d, {}):
                        vec[self.dim_vocab_to_idx[d][t]] = True
                forbidden_local[d] = vec

        return {
            "history_tokens": torch.tensor(hist_tokens, dtype=torch.long),
            "history_dims": torch.tensor(hist_dims, dtype=torch.long),
            "player_card": torch.tensor(self._vocab_idx(player_card), dtype=torch.long),
            "target_scene": target_scene,
            "active_mask": active_mask,
            "scene_type": torch.tensor(self._scene_type_idx(scene_type), dtype=torch.long),
            "forbidden_local": forbidden_local,
        }

    def _build(self, trajectories: Sequence[Any], include_branches: bool):
        for traj in trajectories:
            opening_seq: List[tuple] = []
            for tok in getattr(traj, "opening", []) or []:
                opening_seq.append((self._vocab_idx(tok), self.opening_dim_id))

            history: List[tuple] = list(opening_seq)
            turns = list(getattr(traj, "turns", []) or [])
            sts = getattr(traj, "scene_type_sequence", None) or [None] * len(turns)

            anchor_histories: Dict[int, List[tuple]] = {}  # for branch expansion

            for i, turn in enumerate(turns):
                scene = getattr(turn, "scene", {}) or {}
                if not scene:
                    continue
                player_card = getattr(turn, "player_card", "") or ""
                scene_type = getattr(turn, "scene_type", None) or (sts[i] if i < len(sts) else None)
                forbidden = getattr(turn, "forbidden_dims", None)

                self.examples.append(self._make_example(
                    history, player_card, scene, scene_type, forbidden,
                ))

                # Cache history at this turn index for branch expansion.
                if include_branches:
                    anchor_histories[i] = list(history)

                # Roll history forward.
                history.append((self._vocab_idx(player_card), self.player_card_dim_id))
                history.extend(self._encode_scene_to_history(scene))

            # Counterfactual branches.
            if include_branches:
                for branch in getattr(traj, "counterfactual_branches", None) or []:
                    anchor = int(getattr(branch, "anchor_turn", 0))
                    if anchor not in anchor_histories:
                        continue
                    b_hist = list(anchor_histories[anchor])
                    for j, b_turn in enumerate(branch.turns):
                        b_scene = getattr(b_turn, "scene", {}) or {}
                        if not b_scene:
                            continue
                        b_card = getattr(b_turn, "player_card", "") or ""
                        b_st = getattr(b_turn, "scene_type", None)
                        b_fd = getattr(b_turn, "forbidden_dims", None)
                        self.examples.append(self._make_example(
                            b_hist, b_card, b_scene, b_st, b_fd,
                        ))
                        b_hist.append((self._vocab_idx(b_card), self.player_card_dim_id))
                        b_hist.extend(self._encode_scene_to_history(b_scene))

    # ── Protocol ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.examples[i]

    # ── Collate ──────────────────────────────────────────────────────────
    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_len = max(int(ex["history_tokens"].numel()) for ex in batch)
        max_len = max(max_len, 1)

        history_tokens = torch.zeros((B, max_len), dtype=torch.long)
        history_dims = torch.full((B, max_len), fill_value=self.pad_dim_id, dtype=torch.long)
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
            d: torch.stack([ex["target_scene"][d] for ex in batch], dim=0)
            for d in self.dim_order
        }
        active_mask = {
            d: torch.stack([ex["active_mask"][d] for ex in batch], dim=0)
            for d in self.dim_order
        }
        scene_type = torch.stack([ex["scene_type"] for ex in batch], dim=0)

        # Forbidden-local: dict[dim] -> (B, |dim_vocab|) bool mask. Default
        # all False if no example has it for that dim.
        forbidden_local: Dict[str, torch.Tensor] = {}
        for d in self.dim_order:
            V = len(self.dim_vocab[d])
            fl = torch.zeros((B, V), dtype=torch.bool)
            for i, ex in enumerate(batch):
                if d in ex.get("forbidden_local", {}):
                    fl[i] = ex["forbidden_local"][d]
            forbidden_local[d] = fl

        return {
            "history_tokens": history_tokens,
            "history_dims": history_dims,
            "padding_mask": padding_mask,
            "player_card": player_card,
            "target_scene": target_scene,
            "active_mask": active_mask,
            "scene_type": scene_type,
            "forbidden_local": forbidden_local,
        }


# ── Cross-case combiner ─────────────────────────────────────────────────────
def build_universal_corpus(
    case_trajectories: Dict[str, Sequence[Any]],
    dim_order: List[str],
    dim_vocab: Dict[str, List[str]],
    full_vocab_to_idx: Dict[str, int],
    dim_vocab_to_idx: Dict[str, Dict[str, int]],
    universal_dims: List[str],
    max_history: int = 160,
) -> "TrajectoryDatasetV2":
    """Build one TrajectoryDatasetV2 across ALL cases, with supervision
    restricted to universal_dims. The resulting dataset is what stage-1
    base pretrain trains on.

    NOTE: the caller is responsible for ensuring that `dim_vocab` and
    `full_vocab_to_idx` cover the union of all cases' tokens.
    """
    flat: List[Any] = []
    for trajs in case_trajectories.values():
        flat.extend(trajs)
    return TrajectoryDatasetV2(
        trajectories=flat,
        dim_order=dim_order,
        dim_vocab=dim_vocab,
        full_vocab_to_idx=full_vocab_to_idx,
        dim_vocab_to_idx=dim_vocab_to_idx,
        max_history=max_history,
        universal_dims=universal_dims,
        universal_only=True,
        truncate_history_p=0.3,
        include_branches=True,
    )


# ── Selftest ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path
    import sys
    ROOT = Path("/Users/juan-garassino/Code/005-products/010-more-than-words")
    sys.path.insert(0, str(ROOT))
    from living_tales.trainer.generator.trajectory_loader import TrajectoryLoader  # noqa: E402
    case = "amber_cipher"
    L = TrajectoryLoader(case, ROOT)
    trajs = L.load_all()

    dims_path = ROOT / "living_tales/trainer/cases" / case / "dimensions.json"
    with open(dims_path) as f:
        dims_json = json.load(f)
    dim_order = [d["name"] for d in dims_json["dimensions"]]
    dim_vocab = {d["name"]: list(d["vocab"]) for d in dims_json["dimensions"]}
    universal = [d["name"] for d in dims_json["dimensions"] if d.get("universal")]

    seen, full = set(), []
    for d in dim_order:
        for t in dim_vocab[d]:
            if t not in seen:
                seen.add(t); full.append(t)
    full_vocab_to_idx = {t: i for i, t in enumerate(full)}
    dim_vocab_to_idx = {d: {t: i for i, t in enumerate(toks)} for d, toks in dim_vocab.items()}

    ds = TrajectoryDatasetV2(
        trajs, dim_order, dim_vocab, full_vocab_to_idx, dim_vocab_to_idx,
        max_history=160, universal_dims=universal, universal_only=False,
        truncate_history_p=0.3,
    )
    print(f"[OK] {case}: {len(ds)} training examples (full-supervision).")

    ds_uni = TrajectoryDatasetV2(
        trajs, dim_order, dim_vocab, full_vocab_to_idx, dim_vocab_to_idx,
        max_history=160, universal_dims=universal, universal_only=True,
        truncate_history_p=0.3,
    )
    print(f"[OK] {case}: {len(ds_uni)} examples (universal_only).")
    print(f"[OK] supervised dims (universal_only): {ds_uni.supervised_dims}")

    # Check active_mask: in universal_only mode, non-universal dims should
    # have zero active_mask.
    if ds_uni.examples:
        ex = ds_uni.examples[0]
        for d in dim_order:
            am = float(ex["active_mask"][d].item())
            mark = "OK" if (am == 1.0) == (d in universal) else "FAIL"
            print(f"     active_mask[{d}] = {am}  ({mark})")

    # Collate sanity.
    batch = ds.collate([ds[i] for i in range(min(4, len(ds)))])
    print(f"[OK] collate OK: history_tokens.shape={tuple(batch['history_tokens'].shape)}, "
          f"scene_type.shape={tuple(batch['scene_type'].shape)}")
