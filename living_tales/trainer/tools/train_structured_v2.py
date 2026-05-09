"""
train_structured_v2.py
======================
Two-stage training for the V2 structured scene engine.

Stage 1 — base pretrain:
    python tools/train_structured_v2.py base \
        --cases amber_cipher,attended_hour,venetian_mirror \
        --epochs 80 --batch-size 32 --lr 1e-3

    Trains a single shared StructuredSceneTransformerV2 on the universal-core
    corpus across all cases (case-specific dims masked out via active_mask).
    Saves outputs/_base/base_universal.pt.

Stage 2 — per-case adapter + heads:
    python tools/train_structured_v2.py adapter \
        --case amber_cipher --epochs 150 --batch-size 16 --lr 5e-4 \
        --base-checkpoint outputs/_base/base_universal.pt \
        --lora-rank 8

    Loads the base, freezes everything except LoRA branches + case-specific
    heads, and overfits the case. Saves outputs/<case>/adapter.pt.

Losses
------
- Per-dim cross-entropy on supervised dims (active_mask=1).
- Diversity loss: lambda_div * mean batch entropy of each dim's logits — pulls
  per-dim distributions away from collapsed modes.
- Latent z classification loss on scene_type label.
- Optional contrastive loss using forbidden_local: pushes logits down at
  forbidden token positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
TRAINER_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(TRAINER_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from generator.trajectory_loader import TrajectoryLoader  # noqa: E402
from trainer.trajectory_dataset_v2 import TrajectoryDatasetV2  # noqa: E402
from trainer.structured_scene_model_v2 import (  # noqa: E402
    StructuredSceneTransformerV2,
    V2Config,
)


# ── Vocab assembly ──────────────────────────────────────────────────────────
def load_dims(case_id: str) -> Dict:
    p = REPO_ROOT / "living_tales/trainer/cases" / case_id / "dimensions.json"
    with open(p) as f:
        return json.load(f)


def load_case_data(case_ids: Sequence[str]):
    """Load trajectories + dimensions per case. Returns:
        case_trajs: dict[case -> list of Trajectory]
        per_case_dims: dict[case -> dimensions.json]
    """
    case_trajs = {}
    per_case_dims = {}
    for c in case_ids:
        L = TrajectoryLoader(c, REPO_ROOT)
        case_trajs[c] = L.load_all()
        per_case_dims[c] = load_dims(c)
    return case_trajs, per_case_dims


def build_union_vocab(per_case_dims: Dict[str, Dict]):
    """Merge dim_vocabs across cases into a union dim_vocab + token universe.

    The base model needs a vocab covering tokens from every case it sees.
    Per-case dims (MEDICAL_TELL in attended_hour, ART_TELL in venetian_mirror)
    are kept in dim_order — for cases that don't have them, the active_mask
    is zero so the head never trains on them at base stage.
    """
    union_dim_vocab: Dict[str, List[str]] = {}
    union_universal: List[str] = []
    dim_order: List[str] = []

    for case, dims_json in per_case_dims.items():
        for d in dims_json["dimensions"]:
            name = d["name"]
            if name not in union_dim_vocab:
                union_dim_vocab[name] = []
                dim_order.append(name)
                if d.get("universal"):
                    union_universal.append(name)
            for tok in d.get("vocab", []):
                if tok not in union_dim_vocab[name]:
                    union_dim_vocab[name].append(tok)

    # Token universe = ordered union over dims.
    seen = set()
    full_vocab: List[str] = []
    for d in dim_order:
        for t in union_dim_vocab[d]:
            if t not in seen:
                seen.add(t)
                full_vocab.append(t)

    full_vocab_to_idx = {t: i for i, t in enumerate(full_vocab)}
    dim_vocab_to_idx = {
        d: {t: i for i, t in enumerate(toks)}
        for d, toks in union_dim_vocab.items()
    }
    return dim_order, union_dim_vocab, union_universal, full_vocab, full_vocab_to_idx, dim_vocab_to_idx


# ── Loss components ─────────────────────────────────────────────────────────
def per_dim_ce_loss(
    logits: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    active_mask: Dict[str, torch.Tensor],
    dim_order: List[str],
) -> torch.Tensor:
    """Sum of cross-entropies, weighted by active_mask. active_mask is 0
    when a dim shouldn't be supervised (universal_only mode for case-specific
    dims, or when a target token wasn't authored)."""
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)
    for d in dim_order:
        if d not in logits or d not in target:
            continue
        loss_per = F.cross_entropy(logits[d], target[d], reduction="none")  # (B,)
        m = active_mask[d].to(loss_per.device)
        denom = m.sum().clamp_min(1.0)
        total = total + (loss_per * m).sum() / denom
    return total


def diversity_loss(logits: Dict[str, torch.Tensor], dim_order: List[str]) -> torch.Tensor:
    """Negative entropy averaged over dims — penalises spike-y / collapsed
    distributions across the batch."""
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)
    n = 0
    for d in dim_order:
        if d not in logits:
            continue
        # Aggregate softmax across batch then take entropy.
        probs = F.softmax(logits[d], dim=-1).mean(dim=0)         # (V,)
        probs = probs.clamp_min(1e-9)
        ent = -(probs * probs.log()).sum()
        total = total - ent                                       # we MINIMISE -entropy => maximise entropy
        n += 1
    return total / max(n, 1)


def contrastive_forbidden_loss(
    logits: Dict[str, torch.Tensor],
    forbidden: Dict[str, torch.Tensor],
    dim_order: List[str],
) -> torch.Tensor:
    """Push logits down at positions flagged forbidden_dims. Implemented as
    mean of (forbidden positions' softmax probability) — minimised pulls the
    forbidden mass to zero."""
    device = next(iter(logits.values())).device
    total = torch.tensor(0.0, device=device)
    n = 0
    for d in dim_order:
        if d not in logits or d not in forbidden:
            continue
        fb = forbidden[d].to(device)                              # (B, V)
        if not fb.any():
            continue
        probs = F.softmax(logits[d], dim=-1)                      # (B, V)
        contribution = (probs * fb.float()).sum(dim=-1).mean()
        total = total + contribution
        n += 1
    return total / max(n, 1) if n > 0 else total


def z_loss(z_logits: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(z_logits, z_target)


# ── Train loop ──────────────────────────────────────────────────────────────
def train_one_epoch(
    model: StructuredSceneTransformerV2,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dim_order: List[str],
    lambda_div: float = 0.05,
    lambda_z: float = 0.5,
    lambda_contrast: float = 0.2,
    device: str = "cpu",
):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        ht = batch["history_tokens"].to(device)
        hd = batch["history_dims"].to(device)
        pc = batch["player_card"].to(device)
        targets = {d: t.to(device) for d, t in batch["target_scene"].items()}
        active = {d: m.to(device) for d, m in batch["active_mask"].items()}
        st = batch["scene_type"].to(device)
        forbidden = {d: f.to(device) for d, f in batch["forbidden_local"].items()}

        out = model(ht, hd, pc, targets, target_scene_type=st)
        z_logits = out.pop("_z_logits")
        loss = per_dim_ce_loss(out, targets, active, dim_order)
        loss = loss + lambda_div * diversity_loss(out, dim_order)
        loss = loss + lambda_z * z_loss(z_logits, st)
        loss = loss + lambda_contrast * contrastive_forbidden_loss(out, forbidden, dim_order)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def per_head_accuracy(
    model: StructuredSceneTransformerV2,
    loader: DataLoader,
    dim_order: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    model.eval()
    correct = {d: 0 for d in dim_order}
    total = {d: 0 for d in dim_order}
    for batch in loader:
        ht = batch["history_tokens"].to(device)
        hd = batch["history_dims"].to(device)
        pc = batch["player_card"].to(device)
        targets = {d: t.to(device) for d, t in batch["target_scene"].items()}
        active = {d: m.to(device) for d, m in batch["active_mask"].items()}
        st = batch["scene_type"].to(device)
        out = model(ht, hd, pc, targets, target_scene_type=st)
        out.pop("_z_logits", None)
        for d in dim_order:
            pred = out[d].argmax(dim=-1)
            mask = active[d] > 0
            correct[d] += int((pred[mask] == targets[d][mask]).sum().item())
            total[d] += int(mask.sum().item())
    return {d: (correct[d] / total[d] if total[d] else 0.0) for d in dim_order}


# ── Stage 1: base pretrain ──────────────────────────────────────────────────
def stage_base(args):
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    print(f"[base] training on cases: {case_ids}")
    case_trajs, per_case_dims = load_case_data(case_ids)
    dim_order, dim_vocab, universal, full_vocab, fv2i, dv2i = build_union_vocab(per_case_dims)
    print(f"[base] dim_order ({len(dim_order)}): {dim_order}")
    print(f"[base] universal ({len(universal)}): {universal}")
    print(f"[base] vocab size: {len(full_vocab)}")

    flat = []
    for c in case_ids:
        flat.extend(case_trajs[c])
    print(f"[base] trajectories: {len(flat)} across {len(case_ids)} cases")

    ds = TrajectoryDatasetV2(
        flat, dim_order, dim_vocab, fv2i, dv2i,
        max_history=args.max_history,
        universal_dims=universal, universal_only=True,
        truncate_history_p=0.3, include_branches=True,
    )
    print(f"[base] training pairs: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate)

    cfg = V2Config(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads,
        max_history=args.max_history, lora_rank=0,            # base stage: no LoRA
        universal_dims=universal,
    )
    model = StructuredSceneTransformerV2(dim_vocab, dim_order, full_vocab, cfg)
    print(f"[base] params: {model.num_parameters():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[base] device: {device}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = REPO_ROOT / "living_tales/trainer/outputs/_base"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg = train_one_epoch(
            model, loader, opt, dim_order,
            lambda_div=args.lambda_div, lambda_z=args.lambda_z,
            lambda_contrast=args.lambda_contrast, device=device,
        )
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            acc = per_head_accuracy(model, loader, dim_order, device=device)
            uni_acc = sum(acc[d] for d in universal) / len(universal)
            print(f"[base] epoch {epoch:>4d}/{args.epochs}  loss={avg:.4f}  "
                  f"universal_acc={uni_acc:.3f}")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "dim_order": dim_order,
        "dim_vocab": dim_vocab,
        "universal_dims": universal,
        "full_vocab": full_vocab,
    }
    out_path = out_dir / "base_universal.pt"
    torch.save(ckpt, out_path)
    print(f"[base] saved {out_path}")
    return out_path


# ── Stage 2: adapter + heads ────────────────────────────────────────────────
def stage_adapter(args):
    case = args.case
    print(f"[adapter] case: {case}")
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    dim_order = base_ckpt["dim_order"]
    base_dim_vocab = base_ckpt["dim_vocab"]
    base_full_vocab = base_ckpt["full_vocab"]
    base_universal = base_ckpt["universal_dims"]
    base_cfg_dict = base_ckpt["config"]

    L = TrajectoryLoader(case, REPO_ROOT)
    trajs = L.load_all()
    case_dims = load_dims(case)
    case_dim_vocab = {d["name"]: list(d["vocab"]) for d in case_dims["dimensions"]}
    case_dim_order = [d["name"] for d in case_dims["dimensions"]]

    # Sanity: every case dim is in base dim_order.
    missing = [d for d in case_dim_order if d not in dim_order]
    if missing:
        raise ValueError(f"case {case} dims not in base dim_order: {missing}")
    # Use the BASE vocab so token ids stay consistent with the base model.
    fv2i = {t: i for i, t in enumerate(base_full_vocab)}
    dv2i = {d: {t: i for i, t in enumerate(toks)} for d, toks in base_dim_vocab.items()}

    ds = TrajectoryDatasetV2(
        trajs, dim_order, base_dim_vocab, fv2i, dv2i,
        max_history=base_cfg_dict.get("max_history", 160),
        universal_dims=base_universal, universal_only=False,
        truncate_history_p=0.3, include_branches=True,
    )
    print(f"[adapter] {case}: {len(ds)} training pairs")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate)

    # Build a LoRA-enabled v2 model with the base's hyperparameters.
    cfg = V2Config(
        hidden_dim=base_cfg_dict["hidden_dim"], n_layers=base_cfg_dict["n_layers"],
        n_heads=base_cfg_dict["n_heads"], max_history=base_cfg_dict["max_history"],
        lora_rank=args.lora_rank, universal_dims=base_universal,
    )
    model = StructuredSceneTransformerV2(base_dim_vocab, dim_order, base_full_vocab, cfg)
    # Load base weights into the LoRA model. LoRA branches stay zero-init.
    missing_keys, unexpected = model.load_state_dict(base_ckpt["state_dict"], strict=False)
    if unexpected:
        print(f"[adapter] WARN unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[adapter] LoRA-enabled model params: {model.num_parameters():,}")
    print(f"[adapter] missing-from-base keys (expected: LoRA branches): {len(missing_keys)}")

    # Stage 2: freeze base; train LoRA + case-specific heads.
    model.freeze_base_for_adapter_stage()
    case_specific_dims = [d for d in case_dim_order if d not in base_universal]
    print(f"[adapter] unfreezing case-specific heads: {case_specific_dims}")
    model.unfreeze_dims(case_specific_dims)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[adapter] trainable params: {trainable:,} / {model.num_parameters():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    out_dir = REPO_ROOT / "living_tales/trainer/outputs" / case
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg = train_one_epoch(
            model, loader, opt, dim_order,
            lambda_div=args.lambda_div, lambda_z=args.lambda_z,
            lambda_contrast=args.lambda_contrast, device=device,
        )
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            acc = per_head_accuracy(model, loader, dim_order, device=device)
            avg_acc = sum(acc.values()) / max(len(acc), 1)
            print(f"[adapter] epoch {epoch:>4d}/{args.epochs}  loss={avg:.4f}  "
                  f"avg_acc={avg_acc:.3f}")

    # Save: full state_dict (compact; ~5M floats either way) plus the LoRA-only
    # delta as separate file for clarity.
    full_path = out_dir / "v2_full.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "dim_order": dim_order,
        "dim_vocab": base_dim_vocab,
        "full_vocab": base_full_vocab,
        "universal_dims": base_universal,
        "case": case,
    }, full_path)
    print(f"[adapter] saved {full_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    pb = sub.add_parser("base")
    pb.add_argument("--cases", required=True, help="comma-separated")
    pb.add_argument("--epochs", type=int, default=80)
    pb.add_argument("--batch-size", type=int, default=32)
    pb.add_argument("--lr", type=float, default=1e-3)
    pb.add_argument("--max-history", type=int, default=160)
    pb.add_argument("--hidden-dim", type=int, default=256)
    pb.add_argument("--n-layers", type=int, default=4)
    pb.add_argument("--n-heads", type=int, default=8)
    pb.add_argument("--lambda-div", type=float, default=0.05)
    pb.add_argument("--lambda-z", type=float, default=0.5)
    pb.add_argument("--lambda-contrast", type=float, default=0.2)

    pa = sub.add_parser("adapter")
    pa.add_argument("--case", required=True)
    pa.add_argument("--base-checkpoint", required=True)
    pa.add_argument("--epochs", type=int, default=150)
    pa.add_argument("--batch-size", type=int, default=16)
    pa.add_argument("--lr", type=float, default=5e-4)
    pa.add_argument("--lora-rank", type=int, default=8)
    pa.add_argument("--lambda-div", type=float, default=0.05)
    pa.add_argument("--lambda-z", type=float, default=0.5)
    pa.add_argument("--lambda-contrast", type=float, default=0.2)

    args = p.parse_args()
    if args.stage == "base":
        stage_base(args)
    elif args.stage == "adapter":
        stage_adapter(args)


if __name__ == "__main__":
    main()
