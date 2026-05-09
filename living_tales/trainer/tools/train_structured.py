"""
train_structured.py
===================
Train the StructuredSceneTransformer on hand-authored trajectories
for a Living Tales case.

Usage
-----
    python tools/train_structured.py amber_cipher [--epochs 200] \
        [--lr 1e-3] [--batch-size 16] [--max-history 80]

Pipeline
--------
1.  Load `cases/<case>/dimensions.json` and per-trajectory JSONs via
    `TrajectoryLoader`.
2.  Build the global vocab (union of every dim's vocab + every player_card
    token seen) and per-dim local vocab indices.
3.  Build a `TrajectoryDataset` (one example per turn).
4.  Instantiate `StructuredSceneTransformer` from
    `trainer.structured_scene_model`.
5.  Train with summed per-dim cross-entropy loss.
6.  Save checkpoint to `outputs/<case>/structured_scene_model.pt`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Path setup ──────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
TRAINER_ROOT = THIS_FILE.parents[1]                  # living_tales/trainer
PROJECT_ROOT = THIS_FILE.parents[2]                  # living_tales
REPO_ROOT = THIS_FILE.parents[3]                     # repo root (.../010-more-than-words)
sys.path.insert(0, str(TRAINER_ROOT))

from generator.trajectory_loader import TrajectoryLoader  # noqa: E402
from trainer.trajectory_dataset import (                  # noqa: E402
    TrajectoryDataset,
    DIM_ORDER,
)


# ── Vocab loading ───────────────────────────────────────────────────────────
def load_dim_vocab(case_id: str, repo_root: Path) -> Dict[str, List[str]]:
    """Read `dimensions.json` and return {dim_name: [token, ...]}."""
    dims_path = (
        repo_root
        / "living_tales" / "trainer" / "cases" / case_id / "dimensions.json"
    )
    with open(dims_path) as f:
        data = json.load(f)

    out: Dict[str, List[str]] = {}
    for d in data.get("dimensions", []):
        name = d["name"]
        if name not in DIM_ORDER:
            continue
        out[name] = list(d.get("vocab", []))

    # Ensure every required dim is present.
    for d in DIM_ORDER:
        if d not in out:
            raise ValueError(
                f"dimensions.json for {case_id} is missing required dim '{d}'"
            )
    return out


def collect_player_card_tokens(trajectories) -> List[str]:
    """Tokens that appear as `player_card` but might not live in any dim
    vocab (travel cards, motives used as cards, ACCUSE: prefixed tokens).
    These still need a stable global vocab entry."""
    seen = set()
    for t in trajectories:
        for turn in t.turns:
            tok = turn.player_card or ""
            if tok.startswith("ACCUSE:"):
                tok = tok[len("ACCUSE:"):]
            if tok:
                seen.add(tok)
        for tok in t.opening or []:
            if tok:
                seen.add(tok)
    return sorted(seen)


# ── Evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_per_head_accuracy(
    model, dataset, batch_size: int = 32
) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=TrajectoryDataset.collate,
    )
    correct = {d: 0 for d in DIM_ORDER}
    total = 0
    for batch in loader:
        scene_logits = model(
            batch["history_tokens"],
            batch["history_dims"],
            batch["player_card"],
            target_scene=batch["target_scene"],
        )
        bs = batch["player_card"].shape[0]
        total += bs
        for d in DIM_ORDER:
            pred = scene_logits[d].argmax(dim=-1)
            correct[d] += int((pred == batch["target_scene"][d]).sum().item())
    model.train()
    return {d: (correct[d] / max(total, 1)) for d in DIM_ORDER}


def fmt_acc(acc: Dict[str, float]) -> str:
    parts = [f"{d[:3]}={acc[d]*100:.0f}" for d in DIM_ORDER]
    mean = sum(acc.values()) / len(acc)
    return f"mean={mean*100:.1f}% [" + " ".join(parts) + "]"


# ── Main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("case_id")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-history", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    case_id = args.case_id
    repo_root = REPO_ROOT

    # 1. Load dimensions + trajectories
    dim_vocab = load_dim_vocab(case_id, repo_root)
    loader = TrajectoryLoader(case_id, repo_root)
    trajectories = loader.load_all()
    n_turns = sum(len(t.turns) for t in trajectories)
    print(f"Loaded {len(trajectories)} trajectories ({n_turns} turn pairs)")

    # 2. Build global + per-dim vocab indices
    all_dim_tokens = set()
    for toks in dim_vocab.values():
        all_dim_tokens.update(toks)
    extra_tokens = set(collect_player_card_tokens(trajectories))
    full_vocab = sorted(all_dim_tokens | extra_tokens)
    full_vocab_to_idx = {t: i for i, t in enumerate(full_vocab)}
    dim_vocab_to_idx = {
        d: {t: i for i, t in enumerate(toks)}
        for d, toks in dim_vocab.items()
    }
    print(
        f"Vocab: full={len(full_vocab)} | "
        f"per-dim sizes=" +
        ", ".join(f"{d}:{len(v)}" for d, v in dim_vocab.items())
    )

    # 3. Build dataset
    dataset = TrajectoryDataset(
        trajectories, dim_vocab, full_vocab_to_idx, dim_vocab_to_idx,
        max_history=args.max_history,
    )
    print(f"Dataset: {len(dataset)} training examples")
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=TrajectoryDataset.collate,
    )

    # 4. Build model (lazy import — file may not exist yet)
    try:
        from trainer.structured_scene_model import StructuredSceneTransformer
    except ImportError as e:
        print(
            f"\n[BLOCKER] structured_scene_model.py not available yet: {e}\n"
            "Dataset + training script are ready; re-run once the model "
            "file lands."
        )
        sys.exit(2)

    model = StructuredSceneTransformer(
        dim_vocab=dim_vocab, full_vocab=full_vocab,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: StructuredSceneTransformer ({n_params:,} params)")

    # 5. Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            scene_logits = model(
                batch["history_tokens"],
                batch["history_dims"],
                batch["player_card"],
                target_scene=batch["target_scene"],
            )
            loss = sum(
                F.cross_entropy(scene_logits[d], batch["target_scene"][d])
                for d in DIM_ORDER
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            acc = evaluate_per_head_accuracy(model, dataset)
            print(
                f"Epoch {epoch+1:3d}/{args.epochs}: "
                f"loss={avg:.4f}  per-head acc={fmt_acc(acc)}"
            )

    # 6. Save checkpoint
    out_dir = repo_root / "living_tales" / "trainer" / "outputs" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structured_scene_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "dim_vocab": dim_vocab,
        "full_vocab": full_vocab,
        "dim_vocab_to_idx": dim_vocab_to_idx,
        "full_vocab_to_idx": full_vocab_to_idx,
        "config": {
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "max_history": args.max_history,
        },
    }, out_path)
    print(f"\nSaved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
