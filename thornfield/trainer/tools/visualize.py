"""
Visualization tools for Thornfield dialogue training.

Three matplotlib-based diagnostics (Agg backend, saves PNG):
1. Training curves (loss/reward over epochs)
2. Dialogue trajectories (convergence + energy over turns)
3. Token prediction heatmap (P(engine_token | player_token))

Usage:
    cd thornfield/trainer
    python3 tools/visualize.py training outputs/amber_cipher/history.json
    python3 tools/visualize.py trajectories amber_cipher
    python3 tools/visualize.py heatmap amber_cipher
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def plot_training_curves(history_path: str, output_path: str) -> None:
    """Load history.json and plot loss over epochs."""
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Supervised losses
    if "supervised" in history and "epoch_losses" in history["supervised"]:
        losses = history["supervised"]["epoch_losses"]
        axes[0].plot(losses, "b-", linewidth=1.5)
        axes[0].set_title("Supervised KD Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
    elif "epoch_losses" in history:
        losses = history["epoch_losses"]
        axes[0].plot(losses, "b-", linewidth=1.5)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

    # RL returns
    if "rl" in history and "episode_returns" in history["rl"]:
        returns = history["rl"]["episode_returns"]
        if returns:
            axes[1].plot(returns, "g-", alpha=0.4, linewidth=0.5)
            window = min(20, len(returns))
            if len(returns) >= window:
                smoothed = np.convolve(returns, np.ones(window)/window, mode="valid")
                axes[1].plot(range(window-1, len(returns)), smoothed, "g-", linewidth=2)
            axes[1].set_title("RL Episode Returns")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Return")
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_dialogue_trajectories(case_id: str, output_dir: str, n_games: int = 20) -> None:
    """Run games and plot convergence + energy over turns."""
    from core.cartridge import CartridgeSpec
    from generator.dialogue_sampler import DialogueSampler

    case_dir = _HERE.parent / "cases" / case_id
    spec = CartridgeSpec.load(str(case_dir / "spec.json"))
    sampler = DialogueSampler(spec, strategy="energy")
    paths = sampler.sample_batch(n_games, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence
    for p in paths:
        turns = range(len(p.turns))
        conv = [t.convergence_at_step for t in p.turns]
        axes[0].plot(turns, conv, alpha=0.3, linewidth=1)
    axes[0].axhline(y=spec.convergence_threshold, color="r", linestyle="--", alpha=0.5, label="threshold")
    axes[0].set_title(f"Convergence ({len(paths)} dialogues)")
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Convergence Score (min dim)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Energy
    for p in paths:
        turns = range(len(p.turns))
        energy = [t.energy_at_step for t in p.turns]
        axes[1].plot(turns, energy, alpha=0.3, linewidth=1)
    axes[1].set_title(f"Hopfield Energy ({len(paths)} dialogues)")
    axes[1].set_xlabel("Turn")
    axes[1].set_ylabel("Subgraph Energy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / f"{case_id}_trajectories.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_token_prediction_heatmap(case_id: str, model_path: str, output_path: str) -> None:
    """For each player token as single-token context, show P(engine_token)."""
    import torch
    from core.cartridge import CartridgeSpec
    from core.token import TokenAgency, TokenStream
    from trainer.dialogue_model import DialogueTransformer
    from trainer.train_dialogue import _build_mappings, _encode_token

    case_dir = _HERE.parent / "cases" / case_id
    spec = CartridgeSpec.load(str(case_dir / "spec.json"))

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DialogueTransformer(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["embedding_dim"],
        context_dim=ckpt["context_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mappings = _build_mappings(spec.tokens)
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = mappings

    player_tokens = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]
    engine_tokens = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]

    # Build opening context
    opening_encs = []
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        opening_encs.append(_encode_token(tok, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx))

    engine_indices = [id_to_idx[t.id] for t in engine_tokens]
    n_player = min(len(player_tokens), 30)
    n_engine = min(len(engine_tokens), 30)

    heatmap = np.zeros((n_engine, n_player), dtype=np.float32)

    with torch.no_grad():
        for pi, ptok in enumerate(player_tokens[:n_player]):
            # Context: opening + this player token
            p_enc = _encode_token(ptok, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx)
            all_encs = opening_encs + [p_enc]

            inp_t = torch.tensor([[e[0] for e in all_encs]], dtype=torch.long)
            inp_c = torch.tensor([[e[1] for e in all_encs]], dtype=torch.long)
            inp_p = torch.tensor([[e[2] for e in all_encs]], dtype=torch.long)
            inp_s = torch.tensor([[e[3] for e in all_encs]], dtype=torch.long)
            inp_a = torch.tensor([[e[4] for e in all_encs]], dtype=torch.long)
            pad = torch.zeros(1, len(all_encs), dtype=torch.bool)

            logits = model(inp_t, inp_c, inp_p, inp_s, inp_a, pad)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

            for ei, etok in enumerate(engine_tokens[:n_engine]):
                heatmap[ei, pi] = probs[id_to_idx[etok.id]].item()

    fig, ax = plt.subplots(figsize=(max(12, n_player * 0.4), max(8, n_engine * 0.3)))

    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_player))
    ax.set_xticklabels(
        [t.id.split(":")[-1][:12] for t in player_tokens[:n_player]],
        rotation=90, fontsize=6,
    )
    ax.set_yticks(range(n_engine))
    ax.set_yticklabels(
        [t.id.split(":")[-1][:12] for t in engine_tokens[:n_engine]],
        fontsize=6,
    )
    ax.set_xlabel("Player Token")
    ax.set_ylabel("Engine Token")
    ax.set_title(f"P(engine | opening + player) — {spec.title}")
    plt.colorbar(im, ax=ax, label="Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Thornfield visualization tools.")
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("training", help="Plot training curves from history.json")
    p1.add_argument("history_path")
    p1.add_argument("--output", default="training_curves.png")

    p2 = sub.add_parser("trajectories", help="Plot dialogue trajectories")
    p2.add_argument("case_id")
    p2.add_argument("--output-dir", default="outputs")
    p2.add_argument("--n-games", type=int, default=20)

    p3 = sub.add_parser("heatmap", help="Token prediction heatmap")
    p3.add_argument("case_id")
    p3.add_argument("--model-path", default=None)
    p3.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "training":
        plot_training_curves(args.history_path, args.output)
    elif args.command == "trajectories":
        plot_dialogue_trajectories(args.case_id, args.output_dir, args.n_games)
    elif args.command == "heatmap":
        model_path = args.model_path or str(_HERE.parent / "outputs" / args.case_id / "dialogue_model.pt")
        output = args.output or f"{args.case_id}_heatmap.png"
        plot_token_prediction_heatmap(args.case_id, model_path, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
