"""
Living Tales — Dialogue Play
===========================
Interactive TUI for testing the dialogue transformer.

The player and the field take turns exchanging symbolic tokens.
Each token is an event, suspect, location, object, etc. — not text.
The field (model) responds to the player's token with its own,
building the mystery narrative together.

Usage
-----
    cd living_tales/trainer
    python3 tools/play_dialogue.py amber_cipher
    python3 tools/play_dialogue.py amber_cipher --model-path outputs/amber_cipher/dialogue_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from core.cartridge import CartridgeSpec
from core.token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers — symbolic tokens, not text
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = {
    "SUSPECT":  "bold red",
    "MOTIVE":   "bold magenta",
    "EVENT":    "bold yellow",
    "LOCATION": "cyan",
    "OBJECT":   "blue",
    "ACTION":   "green",
    "EMOTION":  "bright_magenta",
    "MODIFIER": "dim white",
    "WITNESS":  "bold cyan",
    "TIME":     "yellow",
    "ACCOMPLICE": "red",
    "NEED":     "green",
    "STATE":    "white",
    "OFFERING": "bright_blue",
    "UNKNOWN":  "white",
}

_CLASS_ICONS = {
    "SUSPECT":  "◈",
    "MOTIVE":   "◇",
    "EVENT":    "◆",
    "LOCATION": "⬡",
    "OBJECT":   "□",
    "ACTION":   "→",
    "EMOTION":  "~",
    "MODIFIER": "·",
    "WITNESS":  "◎",
    "TIME":     "◷",
    "ACCOMPLICE": "◈",
    "NEED":     "△",
    "STATE":    "○",
    "OFFERING": "☆",
}

_ARTICLE_CLASSES = {TokenClass.OBJECT, TokenClass.LOCATION, TokenClass.EVENT}

console = Console() if HAS_RICH else None


def _token_name(tok: Token) -> str:
    """Human-readable name for a symbolic token."""
    if tok.surface_expression:
        return tok.surface_expression
    raw = tok.id.split(":")[-1].replace("_", " ").title()
    if tok.token_class in _ARTICLE_CLASSES:
        return f"The {raw}"
    return raw


def _token_rich(tok: Token) -> str:
    """Rich markup for a symbolic token: icon + name + class tag."""
    name = _token_name(tok)
    color = _CLASS_COLORS.get(tok.token_class.value, "white")
    icon = _CLASS_ICONS.get(tok.token_class.value, "·")
    return f"[{color}]{icon} {name}[/{color}]  [dim]{tok.token_class.value}[/dim]"


def _convergence_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{score:.0%}"
    return f"[{bar}] {pct}"


# ─────────────────────────────────────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────────────────────────────────────

HAND_SIZE = 7


def _load_model_and_spec(case_id: str, model_path: Optional[str]):
    """Load CartridgeSpec and optionally the dialogue transformer."""
    case_dir = _HERE.parent / "cases" / case_id
    spec_path = case_dir / "spec.json"
    if not spec_path.exists():
        print(f"Error: {spec_path} not found. Run 'make ac-s02-pack' first.")
        sys.exit(1)
    spec = CartridgeSpec.load(str(spec_path))

    model = None
    mappings = None
    model_type = None
    if model_path and HAS_TORCH:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model_type = ckpt.get("model_type", "dialogue")

        sd = ckpt["state_dict"]
        # Infer actual max_seq_len from position encoding in state_dict
        pe_key = "position_encoding.pe"
        actual_seq_len = sd[pe_key].shape[1] if pe_key in sd else ckpt.get("max_seq_len", 256)

        if model_type == "scene":
            from trainer.dialogue_model import SceneTransformer
            head_vocab_masks = sd.get("head_vocab_masks")
            model = SceneTransformer(
                vocab_size=ckpt["vocab_size"],
                embedding_dim=ckpt["embedding_dim"],
                context_dim=ckpt["context_dim"],
                n_heads=ckpt.get("n_heads", 6),
                n_layers=ckpt.get("n_layers", 6),
                n_output_heads=ckpt.get("n_output_heads", 3),
                head_vocab_masks=head_vocab_masks,
                max_seq_len=actual_seq_len,
            )
        else:
            from trainer.dialogue_model import DialogueTransformer
            model = DialogueTransformer(
                vocab_size=ckpt["vocab_size"],
                embedding_dim=ckpt["embedding_dim"],
                context_dim=ckpt["context_dim"],
                n_layers=ckpt.get("n_layers", 4),
                n_heads=ckpt.get("n_heads", 4),
                max_seq_len=actual_seq_len,
            )

        model.load_state_dict(sd)
        model.eval()
        mappings = {
            "id_to_idx": ckpt["id_to_idx"],
            "class_to_idx": ckpt["class_to_idx"],
            "phase_to_idx": ckpt["phase_to_idx"],
            "stream_to_idx": ckpt["stream_to_idx"],
            "agency_to_idx": ckpt["agency_to_idx"],
        }
        print(f"Loaded {model_type} model from {model_path}")
    elif model_path and not HAS_TORCH:
        print("Warning: torch not available, running without model.")

    return spec, model, mappings


def _encode_token(tok: Token, m: dict) -> Tuple[int, int, int, int, int]:
    return (
        m["id_to_idx"][tok.id],
        m["class_to_idx"][tok.token_class.value],
        m["phase_to_idx"][tok.phase.value],
        m["stream_to_idx"][tok.stream.value],
        m["agency_to_idx"][tok.agency.value],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fallback engine (no model — energy-based greedy)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_engine_pick(
    spec: CartridgeSpec,
    engine_pool: List[Token],
    context_ids: List[str],
    game_turn: int,
) -> Optional[Token]:
    """Pick engine token by energy minimization when no model is loaded."""
    candidates = [
        t for t in engine_pool if t.is_available_at_turn(game_turn)
    ]
    if not candidates:
        return None
    scored = []
    for t in candidates:
        e = spec.token_graph.induced_subgraph_energy([t.id], context_ids)
        scored.append((t, e))
    scored.sort(key=lambda x: x[1])
    return scored[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Main game loop
# ─────────────────────────────────────────────────────────────────────────────

RED_HERRING_TAGS = {'surface', 'plausible', 'dramatic'}


def game_loop(spec: CartridgeSpec, model, mappings: Optional[dict]):
    assert console is not None, "Rich library required. pip install rich"

    is_creature = getattr(spec, 'mode', 'converging') == 'oscillating'

    # Partition tokens
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

    np.random.shuffle(player_tokens)
    hand = player_tokens[:HAND_SIZE]
    deck = player_tokens[HAND_SIZE:]
    engine_pool = list(engine_tokens)

    convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
    placed_ids: set = set()
    context_ids: List[str] = []
    dialogue_history: List[Tuple[str, Token]] = []  # (role, token)

    # Model inference sequences
    seq_t, seq_c, seq_p, seq_s, seq_a = [], [], [], [], []

    max_turns = spec.max_turns * 2
    turn = 0

    # ── Opening ──
    console.print()
    if is_creature:
        _intro = (
            f"[bold]{spec.title}[/bold]\n\n"
            f"[dim]You care for a creature. Feed, play, comfort.\n"
            f"The creature responds. Wellbeing oscillates.[/dim]"
        )
    else:
        _intro = (
            f"[bold]{spec.title}[/bold]\n\n"
            f"[dim]You are the detective. Play symbolic tokens.\n"
            f"The field responds. The truth converges.[/dim]"
        )
    console.print(Panel(_intro, border_style="bright_blue"))

    console.print("\n[bold]OPENING SCENE[/bold]")
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        placed_ids.add(tok.id)
        context_ids.append(tok.id)
        convergence_dims = np.minimum(
            1.0, convergence_dims + tok.attractor_weights * spec.convergence_rate,
        )
        dialogue_history.append(("FIELD", tok))
        console.print(f"  FIELD:  {_token_rich(tok)}")
        if mappings:
            enc = _encode_token(tok, mappings)
            seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
            seq_s.append(enc[3]); seq_a.append(enc[4])
        turn += 1

    console.print()

    # ── Main loop ──
    while turn < max_turns:
        game_turn = turn // 2
        conv_score = float(convergence_dims.min())

        # ── Display state ──
        console.print(Rule(f"Turn {turn}/{max_turns}"))

        # Convergence / Wellbeing
        _conv_label = "Wellbeing" if is_creature else "Convergence"
        console.print(f"  {_conv_label}: {_convergence_bar(conv_score)}")

        # Creature token recycling: reset placed_ids every 10 steps
        step = turn // 2
        if is_creature and step > 0 and step % 10 == 0:
            placed_ids = set(spec.opening_token_ids)

        # Hand (creatures skip phase gating for more variety)
        if is_creature:
            valid_hand = [t for t in hand if t.id not in placed_ids]
        else:
            valid_hand = [t for t in hand if t.is_available_at_turn(game_turn)]
        if not valid_hand:
            # Try refilling from deck
            while len(valid_hand) < HAND_SIZE and deck:
                card = deck.pop(0)
                if is_creature or card.is_available_at_turn(game_turn):
                    valid_hand.append(card)
                    hand.append(card)

        console.print()
        console.print("[bold]YOUR HAND[/bold]")
        for i, tok in enumerate(valid_hand):
            console.print(f"  [{i + 1}] {_token_rich(tok)}")

        if not valid_hand:
            console.print("  [dim]No tokens available.[/dim]")
            break

        # Check if accusation available (mysteries only)
        if not is_creature and conv_score >= spec.convergence_threshold:
            console.print(
                f"\n  [bold green]The field has converged ({conv_score:.0%}). "
                f"You may [bold]accuse[/bold] or keep investigating.[/bold green]"
            )

        # ── Player input ──
        console.print()
        if is_creature:
            choice = Prompt.ask(
                "[bold]Play a token[/bold] (number), [dim]rest[/dim], or [dim]quit[/dim]",
                default="1",
            )
        else:
            choice = Prompt.ask(
                "[bold]Play a token[/bold] (number), [dim]accuse[/dim], or [dim]quit[/dim]",
                default="1",
            )

        if choice.lower() in ("q", "quit"):
            if is_creature:
                console.print("\n[dim]You step away. The creature watches you go.[/dim]")
            else:
                console.print("\n[dim]You walk away from the case.[/dim]")
            break

        if choice.lower() == "rest" and is_creature:
            # Skip turn, apply decay
            convergence_dims = np.maximum(0.0, convergence_dims - 0.01)
            console.print("\n  [dim]You rest. The creature stirs quietly.[/dim]")
            turn += 2  # skip both player and engine turn
            console.print()
            continue

        if choice.lower() == "accuse":
            if is_creature:
                console.print("[yellow]No accusations here -- tend to your creature.[/yellow]")
                continue
            _handle_accusation(spec, convergence_dims, console)
            break

        # Parse card selection
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(valid_hand):
                console.print("[red]Invalid card number.[/red]")
                continue
        except ValueError:
            if is_creature:
                console.print("[red]Enter a number, 'rest', or 'quit'.[/red]")
            else:
                console.print("[red]Enter a number, 'accuse', or 'quit'.[/red]")
            continue

        player_tok = valid_hand[idx]
        hand.remove(player_tok)

        # Place player token
        placed_ids.add(player_tok.id)
        context_ids.append(player_tok.id)
        convergence_dims = np.minimum(
            1.0, convergence_dims + player_tok.attractor_weights * spec.convergence_rate,
        )

        # Red herring penalty (mysteries only)
        if not is_creature and set(getattr(player_tok, 'affinity_tags', [])) & RED_HERRING_TAGS:
            convergence_dims = np.maximum(
                0.0,
                convergence_dims - abs(np.array(player_tok.attractor_weights)) * spec.convergence_rate * 0.5,
            )

        dialogue_history.append(("YOU", player_tok))
        console.print(f"\n  YOU:    {_token_rich(player_tok)}")

        if mappings:
            enc = _encode_token(player_tok, mappings)
            seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
            seq_s.append(enc[3]); seq_a.append(enc[4])
        turn += 1

        # Refill hand
        while len(hand) < HAND_SIZE and deck:
            hand.append(deck.pop(0))

        # ── Engine response ──
        engine_tok = None
        scene_already_placed = False
        game_turn_engine = turn // 2

        if model is not None and mappings is not None and HAS_TORCH:
            # Model-driven response
            model_device = next(model.parameters()).device
            inp_t = torch.tensor([seq_t], dtype=torch.long, device=model_device)
            inp_c = torch.tensor([seq_c], dtype=torch.long, device=model_device)
            inp_p = torch.tensor([seq_p], dtype=torch.long, device=model_device)
            inp_s = torch.tensor([seq_s], dtype=torch.long, device=model_device)
            inp_a = torch.tensor([seq_a], dtype=torch.long, device=model_device)

            # Build valid mask: engine tokens, phase-valid, not placed
            valid_mask = torch.zeros(spec.vocab_size, dtype=torch.bool, device=model_device)
            id_to_idx = mappings["id_to_idx"]
            for t in engine_pool:
                if t.id not in placed_ids and t.is_available_at_turn(game_turn_engine):
                    valid_mask[id_to_idx[t.id]] = True

            idx_to_id = {v: k for k, v in id_to_idx.items()}
            is_scene = hasattr(model, "predict_scene")

            if is_scene and valid_mask.any():
                # SceneTransformer: predict N tokens in parallel (one per head)
                n_heads = model.n_output_heads
                per_head_valid = [valid_mask.clone() for _ in range(n_heads)]
                scene_results = model.predict_scene(
                    inp_t, inp_c, inp_p, inp_s, inp_a,
                    per_head_valid=per_head_valid, temperature=0.8,
                )
                # Collect all scene tokens (skip heads with no valid token)
                scene_tokens = []
                for head_idx, (chosen_idx, probs) in enumerate(scene_results):
                    if chosen_idx < 0:
                        continue
                    chosen_id = idx_to_id.get(chosen_idx)
                    if chosen_id is None or chosen_id in placed_ids:
                        continue
                    scene_tok = spec.get_token(chosen_id)
                    scene_tokens.append(scene_tok)
                    placed_ids.add(scene_tok.id)
                if scene_tokens:
                    engine_tok = scene_tokens[0]  # primary response token
                    scene_already_placed = True
                    # Place all scene tokens
                    for stok in scene_tokens:
                        context_ids.append(stok.id)
                        convergence_dims = np.minimum(
                            1.0, convergence_dims + stok.attractor_weights * spec.convergence_rate,
                        )
                        dialogue_history.append(("FIELD", stok))
                        if stok != engine_tok:
                            console.print(f"  FIELD:  {_token_rich(stok)}")
                        if mappings:
                            enc = _encode_token(stok, mappings)
                            seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
                            seq_s.append(enc[3]); seq_a.append(enc[4])
            elif not is_scene and valid_mask.any():
                # DialogueTransformer: single next-token prediction
                chosen_idx, probs = model.predict_next(
                    inp_t, inp_c, inp_p, inp_s, inp_a,
                    valid_mask=valid_mask, temperature=0.8,
                )
                chosen_id = idx_to_id[chosen_idx]
                engine_tok = spec.get_token(chosen_id)
        else:
            # Fallback: energy-based greedy
            available_engine = [t for t in engine_pool if t.id not in placed_ids]
            engine_tok = _fallback_engine_pick(
                spec, available_engine, context_ids, game_turn_engine,
            )

        if engine_tok is not None:
            if not scene_already_placed:
                # DialogueTransformer / fallback: place single token
                placed_ids.add(engine_tok.id)
                context_ids.append(engine_tok.id)
                convergence_dims = np.minimum(
                    1.0, convergence_dims + engine_tok.attractor_weights * spec.convergence_rate,
                )
                dialogue_history.append(("FIELD", engine_tok))
                if mappings:
                    enc = _encode_token(engine_tok, mappings)
                    seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
                    seq_s.append(enc[3]); seq_a.append(enc[4])
            # Print primary engine token (scene extras already printed above)
            console.print(f"  FIELD:  {_token_rich(engine_tok)}")
            turn += 1
        else:
            console.print("  [dim]The field is silent.[/dim]")
            turn += 1

        # Mystery dimension decay (0.01/step)
        if not is_creature:
            convergence_dims = np.maximum(0.0, convergence_dims - 0.01)

        console.print()

    # ── End ──
    conv_score = float(convergence_dims.min())
    console.print()
    if is_creature:
        console.print(Rule("SESSION OVER"))
        console.print(f"  Final wellbeing: {_convergence_bar(conv_score)}")
    else:
        if conv_score < spec.convergence_threshold:
            console.print(Rule("THE TRAIL HAS GONE COLD"))
            console.print("  [bold red]The trail has gone cold. The case remains unsolved.[/bold red]")
        else:
            console.print(Rule("CASE CLOSED"))
        console.print(f"  Final convergence: {_convergence_bar(conv_score)}")
    console.print(f"  Tokens exchanged: {len(dialogue_history)}")
    console.print()

    console.print("[bold]DIALOGUE TRANSCRIPT[/bold]")
    for role, tok in dialogue_history:
        tag = "[bold blue]YOU[/bold blue]  " if role == "YOU" else "[bold cyan]FIELD[/bold cyan]"
        console.print(f"  {tag}  {_token_rich(tok)}")


def _handle_accusation(spec: CartridgeSpec, convergence_dims: np.ndarray, con: Console):
    """Accusation prompt -- wrong guess ends the game immediately."""
    # All suspects in the case (including invariant, for the list)
    suspects = [t for t in spec.tokens if t.token_class == TokenClass.SUSPECT]

    # The correct culprit is the first invariant token
    correct_id = spec.invariant_token_ids[0] if spec.invariant_token_ids else None
    correct_tok = spec.get_token(correct_id) if correct_id else None

    con.print("\n[bold]ACCUSATION[/bold]")
    con.print("[dim]Name the culprit.[/dim]\n")

    con.print("  [bold]Suspects:[/bold]")
    for i, t in enumerate(suspects):
        con.print(f"    [{i + 1}] {_token_rich(t)}")

    con.print()
    try:
        s = int(Prompt.ask("Suspect #")) - 1
    except (ValueError, IndexError):
        con.print("[red]Invalid input.[/red]")
        return

    if s < 0 or s >= len(suspects):
        con.print("[red]Invalid selection.[/red]")
        return

    chosen = suspects[s]
    chosen_name = _token_name(chosen)

    if chosen.id == correct_id:
        con.print(f"\n[bold green]CORRECT! {chosen_name} is the culprit. Case solved![/bold green]")
    else:
        correct_name = _token_name(correct_tok) if correct_tok else "unknown"
        con.print(
            f"\n[bold red]WRONG ACCUSATION -- CASE DISMISSED. "
            f"The real culprit was {correct_name}.[/bold red]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play a Living Tales dialogue.")
    parser.add_argument("case_id", help="Case ID (e.g. amber_cipher)")
    parser.add_argument("--model-path", help="Path to dialogue_model.pt")
    args = parser.parse_args()

    # Auto-detect model if not specified
    model_path = args.model_path
    if model_path is None:
        default = _HERE.parent / "outputs" / args.case_id / "dialogue_model.pt"
        if default.exists():
            model_path = str(default)
            print(f"Auto-detected model: {model_path}")

    if not HAS_RICH:
        print("Error: Rich library required. pip install rich")
        sys.exit(1)

    spec, model, mappings = _load_model_and_spec(args.case_id, model_path)
    game_loop(spec, model, mappings)


if __name__ == "__main__":
    main()
