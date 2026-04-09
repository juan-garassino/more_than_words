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
    if model_path and HAS_TORCH:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        from trainer.dialogue_model import DialogueTransformer
        model = DialogueTransformer(
            vocab_size=ckpt["vocab_size"],
            embedding_dim=ckpt["embedding_dim"],
            context_dim=ckpt["context_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        mappings = {
            "id_to_idx": ckpt["id_to_idx"],
            "class_to_idx": ckpt["class_to_idx"],
            "phase_to_idx": ckpt["phase_to_idx"],
            "stream_to_idx": ckpt["stream_to_idx"],
            "agency_to_idx": ckpt["agency_to_idx"],
        }
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

def game_loop(spec: CartridgeSpec, model, mappings: Optional[dict]):
    assert console is not None, "Rich library required. pip install rich"

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
    console.print(Panel(
        f"[bold]{spec.title}[/bold]\n\n"
        f"[dim]You are the detective. Play symbolic tokens.\n"
        f"The field responds. The truth converges.[/dim]",
        border_style="bright_blue",
    ))

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

        # Convergence
        console.print(f"  Convergence: {_convergence_bar(conv_score)}")

        # Hand (only phase-valid tokens)
        valid_hand = [t for t in hand if t.is_available_at_turn(game_turn)]
        if not valid_hand:
            # Try refilling from deck
            while len(valid_hand) < HAND_SIZE and deck:
                card = deck.pop(0)
                if card.is_available_at_turn(game_turn):
                    valid_hand.append(card)
                    hand.append(card)

        console.print()
        console.print("[bold]YOUR HAND[/bold]")
        for i, tok in enumerate(valid_hand):
            console.print(f"  [{i + 1}] {_token_rich(tok)}")

        if not valid_hand:
            console.print("  [dim]No tokens available.[/dim]")
            break

        # Check if accusation available
        if conv_score >= spec.convergence_threshold:
            console.print(
                f"\n  [bold green]The field has converged ({conv_score:.0%}). "
                f"You may [bold]accuse[/bold] or keep investigating.[/bold green]"
            )

        # ── Player input ──
        console.print()
        choice = Prompt.ask(
            "[bold]Play a token[/bold] (number), [dim]accuse[/dim], or [dim]quit[/dim]",
            default="1",
        )

        if choice.lower() in ("q", "quit"):
            console.print("\n[dim]You walk away from the case.[/dim]")
            break

        if choice.lower() == "accuse":
            if conv_score < spec.convergence_threshold:
                console.print("[yellow]The field hasn't converged yet. Keep investigating.[/yellow]")
                continue
            _handle_accusation(spec, console)
            break

        # Parse card selection
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(valid_hand):
                console.print("[red]Invalid card number.[/red]")
                continue
        except ValueError:
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
        game_turn_engine = turn // 2

        if model is not None and mappings is not None and HAS_TORCH:
            # Model-driven response
            inp_t = torch.tensor([seq_t], dtype=torch.long)
            inp_c = torch.tensor([seq_c], dtype=torch.long)
            inp_p = torch.tensor([seq_p], dtype=torch.long)
            inp_s = torch.tensor([seq_s], dtype=torch.long)
            inp_a = torch.tensor([seq_a], dtype=torch.long)

            # Build valid mask: engine tokens, phase-valid, not placed
            valid_mask = torch.zeros(spec.vocab_size, dtype=torch.bool)
            id_to_idx = mappings["id_to_idx"]
            for t in engine_pool:
                if t.id not in placed_ids and t.is_available_at_turn(game_turn_engine):
                    valid_mask[id_to_idx[t.id]] = True

            if valid_mask.any():
                idx_to_id = {v: k for k, v in id_to_idx.items()}
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
            placed_ids.add(engine_tok.id)
            context_ids.append(engine_tok.id)
            convergence_dims = np.minimum(
                1.0, convergence_dims + engine_tok.attractor_weights * spec.convergence_rate,
            )
            dialogue_history.append(("FIELD", engine_tok))
            console.print(f"  FIELD:  {_token_rich(engine_tok)}")

            if mappings:
                enc = _encode_token(engine_tok, mappings)
                seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
                seq_s.append(enc[3]); seq_a.append(enc[4])
            turn += 1
        else:
            console.print("  [dim]The field is silent.[/dim]")
            turn += 1

        console.print()

    # ── End ──
    conv_score = float(convergence_dims.min())
    console.print()
    console.print(Rule("CASE CLOSED"))
    console.print(f"  Final convergence: {_convergence_bar(conv_score)}")
    console.print(f"  Tokens exchanged: {len(dialogue_history)}")
    console.print()

    console.print("[bold]DIALOGUE TRANSCRIPT[/bold]")
    for role, tok in dialogue_history:
        tag = "[bold blue]YOU[/bold blue]  " if role == "YOU" else "[bold cyan]FIELD[/bold cyan]"
        console.print(f"  {tag}  {_token_rich(tok)}")


def _handle_accusation(spec: CartridgeSpec, con: Console):
    """Simple accusation prompt."""
    suspects = [t for t in spec.tokens if t.token_class == TokenClass.SUSPECT and not t.is_invariant]
    events = [t for t in spec.tokens if t.token_class == TokenClass.EVENT and not t.is_invariant]
    motives = [t for t in spec.tokens if t.token_class == TokenClass.MOTIVE and not t.is_invariant]

    con.print("\n[bold]ACCUSATION[/bold]")
    con.print("[dim]Name the killer, the mechanism, and the motive.[/dim]\n")

    for label, pool, dim in [("Suspect", suspects, 0), ("Event", events, 1), ("Motive", motives, 2)]:
        con.print(f"  [bold]{label}s:[/bold]")
        for i, t in enumerate(pool):
            con.print(f"    [{i + 1}] {_token_rich(t)}")

    con.print()
    try:
        s = int(Prompt.ask("Suspect #")) - 1
        e = int(Prompt.ask("Event #")) - 1
        m = int(Prompt.ask("Motive #")) - 1
    except (ValueError, IndexError):
        con.print("[red]Invalid input.[/red]")
        return

    guesses = []
    for idx, pool in [(s, suspects), (e, events), (m, motives)]:
        if 0 <= idx < len(pool):
            guesses.append(pool[idx].id)
        else:
            con.print("[red]Invalid selection.[/red]")
            return

    correct = list(spec.invariant_token_ids)
    if guesses == correct:
        con.print("\n[bold green]CORRECT. The case is solved.[/bold green]")
        for tid in correct:
            tok = spec.get_token(tid)
            con.print(f"  {_token_rich(tok)}")
    else:
        con.print("\n[bold red]WRONG.[/bold red]")
        wrong = [i for i, (g, c) in enumerate(zip(guesses, correct)) if g != c]
        dims = ["Suspect", "Event", "Motive"]
        for i in wrong:
            con.print(f"  [red]{dims[i]} is incorrect.[/red]")


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
