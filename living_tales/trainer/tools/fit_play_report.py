"""
Living Tales — Fit · Play · Report
=================================
One command: trains a dialogue transformer on a case, plays N automated
games with it, and prints a rich report showing how the model performs
as a narrative partner.

Usage:
    cd living_tales/trainer
    python3 tools/fit_play_report.py amber_cipher
    python3 tools/fit_play_report.py amber_cipher --paths 200 --epochs 20 --rl-episodes 50 --games 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import torch
from core.cartridge import CartridgeSpec
from core.token import Token, TokenAgency, TokenClass, TokenStream

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = {
    "SUSPECT": "bold red", "MOTIVE": "bold magenta", "EVENT": "bold yellow",
    "LOCATION": "cyan", "OBJECT": "blue", "ACTION": "green",
    "EMOTION": "bright_magenta", "MODIFIER": "dim white", "WITNESS": "bold cyan",
    "TIME": "yellow", "ACCOMPLICE": "red", "UNKNOWN": "white",
}
_CLASS_ICONS = {
    "SUSPECT": "◈", "MOTIVE": "◇", "EVENT": "◆", "LOCATION": "⬡",
    "OBJECT": "□", "ACTION": "→", "EMOTION": "~", "MODIFIER": "·",
    "WITNESS": "◎", "TIME": "◷", "ACCOMPLICE": "◈",
}


def _tok_display(tok: Token) -> str:
    name = tok.surface_expression or tok.id.split(":")[-1].replace("_", " ").title()
    color = _CLASS_COLORS.get(tok.token_class.value, "white")
    icon = _CLASS_ICONS.get(tok.token_class.value, "·")
    return f"[{color}]{icon} {name}[/{color}]  [dim]{tok.token_class.value}[/dim]"


def _convergence_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_model(case_id: str, args) -> tuple:
    """Train and return (model, spec, mappings)."""
    case_dir = _HERE.parent / "cases" / case_id
    spec_path = case_dir / "spec.json"
    output_dir = _HERE.parent / "outputs" / case_id

    if not spec_path.exists():
        print(f"Error: {spec_path} not found. Run 'make ac-s02-pack' first.")
        sys.exit(1)

    from trainer.train_dialogue import train_dialogue_cartridge, _build_mappings

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model, history = train_dialogue_cartridge(
        spec_path=str(spec_path),
        output_dir=str(output_dir),
        n_dialogues=args.paths,
        n_epochs=args.epochs,
        n_rl_episodes=args.rl_episodes,
        device=args.device,
    )

    spec = CartridgeSpec.load(str(spec_path))
    mappings = dict(zip(
        ["id_to_idx", "class_to_idx", "phase_to_idx", "stream_to_idx", "agency_to_idx"],
        _build_mappings(spec.tokens),
    ))

    return model, spec, mappings, history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: PLAY (automated dialogue games)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_token(tok, mappings):
    return (
        mappings["id_to_idx"][tok.id],
        mappings["class_to_idx"][tok.token_class.value],
        mappings["phase_to_idx"][tok.phase.value],
        mappings["stream_to_idx"][tok.stream.value],
        mappings["agency_to_idx"][tok.agency.value],
    )


def play_game(model, spec, mappings, seed: int, max_turns: int = 60):
    """
    Play one automated dialogue game. Returns structured game data.

    For oscillating (creature) mode: the game runs for max_turns with a rolling
    context window — tokens can be replayed and the creature lives indefinitely.
    For converging (mystery) mode: stops when convergence is reached.
    """
    rng = np.random.RandomState(seed)
    model_device = next(model.parameters()).device
    id_to_idx = mappings["id_to_idx"]
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    is_creature = getattr(spec, "mode", "converging") == "oscillating"
    # Creatures: longer auto-play to show the loop; mysteries: stop at convergence
    effective_max = max_turns * 2 if is_creature else max_turns
    # Rolling context window size for the transformer
    context_window = 64

    player_pool = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]
    engine_pool = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]

    convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
    placed_ids = set()
    context_ids = []
    transcript = []  # list of (role, token, convergence)

    seq_t, seq_c, seq_p, seq_s, seq_a = [], [], [], [], []

    # Opening
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        placed_ids.add(tok.id)
        context_ids.append(tok.id)
        convergence_dims = np.minimum(
            1.0, convergence_dims + tok.attractor_weights * spec.convergence_rate,
        )
        enc = _encode_token(tok, mappings)
        seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
        seq_s.append(enc[3]); seq_a.append(enc[4])
        transcript.append(("FIELD", tok, float(convergence_dims.min())))

    is_player = True
    for step in range(len(transcript), effective_max):
        game_turn = step // 2
        conv_score = float(convergence_dims.min())

        # Mystery mode: stop at convergence
        if not is_creature and conv_score >= spec.convergence_threshold and game_turn >= spec.min_turns:
            break

        # Creature mode: reset placed_ids periodically so tokens can replay
        # (the creature's needs recur — you feed it again, play again, etc.)
        if is_creature and step > 0 and step % 20 == 0:
            # Keep opening tokens placed, reset everything else
            placed_ids = set(spec.opening_token_ids)

        if is_player:
            candidates = [
                t for t in player_pool
                if t.id not in placed_ids and t.is_available_at_turn(game_turn)
            ]
            if not candidates and is_creature:
                # Creature: reset placed_ids and try again
                placed_ids = set(spec.opening_token_ids)
                candidates = [
                    t for t in player_pool
                    if t.id not in placed_ids and t.is_available_at_turn(game_turn)
                ]
            if not candidates:
                break
            chosen = candidates[rng.randint(len(candidates))]
        else:
            # Model picks engine token(s) — use rolling window for transformer input
            model.eval()
            win_t = seq_t[-context_window:]
            win_c = seq_c[-context_window:]
            win_p = seq_p[-context_window:]
            win_s = seq_s[-context_window:]
            win_a = seq_a[-context_window:]

            inp_t = torch.tensor([win_t], dtype=torch.long, device=model_device)
            inp_c = torch.tensor([win_c], dtype=torch.long, device=model_device)
            inp_p = torch.tensor([win_p], dtype=torch.long, device=model_device)
            inp_s = torch.tensor([win_s], dtype=torch.long, device=model_device)
            inp_a = torch.tensor([win_a], dtype=torch.long, device=model_device)

            # Check if this is a SceneTransformer (multi-head)
            is_scene_model = hasattr(model, 'predict_scene')

            if is_scene_model:
                # SceneTransformer: predict N tokens (one per head/dimension)
                with torch.no_grad():
                    results = model.predict_scene(
                        inp_t, inp_c, inp_p, inp_s, inp_a, temperature=0.8,
                    )

                # Place all scene tokens
                scene_tokens_placed = 0
                for d, (chosen_idx, probs) in enumerate(results):
                    if chosen_idx < 0:
                        continue
                    chosen_tok = spec.get_token(idx_to_id[chosen_idx])
                    if chosen_tok.id in placed_ids and not is_creature:
                        continue  # skip duplicates for mysteries

                    placed_ids.add(chosen_tok.id)
                    context_ids.append(chosen_tok.id)
                    convergence_dims = np.minimum(
                        1.0, convergence_dims + chosen_tok.attractor_weights * spec.convergence_rate,
                    )
                    if is_creature:
                        convergence_dims = np.maximum(0.0, convergence_dims - 0.01)

                    enc = _encode_token(chosen_tok, mappings)
                    seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
                    seq_s.append(enc[3]); seq_a.append(enc[4])

                    role = "FIELD"
                    transcript.append((role, chosen_tok, float(convergence_dims.min())))
                    scene_tokens_placed += 1

                if scene_tokens_placed == 0:
                    break
                is_player = not is_player
                continue  # skip the single-token placement below

            # Single-token model fallback
            valid_mask = torch.zeros(spec.vocab_size, dtype=torch.bool, device=model_device)
            for t in engine_pool:
                if t.id not in placed_ids and t.is_available_at_turn(game_turn):
                    valid_mask[id_to_idx[t.id]] = True

            if not valid_mask.any():
                if is_creature:
                    placed_ids = set(spec.opening_token_ids)
                    for t in engine_pool:
                        if t.id not in placed_ids and t.is_available_at_turn(game_turn):
                            valid_mask[id_to_idx[t.id]] = True
                if not valid_mask.any():
                    break

            with torch.no_grad():
                chosen_idx, probs = model.predict_next(
                    inp_t, inp_c, inp_p, inp_s, inp_a,
                    valid_mask=valid_mask, temperature=0.8,
                )
            chosen = spec.get_token(idx_to_id[chosen_idx])

        placed_ids.add(chosen.id)
        context_ids.append(chosen.id)
        convergence_dims = np.minimum(
            1.0, convergence_dims + chosen.attractor_weights * spec.convergence_rate,
        )
        # Creature mode: dimensions also decay naturally over time
        if is_creature:
            convergence_dims = np.maximum(0.0, convergence_dims - 0.01)

        enc = _encode_token(chosen, mappings)
        seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
        seq_s.append(enc[3]); seq_a.append(enc[4])

        role = "YOU" if is_player else "FIELD"
        transcript.append((role, chosen, float(convergence_dims.min())))
        is_player = not is_player

    final_conv = float(convergence_dims.min())
    energy = spec.token_graph.subgraph_energy(context_ids[-30:]) if context_ids else 0.0

    return {
        "seed": seed,
        "transcript": transcript,
        "converged": final_conv >= spec.convergence_threshold,
        "final_convergence": final_conv,
        "final_energy": energy,
        "n_turns": len(transcript),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(games: list, spec, history: dict):
    """Print a rich report of the automated games."""
    if not HAS_RICH:
        _print_report_plain(games, spec, history)
        return

    console.print()
    console.print(Panel(
        f"[bold]{spec.title}[/bold] — Fit · Play · Report",
        border_style="bright_blue",
    ))

    # ── Training summary ──
    console.print(Rule("TRAINING"))
    sup = history.get("supervised", history)
    losses = sup.get("epoch_losses", [])
    if losses:
        console.print(f"  KD loss:  {losses[0]:.3f} → {losses[-1]:.3f}  ({len(losses)} epochs)")
    rl = history.get("rl", {})
    if rl.get("convergence_rate"):
        console.print(f"  RL convergence: {rl['convergence_rate']:.0%}")

    # ── Game-by-game ──
    converged = sum(1 for g in games if g["converged"])
    console.print()
    console.print(Rule(f"GAMES ({converged}/{len(games)} converged)"))

    for i, game in enumerate(games):
        console.print()
        status = "[green]CONVERGED[/green]" if game["converged"] else "[red]TIMEOUT[/red]"
        console.print(
            f"  [bold]Game {i+1}[/bold] (seed={game['seed']})  "
            f"{status}  "
            f"turns={game['n_turns']}  "
            f"conv={_convergence_bar(game['final_convergence'])}"
        )

        # Transcript
        for role, tok, conv in game["transcript"]:
            if role == "FIELD":
                tag = "[cyan]FIELD[/cyan]"
            else:
                tag = "[blue]YOU  [/blue]"
            bar = "█" * int(conv * 10) + "░" * (10 - int(conv * 10))
            console.print(f"    {tag}  {_tok_display(tok)}  [{bar}]")

    # ── Aggregate stats ──
    console.print()
    console.print(Rule("SUMMARY"))

    conv_rate = converged / len(games) if games else 0
    avg_turns = np.mean([g["n_turns"] for g in games])
    avg_conv = np.mean([g["final_convergence"] for g in games])

    # Phase compliance
    phase_order = {"EARLY": 0, "MID": 1, "LATE": 2, "INVARIANT": 3, "ANY": -1}
    compliant = total = 0
    for g in games:
        max_p = -1
        for _, tok, _ in g["transcript"]:
            pv = phase_order.get(tok.phase.value, -1)
            total += 1
            if pv < 0 or pv >= max_p:
                compliant += 1
            if pv >= 0:
                max_p = max(max_p, pv)
    chronology = compliant / max(total, 1)

    # Token class diversity
    all_classes = []
    for g in games:
        for _, tok, _ in g["transcript"]:
            all_classes.append(tok.token_class.value)
    class_counts = {}
    for c in all_classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    total_c = sum(class_counts.values())
    entropy = -sum((n/total_c) * np.log2(n/total_c) for n in class_counts.values() if n > 0)

    # Engine responsiveness: tag overlap between consecutive player→engine turns
    responsiveness_scores = []
    for g in games:
        trans = g["transcript"]
        for j in range(1, len(trans)):
            if trans[j][0] == "FIELD" and trans[j-1][0] == "YOU":
                p_tags = set(trans[j-1][1].affinity_tags)
                e_tags = set(trans[j][1].affinity_tags)
                union = p_tags | e_tags
                if union:
                    responsiveness_scores.append(len(p_tags & e_tags) / len(union))
    avg_resp = np.mean(responsiveness_scores) if responsiveness_scores else 0.0

    table = Table(box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Convergence rate", f"{conv_rate:.0%}")
    table.add_row("Avg turns", f"{avg_turns:.1f}")
    table.add_row("Avg convergence", f"{avg_conv:.2f}")
    table.add_row("Chronology compliance", f"{chronology:.0%}")
    table.add_row("Class entropy", f"{entropy:.2f}")
    table.add_row("Engine responsiveness", f"{avg_resp:.2f}")
    console.print(table)

    # ── Quality assessment ──
    console.print()
    issues = []
    if conv_rate < 0.8:
        issues.append("Low convergence — model may need more training or higher convergence_rate")
    if chronology < 0.6:
        issues.append("Poor chronology — phase masking may not be enforced at inference")
    if avg_resp < 0.1:
        issues.append("Low responsiveness — engine tokens don't relate to player tokens")
    if entropy < 2.0:
        issues.append("Low class diversity — model may be collapsing to one token type")

    if issues:
        console.print("[yellow]Issues:[/yellow]")
        for issue in issues:
            console.print(f"  ⚠ {issue}")
    else:
        console.print("[green]No issues detected. The dialogue dynamic looks healthy.[/green]")

    console.print()


def _print_report_plain(games, spec, history):
    """Fallback plain-text report when Rich is not available."""
    print(f"\n{'=' * 60}")
    print(f"  {spec.title} — Fit · Play · Report")
    print(f"{'=' * 60}")

    converged = sum(1 for g in games if g["converged"])
    print(f"\n  {converged}/{len(games)} games converged")

    for i, g in enumerate(games):
        status = "CONVERGED" if g["converged"] else "TIMEOUT"
        print(f"\n  Game {i+1} (seed={g['seed']}) [{status}] turns={g['n_turns']} conv={g['final_convergence']:.2f}")
        for role, tok, conv in g["transcript"]:
            name = tok.surface_expression or tok.id
            print(f"    {role:>5s}  {name:<40s}  {tok.token_class.value:<10s}  conv={conv:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Living Tales: Fit → Play → Report")
    parser.add_argument("case_id", help="Case ID (e.g. amber_cipher)")
    parser.add_argument("--paths", type=int, default=500, help="Dialogue paths for KD training")
    parser.add_argument("--epochs", type=int, default=30, help="KD training epochs")
    parser.add_argument("--rl-episodes", type=int, default=100, help="REINFORCE episodes")
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t0 = time.time()

    # ── TRAIN ──
    model, spec, mappings, history = train_model(args.case_id, args)

    train_time = time.time() - t0
    if console:
        console.print(f"\n  [dim]Training took {train_time:.0f}s[/dim]")

    # ── PLAY ──
    games = []
    for seed in range(args.games):
        game = play_game(model, spec, mappings, seed=seed)
        games.append(game)

    # ── REPORT ──
    print_report(games, spec, history)


if __name__ == "__main__":
    main()
