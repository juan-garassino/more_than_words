"""
Thornfield Cartridge Runner
===========================
Interactive terminal mystery game that loads a .cartridge file and lets
you play The Amber Cipher (or any other Thornfield case).

Usage
-----
    cd thornfield/trainer
    python3 tools/run_cartridge.py outputs/amber_cipher/TheAmberCipher.cartridge

Controls
--------
    play 1 3 5      — place triad (tokens from your hand by number)
    hint            — show resonance hints (which tokens the field wants)
    advance         — engine places the next atmosphere token
    accuse          — make your accusation (unlocked near convergence threshold)
    hand            — redisplay your hand
    field           — redisplay everything
    quit / q        — exit

Requirements
------------
    pip install rich numpy torch
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional torch import ────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Rich ─────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.prompt import Prompt
    from rich.columns import Columns
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Install rich for the full UI:  pip install rich", file=sys.stderr)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_TRAINER_ROOT = _HERE.parent
sys.path.insert(0, str(_TRAINER_ROOT))

from core.token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency
from core.hopfield import TokenGraph
from core.casebook import CasebookState


console = Console() if HAS_RICH else None


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameSpec:
    """Lightweight spec reconstructed from a .cartridge file."""
    case_id: str
    title: str
    vocab_size: int
    embedding_dim: int
    context_dim: int
    n_attractor_dims: int
    convergence_threshold: float
    convergence_rate: float
    min_turns: int
    max_turns: int
    tokens: List[Token]
    token_graph: TokenGraph
    opening_token_ids: List[str]
    invariant_token_ids: List[str]

    def get_token(self, token_id: str) -> Token:
        for t in self.tokens:
            if t.id == token_id:
                return t
        raise KeyError(f"Unknown token: {token_id}")


@dataclass
class GameState:
    spec: GameSpec
    casebook: CasebookState
    hand: List[Token]
    atmosphere: List[Token]
    engine_pool: List[Token]
    played_ids: set = field(default_factory=set)
    turn: int = 0
    solved: bool = False
    game_over: bool = False
    message: str = ""  # last action message


# ─────────────────────────────────────────────────────────────────────────────
# Cartridge loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_token(data: dict) -> Token:
    from core.token import compute_narrative_gradient
    weights = np.array(data["attractor_weights"], dtype=np.float32)
    token = Token(
        id=data["id"],
        token_class=TokenClass(data["token_class"]),
        phase=TokenPhase(data["phase"]),
        attractor_weights=weights,
        affinity_tags=data.get("affinity_tags", []),
        repulsion_tags=data.get("repulsion_tags", []),
        temperature=float(data.get("temperature", 0.5)),
        narrative_gradient=float(data.get("narrative_gradient", 0.0)),
        is_invariant=bool(data.get("is_invariant", False)),
        surface_expression=data.get("surface_expression", ""),
        stream=TokenStream(data.get("stream", "EVIDENCE")),
        agency=TokenAgency(data.get("agency", "SHARED")),
    )
    return token


def load_cartridge(path: str) -> Tuple[GameSpec, Optional[object]]:
    """
    Load a .cartridge zip and return (GameSpec, model | None).
    The model is None if torch is unavailable or weights can't be loaded.
    """
    zpath = Path(path)
    if not zpath.exists():
        raise FileNotFoundError(f"Cartridge not found: {zpath}")

    with zipfile.ZipFile(zpath, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        tokens_data = json.loads(zf.read("tokens.json"))
        graph_data = json.loads(zf.read("graph.json"))
        attractor_data = json.loads(zf.read("attractor.json"))
        phases_data = json.loads(zf.read("phases.json"))
        weights_bytes = zf.read("weights.npz")

    tokens = [_load_token(t) for t in tokens_data]

    # Reconstruct opening_token_ids: prefer phases.json entry, else fall back
    # to tokens with stream=OPENING (for cartridges exported by new exporter)
    opening_ids: List[str] = phases_data.get("opening_token_ids", [])
    if not opening_ids:
        opening_ids = [t.id for t in tokens if t.stream == TokenStream.OPENING]

    graph = TokenGraph.from_json(graph_data)

    spec = GameSpec(
        case_id=manifest["case_id"],
        title=manifest["title"],
        vocab_size=manifest["vocab_size"],
        embedding_dim=manifest["embedding_dim"],
        context_dim=manifest["context_dim"],
        n_attractor_dims=manifest["n_attractor_dims"],
        convergence_threshold=manifest["convergence_threshold"],
        convergence_rate=manifest["convergence_rate"],
        min_turns=phases_data["min_turns"],
        max_turns=phases_data["max_turns"],
        tokens=tokens,
        token_graph=graph,
        opening_token_ids=opening_ids,
        invariant_token_ids=attractor_data["invariants"],
    )

    model = _load_model(spec, weights_bytes)
    return spec, model


def _load_model(spec: GameSpec, weights_bytes: bytes) -> Optional[object]:
    if not HAS_TORCH:
        return None
    try:
        from trainer.energy_model import MysteryEnergyModel
        npz = np.load(io.BytesIO(weights_bytes))
        state_dict = {k: torch.tensor(npz[k]) for k in npz.files}

        model = MysteryEnergyModel(
            vocab_size=spec.vocab_size,
            embedding_dim=spec.embedding_dim,
            context_dim=spec.context_dim,
            n_attractor_dims=spec.n_attractor_dims,
        )
        # Strict=False: gracefully handles old cartridges missing retrieval_head
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            pass  # retrieval_head might be new — that's fine
        model.eval()
        return model
    except Exception as exc:
        print(f"[WARN] Could not load model weights: {exc}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Game initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_game(spec: GameSpec, hand_size: int = 12) -> GameState:
    casebook = CasebookState(
        convergence_dimensions=np.zeros(spec.n_attractor_dims, dtype=np.float32),
        convergence_rate=spec.convergence_rate,
    )

    # Opening tokens → atmosphere
    opening = [spec.get_token(tid) for tid in spec.opening_token_ids]

    # Player hand: PLAYER + SHARED non-invariant non-opening tokens
    opening_ids = set(spec.opening_token_ids)
    inv_ids = set(spec.invariant_token_ids)
    player_pool = [
        t for t in spec.tokens
        if not t.is_invariant
        and t.id not in opening_ids
        and t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
    ]
    np.random.shuffle(player_pool)
    hand = player_pool[:hand_size]

    # Engine pool: ENGINE tokens (not opening, not invariant)
    engine_pool = [
        t for t in spec.tokens
        if not t.is_invariant
        and t.id not in opening_ids
        and t.agency == TokenAgency.ENGINE
    ]
    np.random.shuffle(engine_pool)

    return GameState(
        spec=spec,
        casebook=casebook,
        hand=hand,
        atmosphere=list(opening),
        engine_pool=list(engine_pool),
        played_ids={t.id for t in opening},
        turn=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_eval_triad(
    model,
    spec: GameSpec,
    context_tokens: List[Token],
    triad: List[Token],
) -> Tuple[float, List[Tuple[str, float]]]:
    """Run MysteryEnergyModel on a triad. Returns (energy_score, resonance_topk)."""
    if not HAS_TORCH or model is None:
        return 0.5, []

    try:
        from core.token import TokenClass, TokenPhase, TokenStream, TokenAgency

        id_to_idx   = {t.id: i for i, t in enumerate(spec.tokens)}
        cls_to_idx  = {c.value: i for i, c in enumerate(TokenClass)}
        ph_to_idx   = {p.value: i for i, p in enumerate(TokenPhase)}
        str_to_idx  = {s.value: i for i, s in enumerate(TokenStream)}
        agt_to_idx  = {a.value: i for i, a in enumerate(TokenAgency)}

        def _enc(tok_list):
            return (
                [id_to_idx[t.id] for t in tok_list],
                [cls_to_idx[t.token_class.value] for t in tok_list],
                [ph_to_idx[t.phase.value] for t in tok_list],
                [str_to_idx[t.stream.value] for t in tok_list],
                [agt_to_idx[t.agency.value] for t in tok_list],
            )

        def _pad(seqs, pad=0):
            max_l = max(len(s) for s in seqs) if seqs else 1
            arr = np.full((len(seqs), max_l), pad, dtype=np.int64)
            mask = np.ones((len(seqs), max_l), dtype=bool)
            for i, s in enumerate(seqs):
                arr[i, :len(s)] = s
                mask[i, :len(s)] = False
            return torch.tensor(arr), torch.tensor(mask)

        import torch

        # Context
        if context_tokens:
            ti, ci, pi, si, ai = _enc(context_tokens)
            ctx_t, ctx_mask = _pad([ti]), _pad([ci])[0], ...
            ctx_t, ctx_mask = _pad([ti])
            ctx_c, _        = _pad([ci])
            ctx_p, _        = _pad([pi])
            ctx_s, _        = _pad([si])
            ctx_a, _        = _pad([ai])
            n = len(context_tokens)
            pos = torch.zeros(1, n, 2)
        else:
            ctx_t = ctx_c = ctx_p = ctx_s = ctx_a = torch.zeros(1, 1, dtype=torch.long)
            ctx_mask = torch.ones(1, 1, dtype=torch.bool)
            pos = torch.zeros(1, 1, 2)

        # Candidate (triad = 3 tokens as batch B=1, seq=3)
        cti, cci, cpi, csi, cai = _enc(triad)
        cand_t = torch.tensor([cti])
        cand_c = torch.tensor([cci])
        cand_p = torch.tensor([cpi])
        cand_s = torch.tensor([csi])
        cand_a = torch.tensor([cai])

        model.eval()
        with torch.no_grad():
            out = model(
                ctx_t, ctx_c, ctx_p, pos, ctx_mask,
                cand_t, cand_c, cand_p,
                placed_stream_ids=ctx_s,
                placed_agency_ids=ctx_a,
                candidate_stream_ids=cand_s,
                candidate_agency_ids=cand_a,
            )

        energy_score = float(out["energy"][0, 0])

        # Resonance: top-5 from hand
        logits = out["resonance_logits"][0]  # (V,)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        topk = sorted(
            [(spec.tokens[i].id, float(probs[i])) for i in range(len(spec.tokens))],
            key=lambda x: -x[1],
        )[:5]
        return energy_score, topk

    except Exception:
        return 0.5, []


def _model_retrieve_invariants(
    model,
    spec: GameSpec,
    context_tokens: List[Token],
) -> Optional[List[str]]:
    """
    Use HopfieldRetrievalHead to predict which token per dimension is the
    invariant. Returns a list of n_dims token IDs (may be wrong early on).
    Returns None if model unavailable.
    """
    if not HAS_TORCH or model is None or not hasattr(model, "retrieval_head"):
        return None

    try:
        import torch
        from core.token import TokenClass, TokenPhase, TokenStream, TokenAgency

        id_to_idx  = {t.id: i for i, t in enumerate(spec.tokens)}
        cls_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
        ph_to_idx  = {p.value: i for i, p in enumerate(TokenPhase)}
        str_to_idx = {s.value: i for i, s in enumerate(TokenStream)}
        agt_to_idx = {a.value: i for i, a in enumerate(TokenAgency)}

        dummy_triad = context_tokens[:3] if len(context_tokens) >= 3 else spec.tokens[:3]

        def _enc(tok_list):
            return (
                [id_to_idx[t.id] for t in tok_list],
                [cls_to_idx[t.token_class.value] for t in tok_list],
                [ph_to_idx[t.phase.value] for t in tok_list],
                [str_to_idx[t.stream.value] for t in tok_list],
                [agt_to_idx[t.agency.value] for t in tok_list],
            )

        def _pad(seqs, pad=0):
            max_l = max(len(s) for s in seqs)
            arr = np.full((len(seqs), max_l), pad, dtype=np.int64)
            mask = np.ones((len(seqs), max_l), dtype=bool)
            for i, s in enumerate(seqs):
                arr[i, :len(s)] = s
                mask[i, :len(s)] = False
            return torch.tensor(arr), torch.tensor(mask)

        if context_tokens:
            ti, ci, pi, si, ai = _enc(context_tokens)
            ctx_t, ctx_mask = _pad([ti])
            ctx_c, _ = _pad([ci])
            ctx_p, _ = _pad([pi])
            ctx_s, _ = _pad([si])
            ctx_a, _ = _pad([ai])
            pos = torch.zeros(1, len(context_tokens), 2)
        else:
            ctx_t = ctx_c = ctx_p = ctx_s = ctx_a = torch.zeros(1, 1, dtype=torch.long)
            ctx_mask = torch.ones(1, 1, dtype=torch.bool)
            pos = torch.zeros(1, 1, 2)

        cti, cci, cpi, csi, cai = _enc(dummy_triad)
        cand_t = torch.tensor([cti])
        cand_c = torch.tensor([cci])
        cand_p = torch.tensor([cpi])
        cand_s = torch.tensor([csi])
        cand_a = torch.tensor([cai])

        model.eval()
        with torch.no_grad():
            out = model(
                ctx_t, ctx_c, ctx_p, pos, ctx_mask,
                cand_t, cand_c, cand_p,
                placed_stream_ids=ctx_s,
                placed_agency_ids=ctx_a,
                candidate_stream_ids=cand_s,
                candidate_agency_ids=cand_a,
            )

        if "retrieval_logits" not in out:
            return None

        retrieval = out["retrieval_logits"][0]  # (n_dims, V)
        predicted = retrieval.argmax(dim=-1).cpu().tolist()  # n_dims ints
        return [spec.tokens[idx].id for idx in predicted]

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Game actions
# ─────────────────────────────────────────────────────────────────────────────

def action_play_triad(
    state: GameState,
    indices: List[int],
    model,
) -> str:
    """Place a triad of 3 hand tokens. Returns a message string."""
    if len(indices) != 3:
        return "[red]Pick exactly 3 tokens.[/red]"

    # Validate
    for i in indices:
        if i < 0 or i >= len(state.hand):
            return f"[red]Invalid token number {i + 1}. Hand has {len(state.hand)} tokens.[/red]"
    if len(set(indices)) != 3:
        return "[red]Pick 3 different tokens.[/red]"

    triad = [state.hand[i] for i in indices]

    # Evaluate with model
    context_tokens = list(state.atmosphere) + [
        t for t in state.casebook.all_placed_tokens()
    ]
    energy_score, resonance = _model_eval_triad(model, state.spec, context_tokens, triad)

    # Update casebook
    pos = (state.turn // 6, state.turn % 6)
    state.casebook.place_triad(triad, pos)

    # Remove from hand, track played
    for tok in triad:
        state.played_ids.add(tok.id)
    new_hand = [t for t in state.hand if t.id not in state.played_ids]
    state.hand = new_hand

    state.turn += 1

    names = " + ".join(_tok_label(t) for t in triad)
    conv = state.casebook.convergence_score
    msg = (
        f"[bold green]Triad placed:[/bold green] {names}\n"
        f"  energy={energy_score:.3f}  convergence=[bold cyan]{conv:.1%}[/bold cyan]"
    )
    if resonance:
        top3 = resonance[:3]
        hints = "  ".join(f"[dim]{_tok_label_id(tid, state.spec)} ({p:.1%})[/dim]" for tid, p in top3)
        msg += f"\n  resonance hints: {hints}"

    return msg


def action_advance(state: GameState, model) -> str:
    """Engine places the next atmosphere token (energy-guided)."""
    if not state.engine_pool:
        return "[yellow]No more atmosphere tokens available.[/yellow]"

    placed_ids = [t.id for t in state.atmosphere] + list(state.casebook.placed_token_ids())

    # Score engine tokens by how much they lower the Hopfield energy
    candidates = state.engine_pool[:10]
    scored = []
    for tok in candidates:
        energy = state.spec.token_graph.induced_subgraph_energy(
            placed_ids, [tok.id]
        )
        scored.append((tok, energy))
    scored.sort(key=lambda x: x[1])
    chosen = scored[0][0]

    state.engine_pool.remove(chosen)
    state.atmosphere.append(chosen)
    state.played_ids.add(chosen.id)

    return (
        f"[bold yellow]Scene advances:[/bold yellow] {_tok_label(chosen)} "
        f"[dim]({chosen.token_class.value})[/dim] enters the atmosphere."
    )


def action_accuse(state: GameState, suspect_id: str, event_id: str, motive_id: str) -> str:
    """Check the accusation against the invariants."""
    inv_ids = set(state.spec.invariant_token_ids)

    # Identify what each invariant IS by attractor_weights
    invariants = {
        state.spec.get_token(tid): tid
        for tid in state.spec.invariant_token_ids
    }

    suspect_inv = next((t for t in invariants if t.token_class == TokenClass.SUSPECT), None)
    event_inv   = next((t for t in invariants if t.token_class == TokenClass.EVENT), None)
    motive_inv  = next((t for t in invariants if t.token_class == TokenClass.MOTIVE), None)

    correct_s = suspect_inv and suspect_inv.id == suspect_id
    correct_e = event_inv   and event_inv.id   == event_id
    correct_m = motive_inv  and motive_inv.id  == motive_id

    if correct_s and correct_e and correct_m:
        state.solved = True
        state.game_over = True
        return (
            "[bold green]SOLVED.[/bold green] "
            f"[green]{_tok_label_id(suspect_id, state.spec)}[/green] committed the act "
            f"during [green]{_tok_label_id(event_id, state.spec)}[/green] "
            f"driven by [green]{_tok_label_id(motive_id, state.spec)}[/green]."
        )
    else:
        wrongs = []
        if not correct_s:
            wrongs.append(f"suspect is wrong")
        if not correct_e:
            wrongs.append(f"event is wrong")
        if not correct_m:
            wrongs.append(f"motive is wrong")
        return (
            f"[bold red]Accusation fails:[/bold red] {', '.join(wrongs)}. "
            "Keep investigating."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
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
    "UNKNOWN":  "white",
}


def _tok_label(tok: Token) -> str:
    color = _CLASS_COLORS.get(tok.token_class.value, "white")
    label = tok.surface_expression or tok.id
    return f"[{color}]{label}[/{color}]"


def _tok_label_id(token_id: str, spec: GameSpec) -> str:
    try:
        return _tok_label(spec.get_token(token_id))
    except KeyError:
        return token_id


def _conv_bar(score: float, threshold: float, width: int = 20) -> str:
    filled = int(score * width)
    thr_pos = int(threshold * width)
    bar = ""
    for i in range(width):
        if i == thr_pos:
            bar += "|"
        elif i < filled:
            bar += "█"
        else:
            bar += "░"
    pct = f"{score:.0%}"
    color = "green" if score >= threshold else ("yellow" if score > threshold * 0.5 else "red")
    return f"[{color}]{bar} {pct}[/{color}]"


def render_header(state: GameState) -> Panel:
    turn_str = f"Turn [bold]{state.turn}[/bold]/{state.spec.max_turns}"
    conv_str = _conv_bar(state.casebook.convergence_score, state.spec.convergence_threshold)
    threshold_str = f"threshold={state.spec.convergence_threshold:.0%}"

    dims = state.casebook.convergence_dimensions
    dim_str = "  ".join(
        f"dim{d}=[cyan]{v:.2f}[/cyan]" for d, v in enumerate(dims)
    )

    text = Text()
    text.append(f"  {turn_str}    convergence: ")
    text.append_text(Text.from_markup(conv_str))
    text.append(f"  ({threshold_str})\n  {dim_str}")

    return Panel(text, title=f"[bold white]{state.spec.title}[/bold white]", border_style="bright_blue")


def render_atmosphere(state: GameState) -> Panel:
    if not state.atmosphere:
        body = "[dim]( empty )[/dim]"
    else:
        rows = []
        for tok in state.atmosphere:
            label = _tok_label(tok)
            rows.append(f"  {label}  [dim]{tok.token_class.value}[/dim]")
        body = "\n".join(rows)
    return Panel(body, title="[yellow]ATMOSPHERE[/yellow]", border_style="yellow", padding=(0, 1))


def render_hand(state: GameState) -> Panel:
    if not state.hand:
        return Panel("[dim]Hand is empty[/dim]", title="[blue]YOUR HAND[/blue]", border_style="blue")

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("num", style="dim", width=4)
    table.add_column("token", min_width=28)
    table.add_column("class", style="dim", width=10)
    table.add_column("phase", style="dim", width=8)

    for i, tok in enumerate(state.hand, start=1):
        label = _tok_label(tok)
        table.add_row(
            f"[dim][{i}][/dim]",
            Text.from_markup(label),
            tok.token_class.value,
            tok.phase.value,
        )

    return Panel(table, title="[blue]YOUR HAND[/blue]", border_style="blue", padding=(0, 1))


def render_theory(state: GameState) -> Panel:
    placed = state.casebook.placed_triads
    if not placed:
        body = "[dim]( no triads placed yet )[/dim]"
    else:
        lines = []
        for turn_idx, (pos, triad) in enumerate(placed.items(), start=1):
            names = "  +  ".join(_tok_label(t) for t in triad)
            lines.append(f"  [dim]T{turn_idx}[/dim]  {names}")
        body = "\n".join(lines)
    return Panel(body, title="[green]THEORY BOARD[/green]", border_style="green", padding=(0, 1))


def render_help() -> Panel:
    text = (
        "[bold]play[/bold] [cyan]1 3 5[/cyan]    — place a triad (token numbers from hand)\n"
        "[bold]advance[/bold]          — engine places next atmosphere token\n"
        "[bold]accuse[/bold]           — make your accusation (unlocks at ~75% convergence)\n"
        "[bold]hint[/bold]             — show model's resonance hints + retrieval prediction\n"
        "[bold]hand[/bold]             — redisplay your hand\n"
        "[bold]field[/bold]            — redisplay full game state\n"
        "[bold]tokens[/bold]           — list all tokens with IDs (for accusation)\n"
        "[bold]quit[/bold] / [bold]q[/bold]         — exit"
    )
    return Panel(text, title="[dim]COMMANDS[/dim]", border_style="dim", padding=(0, 1))


def render_full_state(state: GameState) -> None:
    if not HAS_RICH:
        _render_plain(state)
        return
    console.print(render_header(state))
    console.print(Columns([render_atmosphere(state), render_theory(state)], equal=True))
    console.print(render_hand(state))


def _render_plain(state: GameState) -> None:
    score = state.casebook.convergence_score
    print(f"\n=== {state.spec.title} | Turn {state.turn}/{state.spec.max_turns} | Conv {score:.1%} ===")
    print("ATMOSPHERE:", [t.id for t in state.atmosphere])
    print("HAND:")
    for i, t in enumerate(state.hand, start=1):
        print(f"  [{i}] {t.id}")
    print("THEORY BOARD:")
    for pos, triad in state.casebook.placed_triads.items():
        print(f"  {[t.id for t in triad]}")


# ─────────────────────────────────────────────────────────────────────────────
# Token listing for accusation
# ─────────────────────────────────────────────────────────────────────────────

def render_token_list(spec: GameSpec, filter_class: Optional[str] = None) -> None:
    table = Table(title="Token IDs", box=box.SIMPLE)
    table.add_column("ID", style="dim", max_width=40)
    table.add_column("Class", width=12)
    table.add_column("Name", min_width=28)
    table.add_column("Phase", width=8)

    classes_of_interest = {"SUSPECT", "EVENT", "MOTIVE"}
    for tok in spec.tokens:
        if filter_class and tok.token_class.value != filter_class:
            continue
        if not filter_class and tok.token_class.value not in classes_of_interest:
            continue
        color = _CLASS_COLORS.get(tok.token_class.value, "white")
        table.add_row(
            tok.id,
            f"[{color}]{tok.token_class.value}[/{color}]",
            tok.surface_expression or tok.id,
            tok.phase.value,
        )
    if HAS_RICH:
        console.print(table)
    else:
        for tok in spec.tokens:
            if tok.token_class.value in classes_of_interest:
                print(f"  {tok.id}  ({tok.token_class.value})")


# ─────────────────────────────────────────────────────────────────────────────
# Accusation wizard
# ─────────────────────────────────────────────────────────────────────────────

def run_accusation_wizard(state: GameState, model) -> str:
    if HAS_RICH:
        console.print("\n[bold yellow]— ACCUSATION WIZARD —[/bold yellow]")
        console.print("[dim]Type the token ID exactly (use 'tokens' to list).[/dim]")

        # Optionally show retrieval prediction
        context_tokens = list(state.atmosphere) + state.casebook.all_placed_tokens()
        predicted = _model_retrieve_invariants(model, state.spec, context_tokens)
        if predicted:
            console.print("[dim]Model retrieval hints (may be wrong):[/dim]")
            for d, pid in enumerate(predicted):
                console.print(f"  dim{d}: [dim]{_tok_label_id(pid, state.spec)} ({pid})[/dim]")

        suspect_id = Prompt.ask("  [red]Suspect[/red] (token ID)")
        event_id   = Prompt.ask("  [yellow]Event / mechanism[/yellow] (token ID)")
        motive_id  = Prompt.ask("  [magenta]Motive[/magenta] (token ID)")
    else:
        print("Enter token IDs for your accusation:")
        suspect_id = input("  Suspect ID: ").strip()
        event_id   = input("  Event/mechanism ID: ").strip()
        motive_id  = input("  Motive ID: ").strip()

    return action_accuse(state, suspect_id, event_id, motive_id)


# ─────────────────────────────────────────────────────────────────────────────
# Main game loop
# ─────────────────────────────────────────────────────────────────────────────

def game_loop(state: GameState, model) -> None:
    if HAS_RICH:
        console.rule("[bold white]THORNFIELD[/bold white]")
        console.print(f"[bold]Case:[/bold] {state.spec.title}  |  "
                      f"[dim]vocab={state.spec.vocab_size}  "
                      f"threshold={state.spec.convergence_threshold:.0%}  "
                      f"turns={state.spec.min_turns}–{state.spec.max_turns}[/dim]")
        console.print(render_help())

    render_full_state(state)

    while not state.game_over:
        # Max-turns check
        if state.turn >= state.spec.max_turns:
            if HAS_RICH:
                console.print("[bold red]Time is up — the case goes cold.[/bold red]")
            else:
                print("Max turns reached.")
            state.game_over = True
            break

        # Prompt
        if HAS_RICH:
            conv = state.casebook.convergence_score
            threshold = state.spec.convergence_threshold
            accuse_hint = " [dim](accuse?)[/dim]" if conv >= threshold * 0.9 else ""
            raw = Prompt.ask(
                f"\n[bold white]>[/bold white]{accuse_hint}",
                default="",
            ).strip()
        else:
            conv = state.casebook.convergence_score
            raw = input(f"\n[turn {state.turn}  conv {conv:.1%}] > ").strip()

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        # ---- parse --------------------------------------------------------
        if cmd in ("quit", "q", "exit"):
            if HAS_RICH:
                console.print("[dim]Farewell, detective.[/dim]")
            break

        elif cmd == "help":
            if HAS_RICH:
                console.print(render_help())

        elif cmd in ("field", "state", "show"):
            render_full_state(state)

        elif cmd == "hand":
            if HAS_RICH:
                console.print(render_hand(state))
            else:
                for i, t in enumerate(state.hand, 1):
                    print(f"  [{i}] {t.id}")

        elif cmd == "tokens":
            render_token_list(state.spec)

        elif cmd == "hint":
            context_tokens = list(state.atmosphere) + state.casebook.all_placed_tokens()
            _, resonance = _model_eval_triad(
                model, state.spec, context_tokens,
                state.hand[:3] if len(state.hand) >= 3 else state.hand,
            )
            predicted = _model_retrieve_invariants(model, state.spec, context_tokens)

            if HAS_RICH:
                if resonance:
                    console.print("[dim]Resonance (field wants):[/dim]")
                    for tid, p in resonance[:5]:
                        console.print(f"  {_tok_label_id(tid, state.spec)} [dim]({p:.1%})[/dim]")
                if predicted:
                    console.print("[dim]Retrieval prediction (Hopfield read):[/dim]")
                    for d, pid in enumerate(predicted):
                        console.print(f"  dim{d} → {_tok_label_id(pid, state.spec)} [dim]({pid})[/dim]")
                if not resonance and not predicted:
                    console.print("[dim]No model loaded — symbolic mode only.[/dim]")
            else:
                print("Resonance:", [tid for tid, _ in resonance[:5]])
                if predicted:
                    print("Retrieval:", predicted)

        elif cmd == "advance":
            msg = action_advance(state, model)
            if HAS_RICH:
                console.print(msg)
                console.print(render_atmosphere(state))
            else:
                print(msg)

        elif cmd == "play":
            # play 1 3 5  OR  play 1 2 3  (1-based)
            if len(parts) < 4:
                if HAS_RICH:
                    console.print("[red]Usage: play <n1> <n2> <n3>[/red]")
                continue
            try:
                nums = [int(p) - 1 for p in parts[1:4]]  # 0-based
            except ValueError:
                if HAS_RICH:
                    console.print("[red]Invalid numbers.[/red]")
                continue

            msg = action_play_triad(state, nums, model)
            if HAS_RICH:
                console.print(msg)
                console.print(render_header(state))
            else:
                print(msg)

            # Check for accuse opportunity
            conv = state.casebook.convergence_score
            if conv >= state.spec.convergence_threshold and HAS_RICH:
                console.print(
                    f"\n[bold green]Convergence reached {conv:.1%}! "
                    "Type [bold]accuse[/bold] to name the culprit.[/bold green]"
                )

        elif cmd == "accuse":
            conv = state.casebook.convergence_score
            if conv < state.spec.convergence_threshold * 0.7 and state.turn < state.spec.min_turns:
                if HAS_RICH:
                    console.print(
                        f"[yellow]Too early — convergence is only {conv:.1%}. "
                        "Keep investigating.[/yellow]"
                    )
                continue

            msg = run_accusation_wizard(state, model)
            if HAS_RICH:
                console.print(msg)
            else:
                print(msg)

            if state.solved:
                break

        else:
            if HAS_RICH:
                console.print(f"[dim]Unknown command '{cmd}'. Type [bold]help[/bold].[/dim]")

    # ── End screen ──────────────────────────────────────────────────────────
    if HAS_RICH:
        if state.solved:
            console.rule("[bold green]CASE CLOSED[/bold green]")
            console.print("[bold green]You solved The Amber Cipher.[/bold green]")
        else:
            console.rule("[bold red]UNSOLVED[/bold red]")
            inv = state.spec.invariant_token_ids
            console.print("[bold red]The truth:[/bold red]")
            for tid in inv:
                tok = state.spec.get_token(tid)
                console.print(f"  {_tok_label(tok)}  [dim]({tid})[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Thornfield cartridge runner")
    parser.add_argument("cartridge", help="Path to .cartridge file")
    parser.add_argument("--hand-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if HAS_RICH:
        with console.status("[bold]Loading cartridge…[/bold]"):
            spec, model = load_cartridge(args.cartridge)
    else:
        print(f"Loading {args.cartridge}…")
        spec, model = load_cartridge(args.cartridge)

    if HAS_RICH:
        model_status = "[green]loaded[/green]" if model else "[yellow]symbolic mode[/yellow]"
        console.print(
            f"[bold]{spec.title}[/bold]  "
            f"vocab={spec.vocab_size}  model={model_status}"
        )

    state = init_game(spec, hand_size=args.hand_size)
    game_loop(state, model)


if __name__ == "__main__":
    main()
