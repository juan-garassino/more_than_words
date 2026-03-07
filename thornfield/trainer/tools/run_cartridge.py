"""
Thornfield — Mystery Card Runner
=================================
You are the detective. Cards are clues. Connect three at a time.
The field responds. The truth converges.

Usage
-----
    cd thornfield/trainer
    python3 tools/run_cartridge.py outputs/amber_cipher/TheAmberCipher.cartridge

Commands
--------
    connect <a> <b> <c>   — lay three cards on the table as a connection
    examine <n>           — inspect a card closely
    scene                 — let the world advance (engine places next token)
    hint                  — the field whispers what it wants next
    hand                  — redisplay your hand
    connections           — show connections made so far
    accuse                — name the culprit (unlocks when field converges)
    quit / q              — walk away
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
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
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.prompt import Prompt
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("pip install rich  for the full UI", file=sys.stderr)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from core.token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency
from core.hopfield import TokenGraph
from core.casebook import CasebookState

console = Console() if HAS_RICH else None

HAND_SIZE = 9       # cards in hand at all times
REFILL_AFTER = True # draw back to HAND_SIZE after each connection


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameSpec:
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

    def get_token(self, tid: str) -> Token:
        for t in self.tokens:
            if t.id == tid:
                return t
        raise KeyError(tid)

    @property
    def invariant_tokens(self) -> List[Token]:
        return [self.get_token(tid) for tid in self.invariant_token_ids]


@dataclass
class PlacedConnection:
    triad: List[Token]
    energy: float
    turn: int


@dataclass
class GameState:
    spec: GameSpec
    casebook: CasebookState
    hand: List[Token]
    deck: List[Token]           # remaining undealt player cards
    atmosphere: List[Token]
    engine_pool: List[Token]
    connections: List[PlacedConnection] = field(default_factory=list)
    played_ids: set = field(default_factory=set)
    turn: int = 0
    solved: bool = False
    game_over: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Cartridge loading
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
    zpath = Path(path)
    if not zpath.exists():
        raise FileNotFoundError(f"Cartridge not found: {zpath}")

    with zipfile.ZipFile(zpath, "r") as zf:
        manifest    = json.loads(zf.read("manifest.json"))
        tokens_data = json.loads(zf.read("tokens.json"))
        graph_data  = json.loads(zf.read("graph.json"))
        attractor   = json.loads(zf.read("attractor.json"))
        phases      = json.loads(zf.read("phases.json"))
        weights_b   = zf.read("weights.npz")

    tokens = [_load_token(t) for t in tokens_data]

    opening_ids: List[str] = phases.get("opening_token_ids", [])
    if not opening_ids:
        opening_ids = [t.id for t in tokens if t.stream == TokenStream.OPENING]

    spec = GameSpec(
        case_id=manifest["case_id"],
        title=manifest["title"],
        vocab_size=manifest["vocab_size"],
        embedding_dim=manifest["embedding_dim"],
        context_dim=manifest["context_dim"],
        n_attractor_dims=manifest["n_attractor_dims"],
        convergence_threshold=manifest["convergence_threshold"],
        convergence_rate=manifest["convergence_rate"],
        min_turns=phases["min_turns"],
        max_turns=phases["max_turns"],
        tokens=tokens,
        token_graph=TokenGraph.from_json(graph_data),
        opening_token_ids=opening_ids,
        invariant_token_ids=attractor["invariants"],
    )
    model = _load_model(spec, weights_b)
    return spec, model


def _load_model(spec: GameSpec, weights_bytes: bytes):
    if not HAS_TORCH:
        return None
    try:
        from trainer.energy_model import MysteryEnergyModel
        npz = np.load(io.BytesIO(weights_bytes))
        state = {k: torch.tensor(npz[k]) for k in npz.files}
        model = MysteryEnergyModel(
            vocab_size=spec.vocab_size,
            embedding_dim=spec.embedding_dim,
            context_dim=spec.context_dim,
            n_attractor_dims=spec.n_attractor_dims,
        )
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Game init
# ─────────────────────────────────────────────────────────────────────────────

def init_game(spec: GameSpec) -> GameState:
    casebook = CasebookState(
        convergence_dimensions=np.zeros(spec.n_attractor_dims, dtype=np.float32),
        convergence_rate=spec.convergence_rate,
    )

    opening_ids = set(spec.opening_token_ids)
    inv_ids     = set(spec.invariant_token_ids)

    player_pool = [
        t for t in spec.tokens
        if not t.is_invariant
        and t.id not in opening_ids
        and t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
    ]
    np.random.shuffle(player_pool)

    engine_pool = [
        t for t in spec.tokens
        if not t.is_invariant
        and t.id not in opening_ids
        and t.agency == TokenAgency.ENGINE
    ]
    np.random.shuffle(engine_pool)

    hand = player_pool[:HAND_SIZE]
    deck = player_pool[HAND_SIZE:]

    opening = [spec.get_token(tid) for tid in spec.opening_token_ids]

    return GameState(
        spec=spec,
        casebook=casebook,
        hand=hand,
        deck=deck,
        atmosphere=list(opening),
        engine_pool=list(engine_pool),
        played_ids={t.id for t in opening},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Card naming
# ─────────────────────────────────────────────────────────────────────────────

_ARTICLE_CLASSES = {
    TokenClass.OBJECT, TokenClass.ACTION, TokenClass.EMOTION,
    TokenClass.MODIFIER, TokenClass.EVENT, TokenClass.MOTIVE,
}

def card_name(tok: Token) -> str:
    """Human-readable card name. Uses surface_expression if available."""
    if tok.surface_expression:
        return tok.surface_expression
    raw = tok.id.split(":")[-1].replace("_", " ").title()
    if tok.token_class in _ARTICLE_CLASSES:
        return f"The {raw}"
    return raw


def card_name_rich(tok: Token) -> str:
    """Rich markup version of card_name with class colour."""
    name = card_name(tok)
    color = _CLASS_COLORS.get(tok.token_class.value, "white")
    return f"[{color}]{name}[/{color}]"


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
}


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_score_triad(model, spec: GameSpec, context: List[Token], triad: List[Token]) -> float:
    """Returns energy score 0–1, or 0.5 if model unavailable."""
    if not HAS_TORCH or model is None:
        return 0.5
    try:
        from core.token import TokenClass, TokenPhase, TokenStream, TokenAgency
        id2i  = {t.id: i for i, t in enumerate(spec.tokens)}
        cl2i  = {c.value: i for i, c in enumerate(TokenClass)}
        ph2i  = {p.value: i for i, p in enumerate(TokenPhase)}
        st2i  = {s.value: i for i, s in enumerate(TokenStream)}
        ag2i  = {a.value: i for i, a in enumerate(TokenAgency)}

        def enc(lst):
            return ([id2i[t.id] for t in lst], [cl2i[t.token_class.value] for t in lst],
                    [ph2i[t.phase.value] for t in lst], [st2i[t.stream.value] for t in lst],
                    [ag2i[t.agency.value] for t in lst])

        def pad(seqs):
            m = max(len(s) for s in seqs)
            a = np.full((len(seqs), m), 0, np.int64)
            mk = np.ones((len(seqs), m), bool)
            for i, s in enumerate(seqs):
                a[i, :len(s)] = s; mk[i, :len(s)] = False
            return torch.tensor(a), torch.tensor(mk)

        if context:
            ti, ci, pi, si, ai = enc(context)
            ct, cm = pad([ti]); cc, _ = pad([ci]); cp, _ = pad([pi])
            cs, _ = pad([si]);  ca, _ = pad([ai])
            pos = torch.zeros(1, len(context), 2)
        else:
            ct = cc = cp = cs = ca = torch.zeros(1, 1, dtype=torch.long)
            cm = torch.ones(1, 1, dtype=torch.bool)
            pos = torch.zeros(1, 1, 2)

        cti, cci, cpi, csi, cai = enc(triad)
        with torch.no_grad():
            out = model(ct, cc, cp, pos, cm,
                        torch.tensor([cti]), torch.tensor([cci]), torch.tensor([cpi]),
                        placed_stream_ids=cs, placed_agency_ids=ca,
                        candidate_stream_ids=torch.tensor([csi]),
                        candidate_agency_ids=torch.tensor([cai]))
        return float(out["energy"][0, 0])
    except Exception:
        return 0.5


def _model_resonance(model, spec: GameSpec, context: List[Token]) -> List[Tuple[str, float]]:
    """Top token IDs the field resonates with given current context."""
    if not HAS_TORCH or model is None:
        return []
    try:
        from core.token import TokenClass, TokenPhase, TokenStream, TokenAgency
        id2i = {t.id: i for i, t in enumerate(spec.tokens)}
        cl2i = {c.value: i for i, c in enumerate(TokenClass)}
        ph2i = {p.value: i for i, p in enumerate(TokenPhase)}
        st2i = {s.value: i for i, s in enumerate(TokenStream)}
        ag2i = {a.value: i for i, a in enumerate(TokenAgency)}

        dummy = context[:3] if len(context) >= 3 else spec.tokens[:3]

        def enc(lst):
            return ([id2i[t.id] for t in lst], [cl2i[t.token_class.value] for t in lst],
                    [ph2i[t.phase.value] for t in lst], [st2i[t.stream.value] for t in lst],
                    [ag2i[t.agency.value] for t in lst])

        def pad(seqs):
            m = max(len(s) for s in seqs)
            a = np.full((len(seqs), m), 0, np.int64)
            mk = np.ones((len(seqs), m), bool)
            for i, s in enumerate(seqs):
                a[i, :len(s)] = s; mk[i, :len(s)] = False
            return torch.tensor(a), torch.tensor(mk)

        if context:
            ti, ci, pi, si, ai = enc(context)
            ct, cm = pad([ti]); cc, _ = pad([ci]); cp, _ = pad([pi])
            cs, _ = pad([si]);  ca, _ = pad([ai])
            pos = torch.zeros(1, len(context), 2)
        else:
            ct = cc = cp = cs = ca = torch.zeros(1, 1, dtype=torch.long)
            cm = torch.ones(1, 1, dtype=torch.bool)
            pos = torch.zeros(1, 1, 2)

        cti, cci, cpi, csi, cai = enc(dummy)
        with torch.no_grad():
            out = model(ct, cc, cp, pos, cm,
                        torch.tensor([cti]), torch.tensor([cci]), torch.tensor([cpi]),
                        placed_stream_ids=cs, placed_agency_ids=ca,
                        candidate_stream_ids=torch.tensor([csi]),
                        candidate_agency_ids=torch.tensor([cai]))

        probs = torch.softmax(out["resonance_logits"][0], dim=-1).cpu().numpy()
        scored = [(spec.tokens[i].id, float(probs[i])) for i in range(len(spec.tokens))
                  if not spec.tokens[i].is_invariant]
        scored.sort(key=lambda x: -x[1])
        return scored[:6]
    except Exception:
        return []


def _model_retrieval(model, spec: GameSpec, context: List[Token]) -> Optional[List[str]]:
    """Hopfield read — predicted invariant token per dimension."""
    if not HAS_TORCH or model is None or not hasattr(model, "retrieval_head"):
        return None
    try:
        from core.token import TokenClass, TokenPhase, TokenStream, TokenAgency
        id2i = {t.id: i for i, t in enumerate(spec.tokens)}
        cl2i = {c.value: i for i, c in enumerate(TokenClass)}
        ph2i = {p.value: i for i, p in enumerate(TokenPhase)}
        st2i = {s.value: i for i, s in enumerate(TokenStream)}
        ag2i = {a.value: i for i, a in enumerate(TokenAgency)}

        dummy = context[:3] if len(context) >= 3 else spec.tokens[:3]

        def enc(lst):
            return ([id2i[t.id] for t in lst], [cl2i[t.token_class.value] for t in lst],
                    [ph2i[t.phase.value] for t in lst], [st2i[t.stream.value] for t in lst],
                    [ag2i[t.agency.value] for t in lst])

        def pad(seqs):
            m = max(len(s) for s in seqs)
            a = np.full((len(seqs), m), 0, np.int64)
            mk = np.ones((len(seqs), m), bool)
            for i, s in enumerate(seqs):
                a[i, :len(s)] = s; mk[i, :len(s)] = False
            return torch.tensor(a), torch.tensor(mk)

        if context:
            ti, ci, pi, si, ai = enc(context)
            ct, cm = pad([ti]); cc, _ = pad([ci]); cp, _ = pad([pi])
            cs, _ = pad([si]);  ca, _ = pad([ai])
            pos = torch.zeros(1, len(context), 2)
        else:
            ct = cc = cp = cs = ca = torch.zeros(1, 1, dtype=torch.long)
            cm = torch.ones(1, 1, dtype=torch.bool)
            pos = torch.zeros(1, 1, 2)

        cti, cci, cpi, csi, cai = enc(dummy)
        with torch.no_grad():
            out = model(ct, cc, cp, pos, cm,
                        torch.tensor([cti]), torch.tensor([cci]), torch.tensor([cpi]),
                        placed_stream_ids=cs, placed_agency_ids=ca,
                        candidate_stream_ids=torch.tensor([csi]),
                        candidate_agency_ids=torch.tensor([cai]))

        if "retrieval_logits" not in out:
            return None
        idxs = out["retrieval_logits"][0].argmax(dim=-1).cpu().tolist()
        return [spec.tokens[i].id for i in idxs]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Narrative feedback
# ─────────────────────────────────────────────────────────────────────────────

def _energy_narrative(energy: float) -> Tuple[str, str]:
    """
    Returns (narrative, delta_colour).
    The model outputs LOW energy for correct triads (Hopfield convention:
    low energy = stable attractor basin).  Coherence = 1 - energy.
    """
    coherence = 1.0 - energy
    if coherence >= 0.80:
        return "[bold green]The pieces lock together.[/bold green]", "green"
    if coherence >= 0.60:
        return "[green]The connection holds.[/green]", "green"
    if coherence >= 0.40:
        return "[yellow]A tenuous link — but something is there.[/yellow]", "yellow"
    if coherence >= 0.20:
        return "[red]The field resists. A weak trail.[/red]", "red"
    return "[bold red]A false lead. The field pushes back.[/bold red]", "red"


def _convergence_narrative(score: float, threshold: float) -> str:
    ratio = score / threshold
    if ratio >= 1.0:
        return "[bold green]The picture is complete. You know the truth.[/bold green]"
    if ratio >= 0.85:
        return "[green]The shape of the crime is almost clear.[/green]"
    if ratio >= 0.65:
        return "[yellow]The fog is lifting. Stay the course.[/yellow]"
    if ratio >= 0.40:
        return "[yellow]Fragments. The case is taking shape.[/yellow]"
    if ratio >= 0.20:
        return "[dim]Scattered clues. The truth is buried.[/dim]"
    return "[dim]The field is silent. Everything is still unclear.[/dim]"


def _turns_narrative(turns_left: int) -> str:
    if turns_left <= 2:
        return "[bold red]The window is closing.[/bold red]"
    if turns_left <= 5:
        return "[red]Time is running short.[/red]"
    if turns_left <= 8:
        return "[yellow]Several turns remain.[/yellow]"
    return "[dim]You have time.[/dim]"


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _render_scene(state: GameState) -> Panel:
    lines = []
    for tok in state.atmosphere:
        icon = _CLASS_ICONS.get(tok.token_class.value, "·")
        color = _CLASS_COLORS.get(tok.token_class.value, "white")
        lines.append(f"  [{color}]{icon}  {card_name(tok)}[/{color}]")
    body = "\n".join(lines) if lines else "[dim]  ( the scene is empty )[/dim]"
    return Panel(body, title="[yellow]THE SCENE[/yellow]", border_style="yellow", padding=(0, 1))


def _render_connections(state: GameState) -> Panel:
    if not state.connections:
        body = "[dim]  ( no connections yet — lay three cards )[/dim]"
    else:
        lines = []
        for i, conn in enumerate(state.connections, 1):
            names = "  +  ".join(card_name_rich(t) for t in conn.triad)
            lines.append(f"  [dim]{i}.[/dim]  {names}")
        body = "\n".join(lines)
    return Panel(body, title="[green]YOUR CONNECTIONS[/green]", border_style="green", padding=(0, 1))


def _render_hand(state: GameState) -> Panel:
    if not state.hand:
        return Panel("[dim]  Hand empty — no cards left to play.[/dim]",
                     title="[blue]YOUR HAND[/blue]", border_style="blue")
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("n",    style="dim",  width=4)
    table.add_column("icon", width=2)
    table.add_column("name", min_width=26)
    table.add_column("type", style="dim",  width=10)
    table.add_column("phase", style="dim", width=8)

    for i, tok in enumerate(state.hand, 1):
        color = _CLASS_COLORS.get(tok.token_class.value, "white")
        icon  = _CLASS_ICONS.get(tok.token_class.value, "·")
        table.add_row(
            f"[dim][{i}][/dim]",
            f"[{color}]{icon}[/{color}]",
            Text.from_markup(card_name_rich(tok)),
            tok.token_class.value,
            tok.phase.value,
        )
    return Panel(table, title="[blue]YOUR HAND[/blue]", border_style="blue", padding=(0, 1))


def _render_status(state: GameState) -> str:
    score     = state.casebook.convergence_score
    threshold = state.spec.convergence_threshold
    turns_left = state.spec.max_turns - state.turn
    filled = int((score / threshold) * 16)
    bar    = "█" * min(filled, 16) + "░" * (16 - min(filled, 16))
    color  = "green" if score >= threshold else ("yellow" if score > threshold * 0.5 else "red")
    return (
        f"  Turn [bold]{state.turn}[/bold] / {state.spec.max_turns}   "
        f"Field [{color}]{bar}[/{color}]   "
        + _turns_narrative(turns_left)
    )


def render_full(state: GameState) -> None:
    if not HAS_RICH:
        _render_plain(state)
        return
    console.print()
    console.print(Text.from_markup(_render_status(state)))
    console.print()
    console.print(Columns([_render_scene(state), _render_connections(state)], equal=True))
    console.print(_render_hand(state))
    conv_msg = _convergence_narrative(state.casebook.convergence_score, state.spec.convergence_threshold)
    console.print(f"\n  {conv_msg}")


def _render_plain(state: GameState) -> None:
    score = state.casebook.convergence_score
    print(f"\n=== Turn {state.turn}/{state.spec.max_turns}  Field {score:.0%} ===")
    print("SCENE:", [card_name(t) for t in state.atmosphere])
    print("HAND:")
    for i, t in enumerate(state.hand, 1):
        print(f"  [{i}] {card_name(t)}  ({t.token_class.value})")


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────

def _refill_hand(state: GameState) -> List[str]:
    """Draw from deck to bring hand back to HAND_SIZE. Returns names of new cards."""
    drawn = []
    while len(state.hand) < HAND_SIZE and state.deck:
        card = state.deck.pop(0)
        state.hand.append(card)
        drawn.append(card_name(card))
    return drawn


def action_connect(state: GameState, indices: List[int], model) -> str:
    if len(indices) != 3:
        return "[red]Choose exactly 3 cards.[/red]"
    for i in indices:
        if i < 0 or i >= len(state.hand):
            return f"[red]No card [{i+1}] in hand.[/red]"
    if len(set(indices)) != 3:
        return "[red]Choose 3 different cards.[/red]"

    triad = [state.hand[i] for i in indices]

    context = list(state.atmosphere) + state.casebook.all_placed_tokens()
    energy  = _model_score_triad(model, state.spec, context, triad)

    # ── Energy-gated convergence ──────────────────────────────────────────
    # Hopfield convention: LOW energy = stable basin = correct connection.
    # Coherence = 1 − energy.
    #
    # Fallback: if the cartridge lacks stream/agency (old export format),
    # the model outputs degenerate all-high energy.  Detect this and fall
    # back to full advance so the game remains playable.  Retrain with the
    # updated exporter to get real strategic gating.
    coherence = 1.0 - energy
    _model_degenerate = coherence < 0.05
    if _model_degenerate:
        coherence = 1.0   # full advance — old cartridge compatibility
    if coherence >= 0.60:
        scale = 1.0
    elif coherence >= 0.20:
        scale = (coherence - 0.20) / 0.40   # 0 → 1 in [0.20, 0.60]
    else:
        scale = 0.0

    prev_score = state.casebook.convergence_score

    # Record the triad in placed_triads (for Lyapunov checks etc.)
    pos = (state.turn // 6, state.turn % 6)
    state.casebook.placed_triads[pos] = list(triad)
    state.casebook.turn_count += 1

    if scale > 0:
        contribution = np.stack([t.attractor_weights for t in triad]).mean(axis=0)
        delta = contribution * state.spec.convergence_rate * scale
        state.casebook.convergence_dimensions = np.minimum(
            1.0, state.casebook.convergence_dimensions + delta
        )

    state.connections.append(PlacedConnection(triad=triad, energy=energy, turn=state.turn))

    for tok in triad:
        state.played_ids.add(tok.id)
    state.hand = [t for t in state.hand if t.id not in state.played_ids]
    state.turn += 1

    drawn = _refill_hand(state) if REFILL_AFTER else []

    new_score = state.casebook.convergence_score
    diff      = new_score - prev_score

    if _model_degenerate:
        narrative, delta_color = "[dim]The connection is placed.[/dim]", "white"
    else:
        narrative, delta_color = _energy_narrative(energy)
    names = "  +  ".join(card_name_rich(t) for t in triad)

    if scale == 0.0:
        delta_str = f"[{delta_color}]— no advance[/{delta_color}]"
    else:
        delta_str = f"[{delta_color}]+{diff:.0%}[/{delta_color}]"

    msg = f"{names}\n  {narrative}  {delta_str}"

    if drawn:
        drawn_str = "  ".join(f"[dim]{n}[/dim]" for n in drawn)
        msg += f"\n  [dim]Drew: {drawn_str}[/dim]"

    return msg


def action_scene(state: GameState, model) -> str:
    """Engine advances the scene — places the next atmosphere token."""
    if not state.engine_pool:
        return "[yellow]Nothing more stirs in the world tonight.[/yellow]"

    placed_ids = [t.id for t in state.atmosphere] + list(state.casebook.placed_token_ids())
    candidates = state.engine_pool[:10]
    scored = [
        (tok, state.spec.token_graph.induced_subgraph_energy(placed_ids, [tok.id]))
        for tok in candidates
    ]
    scored.sort(key=lambda x: x[1])
    chosen = scored[0][0]
    state.engine_pool.remove(chosen)
    state.atmosphere.append(chosen)
    state.played_ids.add(chosen.id)

    color = _CLASS_COLORS.get(chosen.token_class.value, "white")
    icon  = _CLASS_ICONS.get(chosen.token_class.value, "·")
    return (
        f"[{color}]{icon}  {card_name(chosen)}[/{color}]  "
        f"[dim]enters the scene.[/dim]"
    )


def action_examine(state: GameState, idx: int) -> str:
    if idx < 0 or idx >= len(state.hand):
        return f"[red]No card [{idx+1}] in hand.[/red]"
    tok = state.hand[idx]
    color = _CLASS_COLORS.get(tok.token_class.value, "white")
    icon  = _CLASS_ICONS.get(tok.token_class.value, "·")
    tags  = "  ".join(tok.affinity_tags[:4]) if tok.affinity_tags else "—"
    lines = [
        f"  [{color}]{icon}  {card_name(tok)}[/{color}]",
        f"  [dim]Class  :[/dim]  {tok.token_class.value}",
        f"  [dim]Phase  :[/dim]  {tok.phase.value}",
        f"  [dim]Tags   :[/dim]  {tags}",
    ]
    if tok.surface_expression:
        lines.insert(1, f"  [dim italic]{tok.surface_expression}[/dim italic]")
    return "\n".join(lines)


def action_hint(state: GameState, model) -> str:
    context   = list(state.atmosphere) + state.casebook.all_placed_tokens()
    resonance = _model_resonance(model, state.spec, context)
    retrieval = _model_retrieval(model, state.spec, context)

    lines = []

    if resonance:
        lines.append("[dim]The field responds to:[/dim]")
        for tid, _ in resonance[:4]:
            try:
                tok = state.spec.get_token(tid)
                lines.append(f"  {card_name_rich(tok)}")
            except KeyError:
                pass

    if retrieval:
        lines.append("[dim]The network sees:[/dim]")
        dim_names = ["Suspect", "Event", "Motive"]
        for d, tid in enumerate(retrieval):
            label = dim_names[d] if d < len(dim_names) else f"dim{d}"
            try:
                tok = state.spec.get_token(tid)
                lines.append(f"  [dim]{label}:[/dim]  {card_name_rich(tok)}")
            except KeyError:
                pass

    if not lines:
        return "[dim]The field is quiet. Place more connections.[/dim]"
    return "\n".join(lines)


def action_accuse(state: GameState, model) -> str:
    inv_ids   = state.spec.invariant_token_ids
    invariants = [state.spec.get_token(tid) for tid in inv_ids]

    suspect_tok = next((t for t in invariants if t.token_class == TokenClass.SUSPECT), None)
    event_tok   = next((t for t in invariants if t.token_class == TokenClass.EVENT),   None)
    motive_tok  = next((t for t in invariants if t.token_class == TokenClass.MOTIVE),  None)

    # Show all known suspects / events / motives from the full vocab
    def _list_class(cls: TokenClass) -> List[Token]:
        return [t for t in state.spec.tokens
                if t.token_class == cls and not t.is_invariant]

    if HAS_RICH:
        console.print("\n[bold yellow]── ACCUSATION ──────────────────────────────[/bold yellow]")
        console.print("[dim]Name the suspect, the moment, the reason.[/dim]\n")

        # Show retrieval hint
        context = list(state.atmosphere) + state.casebook.all_placed_tokens()
        retrieval = _model_retrieval(model, state.spec, context)
        if retrieval:
            console.print("[dim]What the field points to:[/dim]")
            dim_names = ["Suspect", "Event", "Motive"]
            for d, tid in enumerate(retrieval):
                try:
                    tok = state.spec.get_token(tid)
                    console.print(f"  [dim]{dim_names[d]}:[/dim]  {card_name_rich(tok)}")
                except KeyError:
                    pass
            console.print()

        def _pick(label: str, cls: TokenClass, correct: Optional[Token]) -> bool:
            candidates = [t for t in state.spec.tokens if t.token_class == cls]
            console.print(f"[bold]{label}[/bold]")
            for i, t in enumerate(candidates, 1):
                marker = " [dim](invariant)[/dim]" if t.is_invariant else ""
                console.print(f"  [{i}] {card_name_rich(t)}{marker}")
            raw = Prompt.ask(f"  Number")
            try:
                idx = int(raw) - 1
                chosen = candidates[idx]
                return chosen.id == (correct.id if correct else "")
            except (ValueError, IndexError):
                return False

        correct_s = _pick("Who committed the act?",    TokenClass.SUSPECT, suspect_tok)
        correct_e = _pick("During which moment?",      TokenClass.EVENT,   event_tok)
        correct_m = _pick("Driven by what motive?",    TokenClass.MOTIVE,  motive_tok)
    else:
        suspect_id = input("Suspect ID: ").strip()
        event_id   = input("Event ID: ").strip()
        motive_id  = input("Motive ID: ").strip()
        correct_s  = suspect_tok and suspect_tok.id == suspect_id
        correct_e  = event_tok   and event_tok.id   == event_id
        correct_m  = motive_tok  and motive_tok.id  == motive_id

    if correct_s and correct_e and correct_m:
        state.solved   = True
        state.game_over = True
        s = card_name_rich(suspect_tok) if suspect_tok else "?"
        e = card_name_rich(event_tok)   if event_tok   else "?"
        m = card_name_rich(motive_tok)  if motive_tok  else "?"
        return (
            f"\n[bold green]CORRECT.[/bold green]\n\n"
            f"  {s} killed Aldous Verne\n"
            f"  during {e}\n"
            f"  to conceal {m}.\n"
        )
    else:
        wrongs = []
        if not correct_s: wrongs.append("suspect")
        if not correct_e: wrongs.append("moment")
        if not correct_m: wrongs.append("motive")
        return (
            f"[bold red]Wrong.[/bold red]  "
            f"[red]{', '.join(wrongs)} {'is' if len(wrongs)==1 else 'are'} incorrect.[/red]  "
            "Keep digging."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

_HELP = (
    "[bold]connect[/bold] [cyan]1 3 5[/cyan]  — lay three cards as a connection\n"
    "[bold]examine[/bold] [cyan]2[/cyan]        — inspect a card\n"
    "[bold]scene[/bold]            — the world advances\n"
    "[bold]hint[/bold]             — the field whispers\n"
    "[bold]connections[/bold]      — show all connections made\n"
    "[bold]hand[/bold]             — redisplay your hand\n"
    "[bold]field[/bold]            — redisplay everything\n"
    "[bold]accuse[/bold]           — make your accusation\n"
    "[bold]quit[/bold]             — leave the case"
)


# ─────────────────────────────────────────────────────────────────────────────
# Opening sequence
# ─────────────────────────────────────────────────────────────────────────────

def opening_sequence(state: GameState) -> None:
    if not HAS_RICH:
        print(f"\n=== {state.spec.title} ===")
        return

    console.print()
    console.rule(f"[bold white]{state.spec.title.upper()}[/bold white]")
    console.print()

    opening_lines = [
        "A body on the platform.",
        "The last train has left. The next has not arrived.",
        "Three things remain hidden:  a hand, a moment, a reason.",
        "",
        "You have the cards. You have the turns.",
        "Connect what belongs together. The field will converge to the truth.",
    ]
    for line in opening_lines:
        console.print(f"  [italic]{line}[/italic]")

    console.print()

    if state.atmosphere:
        console.print("[dim]The scene is set:[/dim]")
        for tok in state.atmosphere:
            color = _CLASS_COLORS.get(tok.token_class.value, "white")
            icon  = _CLASS_ICONS.get(tok.token_class.value, "·")
            console.print(f"    [{color}]{icon}  {card_name(tok)}[/{color}]")
    else:
        console.print("[dim]The scene is empty. Use [bold]scene[/bold] to advance.[/dim]")

    console.print()
    console.print(f"  [dim]{len(state.hand)} cards in hand.  "
                  f"{state.spec.max_turns} turns.  "
                  f"Threshold: {state.spec.convergence_threshold:.0%}.[/dim]")
    console.print()
    console.print(Panel(_HELP, title="[dim]HOW TO PLAY[/dim]", border_style="dim", padding=(0, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Game loop
# ─────────────────────────────────────────────────────────────────────────────

def game_loop(state: GameState, model) -> None:
    opening_sequence(state)
    render_full(state)

    while not state.game_over:
        if state.turn >= state.spec.max_turns:
            if HAS_RICH:
                console.print("\n[bold red]The window closes. The case goes cold.[/bold red]")
                _reveal_truth(state)
            state.game_over = True
            break

        score     = state.casebook.convergence_score
        threshold = state.spec.convergence_threshold
        can_accuse = score >= threshold * 0.8 or state.turn >= state.spec.min_turns

        if HAS_RICH:
            suffix = " [dim](accuse?)[/dim]" if can_accuse and score >= threshold * 0.9 else ""
            raw = Prompt.ask(f"\n[bold white]>[/bold white]{suffix}").strip()
        else:
            raw = input(f"\n[{state.turn}] > ").strip()

        if not raw:
            continue

        parts = raw.split()

        # Shorthands: bare numbers
        # "3"       → examine 3
        # "1 3 5"   → connect 1 3 5
        if all(p.isdigit() for p in parts):
            if len(parts) == 1:
                parts = ["examine", parts[0]]
            elif len(parts) == 3:
                parts = ["connect"] + parts

        cmd = parts[0].lower()

        if cmd in ("quit", "q", "exit"):
            if HAS_RICH:
                console.print("\n[dim]You close the casebook.[/dim]")
            break

        elif cmd == "help":
            if HAS_RICH:
                console.print(Panel(_HELP, border_style="dim", padding=(0, 2)))

        elif cmd in ("field", "show", "state"):
            render_full(state)

        elif cmd == "hand":
            if HAS_RICH:
                console.print(_render_hand(state))
            else:
                for i, t in enumerate(state.hand, 1):
                    print(f"  [{i}] {card_name(t)}")

        elif cmd == "connections":
            if HAS_RICH:
                console.print(_render_connections(state))
            else:
                for c in state.connections:
                    print([card_name(t) for t in c.triad])

        elif cmd == "scene":
            msg = action_scene(state, model)
            if HAS_RICH:
                console.print(f"\n  {msg}")
            else:
                print(msg)

        elif cmd == "examine":
            if len(parts) < 2:
                if HAS_RICH: console.print("[red]examine <n>[/red]")
                continue
            try:
                idx = int(parts[1]) - 1
            except ValueError:
                if HAS_RICH: console.print("[red]examine <n>[/red]")
                continue
            msg = action_examine(state, idx)
            if HAS_RICH:
                console.print(Panel(msg, border_style="dim", padding=(0, 1)))
            else:
                print(msg)

        elif cmd == "hint":
            msg = action_hint(state, model)
            if HAS_RICH:
                console.print(Panel(msg, title="[dim]field whispers[/dim]",
                                    border_style="dim", padding=(0, 1)))
            else:
                print(msg)

        elif cmd == "connect":
            if len(parts) < 4:
                if HAS_RICH: console.print("[red]connect <a> <b> <c>[/red]")
                continue
            try:
                nums = [int(p) - 1 for p in parts[1:4]]
            except ValueError:
                if HAS_RICH: console.print("[red]connect <a> <b> <c>[/red]")
                continue

            msg = action_connect(state, nums, model)
            if HAS_RICH:
                console.print(f"\n  {msg}")
                # Show updated status line
                console.print(f"\n  {Text.from_markup(_render_status(state))}")
                console.print(f"  {_convergence_narrative(state.casebook.convergence_score, threshold)}")

                if state.casebook.convergence_score >= threshold:
                    console.print(
                        "\n  [bold green]The field has converged. "
                        "Type [bold]accuse[/bold] when ready.[/bold green]"
                    )
            else:
                print(msg)

        elif cmd == "accuse":
            if not can_accuse:
                if HAS_RICH:
                    console.print(
                        f"\n  [yellow]The picture is not clear yet. "
                        f"Field at {state.casebook.convergence_score:.0%} — keep connecting.[/yellow]"
                    )
                continue

            msg = action_accuse(state, model)
            if HAS_RICH:
                console.print(msg)
            else:
                print(msg)

            if state.solved:
                if HAS_RICH:
                    _closing_sequence(state)
                break

        else:
            if HAS_RICH:
                console.print(f"[dim]Unknown: '{cmd}'. Type [bold]help[/bold].[/dim]")

    if not state.solved and not state.game_over:
        if HAS_RICH:
            _reveal_truth(state)


def _reveal_truth(state: GameState) -> None:
    console.print()
    console.rule("[bold red]THE TRUTH[/bold red]")
    console.print()
    for tid in state.spec.invariant_token_ids:
        tok = state.spec.get_token(tid)
        console.print(f"  {card_name_rich(tok)}  [dim]({tok.token_class.value})[/dim]")
    console.print()


def _closing_sequence(state: GameState) -> None:
    console.print()
    console.rule("[bold green]CASE CLOSED[/bold green]")
    console.print()
    console.print(
        f"  [dim]Solved in[/dim] [bold]{state.turn}[/bold] [dim]turns.[/dim]  "
        f"[dim]{len(state.connections)} connections made.[/dim]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Thornfield mystery runner")
    parser.add_argument("cartridge", help="Path to .cartridge file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible hands")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if HAS_RICH:
        with console.status("[bold]Loading cartridge…[/bold]", spinner="dots"):
            spec, model = load_cartridge(args.cartridge)
    else:
        spec, model = load_cartridge(args.cartridge)

    state = init_game(spec)
    game_loop(state, model)


if __name__ == "__main__":
    main()
