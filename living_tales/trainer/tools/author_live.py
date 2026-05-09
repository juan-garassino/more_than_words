"""
author_live.py — Live-authoring CLI for Living Tales trajectories.

Watches a single trajectory JSON file and re-renders the last 3 turns
(composed prose + convergence + lightweight validation) on every save.

Usage:
    python tools/author_live.py <case_id> <traj_id> [--lang en|es]

No external deps — polls mtime every 1 second.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `generator` package importable when run from anywhere.
_HERE = Path(__file__).resolve().parent
_TRAINER = _HERE.parent
if str(_TRAINER) not in sys.path:
    sys.path.insert(0, str(_TRAINER))

from generator.structured_scene_composer import SceneComposer  # noqa: E402

# ─── ANSI ──────────────────────────────────────────────────────────────────
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
GREY = "\x1b[90m"
CLEAR = "\x1b[2J\x1b[H"

REQUIRED_DIMS = [
    "LOCATION", "TRANSITION", "CAUSE", "PRESENCE", "STANCE",
    "ACTION", "OBJECT_FOCUS", "TELL", "ATMOSPHERE", "REVELATION", "BEAT",
]


def load_dim_vocabs(case_id: str) -> dict:
    """Return {dim_name: set(token_ids)} from cases/<case>/dimensions.json."""
    path = _TRAINER / "cases" / case_id / "dimensions.json"
    with open(path) as f:
        data = json.load(f)
    out = {}
    for dim in data.get("dimensions", []):
        out[dim["name"]] = set(dim.get("vocab", []))
    return out


def parse_json_with_context(path: Path):
    """Parse a JSON file, returning (data, error_str)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None, f"file not found: {path}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        # Provide line context.
        lines = text.splitlines()
        ln = e.lineno
        lo = max(0, ln - 3)
        hi = min(len(lines), ln + 2)
        ctx = []
        for i in range(lo, hi):
            marker = ">>" if (i + 1) == ln else "  "
            ctx.append(f"{marker} {i+1:4d}| {lines[i]}")
        msg = f"JSON parse error: {e.msg} (line {e.lineno} col {e.colno})\n" + "\n".join(ctx)
        return None, msg


def validate_last_turns(turns: list, vocabs: dict) -> list:
    """Return list of warning strings for the trajectory's recent turns."""
    warnings = []
    if not turns:
        warnings.append("no turns yet")
        return warnings

    # 1. Per-turn slot + vocab membership (check all turns; cheap)
    for t in turns:
        tn = t.get("turn", "?")
        scene = t.get("scene", {})
        missing = [d for d in REQUIRED_DIMS if d not in scene]
        if missing:
            warnings.append(f"turn {tn}: missing dim slots: {', '.join(missing)}")
        for dim, tok in scene.items():
            if dim not in vocabs:
                continue
            if tok not in vocabs[dim]:
                warnings.append(f"turn {tn}: {dim}={tok!r} not in vocab")

    # 2. Lyapunov monotonicity over the last 3 convergence_after vectors
    last3 = turns[-3:]
    convs = [t.get("convergence_after") for t in last3]
    if all(isinstance(c, list) for c in convs) and len(convs) >= 2:
        for i in range(1, len(convs)):
            prev, curr = convs[i - 1], convs[i]
            if len(prev) != len(curr):
                warnings.append(
                    f"turn {last3[i].get('turn','?')}: convergence vector length changed")
                continue
            for j, (a, b) in enumerate(zip(prev, curr)):
                if b + 1e-6 < a:
                    warnings.append(
                        f"turn {last3[i].get('turn','?')}: Lyapunov violated "
                        f"on dim {j} ({a:.3f} -> {b:.3f})")
    return warnings


def render(path: Path, case_id: str, traj_id: str, lang: str,
           composer: SceneComposer, vocabs: dict) -> None:
    sys.stdout.write(CLEAR)

    data, err = parse_json_with_context(path)
    if err:
        print(f"{RED}{BOLD}[parse error]{RESET} {RED}{err}{RESET}")
        print()
        print(f"{DIM}watching {path} — save the file to retry.{RESET}")
        sys.stdout.flush()
        return

    turns = data.get("turns", [])
    outcome = data.get("outcome", "?")
    desc = data.get("description", "")

    # Header
    print(f"{CYAN}{BOLD}━━━ {traj_id} ({case_id}) ━━━{RESET}")
    print(f"{CYAN}outcome:{RESET} {outcome}   "
          f"{CYAN}turns:{RESET} {len(turns)}   "
          f"{CYAN}lang:{RESET} {lang}")
    if desc:
        snip = desc if len(desc) <= 110 else desc[:107] + "..."
        print(f"{DIM}{snip}{RESET}")
    print()

    if not turns:
        print(f"{YELLOW}(no turns yet — start writing){RESET}")
        sys.stdout.flush()
        return

    last3 = turns[-3:]
    print(f"{CYAN}{BOLD}Last {len(last3)} turn(s):{RESET}")
    print()
    for t in last3:
        tn = t.get("turn", "?")
        pcard = t.get("player_card", "?")
        scene = t.get("scene", {})
        print(f"{CYAN}Turn {tn}{RESET}  {GREY}player_card: {pcard}{RESET}")
        try:
            prose = composer.compose(scene)
        except Exception as e:  # noqa: BLE001
            prose = f"{RED}[compose error: {e}]{RESET}"
        # Indent prose
        for line in (prose or "(empty)").split("\n"):
            print(f"  {line}")
        note = t.get("_note")
        if note:
            print(f"  {DIM}note: {note}{RESET}")
        print()

    # Convergence
    last = last3[-1]
    conv = last.get("convergence_after")
    if conv is not None:
        if isinstance(conv, list):
            fmt = "[" + ", ".join(f"{x:.3f}" for x in conv) + "]"
        else:
            fmt = str(conv)
        print(f"{CYAN}convergence_after (turn {last.get('turn','?')}):{RESET} {fmt}")
    else:
        print(f"{YELLOW}convergence_after: (none on last turn){RESET}")

    # Validation
    warnings = validate_last_turns(turns, vocabs)
    print()
    if not warnings:
        print(f"{GREEN}validation: OK — all dims filled, vocab clean, Lyapunov non-decreasing{RESET}")
    else:
        print(f"{YELLOW}validation: {len(warnings)} warning(s){RESET}")
        for w in warnings[:8]:
            print(f"  {YELLOW}- {w}{RESET}")
        if len(warnings) > 8:
            print(f"  {DIM}... and {len(warnings)-8} more{RESET}")

    print()
    print(f"{DIM}watching {path.name} — Ctrl-C to exit{RESET}")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live trajectory authoring watcher.")
    ap.add_argument("case_id")
    ap.add_argument("traj_id")
    ap.add_argument("--lang", choices=["en", "es"], default="en")
    args = ap.parse_args()

    path = _TRAINER / "cases" / args.case_id / "trajectories" / f"{args.traj_id}.json"

    try:
        composer = SceneComposer.load(args.case_id, lang=args.lang)
    except Exception as e:  # noqa: BLE001
        print(f"{RED}failed to load composer: {e}{RESET}")
        return 1

    try:
        vocabs = load_dim_vocabs(args.case_id)
    except Exception as e:  # noqa: BLE001
        print(f"{RED}failed to load dimensions.json: {e}{RESET}")
        return 1

    last_mtime = -1.0
    try:
        while True:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                if last_mtime != -2.0:
                    sys.stdout.write(CLEAR)
                    print(f"{RED}trajectory file not found:{RESET} {path}")
                    print(f"{DIM}create it and the watcher will pick it up.{RESET}")
                    sys.stdout.flush()
                    last_mtime = -2.0
                time.sleep(1.0)
                continue

            if mtime != last_mtime:
                last_mtime = mtime
                render(path, args.case_id, args.traj_id, args.lang, composer, vocabs)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print()
        print(f"{DIM}exit.{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
