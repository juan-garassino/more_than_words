"""
Playtest judge
==============
Reads a `playtest_transcripts.md` produced by `playtest_simulate.py`
and asks an LLM judge to score the playthroughs against a detective-game
rubric. The judge is mandatory; the backend is pluggable.

Backends
--------
  --backend claude_subagent   (default; uses Claude Code's Agent tool)
  --backend openai            (uses OPENAI_API_KEY env var, model gpt-4o-mini)
  --backend gemini            (uses GEMINI_API_KEY env var, model gemini-2.0-flash)
  --backend stdout            (just prints the prompt + transcript; no API call)

The `claude_subagent` backend does not call any API directly. It writes a
"judging packet" (the rubric + transcript) to disk and emits the launch
instructions for a Claude subagent. The parent Claude Code session reads
those instructions and invokes its `Agent` tool. This keeps API key
management out of the script for the default flow.

Rubric
------
The judge scores each playthrough on five axes (0-5 each):
  1. Narrative coherence       — does it read as one connected story?
  2. Voice consistency         — do NPCs sound like themselves throughout?
  3. Constraint integrity      — any logical contradictions?
  4. Detective feel            — does the player seem to be solving a case?
  5. Closing arc               — does the ending land (or fail to land) deliberately?

Plus a free-text "issues" field flagging the top 3 problems and a "highlights"
field for what the engine did well.

Usage
-----
    python tools/playtest_judge.py amber_cipher
    python tools/playtest_judge.py amber_cipher --backend openai
    python tools/playtest_judge.py amber_cipher --backend gemini
    python tools/playtest_judge.py amber_cipher --backend stdout

Output
------
Writes `outputs/<case>/playtest_judgment.md` with per-playthrough scores
and an aggregate verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent

RUBRIC = """\
You are a detective-game playtest judge. You will be shown the full transcript
of a simulated playthrough of a Living Tales case. Your job is to read it as
if it were a real player's run and score it against a rubric.

The Living Tales engine is a structured-scene transformer that emits one token
per dimension per turn (LOCATION, TRANSITION, CAUSE, PRESENCE, STANCE, ACTION,
OBJECT_FOCUS, TELL, ATMOSPHERE, REVELATION, BEAT). The composer renders the
tuple as Obra-Dinn-style third-person past prose. The case is a hand-authored
mystery. The player picks a card; the engine responds with a scene; repeat.

Score each playthrough on five axes (0-5 each, integer). Be honest — if a
playthrough is mostly random or confused, score low. If it reads like a real
detective experience, score high.

  1. NARRATIVE_COHERENCE — does it read as one connected story, or as a string
     of disconnected scenes?
  2. VOICE_CONSISTENCY — do NPCs sound like themselves across appearances?
     Does an NPC's voice/tells match across scenes? Or does it bleed into
     other NPCs' lexicons?
  3. CONSTRAINT_INTEGRITY — any logical contradictions you can spot? E.g., a
     hostile stance with no visible tell, a player questioning an empty room,
     a "still at X" while the location changed, dawn at turn 5, a name uncovered
     while the detective is alone with no evidence in hand.
  4. DETECTIVE_FEEL — does the player seem to be SOLVING a case? Does
     evidence accumulate? Are revelations earned or arbitrary? Does the
     player's choice visibly shape what comes next?
  5. CLOSING_ARC — does the playthrough end deliberately (verdict, cold trail,
     red herring trap, etc.) or just stop? Does the BEAT progression match
     the convergence story?

Then list:
  - HIGHLIGHTS: up to 3 specific moments that read as real detective fiction
  - ISSUES: up to 3 specific problems (turn numbers, NPCs, dim violations)

Format your response EXACTLY like this for EACH playthrough:

---
### Playthrough N
NARRATIVE_COHERENCE: <0-5>  — <one sentence>
VOICE_CONSISTENCY: <0-5>    — <one sentence>
CONSTRAINT_INTEGRITY: <0-5> — <one sentence>
DETECTIVE_FEEL: <0-5>       — <one sentence>
CLOSING_ARC: <0-5>          — <one sentence>
TOTAL: <0-25>

HIGHLIGHTS:
  - [turn N] <quote or paraphrase>
  - ...

ISSUES:
  - [turn N] <quote or paraphrase>
  - ...
---

After all playthroughs, give an aggregate verdict (50-150 words):
  - What does the engine do well right now?
  - What is the single most important thing to fix before pygame playtest?
  - Is the case ready for human playtest? (yes / no / yes-with-caveats)

Begin.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────


def judge_claude_subagent(rubric: str, transcript: str, packet_out: Path) -> str:
    """Write a judging packet and return launch instructions for the parent
    Claude Code session to invoke its Agent tool. No API call from this
    script. The parent session reads the file, invokes Agent(), then writes
    the judgment back to disk."""
    packet_out.write_text(rubric + "\n\n" + transcript)
    return (
        f"=== Claude subagent backend ===\n"
        f"Judging packet written to:\n  {packet_out}\n\n"
        f"In the parent Claude Code session, invoke:\n\n"
        f"    Agent(\n"
        f"        subagent_type='general-purpose',\n"
        f"        description='Playtest judge: <case_id>',\n"
        f"        prompt='Read the file {packet_out} (rubric + transcripts). "
        f"Score per the rubric. Write your judgment to "
        f"{packet_out.parent / \"playtest_judgment.md\"}.'\n"
        f"    )\n\n"
        f"The agent will read the packet, score per the rubric, and write the "
        f"judgment to disk."
    )


def judge_openai(rubric: str, transcript: str, model: str = "gpt-4o-mini") -> str:
    """Call the OpenAI API. Requires OPENAI_API_KEY env var."""
    import urllib.request
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in env.")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.2,
        }).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def judge_gemini(rubric: str, transcript: str,
                 model: str = "gemini-2.0-flash") -> str:
    """Call the Gemini API. Requires GEMINI_API_KEY env var."""
    import urllib.request
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in env.")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps({
            "contents": [{
                "role": "user",
                "parts": [{"text": rubric + "\n\n" + transcript}],
            }],
            "generationConfig": {"temperature": 0.2},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["candidates"][0]["content"]["parts"][0]["text"]


def judge_stdout(rubric: str, transcript: str) -> str:
    """No API call — just emit the prompt + transcript so a human can paste
    them into a chat manually."""
    return rubric + "\n\n=== TRANSCRIPT ===\n\n" + transcript


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case_id")
    p.add_argument(
        "--backend",
        choices=["claude_subagent", "openai", "gemini", "stdout"],
        default="claude_subagent",
    )
    p.add_argument("--transcript", default=None,
                   help="Path to playtest_transcripts.md (default: outputs/<case>/playtest_transcripts.md)")
    p.add_argument("--out", default=None,
                   help="Path to write judgment (default: outputs/<case>/playtest_judgment.md)")
    args = p.parse_args()

    project_root = _HERE.parent.parent.parent
    transcript_path = (
        Path(args.transcript) if args.transcript
        else project_root / "living_tales/trainer/outputs"
             / args.case_id / "playtest_transcripts.md"
    )
    judgment_path = (
        Path(args.out) if args.out
        else project_root / "living_tales/trainer/outputs"
             / args.case_id / "playtest_judgment.md"
    )
    packet_path = judgment_path.parent / "playtest_packet.md"
    judgment_path.parent.mkdir(parents=True, exist_ok=True)

    if not transcript_path.exists():
        print(f"No transcript at {transcript_path}.", file=sys.stderr)
        print(f"Run `python tools/playtest_simulate.py {args.case_id}` first.",
              file=sys.stderr)
        sys.exit(1)

    transcript = transcript_path.read_text()
    print(f"transcript: {transcript_path} ({len(transcript)} chars)")
    print(f"backend:    {args.backend}")
    print(f"judgment:   {judgment_path}")
    print()

    if args.backend == "claude_subagent":
        instructions = judge_claude_subagent(RUBRIC, transcript, packet_path)
        print(instructions)
    elif args.backend == "openai":
        text = judge_openai(RUBRIC, transcript)
        judgment_path.write_text(text)
        print(f"openai judgment → {judgment_path}")
    elif args.backend == "gemini":
        text = judge_gemini(RUBRIC, transcript)
        judgment_path.write_text(text)
        print(f"gemini judgment → {judgment_path}")
    elif args.backend == "stdout":
        print(judge_stdout(RUBRIC, transcript))


if __name__ == "__main__":
    main()
