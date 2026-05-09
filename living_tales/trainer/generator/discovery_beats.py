"""
Discovery beats — runtime convergence-threshold scaffolding.

When the engine's convergence on a dim crosses an authored threshold (and the
session is past `min_turn`), the corresponding beat fires once, force-injecting
REVELATION/BEAT tokens into the next emitted scene. Beats give the player a
visible closing arc even when the trained model alone fails to produce one.

beats.json schema (per case)
----------------------------
    {
      "case_id": "amber_cipher",
      "convergence_dim_names": ["suspect", "evidence", "motive"],
      "beats": [
        {
          "id": "verdict_ready",
          "trigger": { "min_dim_gte": 0.8 },     OR { "dim": "suspect", "gte": 0.7 }
                                                 OR { "any_dim_gte": 0.4 },
          "min_turn": 24,
          "fire_once": true,
          "inject": { "BEAT": "beat:verdict_ready" }
        }
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class _Beat:
    id: str
    trigger: Dict
    min_turn: int = 0
    fire_once: bool = True
    inject: Dict[str, str] = field(default_factory=dict)


@dataclass
class DiscoveryBeats:
    case_id: str
    convergence_dim_names: List[str]
    beats: List[_Beat] = field(default_factory=list)
    fired: set = field(default_factory=set)

    @classmethod
    def load(cls, case_id: str, repo_root: Path) -> "Optional[DiscoveryBeats]":
        path = (
            repo_root / "living_tales" / "trainer" / "cases"
            / case_id / "beats.json"
        )
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        beats = [
            _Beat(
                id=b["id"],
                trigger=dict(b.get("trigger", {})),
                min_turn=int(b.get("min_turn", 0)),
                fire_once=bool(b.get("fire_once", True)),
                inject=dict(b.get("inject", {})),
            )
            for b in data.get("beats", [])
        ]
        return cls(
            case_id=data.get("case_id", case_id),
            convergence_dim_names=list(data.get("convergence_dim_names", [])),
            beats=beats,
        )

    # ── Trigger evaluation ──────────────────────────────────────────────
    def _trigger_satisfied(
        self, beat: _Beat, convergence: Sequence[float],
    ) -> bool:
        trig = beat.trigger
        if not trig:
            return False
        if "min_dim_gte" in trig:
            thr = float(trig["min_dim_gte"])
            return all(c >= thr for c in convergence) if convergence else False
        if "any_dim_gte" in trig:
            thr = float(trig["any_dim_gte"])
            return any(c >= thr for c in convergence) if convergence else False
        if "dim" in trig and "gte" in trig:
            try:
                idx = self.convergence_dim_names.index(trig["dim"])
            except ValueError:
                return False
            if idx >= len(convergence):
                return False
            return float(convergence[idx]) >= float(trig["gte"])
        return False

    # ── Apply ───────────────────────────────────────────────────────────
    def apply(
        self,
        scene: Dict[str, str],
        convergence: Sequence[float],
        turn_idx: int,
    ) -> Dict[str, str]:
        """Return a possibly-modified scene dict with beat injections.

        Mutates `self.fired` to record which beats have already fired. Only
        fires beats whose `min_turn` is satisfied, whose trigger fires, and
        (if `fire_once`) that haven't fired before.
        """
        out = dict(scene)
        for b in self.beats:
            if turn_idx < b.min_turn:
                continue
            if b.fire_once and b.id in self.fired:
                continue
            if not self._trigger_satisfied(b, convergence):
                continue
            for dim, tok in b.inject.items():
                out[dim] = tok
            self.fired.add(b.id)
        return out

    def reset(self) -> None:
        self.fired.clear()


# ── Selftest ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = Path("/Users/juan-garassino/Code/005-products/010-more-than-words")

    for case in ("amber_cipher", "attended_hour", "venetian_mirror"):
        db = DiscoveryBeats.load(case, ROOT)
        if db is None:
            print(f"[FAIL] {case}: beats.json not found")
            sys.exit(1)
        print(f"[OK] {case}: {len(db.beats)} beats loaded; convergence dims = {db.convergence_dim_names}")
        # Drive a synthetic convergence trajectory and observe firing order.
        scene = {"BEAT": "beat:investigation"}
        for turn in range(0, 30):
            conv = [min(0.05 * turn, 0.95)] * 3
            new = db.apply(scene, conv, turn_idx=turn)
            if new != scene:
                fired_now = db.fired
                print(f"     turn={turn:>2d} conv={conv[0]:.2f}  injected={new}  fired={sorted(fired_now)}")
                scene = new
