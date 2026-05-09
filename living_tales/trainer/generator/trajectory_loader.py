"""
Trajectory Loader
=================
Loads hand-authored training trajectories for a Living Tales case.

A trajectory is one complete playthrough from opening to ending, expressed
as a sequence of (player_card, scene_tuple) pairs. See
`living_tales/AUTHORING_TRAJECTORIES.md` for the schema.

Usage
-----
    loader = TrajectoryLoader("amber_cipher", project_root=Path("/path/to/repo"))
    manifest_entries = loader.list_trajectories()
    traj = loader.load("voss_via_cufflink")
    all_trajs = loader.load_all()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ─── Dataclasses ─────────────────────────────────────────────────────────────
@dataclass
class Turn:
    turn: int
    player_card: str
    scene: Dict[str, str]
    convergence_after: Optional[List[float]] = None
    note: Optional[str] = None
    # v2 schema additions (all optional; legacy trajectories omit them).
    scene_type: Optional[str] = None                 # one of the 8 latent z modes
    forbidden_dims: Optional[Dict[str, List[str]]] = None  # negative supervision


@dataclass
class CounterfactualBranch:
    """Alternate 5-turn continuation from a given anchor turn index."""
    anchor_turn: int                # index in parent turns from which the branch diverges
    turns: List[Turn]
    note: Optional[str] = None


@dataclass
class Trajectory:
    trajectory_id: str
    case_id: str
    outcome: str
    description: str
    tags: List[str]
    opening: List[str]
    turns: List[Turn]
    ending: Dict[str, Any]
    starting_hypothesis: Optional[str] = None
    schema_version: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)
    # v2 additions.
    scene_type_sequence: Optional[List[str]] = None  # one z-mode per turn (len == len(turns))
    counterfactual_branches: List[CounterfactualBranch] = field(default_factory=list)


# ─── Loader ──────────────────────────────────────────────────────────────────
class TrajectoryLoader:
    """Reads a case's trajectory portfolio from disk."""

    def __init__(self, case_id: str, project_root: Union[str, Path]):
        self.case_id = case_id
        self.project_root = Path(project_root)
        self.case_dir = (
            self.project_root
            / "living_tales"
            / "trainer"
            / "cases"
            / case_id
        )
        self.traj_dir = self.case_dir / "trajectories"
        self.manifest_path = self.traj_dir / "manifest.json"

    # ── Manifest ──
    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found at {self.manifest_path}"
            )
        with open(self.manifest_path) as f:
            return json.load(f)

    def list_trajectories(self) -> List[Dict[str, Any]]:
        """Return the manifest's trajectory list (raw dicts)."""
        return list(self._load_manifest().get("trajectories", []))

    # ── Single load ──
    def load(self, traj_id: str) -> Trajectory:
        path = self.traj_dir / f"{traj_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return self._parse(data)

    def load_all(self) -> List[Trajectory]:
        out: List[Trajectory] = []
        for entry in self.list_trajectories():
            tid = entry.get("id")
            if not tid:
                continue
            try:
                out.append(self.load(tid))
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
        return out

    # ── Parsing ──
    @staticmethod
    def _parse(data: Dict[str, Any]) -> Trajectory:
        opening_block = data.get("opening", {}) or {}
        if isinstance(opening_block, list):
            opening_tokens = list(opening_block)
        else:
            opening_tokens = list(opening_block.get("tokens", []))

        def _parse_turn(t: Dict[str, Any], default_idx: int) -> Turn:
            forbidden = t.get("forbidden_dims")
            if forbidden is not None and not isinstance(forbidden, dict):
                forbidden = None
            return Turn(
                turn=t.get("turn", default_idx),
                player_card=t.get("player_card", ""),
                scene=dict(t.get("scene", {})),
                convergence_after=t.get("convergence_after"),
                note=t.get("_note") or t.get("note"),
                scene_type=t.get("scene_type"),
                forbidden_dims=forbidden,
            )

        turns: List[Turn] = []
        for t in data.get("turns", []):
            turns.append(_parse_turn(t, len(turns) + 1))

        # Top-level scene_type_sequence (alternative to per-turn scene_type).
        sts = data.get("scene_type_sequence")
        if sts is not None:
            sts = list(sts)
            for i, mode in enumerate(sts):
                if i < len(turns) and turns[i].scene_type is None:
                    turns[i].scene_type = mode

        # Counterfactual branches.
        branches: List[CounterfactualBranch] = []
        for b in data.get("counterfactual_branches", []) or []:
            anchor = int(b.get("anchor_turn", 0))
            b_turns = [_parse_turn(t, i + 1) for i, t in enumerate(b.get("turns", []))]
            branches.append(CounterfactualBranch(
                anchor_turn=anchor, turns=b_turns, note=b.get("note"),
            ))

        return Trajectory(
            trajectory_id=data.get("trajectory_id", ""),
            case_id=data.get("case_id", ""),
            outcome=data.get("outcome", ""),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            opening=opening_tokens,
            turns=turns,
            ending=dict(data.get("ending", {})),
            starting_hypothesis=data.get("starting_hypothesis"),
            schema_version=int(data.get("schema_version", 1)),
            raw=data,
            scene_type_sequence=sts,
            counterfactual_branches=branches,
        )
