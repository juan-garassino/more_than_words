from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class TokenClass(str, Enum):
    SUSPECT = "SUSPECT"
    LOCATION = "LOCATION"
    OBJECT = "OBJECT"
    EMOTION = "EMOTION"
    MODIFIER = "MODIFIER"
    ACTION = "ACTION"
    TIME = "TIME"
    MOTIVE = "MOTIVE"
    WITNESS = "WITNESS"
    EVENT = "EVENT"
    NEED = "NEED"
    STATE = "STATE"
    OFFERING = "OFFERING"
    UNKNOWN = "UNKNOWN"


class TokenPhase(str, Enum):
    EARLY = "EARLY"
    MID = "MID"
    LATE = "LATE"
    INVARIANT = "INVARIANT"
    ANY = "ANY"


class TokenStream(str, Enum):
    EVIDENCE = "EVIDENCE"      # Detective-navigated: objects, deduced actions, motives
    ATMOSPHERE = "ATMOSPHERE"  # World-controlled: weather, time, location, structure
    OPENING = "OPENING"        # Fixed scene-setting tokens
    INVARIANT = "INVARIANT"    # Solution tokens (killer, mechanism, motive)


class TokenAgency(str, Enum):
    PLAYER = "PLAYER"  # Only the detective/player can surface
    ENGINE = "ENGINE"  # Only the world/engine surfaces
    SHARED = "SHARED"  # Either player or engine can surface


@dataclass
class Token:
    id: str
    token_class: TokenClass
    phase: TokenPhase
    attractor_weights: np.ndarray
    affinity_tags: List[str]
    repulsion_tags: List[str]
    temperature: float
    narrative_gradient: float
    is_invariant: bool
    surface_expression: str
    stream: TokenStream = TokenStream.EVIDENCE
    agency: TokenAgency = TokenAgency.SHARED
    embedding: Optional[np.ndarray] = None

    def is_available_at_turn(self, turn: int) -> bool:
        if self.phase == TokenPhase.ANY:
            return True
        if self.phase == TokenPhase.INVARIANT:
            return False
        if self.phase == TokenPhase.EARLY:
            return turn <= 8
        if self.phase == TokenPhase.MID:
            return 5 <= turn <= 14
        if self.phase == TokenPhase.LATE:
            return turn >= 10
        return True


def compute_narrative_gradient(token: Token, n_dims: int) -> float:
    invariant_vec = np.ones(n_dims, dtype=np.float32)
    w = token.attractor_weights
    if np.linalg.norm(w) < 1e-8:
        return 0.0
    return float(
        np.dot(w, invariant_vec) / (np.linalg.norm(w) * np.sqrt(n_dims))
    )
