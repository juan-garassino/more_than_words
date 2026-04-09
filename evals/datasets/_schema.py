"""Data schema for Thornfield game evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DialogueTurn:
    token_id: str
    role: str           # "player" or "engine"
    token_class: str
    phase: str
    energy_at_step: float
    convergence_at_step: float


@dataclass
class GameResult:
    seed: int
    turns: List[DialogueTurn]
    converged: bool
    final_convergence: float
    final_token_ids: List[str]
    total_energy: float


@dataclass
class EvalResult:
    test_name: str
    metric_name: str
    passed: bool
    value: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
