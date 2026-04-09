"""Data schema for Living Tales game evaluation."""
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
    token_role: Optional[str] = None
    dimension_snapshot: Optional[List[float]] = None


@dataclass
class GameResult:
    seed: int
    turns: List[DialogueTurn]
    converged: bool
    final_convergence: float
    final_token_ids: List[str]
    total_energy: float
    mode: str = "converging"


@dataclass
class EvalResult:
    test_name: str
    metric_name: str
    passed: bool
    value: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
