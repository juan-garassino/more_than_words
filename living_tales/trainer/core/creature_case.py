from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


CREATURE_ROLE_MAP = {
    "state": "state",
    "action": "action",
    "recovery": "recovery",
    "decay": "decay",
    "decline": "decline",
    "combo": "combo",
    "mischief": "combo",
    "need": "need",
    "mood": "mood",
    "context": "context",
    "location": "context",
    "time": "context",
    "object": "object",
    "companion": "companion",
    "event": "event",
    "trait": "trait",
    "memory": "memory",
    "stage": "stage",
}

CREATURE_M_REQUIRED_ROLES = {
    "state",
    "action",
    "recovery",
    "decay",
    "decline",
    "combo",
    "context",
}

CREATURE_M_REQUIRED_ARCS = (
    ("decay", "decline", "action", "recovery"),
    ("decay", "combo", "action", "recovery"),
)


def token_prefix(token_id: str) -> str:
    return token_id.split(":", 1)[0]


def classify_creature_token_role(token_id: str) -> str:
    return CREATURE_ROLE_MAP.get(token_prefix(token_id), "context")


def creature_role_counts(token_ids: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(classify_creature_token_role(token_id) for token_id in token_ids))


def has_creature_arc(role_sequence: List[str], arc: Iterable[str]) -> bool:
    required = list(arc)
    index = 0
    for role in role_sequence:
        if role == required[index]:
            index += 1
            if index == len(required):
                return True
    return False
