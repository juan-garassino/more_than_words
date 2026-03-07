from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .token import Token, TokenClass


class RelationType(IntEnum):
    BELONGS_TO = 0    # object/modifier → suspect
    HAPPENED_AT = 1   # action/event → location
    DURING = 2        # action/event → time
    EXPLAINS = 3      # motive → action
    DESCRIBES = 4     # modifier → object
    INDICATES = 5     # emotion → suspect
    OBSERVED = 6      # witness → event
    ENABLES = 7       # object → motive
    IMPLICATES = 8    # evidence → suspect
    SCENE_OF = 9      # location → event
    COMMITTED = 10    # suspect → action
    DRIVES = 11       # motive → suspect


N_RELATIONS = 12

# Auto-derives the most plausible relation from a (source_class, target_class) pair.
RELATION_MAP: Dict[Tuple[TokenClass, TokenClass], RelationType] = {
    (TokenClass.OBJECT, TokenClass.SUSPECT): RelationType.BELONGS_TO,
    (TokenClass.MODIFIER, TokenClass.SUSPECT): RelationType.BELONGS_TO,
    (TokenClass.ACTION, TokenClass.LOCATION): RelationType.HAPPENED_AT,
    (TokenClass.EVENT, TokenClass.LOCATION): RelationType.HAPPENED_AT,
    (TokenClass.ACTION, TokenClass.TIME): RelationType.DURING,
    (TokenClass.EVENT, TokenClass.TIME): RelationType.DURING,
    (TokenClass.MOTIVE, TokenClass.ACTION): RelationType.EXPLAINS,
    (TokenClass.MODIFIER, TokenClass.OBJECT): RelationType.DESCRIBES,
    (TokenClass.EMOTION, TokenClass.SUSPECT): RelationType.INDICATES,
    (TokenClass.WITNESS, TokenClass.EVENT): RelationType.OBSERVED,
    (TokenClass.OBJECT, TokenClass.MOTIVE): RelationType.ENABLES,
    (TokenClass.SUSPECT, TokenClass.ACTION): RelationType.COMMITTED,
    (TokenClass.MOTIVE, TokenClass.SUSPECT): RelationType.DRIVES,
    (TokenClass.LOCATION, TokenClass.EVENT): RelationType.SCENE_OF,
    # Evidence-based implication
    (TokenClass.OBJECT, TokenClass.ACTION): RelationType.IMPLICATES,
    (TokenClass.EMOTION, TokenClass.ACTION): RelationType.INDICATES,
    (TokenClass.WITNESS, TokenClass.SUSPECT): RelationType.IMPLICATES,
}

# Cache for valid_relations lookups
_VALID_RELATION_CACHE: Dict[Tuple[TokenClass, TokenClass], List[RelationType]] = {}


def auto_relation(a: Token, b: Token) -> Optional[RelationType]:
    """Derive the most plausible relation from the token class pair (a → b)."""
    r = RELATION_MAP.get((a.token_class, b.token_class))
    if r is not None:
        return r
    # Try reverse direction
    return RELATION_MAP.get((b.token_class, a.token_class))


def valid_relations(a: Token, b: Token) -> List[RelationType]:
    """Return all semantically valid relations between two tokens."""
    key = (a.token_class, b.token_class)
    if key not in _VALID_RELATION_CACHE:
        relations: List[RelationType] = []
        r_fwd = RELATION_MAP.get((a.token_class, b.token_class))
        if r_fwd is not None:
            relations.append(r_fwd)
        r_rev = RELATION_MAP.get((b.token_class, a.token_class))
        if r_rev is not None and r_rev not in relations:
            relations.append(r_rev)
        _VALID_RELATION_CACHE[key] = relations if relations else [RelationType.BELONGS_TO]
    return _VALID_RELATION_CACHE[key]


@dataclass
class Connection:
    token_a: Token
    relation: RelationType
    token_b: Token
    coherence: float = 0.0        # model output, 0–1
    is_contradiction: bool = False
