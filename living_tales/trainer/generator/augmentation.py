from __future__ import annotations

from typing import List

from core.token import Token


def jitter_temperature(tokens: List[Token], amount: float = 0.05) -> List[Token]:
    for token in tokens:
        token.temperature = max(0.0, min(1.0, token.temperature + amount))
    return tokens
