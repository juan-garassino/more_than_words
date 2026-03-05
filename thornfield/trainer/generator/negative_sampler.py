from __future__ import annotations

from typing import List

import numpy as np

from core.token import Token


def sample_negative_triads(tokens: List[Token], n_samples: int = 256) -> List[List[Token]]:
    rng = np.random.default_rng()
    negatives: List[List[Token]] = []
    if len(tokens) < 3:
        return negatives

    for _ in range(n_samples):
        trio = rng.choice(tokens, size=3, replace=False).tolist()
        negatives.append(trio)

    return negatives
