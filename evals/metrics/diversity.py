"""Path diversity metrics: Jaccard distance and structural variety."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from datasets._schema import GameResult


def compute_path_jaccard_distance(results: List[GameResult]) -> float:
    """
    Average pairwise Jaccard distance between game token sequences.
    Higher = more diverse paths through the story.
    """
    if len(results) < 2:
        return 0.0

    token_sets = [set(t.token_id for t in r.turns) for r in results]
    distances = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            inter = len(a & b)
            if union > 0:
                distances.append(1.0 - inter / union)
    return float(np.mean(distances)) if distances else 0.0


def compute_structural_diversity(results: List[GameResult]) -> Dict[str, float]:
    """Multiple diversity metrics."""
    if not results:
        return {"unique_tokens_ratio": 0.0, "class_entropy": 0.0, "mean_length": 0.0}

    # Unique tokens across all games
    all_tokens = set()
    all_classes = []
    lengths = []
    for r in results:
        for t in r.turns:
            all_tokens.add(t.token_id)
            all_classes.append(t.token_class)
        lengths.append(len(r.turns))

    # Class entropy
    class_counts: Dict[str, int] = {}
    for c in all_classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    total = sum(class_counts.values())
    entropy = 0.0
    for count in class_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return {
        "unique_tokens_ratio": len(all_tokens) / max(1, sum(lengths)),
        "class_entropy": entropy,
        "mean_length": float(np.mean(lengths)),
    }
