"""Eval Pass C: Path diversity — structural variety across games."""
from metrics.diversity import compute_path_jaccard_distance, compute_structural_diversity


def test_path_jaccard_distance(game_results):
    jaccard = compute_path_jaccard_distance(game_results)
    assert jaccard > 0.2, f"Mean Jaccard distance {jaccard:.3f} too low (need > 0.2)"


def test_structural_diversity(game_results):
    metrics = compute_structural_diversity(game_results)
    assert metrics["class_entropy"] > 2.0, (
        f"Class entropy {metrics['class_entropy']:.2f} too low (need > 2.0)"
    )
