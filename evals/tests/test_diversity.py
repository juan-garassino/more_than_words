"""Eval Pass C: Path diversity relative to baselines."""
from metrics.diversity import compute_path_jaccard_distance, compute_structural_diversity


def test_path_jaccard_distance(game_results, baselines):
    jaccard = compute_path_jaccard_distance(game_results)
    # Model should be at least as diverse as random play
    threshold = baselines["random_jaccard_distance"] * 0.8
    assert jaccard > threshold, (
        f"Jaccard distance {jaccard:.3f} below threshold {threshold:.3f} "
        f"(baseline random={baselines['random_jaccard_distance']:.3f})"
    )


def test_structural_diversity(game_results, baselines):
    metrics = compute_structural_diversity(game_results)
    threshold = baselines["random_class_entropy"] * 0.8
    assert metrics["class_entropy"] > threshold, (
        f"Class entropy {metrics['class_entropy']:.2f} below threshold {threshold:.2f} "
        f"(baseline random={baselines['random_class_entropy']:.2f})"
    )
