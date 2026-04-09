"""Eval Pass A: Overfitting quality — perplexity and accuracy relative to baselines."""
from metrics.overfitting import compute_perplexity, compute_next_token_accuracy


def test_perplexity_below_threshold(trained_model, eval_dialogues, baselines):
    ppl = compute_perplexity(trained_model, eval_dialogues)
    threshold = baselines["random_perplexity"] * 0.5
    assert ppl < threshold, (
        f"Perplexity {ppl:.2f} exceeds threshold {threshold:.1f} "
        f"(baseline random={baselines['random_perplexity']:.1f})"
    )


def test_next_token_accuracy(trained_model, eval_dialogues, baselines):
    acc = compute_next_token_accuracy(trained_model, eval_dialogues)
    threshold = baselines["random_accuracy"] * 3.0
    assert acc > threshold, (
        f"Accuracy {acc:.2%} below threshold {threshold:.2%} "
        f"(baseline random={baselines['random_accuracy']:.2%})"
    )
