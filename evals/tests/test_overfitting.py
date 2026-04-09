"""Eval Pass A: Overfitting quality — perplexity and accuracy."""
from metrics.overfitting import compute_perplexity, compute_next_token_accuracy


def test_perplexity_below_threshold(trained_model, eval_dialogues):
    ppl = compute_perplexity(trained_model, eval_dialogues)
    assert ppl < 10.0, f"Perplexity {ppl:.2f} exceeds threshold 10.0"


def test_next_token_accuracy(trained_model, eval_dialogues):
    acc = compute_next_token_accuracy(trained_model, eval_dialogues)
    assert acc > 0.5, f"Accuracy {acc:.2%} below threshold 50%"
