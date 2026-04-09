"""Overfitting quality metrics: perplexity and next-token accuracy."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def compute_perplexity(
    model, dialogues: List[Dict[str, torch.Tensor]], device: str = "cpu",
) -> float:
    """
    Per-token perplexity on held-out dialogue trajectories.
    Lower = more overfitted to the case.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dialogues:
            logits = model(
                batch["token_ids"].to(device),
                batch["class_ids"].to(device),
                batch["phase_ids"].to(device),
                batch["stream_ids"].to(device),
                batch["agency_ids"].to(device),
                batch["padding_mask"].to(device),
            )  # (B, S, V)

            targets = batch["targets"].to(device)  # (B, S)
            B, S, V = logits.shape

            nll = F.cross_entropy(
                logits.reshape(B * S, V),
                targets.reshape(B * S),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (targets != -100).sum().item()
            total_nll += nll.item()
            total_tokens += n_valid

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def compute_next_token_accuracy(
    model, dialogues: List[Dict[str, torch.Tensor]], device: str = "cpu",
) -> float:
    """Top-1 accuracy of next-token prediction."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dialogues:
            logits = model(
                batch["token_ids"].to(device),
                batch["class_ids"].to(device),
                batch["phase_ids"].to(device),
                batch["stream_ids"].to(device),
                batch["agency_ids"].to(device),
                batch["padding_mask"].to(device),
            )

            targets = batch["targets"].to(device)
            preds = logits.argmax(dim=-1)  # (B, S)
            valid = targets != -100
            correct += (preds[valid] == targets[valid]).sum().item()
            total += valid.sum().item()

    return correct / max(total, 1)
