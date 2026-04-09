"""
Shared pytest fixtures for Thornfield evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add trainer to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "thornfield" / "trainer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from core.cartridge import CartridgeSpec
from trainer.dialogue_model import DialogueTransformer
from trainer.train_dialogue import _build_mappings, _dialogue_to_example, _collate_dialogues, PAD_IDX
from generator.dialogue_sampler import DialogueSampler
from utils.game_runner import DialogueGameRunner


CASE_ID = "amber_cipher"
MODEL_DIR = _ROOT / "thornfield" / "trainer" / "outputs" / CASE_ID
CASE_DIR = _ROOT / "thornfield" / "trainer" / "cases" / CASE_ID


@pytest.fixture(scope="session")
def spec():
    spec_path = CASE_DIR / "spec.json"
    if not spec_path.exists():
        pytest.skip(f"Case not packed: {spec_path}")
    return CartridgeSpec.load(str(spec_path))


@pytest.fixture(scope="session")
def trained_model(spec):
    model_path = MODEL_DIR / "dialogue_model.pt"
    if not model_path.exists():
        pytest.skip(f"Model not trained: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DialogueTransformer(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["embedding_dim"],
        context_dim=ckpt["context_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@pytest.fixture(scope="session")
def mappings(spec):
    return dict(zip(
        ["id_to_idx", "class_to_idx", "phase_to_idx", "stream_to_idx", "agency_to_idx"],
        _build_mappings(spec.tokens),
    ))


@pytest.fixture(scope="session")
def eval_dialogues(spec, trained_model, mappings):
    """Sample 200 held-out dialogues and prepare as batched tensors."""
    sampler = DialogueSampler(spec, player_temperature=1.5, engine_temperature=1.0)
    paths = sampler.sample_batch(200, verbose=False)

    id_to_idx = mappings["id_to_idx"]
    class_to_idx = mappings["class_to_idx"]
    phase_to_idx = mappings["phase_to_idx"]
    stream_to_idx = mappings["stream_to_idx"]
    agency_to_idx = mappings["agency_to_idx"]

    examples = [
        _dialogue_to_example(p, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx)
        for p in paths
    ]

    # Batch into groups of 32
    batches = []
    for i in range(0, len(examples), 32):
        batch = _collate_dialogues(examples[i:i+32], "cpu")
        batches.append(batch)
    return batches


@pytest.fixture(scope="session")
def game_results(trained_model, spec, mappings):
    """Run 100 games with different seeds."""
    runner = DialogueGameRunner(trained_model, spec, mappings)
    return runner.run_batch(100, seeds=list(range(100)))
