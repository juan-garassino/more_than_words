from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from core.cartridge import CartridgeSpec
from core.token import Token, TokenClass, TokenPhase
from generator.path_sampler import PathSampler
from generator.negative_sampler import sample_negative_triads
from trainer.energy_model import MysteryEnergyModel
from trainer.loss import CombinedMysteryLoss


@dataclass
class TrainingExample:
    context_tokens: List[Token]
    context_positions: List[Tuple[int, int]]
    triad: List[Token]
    cumulative_dimensions: np.ndarray
    target_delta: np.ndarray


def _build_mappings(tokens: List[Token]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    id_to_idx = {t.id: i for i, t in enumerate(tokens)}
    class_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
    phase_to_idx = {p.value: i for i, p in enumerate(TokenPhase)}
    return id_to_idx, class_to_idx, phase_to_idx


def _encode_tokens(tokens: List[Token], id_to_idx, class_to_idx, phase_to_idx):
    token_ids = [id_to_idx[t.id] for t in tokens]
    class_ids = [class_to_idx[t.token_class.value] for t in tokens]
    phase_ids = [phase_to_idx[t.phase.value] for t in tokens]
    return token_ids, class_ids, phase_ids


def _pad_sequences(seqs: List[List[int]], pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=1)
    out = np.full((len(seqs), max_len), pad, dtype=np.int64)
    mask = np.ones((len(seqs), max_len), dtype=bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = False
    return torch.tensor(out), torch.tensor(mask)


def _build_examples(spec: CartridgeSpec, n_paths: int) -> List[TrainingExample]:
    sampler = PathSampler(spec, sampling_temperature=1.2)
    paths = sampler.sample_batch(n_paths, verbose=False)
    examples: List[TrainingExample] = []

    for path in paths:
        placed_tokens: List[Token] = []
        placed_positions: List[Tuple[int, int]] = []
        cumulative = np.zeros(spec.n_attractor_dims, dtype=np.float32)

        for turn, triad in enumerate(path):
            if all(t.is_invariant for t in triad):
                break

            contribution = np.stack([t.attractor_weights for t in triad]).mean(axis=0)
            target_delta = np.minimum(1.0, contribution * 0.25)

            examples.append(
                TrainingExample(
                    context_tokens=list(placed_tokens),
                    context_positions=list(placed_positions),
                    triad=list(triad),
                    cumulative_dimensions=np.minimum(
                        1.0, cumulative + contribution * 0.25
                    ),
                    target_delta=target_delta,
                )
            )

            row = min(turn, 7)
            col = turn % 6
            placed_tokens.extend(triad)
            placed_positions.extend([(row, col)] * 3)
            cumulative = np.minimum(1.0, cumulative + contribution * 0.25)

    return examples


def _batchify(
    examples: List[TrainingExample],
    negatives: List[List[Token]],
    id_to_idx,
    class_to_idx,
    phase_to_idx,
    device: str,
) -> Dict[str, torch.Tensor]:
    context_token_ids = []
    context_class_ids = []
    context_phase_ids = []
    context_positions = []

    candidate_token_ids = []
    candidate_class_ids = []
    candidate_phase_ids = []

    negative_token_ids = []
    negative_class_ids = []
    negative_phase_ids = []

    cumulative_dimensions = []
    target_delta = []

    for ex, neg in zip(examples, negatives):
        t_ids, c_ids, p_ids = _encode_tokens(ex.context_tokens, id_to_idx, class_to_idx, phase_to_idx)
        context_token_ids.append(t_ids)
        context_class_ids.append(c_ids)
        context_phase_ids.append(p_ids)
        context_positions.append([list(pos) for pos in ex.context_positions])

        t_ids, c_ids, p_ids = _encode_tokens(ex.triad, id_to_idx, class_to_idx, phase_to_idx)
        candidate_token_ids.append(t_ids)
        candidate_class_ids.append(c_ids)
        candidate_phase_ids.append(p_ids)

        t_ids, c_ids, p_ids = _encode_tokens(neg, id_to_idx, class_to_idx, phase_to_idx)
        negative_token_ids.append(t_ids)
        negative_class_ids.append(c_ids)
        negative_phase_ids.append(p_ids)

        cumulative_dimensions.append(ex.cumulative_dimensions)
        target_delta.append(ex.target_delta)

    context_token_ids, context_mask = _pad_sequences(context_token_ids, pad=0)
    context_class_ids, _ = _pad_sequences(context_class_ids, pad=0)
    context_phase_ids, _ = _pad_sequences(context_phase_ids, pad=0)

    max_len = context_token_ids.size(1)
    pos_arr = np.zeros((len(context_positions), max_len, 2), dtype=np.float32)
    for i, positions in enumerate(context_positions):
        for j, pos in enumerate(positions[:max_len]):
            pos_arr[i, j] = np.array(pos, dtype=np.float32)

    batch = {
        "placed_token_ids": context_token_ids.to(device),
        "placed_class_ids": context_class_ids.to(device),
        "placed_phase_ids": context_phase_ids.to(device),
        "placed_positions": torch.tensor(pos_arr, device=device),
        "placed_mask": context_mask.to(device),
        "candidate_token_ids": torch.tensor(candidate_token_ids, device=device),
        "candidate_class_ids": torch.tensor(candidate_class_ids, device=device),
        "candidate_phase_ids": torch.tensor(candidate_phase_ids, device=device),
        "negative_token_ids": torch.tensor(negative_token_ids, device=device),
        "negative_class_ids": torch.tensor(negative_class_ids, device=device),
        "negative_phase_ids": torch.tensor(negative_phase_ids, device=device),
        "cumulative_dimensions": torch.tensor(np.stack(cumulative_dimensions), device=device),
        "target_delta": torch.tensor(np.stack(target_delta), device=device),
    }

    return batch


def train_mystery_cartridge(
    spec_path: str,
    output_dir: str,
    n_paths: int = 1000,
    n_epochs: int = 50,
    convergence_rate: float = 0.25,
    min_turns: int = 10,
    max_turns: int = 18,
    device: str = "cpu",
) -> Tuple[MysteryEnergyModel, Dict[str, float]]:
    _ = (output_dir, convergence_rate, min_turns, max_turns)

    spec = CartridgeSpec.load(spec_path)
    model = MysteryEnergyModel(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        context_dim=spec.context_dim,
        n_attractor_dims=spec.n_attractor_dims,
        token_graph=spec.token_graph,
    ).to(device)

    id_to_idx, class_to_idx, phase_to_idx = _build_mappings(spec.tokens)
    examples = _build_examples(spec, n_paths)
    if not examples:
        print("No training examples generated. Exiting.", flush=True)
        return model, {"loss": 0.0}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedMysteryLoss()

    batch_size = min(64, len(examples))
    history = {"loss": 0.0}
    total_batches = max(1, (len(examples) + batch_size - 1) // batch_size)
    print(f"Examples: {len(examples)} | Batch size: {batch_size} | Batches/epoch: {total_batches}", flush=True)

    use_rich = False
    progress = None
    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

        progress = Progress(
            TextColumn("[bold]Epoch {task.fields[epoch]}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} batches"),
            TimeElapsedColumn(),
        )
        use_rich = True
    except Exception:
        use_rich = False

    for epoch in range(n_epochs):
        np.random.shuffle(examples)
        epoch_loss = 0.0
        batch_count = total_batches

        if use_rich and progress is not None:
            with progress:
                task_id = progress.add_task("train", total=batch_count, epoch=f"{epoch+1:02d}/{n_epochs}")
                for start in range(0, len(examples), batch_size):
                    batch_examples = examples[start : start + batch_size]
                    negatives = sample_negative_triads(spec.tokens, n_samples=len(batch_examples))

                    batch = _batchify(
                        batch_examples,
                        negatives,
                        id_to_idx,
                        class_to_idx,
                        phase_to_idx,
                        device,
                    )

                    model.train()
                    output_pos = model(
                        batch["placed_token_ids"],
                        batch["placed_class_ids"],
                        batch["placed_phase_ids"],
                        batch["placed_positions"],
                        batch["placed_mask"],
                        batch["candidate_token_ids"],
                        batch["candidate_class_ids"],
                        batch["candidate_phase_ids"],
                    )
                    output_neg = model(
                        batch["placed_token_ids"],
                        batch["placed_class_ids"],
                        batch["placed_phase_ids"],
                        batch["placed_positions"],
                        batch["placed_mask"],
                        batch["negative_token_ids"],
                        batch["negative_class_ids"],
                        batch["negative_phase_ids"],
                    )

                    loss_dict = criterion(
                        {
                            "energy_positive": output_pos["energy"],
                            "energy_negative": output_neg["energy"],
                            "cumulative_dimensions": batch["cumulative_dimensions"],
                            "path_energies": torch.zeros((len(batch_examples), 2), device=device),
                            "predicted_delta": output_pos["convergence_delta"],
                            "target_delta": batch["target_delta"],
                        }
                    )

                    optimizer.zero_grad()
                    loss_dict["total"].backward()
                    optimizer.step()

                    epoch_loss += float(loss_dict["total"].detach().cpu())
                    progress.advance(task_id)
        else:
            for idx, start in enumerate(range(0, len(examples), batch_size), start=1):
                batch_examples = examples[start : start + batch_size]
                negatives = sample_negative_triads(spec.tokens, n_samples=len(batch_examples))

                batch = _batchify(
                    batch_examples,
                    negatives,
                    id_to_idx,
                    class_to_idx,
                    phase_to_idx,
                    device,
                )

                model.train()
                output_pos = model(
                    batch["placed_token_ids"],
                    batch["placed_class_ids"],
                    batch["placed_phase_ids"],
                    batch["placed_positions"],
                    batch["placed_mask"],
                    batch["candidate_token_ids"],
                    batch["candidate_class_ids"],
                    batch["candidate_phase_ids"],
                )
                output_neg = model(
                    batch["placed_token_ids"],
                    batch["placed_class_ids"],
                    batch["placed_phase_ids"],
                    batch["placed_positions"],
                    batch["placed_mask"],
                    batch["negative_token_ids"],
                    batch["negative_class_ids"],
                    batch["negative_phase_ids"],
                )

                loss_dict = criterion(
                    {
                        "energy_positive": output_pos["energy"],
                        "energy_negative": output_neg["energy"],
                        "cumulative_dimensions": batch["cumulative_dimensions"],
                        "path_energies": torch.zeros((len(batch_examples), 2), device=device),
                        "predicted_delta": output_pos["convergence_delta"],
                        "target_delta": batch["target_delta"],
                    }
                )

                optimizer.zero_grad()
                loss_dict["total"].backward()
                optimizer.step()

                epoch_loss += float(loss_dict["total"].detach().cpu())
                if idx % 10 == 0 or idx == batch_count:
                    print(f"Epoch {epoch+1:02d}/{n_epochs} | batch {idx}/{batch_count}", flush=True)

        history["loss"] = epoch_loss / batch_count
        print(f"Epoch {epoch+1:02d}/{n_epochs} | loss={history['loss']:.4f}", flush=True)

    return model, history
