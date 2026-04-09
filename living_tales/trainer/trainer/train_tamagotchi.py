from __future__ import annotations

from typing import List

import numpy as np
import torch

from trainer.tamagotchi_model import TamagotchiStateModel, TamagotchiCoherenceLoss
from core.cartridge import TamagotchiSpec
from core.token import TokenClass, TokenPhase
from generator.trajectory_sampler import TrajectorySampler


def _sample_state_tokens(tokens, n: int, rng: np.random.Generator):
    idx = rng.choice(len(tokens), size=n, replace=False)
    return [tokens[i] for i in idx]


def train_tamagotchi_cartridge(
    spec_path: str,
    output_dir: str,
    n_trajectories: int = 500,
    trajectory_length: int = 200,
    n_epochs: int = 80,
    device: str = "cpu",
) -> TamagotchiStateModel:
    _ = output_dir

    spec = TamagotchiSpec.load(spec_path)
    model = TamagotchiStateModel(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        n_needs=spec.n_needs,
        n_personality_dims=spec.n_personality_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = TamagotchiCoherenceLoss()

    sampler = TrajectorySampler(spec, trajectory_length=trajectory_length)
    rng = np.random.default_rng()

    tokens = spec.tokens
    id_to_idx = {t.id: i for i, t in enumerate(tokens)}
    class_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
    phase_to_idx = {p.value: i for i, p in enumerate(TokenPhase)}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for _ in range(max(1, n_trajectories // 50)):
            traj = sampler.sample()
            step = rng.integers(0, trajectory_length)

            alive_tokens = _sample_state_tokens(tokens, n=3, rng=rng)
            incoherent_tokens = _sample_state_tokens(tokens, n=3, rng=rng)
            dissolved_tokens = _sample_state_tokens(tokens, n=3, rng=rng)

            def encode(triad: List):
                token_ids = [id_to_idx[t.id] for t in triad]
                class_ids = [class_to_idx[t.token_class.value] for t in triad]
                phase_ids = [phase_to_idx[t.phase.value] for t in triad]
                return token_ids, class_ids, phase_ids

            alive_ids, alive_class, alive_phase = encode(alive_tokens)
            inco_ids, inco_class, inco_phase = encode(incoherent_tokens)
            diss_ids, diss_class, diss_phase = encode(dissolved_tokens)

            hour = torch.tensor([traj["hours"][step]], device=device)
            needs = torch.tensor([traj["needs"][step]], device=device)
            personality = torch.tensor([traj["personality"][step]], device=device)

            e_alive = model.coherence_energy(
                torch.tensor([alive_ids], device=device),
                torch.tensor([alive_class], device=device),
                torch.tensor([alive_phase], device=device),
                hour,
                needs,
                personality,
            )
            e_incoherent = model.coherence_energy(
                torch.tensor([inco_ids], device=device),
                torch.tensor([inco_class], device=device),
                torch.tensor([inco_phase], device=device),
                hour,
                needs,
                personality,
            )
            e_dissolved = model.coherence_energy(
                torch.tensor([diss_ids], device=device),
                torch.tensor([diss_class], device=device),
                torch.tensor([diss_phase], device=device),
                hour,
                needs,
                personality,
            )

            loss = criterion(e_alive, e_incoherent, e_dissolved)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        _ = epoch_loss

    return model
