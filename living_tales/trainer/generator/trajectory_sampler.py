from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.cartridge import TamagotchiSpec


class TrajectorySampler:
    """
    Sampler for tamagotchi trajectories over needs/personality/time.
    """

    def __init__(self, spec: TamagotchiSpec, trajectory_length: int = 200):
        self.spec = spec
        self.trajectory_length = trajectory_length

    def sample(self) -> Dict[str, np.ndarray]:
        n_needs = self.spec.n_needs
        n_personality = self.spec.n_personality_dims

        hours = np.linspace(0, 24, self.trajectory_length, dtype=np.float32)
        needs = np.clip(
            np.random.rand(self.trajectory_length, n_needs) * 1.1, 0.0, 1.0
        )
        personality = np.clip(
            np.random.randn(self.trajectory_length, n_personality) * 0.1, -1.0, 1.0
        )

        return {
            "hours": hours,
            "needs": needs,
            "personality": personality,
        }
