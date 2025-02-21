from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_covtype

from neural_bandits.benchmark.datasets.abstract_dataset import AbstractDataset


class CovertypeDataset(AbstractDataset):
    """Loads the Covertype dataset as a PyTorch Dataset from the UCI repository (https://archive.ics.uci.edu/ml/datasets/covertype)."""

    num_actions: int = 7
    context_size: int = 54
    num_samples: int = 581012

    def __init__(self) -> None:
        super().__init__(needs_disjoint_contextualization=True)
        self.data = fetch_covtype()
        X_np = self.data.data.astype(np.float32)
        y_np = self.data.target.astype(np.int64)

        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.X[idx].reshape(1, -1)
        contextualized_actions = self.contextualizer(context).squeeze(0)
        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        return float(self.y[idx] == action + 1)
