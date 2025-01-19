from typing import Any, Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_covtype

from .abstract_dataset import AbstractDataset


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

    def __getitem__(self, idx: int) -> torch.Tensor:
        context = self.X[idx].reshape(1, -1)
        return self.contextualizer(context).squeeze(0)

    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(self.y[idx] == action + 1), dtype=torch.float32)

    def optimal_action(self, idx: int) -> Tuple[int, torch.Tensor]:
        opt_idx = self.y[idx] - 1

        return opt_idx, self[idx, opt_idx]
