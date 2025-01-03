import numpy as np
import torch
from sklearn.datasets import fetch_covtype

from .abstract_dataset import AbstractDataset


class CovertypeDataset(AbstractDataset):
    """Loads the Covertype dataset as a pytorch Dataset from the UCI repository (https://archive.ics.uci.edu/ml/datasets/covertype)."""

    num_actions: int = 7
    context_size: int = 54
    num_samples: int = 581012

    def __init__(self) -> None:
        self.data = fetch_covtype()
        self.X = self.data.data.astype(np.float32)
        self.y = self.data.target.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.zeros(self.num_actions, dtype=torch.float32)
        y_item[self.y[idx] - 1] = 1.0
        return X_item

    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(self.y[idx] == action + 1), dtype=torch.float32)
