import torch
from ucimlrepo import fetch_ucirepo

from .abstract_dataset import AbstractDataset


class StatlogDataset(AbstractDataset):
    """Loads the Statlog (Shuttle) dataset as a PyTorch Dataset from the UCI repository (https://archive.ics.uci.edu/dataset/148/statlog+shuttle)."""

    num_actions: int = 9
    context_size: int = 7
    num_samples: int = 58000

    def __init__(self) -> None:
        super().__init__(needs_disjoint_contextualization=True)
        dataset = fetch_ucirepo(
            id=148
        )  # id=148 specifies the Statlog (Shuttle) dataset
        X = dataset.data.features
        y = dataset.data.targets

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        print(self.X[idx].unsqueeze(0))
        return self.contextualizer(self.X[idx].unsqueeze(0)).squeeze(0)

    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(self.y[idx] == action + 1), dtype=torch.float32)
