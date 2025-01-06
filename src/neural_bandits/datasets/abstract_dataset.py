from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class AbstractDataset(ABC, Dataset[torch.Tensor]):
    """
    Abstract class for a dataset that is derived from PyTorch's Dataset class.
    Additionally, it provides a reward method for the specific bandit setting.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        pass
