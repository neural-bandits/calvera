from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class AbstractDataset(ABC, Dataset[torch.Tensor]):
    """"""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        pass
