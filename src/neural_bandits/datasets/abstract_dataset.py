from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.utils.data import Dataset

from neural_bandits.utils.multiclass import MultiClassContextualizer


class AbstractDataset(ABC, Dataset[torch.Tensor]):
    """
    Abstract class for a dataset that is derived from PyTorch's Dataset class.
    Additionally, it provides a reward method for the specific bandit setting.

    Subclasses should have the following to attributes:
    - num_actions  - The maximum number of actions available to the agent.
    - context_size - The standard size of the context vector.
        If needs_disjoint_contextualization is True, the context size will be multiplied by the number of actions.
    """

    num_actions: int
    context_size: int

    def __init__(self, needs_disjoint_contextualization: bool = False) -> None:
        if needs_disjoint_contextualization:
            self.contextualizer = MultiClassContextualizer(self.num_actions)
        else:
            self.contextualizer = lambda x: x

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple with the context vectors of all available actions and the associated rewards.
        """
        pass

    @abstractmethod
    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
