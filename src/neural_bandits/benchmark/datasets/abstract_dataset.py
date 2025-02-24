from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Tuple

import torch
from torch.utils.data import Dataset

from neural_bandits.bandits.abstract_bandit import ActionInputType
from neural_bandits.benchmark.multiclass import MultiClassContextualizer


class AbstractDataset(
    ABC, Generic[ActionInputType], Dataset[Tuple[ActionInputType, torch.Tensor]]
):
    """
    Abstract class for a dataset that is derived from PyTorch's Dataset class.
    Additionally, it provides a reward method for the specific bandit setting.

    Subclasses should have the following attributes:
    - num_actions  - The maximum number of actions available to the agent.
    - context_size - The standard size of the context vector.
        If needs_disjoint_contextualization is True, the context size will be multiplied by the number of actions.
    - input_size   - The size of the input vector for the neural network.
        If needs_disjoint_contextualization is True, the input size will be context_size * num_actions. Otherwise context_size.

    ActionInputType Generic:
        The type of the contextualized actions that are input to the bandit.
    """

    num_actions: int
    context_size: int
    input_size: int

    def __init__(self, needs_disjoint_contextualization: bool = False) -> None:
        self.contextualizer: MultiClassContextualizer | Callable[[Any], Any]
        if needs_disjoint_contextualization:
            self.contextualizer = MultiClassContextualizer(self.num_actions)
            self.input_size = self.context_size * self.num_actions
        else:
            self.contextualizer = lambda x: x
            self.input_size = self.context_size

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[ActionInputType, torch.Tensor]:
        """
        Returns:
            A tuple with the context vectors of all available actions and the associated rewards.
        """
        pass

    @abstractmethod
    def reward(self, idx: int, action: int) -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
