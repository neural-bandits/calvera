from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic

import torch
from torch.utils.data import Dataset

from calvera.bandits.action_input_type import ActionInputType
from calvera.utils.multiclass import MultiClassContextualizer


class AbstractDataset(ABC, Generic[ActionInputType], Dataset[tuple[ActionInputType, torch.Tensor]]):
    """Abstract class for a dataset that is derived from PyTorch's Dataset class.

    Additionally, it provides a reward method for the specific bandit setting.

    Subclasses should have the following attributes:
    - num_actions  - The maximum number of actions available to the agent.
    - context_size - The standard size of the context vector.
        If needs_disjoint_contextualization is True, the number of features should be multiplied by the number of
        actions.

    ActionInputType Generic:
        The type of the contextualized actions that are input to the bandit.
    """

    num_actions: int
    context_size: int

    def __init__(self, needs_disjoint_contextualization: bool = False) -> None:
        """Initialize the dataset.

        Args:
            needs_disjoint_contextualization: Whether the dataset needs disjoint contextualization.
        """
        self.contextualizer: MultiClassContextualizer | Callable[[Any], Any]
        if needs_disjoint_contextualization:
            self.contextualizer = MultiClassContextualizer(self.num_actions)
        else:
            self.contextualizer = lambda x: x

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of contexts / samples in this dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[ActionInputType, torch.Tensor]:
        """Retrieve the item and the associated rewards for a given index.

        Returns:
            A tuple containing the item and the rewards of the different actions.
        """
        pass

    @abstractmethod
    def reward(self, idx: int, action: int) -> float:
        """Returns the reward for a given index and action."""
        pass

    def sort_key(self, idx: int) -> int:
        """Return the sort key for a given index.

        This is only required as the sort key in the  SortedDataSampler in the benchmark,
        i.e. in a special setting where the data is passed sorted to the model.

        Therefore, we don't require this method to be implemented in all subclasses.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            The sort key for the given index.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}()"
