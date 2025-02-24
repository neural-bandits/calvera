from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Tuple, TypeVar

import torch
from torch.utils.data import Dataset

from neural_bandits.benchmark.multiclass import MultiClassContextualizer

# TextItemType is a tuple of three torch tensors. We use this type to represent the
# input to a model from the `transformers` library. Corresponds to the `input_ids`,
# `attention_mask`, and `token_type_ids`.
TextItemType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

ItemType = TypeVar("ItemType", torch.Tensor, TextItemType)


class AbstractDataset(ABC, Generic[ItemType], Dataset[Tuple[ItemType, torch.Tensor]]):
    """Abstract class for a dataset that is derived from PyTorch's Dataset class.

    Additionally, it provides a reward method for the specific bandit setting.

    Subclasses should have the following to attributes:
    - num_actions  -- The maximum number of actions available to the agent.
    - context_size --  The standard size of the context vector.
        If `needs_disjoint_contextualization` is set to True, the context size will be multiplied by the number of actions.
        This is useful for datasets where the context is the same for all actions like multi-class classification.
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
    def __getitem__(self, idx: int) -> Tuple[ItemType, torch.Tensor]:
        """Retrieve the item and the associated rewards for a given index.

        Returns:
            A tuple containing the item and the rewards of the different actions.
        """
        pass

    @abstractmethod
    def reward(self, idx: int, action: int) -> float:
        """Returns the reward for a given index and action."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}()"
