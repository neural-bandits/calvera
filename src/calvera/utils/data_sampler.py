from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


class AbstractDataSampler(Sampler[int], ABC):
    """Base class for all custom samplers.

    Implements the basic functionality required for sampling from a dataset.
    Subclasses need only implement the _get_iterator method to define
    their specific sampling strategy.
    """

    def __init__(
        self,
        data_source: Dataset[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Initializes the AbstractDataSampler.

        Args:
            data_source: Dataset to sample from
        """
        self.data_source = data_source

    def __len__(self) -> int:
        """Returns the number of elements in the data source."""
        return len(self.data_source)  # type: ignore

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator of the specified `data_source` indices in random order."""
        return self._get_iterator()

    @abstractmethod
    def _get_iterator(self) -> Iterator[int]:
        """Core sampling logic to be implemented by subclasses."""
        pass


class RandomDataSampler(AbstractDataSampler):
    """Samples elements randomly without replacement."""

    def __init__(
        self,
        data_source: Dataset[tuple[torch.Tensor, torch.Tensor]],
        generator: torch.Generator | None = None,
    ) -> None:
        """Initializes the RandomDataSampler.

        Args:
            data_source: Dataset to sample from
            generator: Optional PyTorch Generator for reproducible randomness
        """
        super().__init__(data_source)
        self.generator = generator

    def _get_iterator(self) -> Iterator[int]:
        """Returns an iterator that yields indices in random order."""
        indices = torch.randperm(
            len(self.data_source), generator=self.generator, dtype=torch.int64  # type: ignore
        ).tolist()

        return iter(indices)


class SortedDataSampler(AbstractDataSampler):
    """Samples elements in sorted order based on a key function."""

    def __init__(
        self,
        data_source: Dataset[tuple[torch.Tensor, torch.Tensor]],
        key_fn: Callable[[int], Any],
        reverse: bool = False,
    ) -> None:
        """Initializes the SortedDataSampler.

        Args:
            data_source: Dataset to sample from
            key_fn: Function that returns the sorting key for each dataset index
            reverse: Whether to sort in descending order (default: False)
        """
        super().__init__(data_source)
        self.key_fn = key_fn
        self.reverse = reverse

    def _get_iterator(self) -> Iterator[int]:
        """Returns an iterator that yields indices in sorted order."""
        indices = range(len(self.data_source))  # type: ignore
        sorted_indices = sorted(indices, key=self.key_fn, reverse=self.reverse)
        return iter(sorted_indices)
