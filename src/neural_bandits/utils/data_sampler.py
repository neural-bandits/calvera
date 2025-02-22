from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler


class AbstractDataSampler(Sampler[int], ABC):
    """Base class for all custom samplers.

    Implements the basic functionality required for sampling from a dataset.
    Subclasses need only implement the _get_iterator method to define
    their specific sampling strategy.

    Args:
        data_source: Dataset to sample from
    """

    def __init__(
        self,
        data_source: Dataset[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # super().__init__(data_source)
        self.data_source = data_source

    def __len__(self) -> int:
        return len(self.data_source)  # type: ignore

    def __iter__(self) -> Iterator[int]:
        return self._get_iterator()

    @abstractmethod
    def _get_iterator(self) -> Iterator[int]:
        """Core sampling logic to be implemented by subclasses."""
        pass


class RandomDataSampler(AbstractDataSampler):
    """Samples elements randomly without replacement.

    Args:
        data_source: Dataset to sample from
        generator: Optional PyTorch Generator for reproducible randomness
    """

    def __init__(
        self,
        data_source: Dataset[Tuple[torch.Tensor, torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(data_source)
        self.generator = generator

    def _get_iterator(self) -> Iterator[int]:
        """Returns an iterator that yields indices in random order."""
        indices = torch.randperm(
            len(self.data_source), generator=self.generator, dtype=torch.int64  # type: ignore
        ).tolist()

        return iter(indices)


class SortedDataSampler(AbstractDataSampler):
    """Samples elements in sorted order based on a key function.

    Args:
        data_source: Dataset to sample from
        key_fn: Function that returns the sorting key for each dataset index
        reverse: Whether to sort in descending order (default: False)
    """

    def __init__(
        self,
        data_source: Dataset[Tuple[torch.Tensor, torch.Tensor]],
        key_fn: Callable[[int], Any],
        reverse: bool = False,
    ) -> None:
        super().__init__(data_source)
        self.key_fn = key_fn
        self.reverse = reverse

    def _get_iterator(self) -> Iterator[int]:
        """Returns an iterator that yields indices in sorted order."""
        indices = range(len(self.data_source))  # type: ignore
        sorted_indices = sorted(indices, key=self.key_fn, reverse=self.reverse)
        return iter(sorted_indices)
