from collections.abc import Sized
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from calvera.utils.data_sampler import RandomDataSampler, SortedDataSampler


@pytest.fixture
def sample_data() -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Returns a tuple (dataset, sort_values).
    Dataset contains contexts and rewards, sort_values are used for sorting.
    """
    # contexts shape: (5, 3, 4)
    contexts = torch.randn(5, 3, 4)
    # rewards shape: (5, 3)
    rewards = torch.rand(5, 3)
    # values to sort by (simulating class labels like in MNIST example)
    sort_values = torch.tensor([4, 1, 3, 0, 2])

    dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(contexts, rewards))
    return dataset, sort_values


def test_sorted_sampler_ascending(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that sampler correctly sorts indices in ascending order"""
    dataset, sort_values = sample_data

    def key_fn(idx: int) -> int:
        return int(sort_values[idx])

    sampler = SortedDataSampler(dataset, key_fn=key_fn)

    sorted_indices = list(iter(sampler))
    expected_indices = [3, 1, 4, 2, 0]  # indices that would sort [4,1,3,0,2]

    assert sorted_indices == expected_indices

    sorted_values = [sort_values[i].item() for i in sorted_indices]
    assert sorted_values == sorted(sort_values.tolist())


def test_sorted_sampler_descending(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that sampler correctly sorts indices in descending order"""
    dataset, sort_values = sample_data

    def key_fn(idx: int) -> int:
        return int(sort_values[idx])

    sampler = SortedDataSampler(dataset, key_fn=key_fn, reverse=True)

    sorted_indices = list(iter(sampler))
    expected_indices = [0, 2, 4, 1, 3]  # indices that would sort [4,1,3,0,2] in reverse

    assert sorted_indices == expected_indices

    sorted_values = [sort_values[i].item() for i in sorted_indices]
    assert sorted_values == sorted(sort_values.tolist(), reverse=True)


def test_sorted_sampler_with_dataloader(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that sampler works correctly with DataLoader by checking the order of loaded values"""
    dataset, sort_values = sample_data

    def key_fn(idx: int) -> int:
        return int(sort_values[idx])

    sampler = SortedDataSampler(dataset, key_fn=key_fn)

    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    loaded_sort_values: list[int] = []
    for contexts, rewards in dataloader:
        for context, reward in zip(contexts, rewards, strict=False):
            for i in range(len(cast(Sized, dataset))):
                orig_context, orig_reward = dataset[i]
                if torch.equal(context, orig_context) and torch.equal(reward, orig_reward):
                    loaded_sort_values.append(int(sort_values[i]))
                    break

    assert loaded_sort_values == sorted(sort_values.tolist()), "DataLoader did not maintain sorted order"


def test_sorted_sampler_with_subset(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that sampler works correctly with dataset subset"""
    dataset, sort_values = sample_data
    subset_indices = range(3)
    subset = Subset(dataset, subset_indices)

    def key_fn(idx: int) -> int:
        return int(sort_values[subset_indices[idx]])

    sampler = SortedDataSampler(subset, key_fn=key_fn)

    sorted_indices = list(iter(sampler))
    # Expected sorting of first 3 values [4,1,3]
    expected_indices = [1, 2, 0]

    assert sorted_indices == expected_indices

    dataloader = DataLoader(subset, batch_size=2, sampler=sampler)
    loaded_batches = list(dataloader)
    assert len(loaded_batches) == 2
    assert sum(len(batch[0]) for batch in loaded_batches) == len(subset)


def test_random_sampler_basic(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that random sampler returns all indices in random order"""
    dataset, _ = sample_data
    generator = torch.Generator().manual_seed(42)
    sampler = RandomDataSampler(dataset, generator=generator)

    sampled_indices = list(iter(sampler))

    assert len(sampled_indices) == len(cast(Sized, dataset))
    assert set(sampled_indices) == set(range(len(cast(Sized, dataset))))
    assert sampled_indices != list(range(len(cast(Sized, dataset))))


def test_random_sampler_reproducibility(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that random sampler produces same sequence with same seed"""
    dataset, _ = sample_data
    seed = 42

    generator1 = torch.Generator().manual_seed(seed)
    sampler1 = RandomDataSampler(dataset, generator=generator1)
    indices1 = list(iter(sampler1))

    generator2 = torch.Generator().manual_seed(seed)
    sampler2 = RandomDataSampler(dataset, generator=generator2)
    indices2 = list(iter(sampler2))

    generator3 = torch.Generator().manual_seed(seed + 1)
    sampler3 = RandomDataSampler(dataset, generator=generator3)
    indices3 = list(iter(sampler3))

    assert indices1 == indices2, "Same seed should produce same sequence"
    assert indices1 != indices3, "Different seeds should produce different sequences"


def test_random_sampler_with_dataloader(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that random sampler works correctly with DataLoader"""
    dataset, _ = sample_data
    generator = torch.Generator().manual_seed(42)
    sampler = RandomDataSampler(dataset, generator=generator)

    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    seen_indices: set[int] = set()
    for contexts, rewards in dataloader:
        for context, reward in zip(contexts, rewards, strict=False):
            for i in range(len(cast(Sized, dataset))):
                orig_context, orig_reward = dataset[i]
                if torch.equal(context, orig_context) and torch.equal(reward, orig_reward):
                    seen_indices.add(i)
                    break

    assert seen_indices == set(range(len(cast(Sized, dataset)))), "Not all indices were sampled"


def test_random_sampler_with_subset(
    sample_data: tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
) -> None:
    """Test that random sampler works correctly with dataset subset"""
    dataset, _ = sample_data
    subset_indices = range(3)
    subset = Subset(dataset, subset_indices)

    generator = torch.Generator().manual_seed(42)
    sampler = RandomDataSampler(subset, generator=generator)

    sampled_indices = list(iter(sampler))
    assert len(sampled_indices) == len(subset)
    assert set(sampled_indices) == set(range(len(subset)))

    dataloader = DataLoader(subset, batch_size=2, sampler=sampler)
    loaded_batches = list(dataloader)
    assert len(loaded_batches) == 2
    assert sum(len(batch[0]) for batch in loaded_batches) == len(subset)
