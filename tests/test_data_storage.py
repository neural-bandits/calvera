from typing import Any, Dict

import pytest
import torch
from torch.testing import assert_close

from neural_bandits.utils.data_storage import AllDataBufferStrategy, InMemoryDataBuffer, SlidingWindowBufferStrategy


def test_all_data_strategy() -> None:
    strategy = AllDataBufferStrategy()
    indices = strategy.get_training_indices(5)
    assert torch.equal(indices, torch.arange(5))


@pytest.mark.parametrize(
    "total_samples,window_size,expected",
    [
        (2, 3, torch.arange(2)),  # Less data than window
        (5, 3, torch.arange(2, 5)),  # More data than window
    ],
)
def test_sliding_window_strategy(total_samples: int, window_size: int, expected: torch.Tensor) -> None:
    strategy = SlidingWindowBufferStrategy(window_size=window_size)
    indices = strategy.get_training_indices(total_samples)
    assert torch.equal(indices, expected)


@pytest.fixture
def buffer() -> InMemoryDataBuffer[torch.Tensor]:
    return InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy(), max_size=None)


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    context_dim = 4
    embedding_dim = 3
    batch_size = 2

    return {
        "contextualized_actions": torch.randn(batch_size, context_dim),
        "embedded_actions": torch.randn(batch_size, embedding_dim),
        "rewards": torch.randn(batch_size),
        "context_dim": context_dim,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
    }


def test_initial_state(buffer: InMemoryDataBuffer[torch.Tensor]) -> None:
    assert len(buffer) == 0
    assert buffer.contextualized_actions.shape == torch.Size([0, 0, 0])
    assert buffer.embedded_actions.shape == torch.Size([0, 0])
    assert buffer.rewards.shape == torch.Size([0])


def test_add_batch(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    assert len(buffer) == sample_data["batch_size"]
    assert buffer.contextualized_actions.shape == torch.Size([sample_data["batch_size"], 1, sample_data["context_dim"]])
    assert buffer.embedded_actions.shape == torch.Size([sample_data["batch_size"], sample_data["embedding_dim"]])
    assert buffer.rewards.shape == torch.Size([sample_data["batch_size"]])


def test_add_batch_without_embeddings(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(sample_data["contextualized_actions"], None, sample_data["rewards"])

    assert len(buffer) == sample_data["batch_size"]
    assert buffer.embedded_actions.shape == torch.Size([0, 0])


def test_max_size_limit(sample_data: Dict[str, Any]) -> None:
    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy(), max_size=2)

    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )
    first_batch = buffer.contextualized_actions.clone()

    new_context = torch.randn(sample_data["batch_size"], sample_data["context_dim"])
    new_embedded = torch.randn(sample_data["batch_size"], sample_data["embedding_dim"])
    new_rewards = torch.randn(sample_data["batch_size"])

    buffer.add_batch(new_context, new_embedded, new_rewards)

    # Check that only the most recent data is kept
    assert len(buffer) == 2
    assert not torch.equal(buffer.contextualized_actions, first_batch)


def test_get_batch(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    context_batch, embedded_batch, rewards_batch = buffer.get_batch(1)

    assert context_batch.shape == torch.Size([1, sample_data["context_dim"]])
    if embedded_batch is not None:
        assert embedded_batch.shape == torch.Size([1, sample_data["embedding_dim"]])
    assert rewards_batch.shape == torch.Size([1])


def test_get_batch_error(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    with pytest.raises(ValueError):
        buffer.get_batch(3)  # Request more samples than available


def test_update_embeddings(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    new_embeddings = torch.randn(sample_data["batch_size"], sample_data["embedding_dim"])
    buffer.update_embeddings(new_embeddings)

    assert_close(buffer.embedded_actions, new_embeddings)


def test_state_dict(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    state = buffer.state_dict()

    assert isinstance(state, dict)
    assert torch.equal(state["contextualized_actions"], buffer.contextualized_actions)
    assert torch.equal(state["embedded_actions"], buffer.embedded_actions)
    assert torch.equal(state["rewards"], buffer.rewards)
    assert state["max_size"] == buffer.max_size


def test_load_state_dict(buffer: InMemoryDataBuffer[torch.Tensor], sample_data: Dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    state = buffer.state_dict()

    new_buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy(), max_size=None)
    new_buffer.load_state_dict(state)

    assert torch.equal(new_buffer.contextualized_actions, buffer.contextualized_actions)
    assert torch.equal(new_buffer.embedded_actions, buffer.embedded_actions)
    assert torch.equal(new_buffer.rewards, buffer.rewards)
    assert new_buffer.max_size == buffer.max_size
