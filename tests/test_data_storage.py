from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close

from calvera.bandits.abstract_bandit import _collate_fn
from calvera.utils.data_storage import AllDataBufferStrategy, TensorDataBuffer, SlidingWindowBufferStrategy


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
def buffer() -> TensorDataBuffer[torch.Tensor]:
    return TensorDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy(), max_size=None)


@pytest.fixture
def sample_data() -> dict[str, Any]:
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


def test_initial_state(buffer: TensorDataBuffer[torch.Tensor]) -> None:
    assert len(buffer) == 0
    assert buffer.contextualized_actions is None
    assert buffer.embedded_actions.shape == torch.Size([0, 0])
    assert buffer.rewards.shape == torch.Size([0])


def test_add_batch(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    assert len(buffer) == sample_data["batch_size"]
    assert cast(torch.Tensor, buffer.contextualized_actions).shape == torch.Size(
        [sample_data["batch_size"], 1, sample_data["context_dim"]]
    )
    assert buffer.embedded_actions.shape == torch.Size([sample_data["batch_size"], sample_data["embedding_dim"]])
    assert buffer.rewards.shape == torch.Size([sample_data["batch_size"]])


def test_add_batch_without_embeddings(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(sample_data["contextualized_actions"], None, sample_data["rewards"])

    assert len(buffer) == sample_data["batch_size"]
    assert buffer.embedded_actions.shape == torch.Size([0, 0])


def test_max_size_limit(sample_data: dict[str, Any]) -> None:
    buffer = TensorDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy(), max_size=2)

    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )
    first_batch = cast(torch.Tensor, buffer.contextualized_actions).clone()

    new_context = torch.randn(sample_data["batch_size"], sample_data["context_dim"])
    new_embedded = torch.randn(sample_data["batch_size"], sample_data["embedding_dim"])
    new_rewards = torch.randn(sample_data["batch_size"])

    buffer.add_batch(new_context, new_embedded, new_rewards)

    # Check that only the most recent data is kept
    assert len(buffer) == 2
    assert not torch.equal(cast(torch.Tensor, buffer.contextualized_actions), first_batch)


def test_get_all_data(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    context_data, embedded_data, rewards_data, _ = buffer.get_all_data()

    assert torch.allclose(context_data, sample_data["contextualized_actions"])
    assert embedded_data is not None, "Embedded actions should not be None"
    assert torch.allclose(embedded_data, sample_data["embedded_actions"])
    assert torch.allclose(rewards_data, sample_data["rewards"])

    assert buffer.len_of_all_data() == sample_data["batch_size"]


def test_get_batch(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    context_batch, embedded_batch, rewards_batch, _ = buffer.get_batch(1)

    assert context_batch.shape == torch.Size([1, sample_data["context_dim"]])
    if embedded_batch is not None:
        assert embedded_batch.shape == torch.Size([1, sample_data["embedding_dim"]])
    assert rewards_batch.shape == torch.Size([1])


def test_get_batch_error(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    with pytest.raises(ValueError):
        buffer.get_batch(3)  # Request more samples than available


def test_data_loader(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        None,
        sample_data["rewards"],
    )

    dataloader = torch.utils.data.DataLoader(buffer, batch_size=1, collate_fn=_collate_fn)

    i = 0
    for data in dataloader:
        assert len(data) == 4, "Data should be a tuple of (context, embeddings = None, reward, chosen_actions = None)"
        context, reward = data[0], data[2]
        i += reward.shape[0]
        assert context.shape == torch.Size([1, 1, sample_data["context_dim"]])
        assert reward.shape == torch.Size([1, 1])

    assert i == sample_data["batch_size"]


def test_data_loader_with_embeddings(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    dataloader = torch.utils.data.DataLoader(buffer, batch_size=1, collate_fn=_collate_fn)

    i = 0
    for data in dataloader:
        assert len(data) == 4, "Data should be a tuple of (context, embeddings = None, reward, chosen_actions = None)"
        context, embedding, reward = data[:3]
        i += reward.shape[0]
        assert context.shape == torch.Size([1, 1, sample_data["context_dim"]])
        assert embedding.shape == torch.Size([1, 1, sample_data["embedding_dim"]])
        assert reward.shape == torch.Size([1, 1])

    assert i == sample_data["batch_size"]


def test_update_embeddings(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    new_embeddings = torch.randn(sample_data["batch_size"], sample_data["embedding_dim"])
    buffer.update_embeddings(new_embeddings)

    assert_close(buffer.embedded_actions, new_embeddings)


def test_state_dict(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    state = buffer.state_dict()

    assert isinstance(state, dict)
    assert torch.equal(state["contextualized_actions"], cast(torch.Tensor, buffer.contextualized_actions))
    assert torch.equal(state["embedded_actions"], buffer.embedded_actions)
    assert torch.equal(state["rewards"], buffer.rewards)
    assert state["max_size"] == buffer.max_size


def test_load_state_dict(buffer: TensorDataBuffer[torch.Tensor], sample_data: dict[str, Any]) -> None:
    buffer.add_batch(
        sample_data["contextualized_actions"],
        sample_data["embedded_actions"],
        sample_data["rewards"],
    )

    state = buffer.state_dict()

    new_buffer = TensorDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy(), max_size=None)
    new_buffer.load_state_dict(state)

    assert torch.equal(
        cast(torch.Tensor, new_buffer.contextualized_actions), cast(torch.Tensor, buffer.contextualized_actions)
    )
    assert torch.equal(new_buffer.embedded_actions, buffer.embedded_actions)
    assert torch.equal(new_buffer.rewards, buffer.rewards)
    assert new_buffer.max_size == buffer.max_size
