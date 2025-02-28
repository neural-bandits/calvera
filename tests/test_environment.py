from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from neural_bandits.benchmark.environment import BanditBenchmarkEnvironment


@pytest.fixture
def sample_data() -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Returns a tuple (dataloader, expected_contexts, expected_rewards).
    We'll produce a small dataset with shape (batch_size=2, m=3, context_dim=4).
    """
    # contexts shape: (2, 3, 4)
    contexts = torch.randn(2, 3, 4)
    # rewards shape: (2, 3)
    rewards = torch.tensor([[0.4, 0.2, 0.9], [0.1, 0.8, 0.7]])

    dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(contexts, rewards))
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader, contexts, rewards


def test_environment_iterator_length(
    sample_data: tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor],
) -> None:
    dataloader, contexts, rewards = sample_data
    env = BanditBenchmarkEnvironment(dataloader)
    # We expect to retrieve exactly 2 iterations from the environment
    assert len(list(env)) == 2


def test_environment_iteration(
    sample_data: tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor],
) -> None:
    dataloader, contexts, rewards = sample_data
    env = BanditBenchmarkEnvironment(dataloader)

    # We expect to retrieve exactly 2 iterations from the environment:
    # The environment returns only contextualized_actions on each iteration
    # We check that we get shape (1, 3, 4) each time from batch_size=1
    previous_all_rewards = None
    for batch_contexts in env:
        assert batch_contexts.shape == (1, 3, 4)
        # environment stored them internally
        assert env._last_contextualized_actions is not None and torch.allclose(
            env._last_contextualized_actions, batch_contexts
        )
        assert (previous_all_rewards is None and env._last_all_rewards is not None) or (
            not torch.allclose(env._last_all_rewards, previous_all_rewards)  # type: ignore
        )
        previous_all_rewards = env._last_all_rewards


def test_get_feedback(
    sample_data: tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor],
) -> None:
    dataloader, contexts, rewards = sample_data
    env = BanditBenchmarkEnvironment(dataloader)

    # Step through the environment
    batch_contexts = next(iter(env))
    assert batch_contexts.shape == (1, 3, 4)

    # Choose actions (one-hot) => shape (1, 3)
    chosen_actions = torch.tensor([[1, 0, 1]], dtype=torch.float32)

    # get_feedback returns chosen actions & realized rewards
    chosen_contexts, chosen_rewards = env.get_feedback(chosen_actions)
    # chosen_contexts: shape (1, 2, 4) => within each row: #actions=2
    # chosen_rewards: shape (1, 2, )
    assert chosen_contexts.shape == (1, 2, 4)
    assert chosen_rewards.shape == (
        1,
        2,
    )
    # The actual values: we picked indices 0 and 2 from row 0
    # so we can compare them
    expected_contexts = batch_contexts[:, [0, 2], :]
    expected_rewards = rewards[0, [0, 2]].unsqueeze(0)
    assert torch.allclose(chosen_contexts, expected_contexts)
    assert torch.allclose(chosen_rewards, expected_rewards)


def test_compute_regret(
    sample_data: tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor],
) -> None:
    dataloader, contexts, rewards = sample_data
    env = BanditBenchmarkEnvironment(dataloader)

    # We'll fetch the first batch
    _ = next(iter(env))
    chosen_actions = torch.tensor([[1, 0, 1]], dtype=torch.float32)

    # compute_regret performs a top-k selection. In this batch, we chose 2 actions.
    # For row 0 => rewards: [0.4, 0.2, 0.9]. The best 2 are 0.9 + 0.4 = 1.3.
    # chosen_reward = 0.4 + 0.9 = 1.3 => regret = 1.3 - 1.3 = 0
    regret = env.compute_regret(chosen_actions)
    assert regret.shape == (1,)
    assert torch.allclose(regret, torch.tensor([0.0]))

    # Next batch
    _ = next(env)
    # row 1 => rewards: [0.1, 0.8, 0.7]. The best 2 are 0.8 + 0.7 = 1.5
    # If we choose only 1 action => let's choose index 1 => reward = 0.8
    chosen_actions = torch.tensor([[0, 1, 0]], dtype=torch.float32)
    # chosen_reward = 0.8
    # best_reward_for_1_action = the top 1 => 0.8
    # so regret = 0.8 - 0.8 = 0
    regret = env.compute_regret(chosen_actions)
    assert regret.shape == (1,)
    assert torch.allclose(regret, torch.tensor([0.0]))
