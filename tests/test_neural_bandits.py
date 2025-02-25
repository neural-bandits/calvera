from typing import List, Tuple, Type, Union

import lightning as pl
import pytest
import torch
import torch.nn as nn

from neural_bandits.bandits.neural_bandit import NeuralBandit
from neural_bandits.bandits.neural_ts_bandit import NeuralTSBandit
from neural_bandits.bandits.neural_ucb_bandit import NeuralUCBBandit
from neural_bandits.utils.data_storage import (
    AbstractBanditDataBuffer,
    AllDataBufferStrategy,
    InMemoryDataBuffer,
)


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


@pytest.fixture
def network_and_buffer() -> Tuple[int, nn.Module, AbstractBanditDataBuffer]:
    """
    Create a simple network and buffer for bandit testing
    """
    n_features: int = 4
    network: nn.Module = nn.Sequential(
        nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1)
    )
    buffer: AbstractBanditDataBuffer = InMemoryDataBuffer(
        buffer_strategy=AllDataBufferStrategy()
    )
    return n_features, network, buffer


@pytest.fixture
def small_context_reward_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
]:
    """
    Returns (chosen_contextualized_actions, rewards, dataset):
      chosen_contextualized_actions shape: (batch_size=2, n_chosen_arms=1, n_features=4)
      rewards shape: (2,1)
    """
    batch_size, n_chosen_arms, n_features = 2, 1, 4
    contextualized_actions = torch.randn(batch_size, n_chosen_arms, n_features)
    rewards = torch.randn(batch_size, n_chosen_arms)

    class RandomDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __init__(self, actions: torch.Tensor, rewards: torch.Tensor):
            self.actions = actions
            self.rewards = rewards

        def __len__(self) -> int:
            return len(self.actions)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.actions[idx], self.rewards[idx]

    dataset = RandomDataset(contextualized_actions, rewards)
    return contextualized_actions, rewards, dataset


# ------------------------------------------------------------------------------
# 1) Common Tests for NeuralBandit Base Class
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_forward_shape(
    network_and_buffer: Tuple[int, nn.Module, AbstractBanditDataBuffer],
    bandit_type: str,
) -> None:
    """
    Verify forward() returns a one-hot action (batch_size, n_arms) with correct shape
    for both UCB and TS bandits.
    """
    batch_size, n_arms = 2, 3
    n_features, network, buffer = network_and_buffer

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features, network=network, buffer=buffer
        )
    else:
        bandit = NeuralTSBandit(n_features=n_features, network=network, buffer=buffer)

    contextualized_actions: torch.Tensor = torch.randn(batch_size, n_arms, n_features)
    output, p = bandit.forward(contextualized_actions)

    assert output.shape == (
        batch_size,
        n_arms,
    ), f"Expected shape {(batch_size, n_arms)}, got {output.shape}"
    assert p.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {p.shape}"
    assert torch.all(0 <= p) and torch.all(p <= 1), "Probabilities should be in [0, 1]"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_training_step(
    network_and_buffer: Tuple[int, nn.Module, AbstractBanditDataBuffer],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    bandit_type: str,
) -> None:
    """
    Test that a training step runs without error and updates parameters correctly
    for both UCB and TS bandits.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features, network, buffer = network_and_buffer

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            train_interval=4,
            initial_train_steps=4,
            num_grad_steps=10,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            train_interval=4,
            initial_train_steps=4,
            num_grad_steps=10,
        )

    params_1 = {
        name: param.clone() for name, param in bandit.theta_t.named_parameters()
    }

    assert bandit.buffer.contextualized_actions.numel() == 0
    assert bandit.buffer.rewards.numel() == 0

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=False, num_workers=0
        ),
    )

    assert bandit.buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert bandit.buffer.rewards.shape[0] == rewards.shape[0]

    # Training should happen because we're within initial_train_steps (buffer size = 2 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(
            param, params_1[name]
        ), f"Parameter {name} was not updated"

    params_2 = {
        name: param.clone() for name, param in bandit.theta_t.named_parameters()
    }

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=False, num_workers=0
        ),
    )

    assert bandit.buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert bandit.buffer.rewards.shape[0] == 2 * rewards.shape[0]

    # Training should happen because we're within initial_train_steps (buffer size = 4 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(
            param, params_2[name]
        ), f"Parameter {name} was not updated"

    params_3 = {
        name: param.clone() for name, param in bandit.theta_t.named_parameters()
    }

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=False, num_workers=0
        ),
    )

    assert bandit.buffer.contextualized_actions.shape[0] == 3 * actions.shape[0]
    assert bandit.buffer.rewards.shape[0] == 3 * rewards.shape[0]

    # Training should NOT happen here because 6 % 4 != 0
    # Network parameters should remain unchanged
    for name, param in bandit.theta_t.named_parameters():
        assert torch.allclose(param, params_3[name])

    params_4 = {
        name: param.clone() for name, param in bandit.theta_t.named_parameters()
    }

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=False, num_workers=0
        ),
    )

    assert bandit.buffer.contextualized_actions.shape[0] == 4 * actions.shape[0]
    assert bandit.buffer.rewards.shape[0] == 4 * rewards.shape[0]

    # Training SHOULD happen here because 8 % 4 == 0
    # Network parameters should be updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(
            param, params_4[name]
        ), f"Parameter {name} was not updated"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_hparams_effect(
    network_and_buffer: Tuple[int, nn.Module, AbstractBanditDataBuffer],
    bandit_type: str,
) -> None:
    """
    Verify hyperparameters are saved and affect the bandit behavior
    for both UCB and TS bandits.
    """
    n_features, network, buffer = network_and_buffer

    lambda_: float = 0.1
    nu: float = 0.2
    learning_rate: float = 0.01
    train_batch_size: int = 25
    train_interval: int = 50
    initial_train_steps: int = 100

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            lambda_=lambda_,
            nu=nu,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            train_interval=train_interval,
            initial_train_steps=initial_train_steps,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            lambda_=lambda_,
            nu=nu,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            train_interval=train_interval,
            initial_train_steps=initial_train_steps,
        )

    assert bandit.hparams["n_features"] == n_features
    assert bandit.hparams["lambda_"] == lambda_
    assert bandit.hparams["nu"] == nu
    assert bandit.hparams["train_batch_size"] == train_batch_size
    assert bandit.hparams["learning_rate"] == learning_rate
    assert bandit.hparams["train_interval"] == train_interval
    assert bandit.hparams["initial_train_steps"] == initial_train_steps

    assert torch.all(bandit.Z_t == lambda_), "Z_t should be initialized with lambda_"

    optimizer = bandit.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == learning_rate
    assert optimizer.param_groups[0]["weight_decay"] == lambda_


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_parameter_validation(
    network_and_buffer: Tuple[int, nn.Module, AbstractBanditDataBuffer],
    bandit_type: str,
) -> None:
    """
    Test that the bandit properly validates parameter relationships.
    """
    n_features, network, buffer = network_and_buffer

    BanditClass: Type[Union[NeuralUCBBandit, NeuralTSBandit]]

    if bandit_type == "ucb":
        BanditClass = NeuralUCBBandit
    else:
        BanditClass = NeuralTSBandit

    # This should work fine
    BanditClass(
        n_features=n_features,
        network=network,
        buffer=buffer,
        train_batch_size=16,
        train_interval=32,
        initial_train_steps=48,
    )

    # Invalid: train_interval not divisible by batch_size
    with pytest.raises(
        AssertionError, match="train_interval must be divisible by train_batch_size"
    ):
        BanditClass(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=15,
            train_interval=40,
            initial_train_steps=45,
        )

    # Invalid: initial_train_steps not divisible by batch_size
    with pytest.raises(
        AssertionError,
        match="initial_train_steps must be divisible by train_batch_size",
    ):
        BanditClass(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=17,
            train_interval=34,
            initial_train_steps=50,
        )


# ------------------------------------------------------------------------------
# 2) Specific Tests for each Bandit Type
# ------------------------------------------------------------------------------
def test_ucb_score_method(
    network_and_buffer: Tuple[int, nn.Module, AbstractBanditDataBuffer],
) -> None:
    """
    Test that NeuralUCBBandit._score method correctly implements UCB scoring.
    """
    f_t_a: torch.Tensor = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
    )
    exploration_terms: torch.Tensor = torch.tensor(
        [[0.5, 0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1, 0.05]]
    )

    n_features, network, buffer = network_and_buffer
    bandit: NeuralUCBBandit = NeuralUCBBandit(
        n_features=n_features, network=network, buffer=buffer
    )

    scores: torch.Tensor = bandit._score(f_t_a, exploration_terms)

    expected_scores: torch.Tensor = f_t_a + exploration_terms
    assert torch.allclose(
        scores, expected_scores
    ), "UCB scoring should be f_t_a + exploration_terms"


def test_ts_score_method(
    network_and_buffer: Tuple[int, nn.Module, AbstractBanditDataBuffer],
) -> None:
    """
    Test that NeuralTSBandit._score method correctly implements TS scoring.
    """
    batch_size, n_arms = 3, 4
    f_t_a: torch.Tensor = torch.ones((batch_size, n_arms))  # means all set to 1.0
    exploration_terms: torch.Tensor = (
        torch.ones((batch_size, n_arms)) * 0.1
    )  # std devs all set to 0.1

    n_features, network, buffer = network_and_buffer
    bandit: NeuralTSBandit = NeuralTSBandit(
        n_features=n_features, network=network, buffer=buffer
    )

    scores: torch.Tensor = bandit._score(f_t_a, exploration_terms)

    # We can't test exact values due to randomness, but we can verify:
    # 1. Shape is correct
    assert (
        scores.shape == f_t_a.shape
    ), f"Expected shape {f_t_a.shape}, got {scores.shape}"

    # 2. Values differ from means (extremely unlikely to be exactly equal)
    assert not torch.allclose(
        scores, f_t_a
    ), "TS scores should differ from means due to sampling"

    # 3. Most values should be within 3 standard deviations (99.7% statistically)
    within_bounds: float = (
        ((scores - f_t_a).abs() <= 3 * exploration_terms).float().mean().item()
    )
    assert (
        within_bounds > 0.95
    ), f"Expected >95% of samples within 3σ, got {within_bounds * 100:.2f}%"


def test_neural_ucb_forward_deterministic() -> None:
    """
    Test that NeuralUCBBandit forward pass is deterministic with fixed parameters.
    """
    n_features: int = 2
    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    network[0].weight.data = torch.tensor([[1.0, 0.1]])

    buffer: AbstractBanditDataBuffer = InMemoryDataBuffer(
        buffer_strategy=AllDataBufferStrategy()
    )
    bandit: NeuralUCBBandit = NeuralUCBBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        lambda_=1.0,
        nu=0.1,
    )

    contextualized_actions: torch.Tensor = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    # Forward pass should be deterministic for UCB
    output1, _ = bandit(contextualized_actions)
    output2, _ = bandit(contextualized_actions)

    assert torch.allclose(output1, output2), "UCB forward pass should be deterministic"

    # With the given weights [1.0, 0.1], the first action should be chosen
    # First action: 1.0*1.0 + 0.1*0.0 = 1.0
    # Second action: 1.0*0.0 + 0.1*1.0 = 0.1
    assert torch.all(output1 == torch.tensor([[1, 0]])), "Should select first action"


def test_neural_ts_forward_stochastic() -> None:
    """
    Test that NeuralTSBandit forward pass is stochastic (might choose different actions).
    """
    n_runs: int = 50

    n_features: int = 2
    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    network[0].weight.data = torch.tensor([[1.0, 1.0]])

    buffer: AbstractBanditDataBuffer = InMemoryDataBuffer(
        buffer_strategy=AllDataBufferStrategy()
    )
    bandit: NeuralTSBandit = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        lambda_=1.0,
        nu=1.0,
    )

    # Two actions with equal expected rewards
    contextualized_actions: torch.Tensor = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    outputs: List[torch.Tensor] = []
    for _ in range(n_runs):
        output, _ = bandit(contextualized_actions)
        outputs.append(output)

    outputs_tensor: torch.Tensor = torch.cat(outputs, dim=0)  # shape (n_runs, 2)

    # Count how many times each arm was chosen
    arm0_count = (outputs_tensor[:, 0] == 1).sum().item()
    arm1_count = (outputs_tensor[:, 1] == 1).sum().item()

    # With enough runs, both arms should be chosen at least once
    assert arm0_count > 0, "Arm 0 was never chosen in TS"
    assert arm1_count > 0, "Arm 1 was never chosen in TS"

    # For Thompson sampling with equal rewards, the distribution should be roughly balanced
    assert (
        0.2 <= arm0_count / n_runs <= 0.8
    ), f"Expected balanced choices, got {arm0_count}/{n_runs} for arm 0"
