import lightning as pl
import pytest
import torch
import torch.nn as nn

from neural_bandits.bandits.neural_ucb_bandit import NeuralUCBBandit
from neural_bandits.utils.data_storage import AllDataBufferStrategy, InMemoryDataBuffer


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


# ------------------------------------------------------------------------------
# 1) Tests for NeuralUCBBandit Forward Pass
# ------------------------------------------------------------------------------
def test_neural_ucb_bandit_forward_shape() -> None:
    """
    Verify forward() returns a one-hot action (batch_size, n_arms) with correct shape.
    """
    batch_size, n_arms, n_features = 2, 3, 4

    network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralUCBBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
    )

    contextualized_actions = torch.randn(batch_size, n_arms, n_features)
    output, p = bandit.forward(contextualized_actions)

    assert output.shape == (
        batch_size,
        n_arms,
    ), f"Expected shape {(batch_size, n_arms)}, got {output.shape}"
    assert p.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {p.shape}"
    assert torch.all(0 <= p) and torch.all(p <= 1), "Probabilities should be in [0, 1]"


def test_neural_ucb_bandit_forward_small_sample() -> None:
    """
    Test forward with a small sample data we can reason about.
    """
    n_features = 2

    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    network[0].weight.data = torch.tensor([[1.0, 0.1]])

    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())
    bandit = NeuralUCBBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        lambda_=1.0,
        nu=0.1,
    )

    contextualized_actions = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    output, p = bandit(contextualized_actions)

    # The bandit should select the first action due to higher reward estimate
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    assert torch.all(output == torch.tensor([[1, 0]])), "Should select first action"
    assert torch.sum(output).item() == 1, "Should be one-hot vector"

    assert p.shape == (1,), f"Expected shape (1,), got {p.shape}"
    assert 0 <= p.item() <= 1, "Probability should be in [0, 1]"


# ------------------------------------------------------------------------------
# 2) Tests for updating the NeuralUCB bandit
# ------------------------------------------------------------------------------
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
    batch_size, n_chosen_arms, n_features = 3, 1, 4
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


def test_neural_ucb_bandit_training_step(
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
) -> None:
    """
    Test that a training step runs without error and updates parameters correctly.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features = actions.shape[2]

    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralUCBBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        batch_size=3,
        train_interval=8,
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

    # Training should happen because we're within initial_train_steps (buffer size = 3 <= 4)
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

    # Training should NOT happen here because we've exceeded initial_train_steps (buffer size = 6)
    # but haven't accumulated enough new samples (only 2 samples beyond initial_train_steps)
    # Network parameters should remain unchanged
    for name, param in bandit.theta_t.named_parameters():
        assert torch.allclose(param, params_2[name])

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

    # Training should NOT happen here because we've only accumulated 5 samples
    # after the initial_train_steps (buffer size = 9), which is less than train_interval (8)
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

    # Training SHOULD happen here because we've accumulated 8 samples
    # after the initial_train_steps (buffer size = 12), which equals train_interval (8)
    # Network parameters should be updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(
            param, params_4[name]
        ), f"Parameter {name} was not updated"


def test_neural_ucb_bandit_hparams_effect() -> None:
    """
    Verify hyperparameters are saved and affect the bandit behavior.
    """
    n_features = 4
    network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    lambda_ = 0.1
    nu = 0.2
    learning_rate = 0.01
    train_interval = 50
    initial_train_steps = 100

    bandit = NeuralUCBBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        lambda_=lambda_,
        nu=nu,
        learning_rate=learning_rate,
        train_interval=train_interval,
        initial_train_steps=initial_train_steps,
    )

    assert bandit.hparams["n_features"] == n_features
    assert bandit.hparams["lambda_"] == lambda_
    assert bandit.hparams["nu"] == nu
    assert bandit.hparams["learning_rate"] == learning_rate
    assert bandit.hparams["train_interval"] == train_interval
    assert bandit.hparams["initial_train_steps"] == initial_train_steps

    assert torch.all(bandit.Z_t == lambda_), "Z_t should be initialized with lambda_"

    optimizer = bandit.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == learning_rate
    assert optimizer.param_groups[0]["weight_decay"] == lambda_
