import pytest
from typing import Tuple
import torch
from neural_bandits.algorithms.linear_bandits import (
    LinearBandit,
    LinearTSBandit,
    LinearUCBBandit,
)
from neural_bandits.trainers.linear_bandit_trainer import LinearBanditTrainer


@pytest.fixture
def setup_bandit() -> Tuple[LinearBandit, LinearBanditTrainer]:
    n_features = 5
    bandit = LinearUCBBandit(n_features=n_features)
    trainer = LinearBanditTrainer(bandit=bandit)
    return bandit, trainer


@pytest.fixture
def setup_ts_bandit() -> Tuple[LinearBandit, LinearBanditTrainer]:
    n_features = 5
    bandit = LinearTSBandit(n_features=n_features)
    trainer = LinearBanditTrainer(bandit=bandit)
    return bandit, trainer


@pytest.fixture
def setup_simple_bandit() -> Tuple[LinearBandit, LinearBanditTrainer]:
    n_features = 1
    bandit = LinearUCBBandit(n_features=n_features)
    trainer = LinearBanditTrainer(bandit=bandit)
    return bandit, trainer


def test_training_step_updates_parameters(
    setup_bandit: Tuple[LinearBandit, LinearBanditTrainer]
) -> None:
    bandit, trainer = setup_bandit
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data
    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Save initial state
    initial_precision_matrix = bandit.precision_matrix.clone()
    initial_b = bandit.b.clone()
    initial_theta = bandit.theta.clone()

    # Perform training step
    trainer.training_step(chosen_actions, realized_rewards)

    # Check that parameters have been updated
    assert not torch.equal(
        bandit.precision_matrix, initial_precision_matrix
    ), "Precision matrix should be updated"
    assert not torch.equal(bandit.b, initial_b), "b should be updated"
    assert not torch.equal(bandit.theta, initial_theta), "theta should be updated"


def test_training_step_updates_parameters_ts(
    setup_ts_bandit: Tuple[LinearBandit, LinearBanditTrainer]
) -> None:
    bandit, trainer = setup_ts_bandit
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data
    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Save initial state
    initial_precision_matrix = bandit.precision_matrix.clone()
    initial_b = bandit.b.clone()
    initial_theta = bandit.theta.clone()

    # Perform training step
    trainer.training_step(chosen_actions, realized_rewards)

    # Check that parameters have been updated
    assert not torch.equal(
        bandit.precision_matrix, initial_precision_matrix
    ), "Precision matrix should be updated"
    assert not torch.equal(bandit.b, initial_b), "b should be updated"
    assert not torch.equal(bandit.theta, initial_theta), "theta should be updated"


def test_training_step_correct(
    setup_simple_bandit: Tuple[LinearBandit, LinearBanditTrainer]
) -> None:
    """
    In this simple scenario:
      - n_features = 1
      - precision_matrix = [[1.0]]
      - b = [0.0]
      - chosen_actions = [[2.0]]
      - realized_rewards = [1.0]

    The manual Sherman-Morrison update for M should yield:
      M_new = [[0.2]]
      b_new = [2.0]
      theta_new = [0.4]
    """

    bandit, trainer = setup_simple_bandit

    chosen_actions = torch.tensor([[2.0]])  # shape (1,1)
    realized_rewards = torch.tensor([1.0])  # shape (1,)

    trainer.training_step(chosen_actions, realized_rewards)

    expected_M = torch.tensor([[0.2]])
    expected_b = torch.tensor([2.0])
    expected_theta = torch.tensor([0.4])

    assert torch.allclose(
        bandit.precision_matrix, expected_M, atol=1e-6
    ), f"Expected M={expected_M}, got {bandit.precision_matrix}"
    assert torch.allclose(
        bandit.b, expected_b, atol=1e-6
    ), f"Expected b={expected_b}, got {bandit.b}"
    assert torch.allclose(
        bandit.theta, expected_theta, atol=1e-6
    ), f"Expected theta={expected_theta}, got {bandit.theta}"


def test_training_step_shapes(
    setup_bandit: Tuple[LinearBandit, LinearBanditTrainer]
) -> None:
    bandit, trainer = setup_bandit
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data
    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Perform training step
    trainer.training_step(chosen_actions, realized_rewards)

    # Check shapes of updated parameters
    assert bandit.precision_matrix.shape == (
        n_features,
        n_features,
    ), "Precision matrix should have shape (n_features, n_features)"
    assert bandit.b.shape == (n_features,), "b should have shape (n_features,)"
    assert bandit.theta.shape == (n_features,), "theta should have shape (n_features,)"


def test_training_step_invalid_shapes(
    setup_bandit: Tuple[LinearBandit, LinearBanditTrainer]
) -> None:
    bandit, trainer = setup_bandit
    batch_size = 10
    n_features = bandit.n_features

    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Create invalid dummy data
    chosen_actions_invalid = torch.randn(batch_size, n_features + 1)
    realized_rewards_invalid = torch.randn(batch_size + 1)

    # Check for assertion errors
    with pytest.raises(AssertionError):
        trainer.training_step(chosen_actions_invalid, realized_rewards)

    with pytest.raises(AssertionError):
        trainer.training_step(chosen_actions, realized_rewards_invalid)


def test_training_step_zero_denominator(
    setup_simple_bandit: Tuple[LinearBandit, LinearBanditTrainer]
) -> None:
    bandit, trainer = setup_bandit
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data that will cause a nan denominator
    chosen_actions = torch.zeros(batch_size, n_features)
    realized_rewards = torch.zeros(batch_size)

    chosen_actions[0, 0] = torch.nan

    # Check for assertion error due to nan denominator
    with pytest.raises(AssertionError):
        trainer.training_step(chosen_actions, realized_rewards)

    # Create dummy data that will cause a zero denominator
    chosen_actions = torch.tensor([[2.0]])
    realized_rewards = torch.zeros(1)

    # overwrite negative definite precision matrix for test (its invalid but we want to test the assertion)
    bandit.precision_matrix = torch.tensor([[-0.25]])  # shape (1,1)

    # Check for assertion error due to zero denominator
    with pytest.raises(AssertionError):
        trainer.training_step(chosen_actions, realized_rewards)
