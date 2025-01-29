from typing import Tuple

import pytest
import torch

from neural_bandits.algorithms.linear_bandits import (
    LinearBandit,
    LinearTSBandit,
    LinearUCBBandit,
)
from neural_bandits.modules.linear_bandit_module import LinearBanditModule


@pytest.fixture
def setup_ucb_bandit() -> Tuple[LinearBandit, LinearBanditModule]:
    """
    Setup LinearUCBBandit with n_features=5.
    """
    n_features = 5
    module = LinearBanditModule(
        linear_bandit_type=LinearUCBBandit, n_features=n_features
    )
    return module.bandit, module


@pytest.fixture
def setup_ts_bandit() -> Tuple[LinearBandit, LinearBanditModule]:
    """
    Setup LinearTSBandit with n_features=5.
    """
    n_features = 5
    module = LinearBanditModule(
        linear_bandit_type=LinearTSBandit, n_features=n_features
    )
    return module.bandit, module


@pytest.fixture
def setup_simple_bandit() -> Tuple[LinearBandit, LinearBanditModule]:
    """
    Setup LinearUCBBandit with n_features=1.
    """
    n_features = 1
    module = LinearBanditModule(
        linear_bandit_type=LinearUCBBandit, n_features=n_features
    )
    return module.bandit, module


@pytest.mark.parametrize("fixture_name", ["setup_ucb_bandit", "setup_ts_bandit"])
def test_update_head_updates_parameters_parameterized(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Test if parameters are updated after training step.
    """
    bandit, module = request.getfixturevalue(fixture_name)
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
    module.update_head(chosen_actions, realized_rewards)

    # Check that parameters have been updated
    assert not torch.equal(
        bandit.precision_matrix, initial_precision_matrix
    ), "Precision matrix should be updated"
    assert not torch.equal(bandit.b, initial_b), "b should be updated"
    assert not torch.equal(bandit.theta, initial_theta), "theta should be updated"


def test_update_head_correct(
    setup_simple_bandit: Tuple[LinearBandit, LinearBanditModule]
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

    bandit, module = setup_simple_bandit

    chosen_actions = torch.tensor([[2.0]])  # shape (1,1)
    realized_rewards = torch.tensor([1.0])  # shape (1,)

    module.update_head(chosen_actions, realized_rewards)

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


@pytest.mark.parametrize("fixture_name", ["setup_ucb_bandit", "setup_ts_bandit"])
def test_update_head_shapes_parameterized(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Test if parameters have correct shapes after update.
    """
    bandit, module = request.getfixturevalue(fixture_name)
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data
    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Perform training step
    module.update_head(chosen_actions, realized_rewards)

    # Check shapes of updated parameters
    assert bandit.precision_matrix.shape == (
        n_features,
        n_features,
    ), "Precision matrix should have shape (n_features, n_features)"
    assert bandit.b.shape == (n_features,), "b should have shape (n_features,)"
    assert bandit.theta.shape == (n_features,), "theta should have shape (n_features,)"


@pytest.mark.parametrize("fixture_name", ["setup_ucb_bandit", "setup_ts_bandit"])
def test_update_head_invalid_shapes_parameterized(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Test if assertion errors are raised for invalid input shapes.
    """
    bandit, module = request.getfixturevalue(fixture_name)
    batch_size = 10
    n_features = bandit.n_features

    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Create invalid dummy data
    chosen_actions_invalid = torch.randn(batch_size, n_features + 1)
    realized_rewards_invalid = torch.randn(batch_size + 1)

    # Check for assertion errors
    with pytest.raises(AssertionError):
        module.update_head(chosen_actions_invalid, realized_rewards)

    with pytest.raises(AssertionError):
        module.update_head(chosen_actions, realized_rewards_invalid)


def test_update_head_zero_denominator(
    setup_simple_bandit: Tuple[LinearBandit, LinearBanditModule]
) -> None:
    """
    Test if assertion error is raised when denominator is zero.
    """
    bandit, module = setup_simple_bandit
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data with NaN
    chosen_actions = torch.zeros(batch_size, n_features)
    realized_rewards = torch.zeros(batch_size)
    chosen_actions[0, 0] = torch.nan

    with pytest.raises(AssertionError):
        module.update_head(chosen_actions, realized_rewards)

    # Create dummy data that will cause zero denominator
    chosen_actions = torch.tensor([[2.0]])
    realized_rewards = torch.zeros(1)

    bandit.precision_matrix = torch.tensor([[-0.25]])  # shape (1,1)

    with pytest.raises(AssertionError):
        module.update_head(chosen_actions, realized_rewards)
