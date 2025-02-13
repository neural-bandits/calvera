from typing import TypeVar

import pytest
import pytorch_lightning as pl
import torch

from neural_bandits.bandits.linear_bandit import LinearBandit
from neural_bandits.bandits.linear_ts_bandit import LinearTSBandit
from neural_bandits.bandits.linear_ucb_bandit import LinearUCBBandit

BanditClassType = TypeVar("BanditClassType", bound="LinearBandit")


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


@pytest.fixture
def lin_ucb_bandit() -> LinearUCBBandit:
    """
    Setup LinearUCBBandit with n_features=5.
    """
    n_features = 3
    module = LinearUCBBandit(n_features=n_features)
    return module


@pytest.fixture
def lin_ts_bandit() -> LinearTSBandit:
    """
    Setup LinearTSBandit with n_features=5.
    """
    n_features = 3
    module = LinearTSBandit(n_features=n_features)
    return module


@pytest.fixture
def simple_ucb_bandit() -> LinearUCBBandit:
    """
    Setup LinearUCBBandit with n_features=1.
    """
    n_features = 1
    module = LinearUCBBandit(n_features=n_features)
    return module


# TODO: Also use the fixtures from above here?
@pytest.mark.parametrize("BanditClass", [LinearTSBandit, LinearUCBBandit])
def test_linear_bandits_forward_shapes(BanditClass: BanditClassType) -> None:
    """Check if forward method returns correct shape and handles valid input."""
    n_features = 5
    batch_size = 4
    n_arms = 3

    bandit: LinearBandit = BanditClass(n_features=n_features)

    # Contextualized actions have shape (batch_size, n_arms, n_features)
    # E.g., (4, 3, 5)
    contextualized_actions = torch.randn(batch_size, n_arms, n_features)
    output, p = bandit.forward(contextualized_actions)

    # Forward in both LinearTSBandit and LinearUCBBandit returns a one-hot
    # encoding of shape (batch_size, n_arms).
    assert output.shape == (batch_size, n_arms), (
        f"Expected one-hot shape (batch_size={batch_size}, n_arms={n_arms}), "
        f"got {output.shape}"
    )

    assert p.shape == (batch_size,), (
        f"Expected probability shape (batch_size={batch_size}), " f"got {p.shape}"
    )


@pytest.mark.parametrize("BanditClass", [LinearTSBandit, LinearUCBBandit])
def test_linear_bandits_forward_shape_errors(BanditClass: BanditClassType) -> None:
    """Check if forward method raises assertion error for invalid input shape."""
    n_features = 5
    bandit: LinearBandit = BanditClass(n_features=n_features)

    # Invalid shape: contextualized_actions should have shape (batch_size, n_arms, n_features).
    # Here we intentionally use (batch_size, n_arms, n_features + 1).
    invalid_actions = torch.randn(2, 3, n_features + 1)

    with pytest.raises(AssertionError, match="must have shape"):
        bandit.forward(invalid_actions)


@pytest.mark.parametrize("BanditClass", [LinearTSBandit, LinearUCBBandit])
def test_linear_bandit_defaults(BanditClass: BanditClassType) -> None:
    """
    Test default initialization of base LinearBandit.
    Ensures shapes of precision_matrix, b, theta are correct.
    """
    n_features = 5
    bandit: LinearBandit = BanditClass(n_features=n_features)

    assert bandit.precision_matrix.shape == (
        n_features,
        n_features,
    ), f"precision_matrix should be (n_features, n_features), got {bandit.precision_matrix.shape}"

    assert torch.allclose(
        bandit.precision_matrix, torch.eye(n_features)
    ), "Default precision_matrix should be identity."

    assert bandit.b.shape == (
        n_features,
    ), f"b should be (n_features,), got {bandit.b.shape}"

    assert bandit.theta.shape == (
        n_features,
    ), f"theta should be (n_features,), got {bandit.theta.shape}"


def test_linear_ucb_correct_mean() -> None:
    """
    Test if LinearUCBBandit returns correct values for theta = (0, 0, 1).
    """
    n_features = 3
    bandit = LinearUCBBandit(n_features=n_features)

    # Manually adjust the bandits parameters
    bandit.theta = torch.tensor([0.0, 0.0, 1.0])

    # pass one batch of 3 arms
    contextualized_actions = torch.tensor(
        [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]]
    )
    output, p = bandit.forward(contextualized_actions)

    assert torch.allclose(
        output,
        torch.tensor([[0, 0, 1]]),
    ), "Expected one-hot encoding of the arm with highest UCB."

    assert torch.allclose(
        p,
        torch.ones((1)),
    ), "Expected probability of 1 for all arms."


def test_linear_ucb_correct_variance() -> None:
    """
    Test if LinearUCBBandit returns correct values for same UCB means but different variances.
    """
    n_features = 3
    bandit = LinearUCBBandit(n_features=n_features)

    # Manually adjust the bandits parameters
    bandit.theta = torch.ones(n_features)
    bandit.precision_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    bandit.b = torch.zeros(n_features)

    # pass vectors where without alpha all arms are equally good
    contextualized_actions = torch.tensor(
        [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]]]
    )
    output, p = bandit.forward(contextualized_actions)

    # test that now instead the arm with higher variance is selected
    assert torch.allclose(
        output,
        torch.tensor([[0, 1, 0]]),
    ), "Expected one-hot encoding of the arm with highest variance."

    assert torch.allclose(
        p,
        torch.ones((1)),
    ), "Expected probability of 1 for all arms."


def test_linear_ucb_alpha() -> None:
    """
    Test alpha parameter in LinearUCBBandit to confirm it's settable and used.
    """
    n_features = 3
    alpha = 25.0  # extreme alpha for testing
    bandit = LinearUCBBandit(n_features=n_features, alpha=alpha)

    assert (
        bandit.hparams["alpha"] == alpha
    ), "Bandit's alpha should match the passed value."

    # Manually adjust the bandits parameters
    bandit.theta = torch.ones(n_features)
    bandit.precision_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    bandit.b = torch.zeros(n_features)

    # pass vectors where without alpha all arms are equally good
    contextualized_actions = torch.tensor(
        [[[1.0, 0.5, 0.0], [0.0, 0.9, 0.5], [0.5, 0.0, 1.0]]]
    )
    output, p = bandit.forward(contextualized_actions)

    # test that now instead the arm with higher variance is selected
    assert torch.allclose(
        output,
        torch.tensor([[0, 1, 0]]),
    ), "Expected one-hot encoding of the arm with highest variance due to extreme alpha value."

    assert torch.allclose(
        p,
        torch.ones((1)),
    ), "Expected probability of 1 for all arms."


def test_linear_ts_correct() -> None:
    """
    Test if LinearTSBandit returns correct values for theta = (0, 0, 1).
    """
    n_features = 3
    bandit = LinearTSBandit(n_features=n_features)

    # Manually adjust the bandits parameters
    bandit.theta = torch.tensor([0.0, 0.0, 1.0])

    # pass one batch of 3 arms
    contextualized_actions = torch.tensor(
        [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]]
    )
    output, _ = bandit.forward(contextualized_actions)

    assert torch.allclose(
        output,
        torch.tensor([[0, 0, 1]]),
    ), "Expected one-hot encoding of the arm with highest expected reward."

    # TODO: Test correct computation of probabilities


@pytest.mark.parametrize("BanditClass", [LinearUCBBandit, LinearTSBandit])
def test_update_updates_parameters_parameterized(BanditClass: BanditClassType) -> None:
    """
    Test if parameters are updated after training step.
    """
    bandit: LinearBandit = BanditClass(n_features=3)
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data
    chosen_actions = torch.randn(batch_size, 1, n_features)
    realized_rewards = torch.randn(batch_size, 1)

    # Save initial state
    initial_precision_matrix = bandit.precision_matrix.clone()
    initial_b = bandit.b.clone()
    initial_theta = bandit.theta.clone()

    # Perform training step
    bandit.update(chosen_actions, realized_rewards)

    # Check that parameters have been updated
    assert not torch.equal(
        bandit.precision_matrix, initial_precision_matrix
    ), "Precision matrix should be updated"
    assert not torch.equal(bandit.b, initial_b), "b should be updated"
    assert not torch.equal(bandit.theta, initial_theta), "theta should be updated"


def test_update_correct() -> None:
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

    bandit = LinearUCBBandit(n_features=1)

    chosen_contextualized_actions = torch.tensor([[[2.0]]])  # shape (1,1)
    realized_rewards = torch.tensor([[1.0]])  # shape (1,1)

    bandit.update(chosen_contextualized_actions, realized_rewards)

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


@pytest.mark.parametrize("BanditClass", [LinearUCBBandit, LinearTSBandit])
def test_update_shapes_parameterized(BanditClass: BanditClassType) -> None:
    """
    Test if parameters have correct shapes after update.
    """
    bandit: LinearBandit = BanditClass(n_features=3)
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data
    chosen_actions = torch.randn(batch_size, 1, n_features)
    realized_rewards = torch.randn(batch_size, 1)

    # Perform training step
    bandit.update(chosen_actions, realized_rewards)

    # Check shapes of updated parameters
    assert bandit.precision_matrix.shape == (
        n_features,
        n_features,
    ), "Precision matrix should have shape (n_features, n_features)"
    assert bandit.b.shape == (n_features,), "b should have shape (n_features,)"
    assert bandit.theta.shape == (n_features,), "theta should have shape (n_features,)"


@pytest.mark.parametrize("BanditClass", [LinearUCBBandit, LinearTSBandit])
def test_update_invalid_shapes_parameterized(BanditClass: BanditClassType) -> None:
    """
    Test if assertion errors are raised for invalid input shapes.
    """
    bandit: LinearBandit = BanditClass(n_features=3)
    batch_size = 10
    n_features = bandit.n_features

    chosen_actions = torch.randn(batch_size, n_features)
    realized_rewards = torch.randn(batch_size)

    # Create invalid dummy data
    chosen_actions_invalid = torch.randn(batch_size, n_features + 1)
    realized_rewards_invalid = torch.randn(batch_size + 1)

    # Check for assertion errors
    with pytest.raises(AssertionError):
        bandit.update(chosen_actions_invalid, realized_rewards)

    with pytest.raises(AssertionError):
        bandit.update(chosen_actions, realized_rewards_invalid)


def test_update_zero_denominator(
    simple_ucb_bandit: LinearUCBBandit,
) -> None:
    """
    Test if assertion error is raised when denominator is zero.
    """
    bandit = simple_ucb_bandit
    batch_size = 10
    n_features = bandit.n_features

    # Create dummy data with NaN
    chosen_actions = torch.zeros(batch_size, n_features)
    realized_rewards = torch.zeros(batch_size)
    chosen_actions[0, 0] = torch.nan

    with pytest.raises(AssertionError):
        bandit.update(chosen_actions, realized_rewards)

    # Create dummy data that will cause zero denominator
    chosen_actions = torch.tensor([[2.0]])
    realized_rewards = torch.zeros(1)

    bandit.precision_matrix = torch.tensor([[-0.25]])  # shape (1,1)

    with pytest.raises(AssertionError):
        bandit.update(chosen_actions, realized_rewards)
