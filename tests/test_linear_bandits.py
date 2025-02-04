from typing import TypeVar

import pytest
import torch

from neural_bandits.algorithms.linear_bandits import (
    LinearBandit,
    LinearTSBandit,
    LinearUCBBandit,
)

BanditClassType = TypeVar("BanditClassType", bound="LinearBandit")


@pytest.fixture(autouse=True)
def setup_seed() -> None:
    torch.manual_seed(42)


@pytest.mark.parametrize("BanditClass", [LinearTSBandit, LinearUCBBandit])
def test_linear_bandits_forward_shapes(BanditClass: BanditClassType) -> None:
    """Check if forward method returns correct shape and handles valid input."""
    n_features = 5
    batch_size = 4
    n_arms = 3

    bandit = BanditClass(n_features=n_features)

    # Contextualized actions have shape (batch_size, n_arms, n_features)
    # E.g., (4, 3, 5)
    contextualized_actions = torch.randn(batch_size, n_arms, n_features)
    output = bandit.forward(contextualized_actions)

    # Forward in both LinearTSBandit and LinearUCBBandit returns a one-hot
    # encoding of shape (batch_size, n_arms).
    assert output.shape == (batch_size, n_arms), (
        f"Expected one-hot shape (batch_size={batch_size}, n_arms={n_arms}), "
        f"got {output.shape}"
    )


@pytest.mark.parametrize("BanditClass", [LinearTSBandit, LinearUCBBandit])
def test_linear_bandits_forward_shape_errors(BanditClass: BanditClassType) -> None:
    """Check if forward method raises assertion error for invalid input shape."""
    n_features = 5
    bandit = BanditClass(n_features=n_features)

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
    bandit = BanditClass(n_features=n_features)

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
    output = bandit.forward(contextualized_actions)

    assert torch.allclose(
        output,
        torch.tensor([[0, 0, 1]]),
    ), "Expected one-hot encoding of the arm with highest UCB."


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
    output = bandit.forward(contextualized_actions)

    # test that now instead the arm with higher variance is selected
    assert torch.allclose(
        output,
        torch.tensor([[0, 1, 0]]),
    ), "Expected one-hot encoding of the arm with highest variance."


def test_linear_ucb_alpha() -> None:
    """
    Test alpha parameter in LinearUCBBandit to confirm it's settable and used.
    """
    n_features = 3
    alpha = 25.0  # extreme alpha for testing
    bandit = LinearUCBBandit(n_features=n_features, alpha=alpha)

    assert bandit.alpha == alpha, "Bandit's alpha should match the passed value."

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
    output = bandit.forward(contextualized_actions)

    # test that now instead the arm with higher variance is selected
    assert torch.allclose(
        output,
        torch.tensor([[0, 1, 0]]),
    ), "Expected one-hot encoding of the arm with highest variance due to extreme alpha value."


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
    output = bandit.forward(contextualized_actions)

    assert torch.allclose(
        output,
        torch.tensor([[0, 0, 1]]),
    ), "Expected one-hot encoding of the arm with highest expected reward."
