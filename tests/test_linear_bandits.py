from typing import Any, TypeVar

import pytest
import pytorch_lightning as pl
import torch

from calvera.bandits.linear_bandit import LinearBandit
from calvera.bandits.linear_ts_bandit import (
    DiagonalPrecApproxLinearTSBandit,
    LinearTSBandit,
)
from calvera.bandits.linear_ucb_bandit import (
    DiagonalPrecApproxLinearUCBBandit,
    LinearUCBBandit,
)
from calvera.utils.selectors import ArgMaxSelector, EpsilonGreedySelector

BanditClassType = TypeVar("BanditClassType", bound="LinearBandit[torch.Tensor]")


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


@pytest.fixture
def lin_ucb_bandit() -> LinearUCBBandit:
    """Setup LinearUCBBandit with n_features=5."""
    n_features = 3
    module = LinearUCBBandit(n_features=n_features)
    return module


@pytest.fixture
def lin_ts_bandit() -> LinearTSBandit[torch.Tensor]:
    """
    Setup LinearTSBandit with n_features=3.
    """
    n_features = 3
    module = LinearTSBandit[torch.Tensor](n_features=n_features)
    return module


@pytest.fixture
def approx_lin_ucb_bandit() -> DiagonalPrecApproxLinearUCBBandit:
    """
    Setup DiagonalPrecApproxLinearUCBBandit with n_features=3.
    """
    n_features = 3
    module = DiagonalPrecApproxLinearUCBBandit(n_features=n_features)
    return module


@pytest.fixture
def approx_lin_ts_bandit() -> DiagonalPrecApproxLinearTSBandit:
    """
    Setup DiagonalPrecApproxLinearTSBandit with n_features=3.
    """
    n_features = 3
    module = DiagonalPrecApproxLinearTSBandit(n_features=n_features)
    return module


@pytest.fixture
def simple_ucb_bandit() -> LinearUCBBandit:
    """Setup LinearUCBBandit with n_features=1."""
    n_features = 1
    module = LinearUCBBandit(n_features=n_features, lazy_uncertainty_update=True)
    return module


LinearBanditTypes = [
    LinearTSBandit,
    LinearUCBBandit,
    DiagonalPrecApproxLinearUCBBandit,
    DiagonalPrecApproxLinearTSBandit,
]


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_linear_bandits_forward_shapes(BanditClass: BanditClassType) -> None:
    """Check if forward method returns correct shape and handles valid input."""
    n_features = 5
    batch_size = 4
    n_arms = 3

    bandit: LinearBandit[torch.Tensor] = BanditClass(n_features=n_features)

    # Contextualized actions have shape (batch_size, n_arms, n_features)
    # E.g., (4, 3, 5)
    contextualized_actions = torch.randn(batch_size, n_arms, n_features)
    output, p = bandit.forward(contextualized_actions)

    # Forward in both LinearTSBandit and LinearUCBBandit returns a one-hot
    # encoding of shape (batch_size, n_arms).
    assert output.shape == (batch_size, n_arms), (
        f"Expected one-hot shape (batch_size={batch_size}, n_arms={n_arms}), " f"got {output.shape}"
    )

    assert p.shape == (batch_size,), f"Expected probability shape (batch_size={batch_size}), " f"got {p.shape}"


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_linear_bandits_forward_shape_errors(BanditClass: BanditClassType) -> None:
    """Check if forward method raises assertion error for invalid input shape."""
    n_features = 5
    bandit: LinearBandit[torch.Tensor] = BanditClass(n_features=n_features)

    # Invalid shape: contextualized_actions should have shape (batch_size, n_arms, n_features).
    # Here we intentionally use (batch_size, n_arms, n_features + 1).
    invalid_actions = torch.randn(2, 3, n_features + 1)

    with pytest.raises(AssertionError, match="must have shape"):
        bandit.forward(invalid_actions)


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_linear_bandit_defaults(BanditClass: BanditClassType) -> None:
    """Test default initialization of base LinearBandit.
    Ensures shapes of precision_matrix, b, theta are correct.
    """
    n_features = 5
    bandit: LinearBandit[torch.Tensor] = BanditClass(n_features=n_features, lazy_uncertainty_update=True)

    assert bandit.precision_matrix.shape == (
        n_features,
        n_features,
    ), f"precision_matrix should be (n_features, n_features), got {bandit.precision_matrix.shape}"

    assert torch.allclose(
        bandit.precision_matrix, torch.eye(n_features)
    ), "Default precision_matrix should be identity."

    assert bandit.b.shape == (n_features,), f"b should be (n_features,), got {bandit.b.shape}"

    assert bandit.theta.shape == (n_features,), f"theta should be (n_features,), got {bandit.theta.shape}"


def test_linear_ucb_correct_mean() -> None:
    """Test if LinearUCBBandit returns correct values for theta = (0, 0, 1)."""
    n_features = 3
    bandit = LinearUCBBandit(n_features=n_features)

    # Manually adjust the bandits parameters
    bandit.theta = torch.tensor([0.0, 0.0, 1.0])

    # pass one batch of 3 arms
    contextualized_actions = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]])
    output, p = bandit.forward(contextualized_actions)

    assert torch.allclose(
        output,
        torch.tensor([[0, 0, 1]]),
    ), "Expected one-hot encoding of the arm with highest UCB."

    assert torch.allclose(
        p,
        torch.ones(1),
    ), "Expected probability of 1 for all arms."


def test_linear_ucb_correct_variance() -> None:
    """Test if LinearUCBBandit returns correct values for same UCB means but different variances."""
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

    # pass vectors where without exploration_rate all arms are equally good
    contextualized_actions = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]]])
    output, p = bandit.forward(contextualized_actions)

    # test that now instead the arm with higher variance is selected
    assert torch.allclose(
        output,
        torch.tensor([[0, 1, 0]]),
    ), "Expected one-hot encoding of the arm with highest variance."

    assert torch.allclose(
        p,
        torch.ones(1),
    ), "Expected probability of 1 for all arms."


def test_linear_ucb_exploration_rate() -> None:
    """Test exploration_rate parameter in LinearUCBBandit to confirm it's settable and used."""
    n_features = 3
    exploration_rate = 25.0  # extreme exploration_rate for testing
    bandit = LinearUCBBandit(n_features=n_features, exploration_rate=exploration_rate)

    assert (
        bandit.hparams["exploration_rate"] == exploration_rate
    ), "Bandit's exploration_rate should match the passed value."

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

    # pass vectors where without exploration_rate all arms are equally good
    contextualized_actions = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.9, 0.5], [0.5, 0.0, 1.0]]])
    output, p = bandit.forward(contextualized_actions)

    # test that now instead the arm with higher variance is selected
    assert torch.allclose(
        output,
        torch.tensor([[0, 1, 0]]),
    ), "Expected one-hot encoding of the arm with highest variance due to extreme exploration_rate value."

    assert torch.allclose(
        p,
        torch.ones(1),
    ), "Expected probability of 1 for all arms."


def test_linear_ts_correct() -> None:
    """Test if LinearTSBandit returns correct values for theta = (0, 0, 1)."""
    n_features = 3
    bandit = LinearTSBandit[torch.Tensor](n_features=n_features)

    # Manually adjust the bandits parameters
    bandit.theta = torch.tensor([0.0, 0.0, 1.0])

    # pass one batch of 3 arms
    contextualized_actions = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]])
    output, _ = bandit.forward(contextualized_actions)

    assert torch.allclose(
        output,
        torch.tensor([[0, 0, 1]]),
    ), "Expected one-hot encoding of the arm with highest expected reward."

    # TODO: Test correct computation of probabilities. See issue #72.


@pytest.mark.parametrize("BanditClass", [LinearUCBBandit, LinearTSBandit[torch.Tensor]])
def test_update_updates_parameters_parameterized(BanditClass: BanditClassType) -> None:
    """
    Test if parameters are updated after training step.
    """
    bandit: LinearBandit[torch.Tensor] = BanditClass(n_features=3, lazy_uncertainty_update=True)
    batch_size = 10
    n_features = bandit.hparams["n_features"]

    # Create dummy data
    chosen_actions = torch.randn(batch_size, 1, n_features)
    realized_rewards = torch.randn(batch_size, 1)

    # Save initial state
    initial_precision_matrix = bandit.precision_matrix.clone()
    initial_b = bandit.b.clone()
    initial_theta = bandit.theta.clone()

    # Perform training step
    bandit._perform_update(chosen_actions, realized_rewards)

    # Check that parameters have been updated
    assert not torch.equal(bandit.precision_matrix, initial_precision_matrix), "Precision matrix should be updated"
    assert not torch.equal(bandit.b, initial_b), "b should be updated"
    assert not torch.equal(bandit.theta, initial_theta), "theta should be updated"


def test_update_correct() -> None:
    """In this simple scenario:
      - n_features = 1
      - precision_matrix = [[1.0]]
      - b = [0.0]
      - chosen_actions = [[2.0]]
      - realized_rewards = [1.0]

    The manual update for the precision_matrix should yield:
      M_new = [[0.2]]
      b_new = [2.0]
      theta_new = [0.4]
    """

    bandit = LinearUCBBandit(n_features=1, eps=0.0, lazy_uncertainty_update=True)

    chosen_contextualized_actions = torch.tensor([[[2.0]]])  # shape (1,1)
    realized_rewards = torch.tensor([[1.0]])  # shape (1,1)

    bandit._perform_update(chosen_contextualized_actions, realized_rewards)

    expected_M = torch.tensor([[0.2]])
    expected_b = torch.tensor([2.0])
    expected_theta = torch.tensor([0.4])

    assert torch.allclose(
        bandit.precision_matrix, expected_M, atol=1e-6
    ), f"Expected M={expected_M}, got {bandit.precision_matrix}"
    assert torch.allclose(bandit.b, expected_b, atol=1e-6), f"Expected b={expected_b}, got {bandit.b}"
    assert torch.allclose(
        bandit.theta, expected_theta, atol=1e-6
    ), f"Expected theta={expected_theta}, got {bandit.theta}"


@pytest.mark.parametrize("BanditClass", [LinearUCBBandit, LinearTSBandit])
def test_update_shapes_parameterized(BanditClass: BanditClassType) -> None:
    """Test if parameters have correct shapes after update."""
    bandit: LinearBandit[torch.Tensor] = BanditClass(n_features=3)
    batch_size = 10
    n_features = bandit.hparams["n_features"]

    # Create dummy data
    chosen_actions = torch.randn(batch_size, 1, n_features)
    realized_rewards = torch.randn(batch_size, 1)

    # Perform training step
    bandit._perform_update(chosen_actions, realized_rewards)

    # Check shapes of updated parameters
    assert bandit.precision_matrix.shape == (
        n_features,
        n_features,
    ), "Precision matrix should have shape (n_features, n_features)"
    assert bandit.b.shape == (n_features,), "b should have shape (n_features,)"
    assert bandit.theta.shape == (n_features,), "theta should have shape (n_features,)"


@pytest.mark.parametrize("BanditClass", [LinearUCBBandit, LinearTSBandit])
def test_update_invalid_shapes_parameterized(BanditClass: BanditClassType) -> None:
    """Test if assertion errors are raised for invalid input shapes."""
    bandit: LinearBandit[torch.Tensor] = BanditClass(n_features=3)
    batch_size = 10
    n_features = bandit.hparams["n_features"]

    # Create invalid dummy data
    chosen_actions_invalid = torch.randn(batch_size, 1, n_features + 1)

    realized_rewards = torch.randn(batch_size, 1)

    # Check for assertion errors
    with pytest.raises(AssertionError):
        bandit._perform_update(chosen_actions_invalid, realized_rewards)

    chosen_actions_invalid = torch.randn(batch_size, 2, n_features)

    with pytest.raises(AssertionError):
        bandit._perform_update(chosen_actions_invalid, realized_rewards)


def test_update_zero_denominator(
    simple_ucb_bandit: LinearUCBBandit,
) -> None:
    """Test if assertion error is raised when denominator is zero."""
    bandit = simple_ucb_bandit
    batch_size = 10
    n_features = bandit.hparams["n_features"]

    # Create dummy data with NaN
    chosen_actions = torch.zeros(batch_size, 1, n_features)
    realized_rewards = torch.zeros(batch_size, 1)
    chosen_actions[0, 0] = torch.nan

    with pytest.raises(AssertionError):
        bandit._perform_update(chosen_actions, realized_rewards)

    # Create dummy data that will cause zero denominator
    chosen_actions = torch.tensor([[[2.0]]])
    realized_rewards = torch.zeros(1, 1)

    bandit.save_hyperparameters({"eps": 0.0})
    bandit.precision_matrix = torch.tensor([[-0.25]])  # shape (1,1)

    with pytest.raises((AssertionError, Exception)):
        bandit._perform_update(chosen_actions, realized_rewards)


@pytest.mark.parametrize("BanditClass", [DiagonalPrecApproxLinearUCBBandit, DiagonalPrecApproxLinearTSBandit])
def test_approx_prec_update(BanditClass: BanditClassType) -> None:
    """
    Test if the precision matrix is updated correctly using the diagonal approximation.
    """
    n_features = 3
    bandit = BanditClass(n_features=n_features, lazy_uncertainty_update=True, eps=0.0)
    bandit.theta = torch.tensor([0.0, 0.5, 1.0])

    contextualized_actions = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]])
    output, _ = bandit.forward(contextualized_actions.unsqueeze(0))

    expected_output = torch.tensor([[0, 0, 1]])

    assert torch.allclose(output, expected_output), "Expected one-hot encoding of the arm with highest expected reward."

    # Update precision matrix using the diagonal approximation
    bandit._update_precision_matrix(contextualized_actions)

    expected_precision_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.25]]) + torch.eye(
        n_features
    )

    assert torch.allclose(
        bandit.precision_matrix, expected_precision_matrix
    ), f"Expected precision matrix {expected_precision_matrix}, got {bandit.precision_matrix}"


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_bandit_state_checkpoint(BanditClass: BanditClassType) -> None:
    """Test basic state saving and loading for bandits."""
    n_features = 4
    original_bandit = BanditClass(n_features=n_features)

    original_bandit.precision_matrix = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    original_bandit.b = torch.tensor([0.5, 1.5, 2.5, 3.5])
    original_bandit.theta = torch.tensor([0.25, 0.75, 1.25, 1.75])

    checkpoint: dict[str, Any] = {}
    original_bandit.on_save_checkpoint(checkpoint)

    loaded_bandit = BanditClass(n_features=n_features)
    loaded_bandit.on_load_checkpoint(checkpoint)

    # Verify state was properly restored
    assert torch.allclose(loaded_bandit.precision_matrix, original_bandit.precision_matrix)
    assert torch.allclose(loaded_bandit.b, original_bandit.b)
    assert torch.allclose(loaded_bandit.theta, original_bandit.theta)

    # Verify selector was properly restored
    assert isinstance(loaded_bandit.selector, ArgMaxSelector)
    assert checkpoint["selector_state"]["type"] == "ArgMaxSelector"


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_bandit_selector_checkpoint_epsilon_greedy(BanditClass: BanditClassType) -> None:
    """Test saving and loading EpsilonGreedySelector configuration."""
    n_features = 3
    epsilon = 0.15
    original_bandit = BanditClass(n_features=n_features, selector=EpsilonGreedySelector(epsilon=epsilon))

    original_bandit.selector.generator.get_state()

    checkpoint: dict[str, Any] = {}
    original_bandit.on_save_checkpoint(checkpoint)

    loaded_bandit = BanditClass(n_features=n_features)
    loaded_bandit.on_load_checkpoint(checkpoint)

    # Verify selector was properly restored
    assert isinstance(loaded_bandit.selector, EpsilonGreedySelector)
    assert loaded_bandit.selector.epsilon == epsilon
    assert checkpoint["selector_state"]["type"] == "EpsilonGreedySelector"
    assert checkpoint["selector_state"]["epsilon"] == epsilon

    # Verify random generator state was saved
    assert "generator_state" in checkpoint["selector_state"]


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_bandit_hyperparameters_checkpoint(BanditClass: BanditClassType) -> None:
    """Test that model hyperparameters are properly preserved during checkpointing."""
    n_features = 5
    eps = 1e-3
    lambda_ = 2.0
    exploration_rate = 2.0
    lazy_update = True

    if BanditClass in [LinearUCBBandit, DiagonalPrecApproxLinearUCBBandit]:
        original_bandit = BanditClass(
            n_features=n_features,
            eps=eps,
            lambda_=lambda_,
            exploration_rate=exploration_rate,
            lazy_uncertainty_update=lazy_update,
        )
    else:
        original_bandit = BanditClass(
            n_features=n_features, eps=eps, lambda_=lambda_, lazy_uncertainty_update=lazy_update
        )

    checkpoint: dict[str, Any] = {}
    original_bandit.on_save_checkpoint(checkpoint)

    # Verify hyperparameters are maintained by lightning's hyperparameter saving
    assert original_bandit.hparams["n_features"] == n_features
    assert original_bandit.hparams["eps"] == eps
    assert original_bandit.hparams["lambda_"] == lambda_
    assert original_bandit.hparams["lazy_uncertainty_update"] == lazy_update

    if BanditClass in [LinearUCBBandit, DiagonalPrecApproxLinearUCBBandit]:
        assert original_bandit.hparams["exploration_rate"] == exploration_rate


@pytest.mark.parametrize("BanditClass", LinearBanditTypes)
def test_bandit_end_to_end_checkpoint(BanditClass: BanditClassType) -> None:
    """
    End-to-end test of training, saving checkpoint, and restoring bandit
    with proper behavior preservation.
    """
    n_features = 3
    original_bandit = BanditClass(n_features=n_features, lazy_uncertainty_update=True)

    batch_size = 1
    n_arms = 4
    torch.manual_seed(42)
    test_context = torch.randn(batch_size, n_arms, n_features)

    initial_theta = original_bandit.theta.clone()
    initial_b = original_bandit.b.clone()
    initial_precision = original_bandit.precision_matrix.clone()

    checkpoint: dict[str, Any] = {}
    original_bandit.on_save_checkpoint(checkpoint)

    loaded_bandit = BanditClass(n_features=n_features, lazy_uncertainty_update=True)
    loaded_bandit.on_load_checkpoint(checkpoint)

    if BanditClass in [LinearUCBBandit, DiagonalPrecApproxLinearUCBBandit]:
        # UCB bandits should be deterministic with same parameters
        orig_action, _ = original_bandit.forward(test_context)
        loaded_action, _ = loaded_bandit.forward(test_context)

        assert torch.equal(orig_action, loaded_action), f"Expected {orig_action}, got {loaded_action}"
    else:
        n_samples = 50
        original_choices = []
        loaded_choices = []

        for i in range(n_samples):
            seed = 1000 + i

            torch.manual_seed(seed)
            orig_action, _ = original_bandit.forward(test_context)

            torch.manual_seed(seed)
            loaded_action, _ = loaded_bandit.forward(test_context)

            original_choices.append(orig_action.argmax(dim=1).item())
            loaded_choices.append(loaded_action.argmax(dim=1).item())

        # Verify the pattern of selections is similar (agreement rate > 0.8)
        agreement_rate = (
            sum(
                original_choice == loaded_choice
                for original_choice, loaded_choice in zip(original_choices, loaded_choices, strict=False)
            )
            / n_samples
        )

        assert agreement_rate > 0.8, f"Models only agreed on {agreement_rate:.1%} of selections"

    # Update the model and verify updates work properly
    batch_size = 5
    torch.manual_seed(43)
    chosen_actions = torch.randn(batch_size, 1, n_features)
    realized_rewards = torch.ones(batch_size, 1)

    loaded_bandit._perform_update(chosen_actions, realized_rewards)

    # Verify the parameters has changed
    params_changed = (
        not torch.allclose(loaded_bandit.theta, initial_theta)
        and not torch.allclose(loaded_bandit.b, initial_b)
        and not torch.allclose(loaded_bandit.precision_matrix, initial_precision)
    )

    assert params_changed, "Model parameters should change after update"
