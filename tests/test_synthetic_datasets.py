import pytest
import torch

from calvera.benchmark.datasets.synthetic import (
    CubicSyntheticDataset,
    LinearCombinationSyntheticDataset,
    LinearSyntheticDataset,
    QuadraticSyntheticDataset,
    SinSyntheticDataset,
    SyntheticDataset,
)


@pytest.mark.parametrize(
    "DatasetClass, expected_phi_dim",
    [
        # For a dataset with n_features=2:
        # LinearSyntheticDataset: bias + x → 1 + 2 = 3
        (LinearSyntheticDataset, 3),
        # CubicSyntheticDataset: bias + x + x^2 → 1 + 2 + 2 = 5
        (QuadraticSyntheticDataset, 5),
        # CubicSyntheticDataset: bias + x + x^2 → 1 + 2 + 2 + 2 = 7
        (CubicSyntheticDataset, 7),
        # SinSyntheticDataset: bias + x + sin(x) → 1 + 2 + 2 = 5
        (SinSyntheticDataset, 5),
        # LinearCombinationSyntheticDataset: bias + x + upper triangle of outer product
        # For n_features=2, the upper triangle (including diagonal) has 3 elements → total = 1 + 2 + 3 = 6
        (LinearCombinationSyntheticDataset, 6),
    ],
)
def test_phi(DatasetClass: type[SyntheticDataset], expected_phi_dim: int) -> None:
    # Use a fixed n_features and small num_samples for testing
    n_features = 2
    num_samples = 10
    ds = DatasetClass(n_features=n_features, num_samples=num_samples, noise_std=0.0)
    # Create a dummy input; here we test on a batch of 10 samples
    x = torch.randn(num_samples, n_features)
    phi_out = ds.phi(x)
    assert phi_out.shape == (
        num_samples,
        expected_phi_dim,
    ), f"Expected phi shape {(num_samples, expected_phi_dim)} but got {phi_out.shape}"


@pytest.mark.parametrize(
    "DatasetClass",
    [
        LinearSyntheticDataset,
        QuadraticSyntheticDataset,
        SinSyntheticDataset,
        LinearCombinationSyntheticDataset,
    ],
)
def test_initialization(DatasetClass: type[SyntheticDataset]) -> None:
    # Use n_features=3 and a small dataset; note: num_actions is fixed to 2 in SyntheticDataset
    n_features = 3
    num_samples = 20
    num_actions = 2
    ds = DatasetClass(n_features=n_features, num_samples=num_samples, noise_std=0.0)

    assert ds.num_actions == num_actions, f"Expected num_actions {num_actions} but got {ds.num_actions}"

    assert ds.n_features == n_features, f"Expected n_features {n_features} but got {ds.n_features}"

    assert len(ds) == num_samples, f"Expected len(ds) {num_samples} but got {len(ds)}"

    assert ds.X.shape == (num_samples, n_features), f"Expected X shape {(num_samples, n_features)} but got {ds.X.shape}"

    assert ds.Phi.shape[0] == num_samples, f"Expected Phi shape ({num_samples}, phi_features) but got {ds.Phi.shape}"

    assert ds.Phi.shape[1] >= n_features, f"Expected Phi shape ({num_samples}, phi_features) but got {ds.Phi.shape}"

    assert ds.y.ndim == 1 and ds.y.shape[0] == num_samples, f"Expected y shape {num_samples} but got {ds.y.shape}"


@pytest.mark.parametrize(
    "DatasetClass",
    [
        LinearSyntheticDataset,
        QuadraticSyntheticDataset,
        SinSyntheticDataset,
        LinearCombinationSyntheticDataset,
    ],
)
def test_getitem_returns_expected_shapes(DatasetClass: type[SyntheticDataset]) -> None:
    # Use n_features=3 and a small dataset; note: num_actions is fixed to 2 in SyntheticDataset
    n_features = 3
    num_samples = 20
    num_actions = 2
    ds = DatasetClass(n_features=n_features, num_samples=num_samples, noise_std=0.0)

    # Set contextualizer to identity for testing purposes.
    contextualized_actions, rewards = ds[0]
    # In SyntheticDataset, context_size is computed as n_features * num_actions.
    expected_context_size = n_features * num_actions  # for num_actions=2, expect 6
    assert (
        contextualized_actions.shape[0] == num_actions
    ), f"Expected num_actions {num_actions} but got {contextualized_actions.shape[0]}"
    assert (
        contextualized_actions.shape[1] == expected_context_size
    ), f"Expected contextualized actions length {expected_context_size} but got {contextualized_actions.shape[0]}"

    # Rewards should have shape (num_actions,)
    assert rewards.shape[0] == num_actions, f"Expected rewards shape {(num_actions,)} but got {rewards.shape}"

    # Ensure rewards are float32
    assert rewards.dtype == torch.float32


def test_reward_method() -> None:
    # Test that the reward method returns a float matching the label in self.y.
    n_features = 2
    num_samples = 5
    ds = LinearSyntheticDataset(n_features=n_features, num_samples=num_samples, noise_std=0.0)
    # Set contextualizer to identity (so __getitem__ becomes easy to check)
    ds.contextualizer = lambda x: x
    # For each sample and action, reward should be 1.0 if y[idx]==action, else 0.0.
    for idx in range(len(ds)):
        for action in range(ds.num_actions):
            expected = 1.0 if ds.y[idx].item() == action else 0.0
            reward_val = ds.reward(idx, action)
            assert isinstance(reward_val, float), "Reward value should be a float."
            assert reward_val == expected, f"At idx {idx} and action {action}, expected {expected} but got {reward_val}"
