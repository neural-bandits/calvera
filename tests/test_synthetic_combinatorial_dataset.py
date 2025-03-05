import pytest
import torch
import numpy as np
from math import isclose

from calvera.benchmark.datasets.synthetic_combinatorial import SyntheticCombinatorialDataset


def test_unit_ball_distribution() -> None:
    """Test that _random_unit_ball generates points within a unit ball with proper distribution."""
    context_size = 3
    n_samples = 10000
    seed = 42

    torch.manual_seed(seed)
    ds = SyntheticCombinatorialDataset(n_samples=10, context_size=context_size, seed=seed)

    points = ds._random_unit_ball(n_samples)

    assert points.shape == (n_samples, context_size)

    norms = torch.norm(points, dim=1)
    assert torch.all(norms <= 1.0), "Some points are outside the unit ball"
    assert torch.all(norms > 0.0), "Some points are outside the unit ball"

    assert torch.any(norms < 0.8), "All vectors seem to be on or near the surface of the unit ball"

    # For a uniform distribution within the unit ball:
    # - The distribution of r^d should follow a uniform distribution
    # - Where r is the radius and d is the dimension
    radii_d = norms.pow(context_size)

    hist, _ = np.histogram(radii_d.numpy(), bins=10, range=(0, 1))

    # The bins should have roughly equal counts
    max_diff_ratio = max(hist) / min(hist) if min(hist) > 0 else float("inf")
    assert max_diff_ratio < 2.0, f"Radial distribution r^d is too uneven, max/min ratio: {max_diff_ratio}"


def test_linear_reward_function() -> None:
    """Test that the linear reward function correctly implements h₁(x) = x^T a."""
    n_samples = 20
    num_actions = 10
    context_size = 8
    seed = 42

    torch.manual_seed(seed)
    ds = SyntheticCombinatorialDataset(
        n_samples=n_samples,
        num_actions=num_actions,
        context_size=context_size,
        function_type="linear",
        noise_std=0.0,
        seed=seed,
    )

    expected_rewards = torch.zeros(n_samples, num_actions)
    for i in range(n_samples):
        for j in range(num_actions):
            expected_rewards[i, j] = torch.dot(ds.contexts[i, j], ds.a)

    assert torch.allclose(ds.rewards, expected_rewards), "Linear reward calculation is incorrect"


def test_quadratic_reward_function() -> None:
    """Test that the quadratic reward function correctly implements h₂(x) = (x^T a)²."""
    n_samples = 20
    num_actions = 10
    context_size = 8
    seed = 42

    torch.manual_seed(seed)
    ds = SyntheticCombinatorialDataset(
        n_samples=n_samples,
        num_actions=num_actions,
        context_size=context_size,
        function_type="quadratic",
        noise_std=0.0,
        seed=seed,
    )

    expected_rewards = torch.zeros(n_samples, num_actions)
    for i in range(n_samples):
        for j in range(num_actions):
            inner_prod = torch.dot(ds.contexts[i, j], ds.a)
            expected_rewards[i, j] = inner_prod**2

    assert torch.allclose(ds.rewards, expected_rewards), "Quadratic reward calculation is incorrect"


def test_cosine_reward_function() -> None:
    """Test that the cosine reward function correctly implements h₃(x) = cos(πx^T a)."""
    n_samples = 20
    num_actions = 10
    context_size = 8
    seed = 42

    torch.manual_seed(seed)
    ds = SyntheticCombinatorialDataset(
        n_samples=n_samples,
        num_actions=num_actions,
        context_size=context_size,
        function_type="cosine",
        noise_std=0.0,
        seed=seed,
    )

    expected_rewards = torch.zeros(n_samples, num_actions)
    for i in range(n_samples):
        for j in range(num_actions):
            inner_prod = torch.dot(ds.contexts[i, j], ds.a)
            expected_rewards[i, j] = torch.cos(torch.pi * inner_prod)

    assert torch.allclose(ds.rewards, expected_rewards), "Cosine reward calculation is incorrect"


def test_noise_impact() -> None:
    """Test that noise is correctly added and affects the reward variance."""
    n_samples = 100
    num_actions = 10
    context_size = 5
    function_type = "linear"
    seed = 42

    noise_levels = [0.0, 0.1, 0.5, 1.0]
    reward_stds = []

    for noise_std in noise_levels:
        torch.manual_seed(seed)

        ds = SyntheticCombinatorialDataset(
            n_samples=n_samples,
            num_actions=num_actions,
            context_size=context_size,
            function_type=function_type,
            noise_std=noise_std,
            seed=seed,
        )

        inner_prod = torch.matmul(ds.contexts, ds.a)
        ground_truth = inner_prod  # For linear function

        # Calculate the empirical variance of the noise
        noise = ds.rewards - ground_truth
        empirical_noise_std = noise.std().item()
        reward_stds.append(empirical_noise_std)

        if noise_std == 0.0:
            assert isclose(
                empirical_noise_std, 0.0, abs_tol=1e-5
            ), f"Expected zero noise, got std {empirical_noise_std}"
        else:
            assert isclose(
                empirical_noise_std, noise_std, rel_tol=0.2
            ), f"Expected noise std {noise_std}, got {empirical_noise_std}"

    # Check that higher noise levels result in higher variance
    for i in range(1, len(noise_levels)):
        assert reward_stds[i] > reward_stds[i - 1], "Higher noise level should result in higher variance"


@pytest.mark.parametrize(
    "function_type",
    ["linear", "quadratic", "cosine"],
)
def test_initialization(function_type: str) -> None:
    """Test initialization with various parameters and function types."""
    n_samples = 50
    num_actions = 8
    context_size = 10

    ds = SyntheticCombinatorialDataset(
        n_samples=n_samples,
        num_actions=num_actions,
        context_size=context_size,
        function_type=function_type,
        noise_std=0.1,
        seed=42,
    )

    assert ds.n_samples == n_samples, f"Expected n_samples {n_samples} but got {ds.n_samples}"
    assert ds.num_actions == num_actions, f"Expected num_actions {num_actions} but got {ds.num_actions}"
    assert ds.context_size == context_size, f"Expected context_size {context_size} but got {ds.context_size}"
    assert ds.function_type == function_type, f"Expected function_type {function_type} but got {ds.function_type}"

    assert len(ds) == n_samples, f"Expected len(ds) {n_samples} but got {len(ds)}"

    assert ds.contexts.shape == (
        n_samples,
        num_actions,
        context_size,
    ), f"Expected contexts shape {(n_samples, num_actions, context_size)} but got {ds.contexts.shape}"
    assert ds.rewards.shape == (
        n_samples,
        num_actions,
    ), f"Expected rewards shape {(n_samples, num_actions)} but got {ds.rewards.shape}"


def test_getitem_returns_expected_shapes() -> None:
    """Test that __getitem__ returns tensors with expected shapes and values."""
    n_samples = 30
    num_actions = 6
    context_size = 12

    ds = SyntheticCombinatorialDataset(
        n_samples=n_samples,
        num_actions=num_actions,
        context_size=context_size,
    )

    idx = 0
    contexts, rewards = ds[idx]

    assert contexts.shape == (
        num_actions,
        context_size,
    ), f"Expected contexts shape {(num_actions, context_size)} but got {contexts.shape}"
    assert rewards.shape == (num_actions,), f"Expected rewards shape {(num_actions,)} but got {rewards.shape}"

    assert torch.equal(contexts, ds.contexts[idx]), f"Returned contexts at index {idx} don't match stored contexts"
    assert torch.equal(rewards, ds.rewards[idx]), f"Returned rewards at index {idx} don't match stored rewards"


@pytest.mark.parametrize(
    "function_type",
    ["linear", "quadratic", "cosine"],
)
def test_function_types(function_type: str) -> None:
    """Test that different function types produce appropriate reward values."""
    n_samples = 10
    num_actions = 4
    context_size = 8
    seed = 42

    ds = SyntheticCombinatorialDataset(
        n_samples=n_samples,
        num_actions=num_actions,
        context_size=context_size,
        function_type=function_type,
        noise_std=0.0,
        seed=seed,
    )

    if function_type == "linear":
        assert torch.any(ds.rewards < 0) or torch.any(ds.rewards > 0), "Linear rewards should have variation"
    elif function_type == "quadratic":
        assert torch.all(ds.rewards >= 0), "Quadratic rewards should be non-negative"
    elif function_type == "cosine":
        assert torch.all(ds.rewards >= -1), "Cosine rewards should be >= -1"
        assert torch.all(ds.rewards <= 1), "Cosine rewards should be <= 1"


def test_invalid_function_type() -> None:
    """Test that an invalid function type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown function type: invalid"):
        SyntheticCombinatorialDataset(function_type="invalid")
