import torch

from calvera.benchmark.datasets.wheel import WheelBanditDataset


def test_wheel() -> None:
    n_features = 2
    num_samples = 1000
    ds = WheelBanditDataset(num_samples=num_samples, delta=0.1, seed=42)
    assert len(ds) == num_samples
    assert ds.context_size == n_features * 5
    assert ds.num_actions == 5
    assert ds.num_samples == num_samples
    assert ds.delta == 0.1
    assert ds.mu_small == 1.0
    assert ds.std_small == 0.01
    assert ds.mu_medium == 1.2
    assert ds.std_medium == 0.01
    assert ds.mu_large == 50.0
    assert ds.std_large == 0.01
    assert ds.data.shape == (num_samples, n_features)
    assert ds.rewards.shape == (num_samples, 5)

    assert ds[0][0].shape == (
        5,
        5 * n_features,
    )
    assert torch.allclose(ds[0][1], ds.rewards[0])

    # assert that 5 actions exist
    assert ds.rewards.argmax(dim=1).unique().shape[0] == 5  # type: ignore
    num_medium_reward_1 = (ds.rewards.argmax(dim=1) == 4).sum()

    # Test that the dataset is reproducible
    ds2 = WheelBanditDataset(num_samples=num_samples, delta=0.1, seed=42)
    assert torch.allclose(ds.data, ds2.data)
    assert torch.allclose(ds.rewards, ds2.rewards)

    # Test delta
    ds3 = WheelBanditDataset(num_samples=num_samples, delta=0.2, seed=42)
    assert torch.allclose(ds.data, ds3.data)
    assert not torch.allclose(ds.rewards, ds3.rewards)

    num_medium_reward_2 = (ds3.rewards.argmax(dim=1) == 4).sum()
    assert num_medium_reward_1 < num_medium_reward_2
