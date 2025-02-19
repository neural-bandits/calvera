import pytest
import torch

from neural_bandits.benchmark.feedback_dataset import BanditFeedbackDataset


def test_feedback_dataset_basic():
    n, i, k = 3, 2, 4
    chosen_contextualized_actions = torch.randn(n, i, k)
    realized_rewards = torch.randn(n, i)

    dataset = BanditFeedbackDataset(chosen_contextualized_actions, realized_rewards)
    assert len(dataset) == n

    for idx in range(n):
        x, y = dataset[idx]
        assert x.shape == (i, k)
        assert y.shape == (i,)


def test_feedback_dataset_shape_mismatch():
    with pytest.raises(AssertionError):
        # Mismatch in first dimension
        BanditFeedbackDataset(torch.randn(2, 2, 4), torch.randn(3, 2))

    with pytest.raises(AssertionError):
        # Mismatch in second dimension
        BanditFeedbackDataset(torch.randn(2, 2, 4), torch.randn(2, 3))
