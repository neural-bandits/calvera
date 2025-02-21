import lightning as tl
import pytest
import torch
import torch.nn as nn

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.bandits.linear_ts_bandit import LinearTSBandit
from neural_bandits.bandits.linear_ucb_bandit import LinearUCBBandit
from neural_bandits.bandits.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.bandits.neural_ucb_bandit import NeuralUCBBandit

n_features = 3


@pytest.mark.parametrize(
    "bandit",
    [
        LinearTSBandit(n_features),
        LinearUCBBandit(n_features),
        NeuralLinearBandit(
            network=nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, n_features)
            ),
            n_network_input_size=n_features,
        ),
        NeuralUCBBandit(
            n_features,
            nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 1)),
        ),
    ],
)
def test_trainer_fit_runs(bandit: AbstractBandit) -> None:
    """
    Test if parameters are updated after training step.
    """

    dataset = torch.utils.data.TensorDataset(torch.randn(10, 1, 3), torch.rand(10, 1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    tl.Trainer(fast_dev_run=True).fit(bandit, dataloader)

    contextualized_actions = torch.randn(10, 2, 3)
    result, p = bandit.forward(contextualized_actions)

    assert result.shape == (10, 2)
    assert torch.allclose(
        result.sum(dim=1, dtype=torch.float32), torch.ones(10, dtype=torch.float32)
    )

    assert p.shape == (10,)
    assert p.min() > 0.0 and p.max() <= 1.0
