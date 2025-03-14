import lightning as pl
import pytest
import torch
import torch.nn as nn

from calvera.bandits.abstract_bandit import AbstractBandit, _collate_fn
from calvera.bandits.linear_ts_bandit import (
    DiagonalPrecApproxLinearTSBandit,
    LinearTSBandit,
)
from calvera.bandits.linear_ucb_bandit import (
    DiagonalPrecApproxLinearUCBBandit,
    LinearUCBBandit,
)
from calvera.bandits.neural_linear_bandit import NeuralLinearBandit
from calvera.bandits.neural_ts_bandit import NeuralTSBandit
from calvera.bandits.neural_ucb_bandit import NeuralUCBBandit
from calvera.utils.data_storage import AllDataRetrievalStrategy, TensorDataBuffer

n_features = 3


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


bandits = [
    LinearTSBandit(
        n_features,
    ),
    DiagonalPrecApproxLinearTSBandit(
        n_features,
    ),
    LinearUCBBandit(
        n_features,
    ),
    DiagonalPrecApproxLinearUCBBandit(
        n_features,
    ),
    NeuralLinearBandit(
        network=nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, n_features)),
        buffer=TensorDataBuffer[torch.Tensor](
            AllDataRetrievalStrategy(),
            device=torch.device("cpu"),
        ),
        n_embedding_size=n_features,
        train_batch_size=2,
        min_samples_required_for_training=1024,
    ),
    NeuralUCBBandit(
        n_features,
        nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 1)),
        buffer=TensorDataBuffer[torch.Tensor](
            AllDataRetrievalStrategy(),
            device=torch.device("cpu"),
        ),
        train_batch_size=2,
    ),
    NeuralTSBandit(
        n_features,
        nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 1)),
        buffer=TensorDataBuffer[torch.Tensor](
            AllDataRetrievalStrategy(),
            device=torch.device("cpu"),
        ),
        train_batch_size=2,
    ),
]


@pytest.mark.parametrize("bandit", bandits)
def test_trainer_fit_runs(bandit: AbstractBandit[torch.Tensor]) -> None:
    """
    Test if parameters are updated after training step.
    """
    device = "cpu"
    bandit = bandit.to(device)

    bandit.record_feedback(torch.randn(10, 1, 3, device=device), torch.rand(10, 1, device=device))
    pl.Trainer(fast_dev_run=True).fit(bandit)

    contextualized_actions = torch.randn(10, 2, 3, device=device)
    result, p = bandit.forward(contextualized_actions)

    assert result.shape == (10, 2)
    assert torch.allclose(
        result.sum(dim=1, dtype=torch.float32),
        torch.ones(10, dtype=torch.float32, device=device),
    )

    assert p.shape == (10,)
    assert p.min() > 0.0 and p.max() <= 1.0


@pytest.mark.parametrize("bandit", bandits)
def test_trainer_fit_runs_with_dataloader(
    bandit: AbstractBandit[torch.Tensor],
) -> None:
    """
    Test if parameters are updated after training step.
    """
    device = "cpu"
    bandit = bandit.to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.randn(10, 1, 3, device=device),
        torch.rand(10, 1, 5, device=device),
        torch.rand(10, 1, device=device),
        torch.rand(10, 2, device=device),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)
    pl.Trainer(fast_dev_run=True).fit(bandit, dataloader)

    contextualized_actions = torch.randn(10, 2, 3, device=device)
    result, p = bandit.forward(contextualized_actions)

    assert result.shape == (10, 2)
    assert torch.allclose(
        result.sum(dim=1, dtype=torch.float32),
        torch.ones(10, dtype=torch.float32, device=device),
    )

    assert p.shape == (10,)
    assert p.min() > 0.0 and p.max() <= 1.0
