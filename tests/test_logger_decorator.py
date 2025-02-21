import pytest
from unittest.mock import MagicMock

from lightning.pytorch.loggers.logger import Logger
import torch

from neural_bandits.benchmark.logger_decorator import OnlineBanditLoggerDecorator


def test_online_bandit_logger_decorator_basic() -> None:
    mock_logger = MagicMock(spec=Logger)
    decorator = OnlineBanditLoggerDecorator(mock_logger)

    # 1. Check pre_training_log
    batch = torch.Tensor([123.0, 53.0])
    decorator.pre_training_log({"pre_metric": batch})
    assert decorator.pre_training_metrics == {"pre_metric": batch}

    # 2. Check log_metrics
    decorator.log_metrics({"loss": 0.5}, step=0)
    # Ensure that the wrapped logger’s log_metrics was called with updated metrics
    mock_logger.log_metrics.assert_called_with(
        {"training_run": 0, "loss": 0.5, "pre_metric": 123.0}, 0
    )

    decorator.log_metrics({"loss": 0.9}, step=1)
    mock_logger.log_metrics.assert_called_with(
        {"training_run": 0, "loss": 0.9, "pre_metric": 53.0}, 1
    )

    # assert it throws an AssertionError
    with pytest.raises(AssertionError):
        decorator.log_metrics({"loss": 0.9}, step=2)

    # 3. Check finalize increments training_run
    decorator.finalize(status="finished")
    mock_logger.finalize.assert_called_with("finished")
    assert decorator.training_run == 1
    assert decorator.start_step_of_current_run == 2
    assert decorator.pre_training_metrics is None

    # 4. Check log_hyperparams
    params = {"lr": 0.1}
    decorator.log_hyperparams(params)
    mock_logger.log_hyperparams.assert_called_with(params)

    # 5. Check log_graph
    model = MagicMock()
    decorator.log_graph(model)
    mock_logger.log_graph.assert_called_with(model)

    # 6. Check save
    decorator.save()
    mock_logger.save.assert_called()
