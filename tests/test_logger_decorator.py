import sys
from unittest.mock import MagicMock

from lightning.pytorch.loggers.logger import Logger

from neural_bandits.benchmark.logger_decorator import OnlineBanditLoggerDecorator


def test_online_bandit_logger_decorator_basic() -> None:
    mock_logger = MagicMock(spec=Logger)
    decorator = OnlineBanditLoggerDecorator(mock_logger)

    # 1. Check pre_training_log
    decorator.pre_training_log({"pre_metric": 123.0})
    assert decorator.pre_training_metrics == {"pre_metric": 123.0}

    # 2. Check log_metrics
    decorator.log_metrics({"loss": 0.5}, step=1)
    # Decorator should flush and write to stdout once
    sys.stdout.flush()
    # Ensure that the wrapped logger’s log_metrics was called with updated metrics
    mock_logger.log_metrics.assert_called_with(
        {"training_run": 0, "loss": 0.5, "pre_metric": 123.0}, 1
    )
    assert (
        decorator.pre_training_metrics is None
    )  # pre_metrics cleared after first usage

    # 3. Check finalize increments training_run
    decorator.finalize(status="finished")
    mock_logger.finalize.assert_called_with("finished")
    assert decorator.training_run == 1
    assert decorator.final_step_of_last_run == 1
