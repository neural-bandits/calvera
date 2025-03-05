from unittest.mock import MagicMock

from lightning.pytorch.loggers.logger import Logger

from calvera.benchmark.logger_decorator import OnlineBanditLoggerDecorator


def test_online_bandit_logger_decorator_basic() -> None:
    mock_logger = MagicMock(spec=Logger)
    decorator = OnlineBanditLoggerDecorator(mock_logger)

    # 2. Check log_metrics
    decorator.log_metrics({"loss": 0.5}, step=0)
    # Ensure that the wrapped logger’s log_metrics was called with updated metrics
    mock_logger.log_metrics.assert_called_with({"training_run": 0, "loss": 0.5}, 0)

    decorator.log_metrics({"loss": 0.9}, step=1)
    mock_logger.log_metrics.assert_called_with({"training_run": 0, "loss": 0.9}, 1)

    # 3. Check finalize increments training_run
    decorator.finalize(status="finished")
    mock_logger.finalize.assert_called_with("finished")
    assert decorator.training_run == 1
    assert decorator.start_step_of_current_run == 2

    # 4. Check log_hyperparams
    params = {"lr": 0.1}
    decorator.log_hyperparams(params)
    mock_logger.log_hyperparams.assert_called_with(params)

    # 5. Check log_graph
    model = MagicMock()
    decorator.log_graph(model)
    mock_logger.log_graph.assert_called_with(model, None)

    # 6. Check save
    decorator.save()
    mock_logger.save.assert_called()
