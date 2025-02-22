import sys
from typing import Any, Optional

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class OnlineBanditLoggerDecorator(Logger):
    """
    Uses the Decorator pattern to add online bandit functionality to a pytorch lightning logger.
    Will use stdout flush to only print the metrics of the current training run to the console to prevent too many prints over many training runs.

    Allows for logging over multiple training runs and for logging batch specific metrics before
    the batch is actually started.

    Usage:
    ```python
    from lightning.pytorch.loggers import CSVLogger
    from lightning import Trainer
    from neural_bandits.benchmark.logger import OnlineBanditLoggerDecorator
    import pandas as pd

    logger = CSVLogger("logs")
    online_bandit_logger = OnlineBanditLoggerDecorator(logger)

    # allows for logging over multiple training runs
    for batch in data_loader:
        # log batch specific metrics before the batch is actually started
        logger.pre_training_log({"regret": 1})

        trainer = Trainer(logger=online_bandit_logger)
        trainer.fit(model)

    # now you can load the logs from the CSV file
    df = pd.read_csv(logger.log_dir + "/metrics.csv")
    df["regret"].dropna().plot() # regret is the batch specific metric. It is only added to the first row of a training run of the CSV logger.
    df["training_run"].plot() # idx of how often trainer.fit has been called.
    ```
    """

    def __init__(self, logger: Logger, enable_console_logging: bool = True) -> None:
        """
        Args:
            logger: The logger to decorate.
            enable_console_logging: If True, only the metrics of the current training run will be printed to the console.
        """
        super().__init__()
        self._logger_wrappee = logger
        self.enable_console_logging = enable_console_logging

        self.global_step: int = 0
        self.final_step_of_last_run: int = 0
        self.training_run: int = 0
        self.pre_training_metrics: Optional[dict[str, float]] = None

    def __getattr__(self, name: str) -> Any:
        """
        Automatically delegate to the wrapped logger for any attribute or method
        not found in this decorator class.
        """
        return getattr(self._logger_wrappee, name)

    @property
    def name(self) -> Optional[str]:
        return self._logger_wrappee.name

    @property
    def version(self) -> Optional[str | int]:
        return self._logger_wrappee.version

    @rank_zero_only
    def pre_training_log(self, metrics: dict[str, float]) -> None:
        """Log metrics for an entire training run before the trianing run is actually started.
        The metrics will be added to the first row of this training run.

        Args:
            metrics: The metrics to log.
        """
        self.pre_training_metrics = metrics

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        self._logger_wrappee.log_hyperparams(params, *args, **kwargs)

    @rank_zero_only
    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        if step is not None:
            self.global_step = self.final_step_of_last_run + step
        else:
            assert (
                self.global_step is not None
            ), "Step must be set before logging metrics."

        # add custom metric
        updated_metrics = {
            "training_run": self.training_run,
            **metrics,
        }

        if self.pre_training_metrics:
            updated_metrics.update(self.pre_training_metrics)
            self.pre_training_metrics = None

        if self.enable_console_logging:
            sys.stdout.flush()
            sys.stdout.write(f"\rStep: {self.global_step} {str(updated_metrics)}")

        self._logger_wrappee.log_metrics(updated_metrics, self.global_step)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.training_run += 1
        self.final_step_of_last_run = self.global_step
        self.pre_training_metrics = None
        self._logger_wrappee.finalize(status)
