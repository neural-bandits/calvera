import sys
from typing import Any, cast

import torch
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class OnlineBanditLoggerDecorator(Logger):
    """Uses the Decorator pattern to add online bandit functionality to a pytorch lightning logger.

    Will use stdout flush to only print the metrics of the current training run to the console to prevent too many
    prints over many training runs. Allows for logging over multiple training runs.

    Usage:
    ```python
    from lightning.pytorch.loggers import CSVLogger
    from lightning import Trainer
    from calvera.benchmark.logger import OnlineBanditLoggerDecorator
    import pandas as pd

    logger = CSVLogger("logs")
    online_bandit_logger = OnlineBanditLoggerDecorator(logger)

    # allows for logging over multiple training runs
    for batch in data_loader:
        trainer = Trainer(logger=online_bandit_logger)
        trainer.fit(model)

    # now you can load the logs from the CSV file
    df = pd.read_csv(logger.log_dir + "/metrics.csv")
    # Regret is the batch specific metric.
    # It is only added to the first row of a training run of the CSV logger.
    df["regret"].dropna().plot()
    df["training_run"].plot() # idx of how often trainer.fit has been called.
    ```
    """

    def __init__(self, logger: Logger, enable_console_logging: bool = True) -> None:
        """Initialize the OnlineBanditLoggerDecorator.

        Args:
            logger: The logger to decorate / wrap.
            enable_console_logging: If True, only the metrics of the current training run will be printed to the
                console.
        """
        super().__init__()
        self._logger_wrappee = logger
        self.enable_console_logging = enable_console_logging

        self.global_step: int = 0
        self.start_step_of_current_run: int = 0
        self.training_run: int = 0

    def __getattr__(self, name: str) -> Any:
        """Pass all unknown attributes to the wrapped logger.

        Args:
            name: The attribute name.
        """
        return getattr(self._logger_wrappee, name)

    @property
    def root_dir(self) -> str | None:
        """Return the root directory.

        Returns the root directory where all versions of an experiment get saved, or `None` if the logger does not
        save data locally.
        """
        return self._logger_wrappee.root_dir

    @property
    def log_dir(self) -> str | None:
        """Return directory with the current version of the experiment.

        Returns the directory where the current version of the experiment gets saved, or `None` if the logger does not
        save data locally.
        """
        return self._logger_wrappee.log_dir

    @property
    def save_dir(self) -> str | None:
        """Return the root directory, or `None` if the logger does not save data locally."""
        return self._logger_wrappee.save_dir

    @property
    def group_separator(self) -> str:
        """Return the default separator used by the logger to group the data into subfolders."""
        return self._logger_wrappee.group_separator

    @property
    def name(self) -> str | None:
        """Return the experiment name."""
        return self._logger_wrappee.name

    @property
    def version(self) -> str | int | None:
        """Return the experiment version."""
        return self._logger_wrappee.version

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded

        """
        assert self.global_step is not None, "Step must be set before logging metrics."
        step = cast(int, step)

        # add custom metric
        updated_metrics = {
            "training_run": self.training_run,
            **metrics,
        }

        if self.enable_console_logging:
            sys.stdout.flush()
            sys.stdout.write(f"\rStep: {self.global_step} {str(updated_metrics)}")

        self.global_step = self.start_step_of_current_run + step
        self._logger_wrappee.log_metrics(updated_metrics, self.global_step)

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used

        """
        self._logger_wrappee.log_hyperparams(params, *args, **kwargs)

    @rank_zero_only
    def log_graph(self, model: torch.nn.Module, input_array: torch.Tensor | None = None) -> None:
        """Record model graph.

        Args:
            model: the model with an implementation of ``forward``.
            input_array: input passes to `model.forward`
        """
        self._logger_wrappee.log_graph(model, input_array)

    def save(self) -> None:
        """Save log data."""
        return self._logger_wrappee.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize the experiment. This method is called at the end of training.

        Args:
            status: The status of the training.
        """
        self.training_run += 1
        self.start_step_of_current_run = self.global_step + 1
        self._logger_wrappee.finalize(status)

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance

        """
        return self._logger_wrappee.after_save_checkpoint(checkpoint_callback)
