import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import lightning as pl
import random
from lightning.pytorch.loggers import Logger, CSVLogger
import torch
from torch.utils.data import DataLoader

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.benchmark.environment import BanditBenchmarkEnvironment
from neural_bandits.benchmark.logger_decorator import OnlineBanditLoggerDecorator
from neural_bandits.benchmark.datasets.abstract_dataset import AbstractDataset

from neural_bandits.bandits.linear_ucb_bandit import LinearUCBBandit
from neural_bandits.bandits.linear_ts_bandit import LinearTSBandit
from neural_bandits.bandits.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.bandits.neural_ucb_bandit import NeuralUCBBandit

from neural_bandits.benchmark.datasets.covertype import CovertypeDataset
from neural_bandits.benchmark.datasets.mnist import MNISTDataset
from neural_bandits.benchmark.datasets.statlog import StatlogDataset
from neural_bandits.benchmark.datasets.wheel import WheelBanditDataset


class BanditBenchmark:

    def __init__(
        self,
        BanditClass: type[AbstractBandit],
        dataset: AbstractDataset,
        training_params: Dict[str, Any],
        bandit_hparams: Dict[str, Any],
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            bandit: A PyTorch Lightning module implementing your bandit.
            dataloader: A DataLoader supplying (contextualized_actions, all_rewards) tuples.
            training_params: Dictionary of parameters for training (e.g. batch_size, etc).
            bandit_hparams: Dictionary of bandit hyperparameters.
            logger: Optional Lightning logger to record metrics.
        """
        bandit_hparams["n_features"] = dataset.context_size * dataset.num_actions
        self.bandit = BanditClass(**bandit_hparams)
        # TODO: how to load hyperparams properly from file, cli, sweep, etc.?
        self.training_params = training_params or {}
        self.logger: Optional[OnlineBanditLoggerDecorator] = (
            OnlineBanditLoggerDecorator(logger) if logger is not None else None
        )

        self.dataloader = self._initialize_dataloader(dataset)
        # Wrap the dataloader in an environment to simulate delayed feedback.
        self.environment = BanditBenchmarkEnvironment(self.dataloader)

    def _initialize_dataloader(
        self, dataset: AbstractDataset
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Add a non-iid data loader as a special setting. Then we need to load a special DataLoader.

        max_samples = self.training_params.get("max_samples", 1e6)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subset_indices = indices[:max_samples]
        subset = torch.utils.data.Subset(dataset, subset_indices)

        return DataLoader(
            subset,
            batch_size=self.training_params.get("feedback_delay", 1),
        )

    def run(self) -> None:
        """
        Runs the benchmark training.

        For each iteration (or for a set number of runs) the bandit:
            - Samples contextualized_actions from the environment,
            - Chooses actions by calling its forward() method,
            - Obtains feedback via environment.get_feedback(chosen_actions),
            - Updates itself (e.g. via trainer.fit), and
            - Optionally computes and logs regret and other metrics.

        Metrics are logged and can be analyzed later, e.g. using the BenchmarkAnalyzer.
        """
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.FATAL
        )

        training_batch_size = self.training_params.get("training_batch_size", 1)
        # Iterate over one epoch (or limited iterations) from the environment.
        for contextualized_actions in self.environment:
            chosen_actions = self._predict_actions(contextualized_actions)
            # Optional: compute and log regret.
            if self.logger is not None:
                regret = self.environment.compute_regret(chosen_actions)
                self.logger.pre_training_log({"regret": regret.sum().item()})

            # Get feedback dataset for the chosen actions.
            feedback_dataset = self.environment.get_feedback(chosen_actions)
            assert training_batch_size <= contextualized_actions.size(
                0
            ), "training_batch_size must be lower than or equal to the data loaders batch_size (feedback_delay)."
            feedback_loader = DataLoader(
                feedback_dataset, batch_size=training_batch_size
            )

            trainer = pl.Trainer(
                max_epochs=1,
                logger=self.logger,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                log_every_n_steps=self.training_params.get("log_every_n_steps", 1),
            )

            # Train the bandit on the current feedback.
            trainer.fit(self.bandit, feedback_loader)

    def _predict_actions(self, contextualized_actions: torch.Tensor) -> torch.Tensor:
        """
        Predicts actions for the given contextualized_actions.
        Predictions are made in batches of size 'forward_batch_size'. Therefore, the input batch size must be divisible by 'forward_batch_size'.

        Args:
            contextualized_actions: A tensor of contextualized actions.
        """
        forward_batch_size = self.training_params.get("forward_batch_size", 1)
        batch_size = contextualized_actions.size(0)

        if forward_batch_size == contextualized_actions.size(0):
            # Forward pass: bandit chooses actions.
            chosen_actions, _ = self.bandit.forward(contextualized_actions)
            return chosen_actions
        elif forward_batch_size < batch_size:
            # Split the batch into smaller forward_batch_size chunks. Process each chunk separately. e.g. we always predict for a single sample but then later train on a batch of samples.
            assert (
                batch_size % forward_batch_size == 0
            ), "data loaders batch_size (feedback_delay) must be divisible by forward_batch_size."
            chosen_actions = torch.tensor([], device=contextualized_actions.device)
            for i in range(0, batch_size, forward_batch_size):
                actions, _ = self.bandit.forward(
                    contextualized_actions[i : i + forward_batch_size]
                )
                chosen_actions = torch.cat((chosen_actions, actions), dim=0)

            return chosen_actions
        else:
            raise ValueError(
                "forward_batch_size must be smaller than the data loaders batch_size (feedback_delay)."
            )


class BenchmarkAnalyzer:
    """
    Separates out the analysis of CSV logs produced during benchmark training.

    This class reads the CSV logs output by the logger (for example, a CSVLogger)
    and produces metrics, plots, or statistics exactly as you need.

    Keeping analysis separate from training improves modularity.
    """

    def __init__(self, log_path: str, metrics_file: str="metrics.csv") -> None:
        """
        Args:
            log_path: Path to the log data.
                Will also be output directory for plots.
                Most likely the log_dir where metrics.csv from your CSV logger is located.
            metrics_file: Name of the metrics file. Default is "metrics.csv".
        """
        self.log_path = log_path
        self.metrics_file = metrics_file
        self.df = self.load_logs()

    def load_logs(self) -> Any:
        # Load CSV data (e.g., using pandas)
        return pd.read_csv(os.path.join(self.log_path, self.metrics_file))

    def plot_accumulated_metric(self, metric_name: str) -> None:
        accumulated_metric = self.df[metric_name].fillna(0).cumsum()

        plt.figure(figsize=(10, 5))
        plt.plot(accumulated_metric)
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.title(f"Accumulated {metric_name} over training steps")
        plt.show()

    def plot_average_metric(self, metric_name: str) -> None:
        # Print average metrics
        valid_idx = self.df[metric_name].dropna().index
        accumulated_metric = self.df.loc[valid_idx, metric_name].cumsum()
        steps = self.df.loc[valid_idx, "step"]

        # Plot how average changes over time
        plt.figure(figsize=(10, 5))
        plt.plot(accumulated_metric / steps)
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.title(f"Average {metric_name} over training steps")
        plt.show()

    def plot_loss(self) -> None:
        # Generate a plot for the loss
        if "loss" not in self.df.columns:
            print("\nNo loss data found in logs.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.df["loss"])
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss over training steps")


bandits: dict[str, type[AbstractBandit]] = {
    "lin_ucb": LinearUCBBandit,
    "lin_ts": LinearTSBandit,
    "neural_linear": NeuralLinearBandit,
    "neural_ucb": NeuralUCBBandit,
}

datasets = {
    "covertype": CovertypeDataset,
    "mnist": MNISTDataset,
    "statlog": StatlogDataset,
    "wheel": WheelBanditDataset,
}


def run(
    bandit_name: str,
    dataset_name: str,
    training_params: dict[str, Any] = {},
    bandit_hparams: dict[str, Any] = {},
) -> None:
    pl.seed_everything(42)

    Bandit = bandits[bandit_name]
    dataset = datasets[dataset_name]()

    logger = CSVLogger("logs/")
    benchmark = BanditBenchmark(
        Bandit, dataset, training_params, bandit_hparams, logger
    )
    benchmark.run()

    analyzer = BenchmarkAnalyzer(logger.log_dir, "metrics.csv")
    analyzer.plot_accumulated_metric("reward")
    analyzer.plot_accumulated_metric("regret")
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()


if __name__ == "__main__":
    run(
        "lin_ucb",
        "covertype",
        {
            "max_samples": 5000,
            "batch_size": 1,
            "forward_batch_size": 1,
            "feedback_delay": 1,
        },  # training parameters
        {
            "alpha": 1.0,
        },  # bandit hyperparameters
    )
