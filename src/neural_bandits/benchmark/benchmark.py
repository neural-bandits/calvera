import inspect
import logging
import os
import random
from typing import Any, Callable, Dict, Generic, Optional

import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from transformers import BertModel
except Exception as e:
    logging.warning(
        "Importing BertModel failed. Make sure transformers is installed and cuda is set up correctly."
    )
    logging.warning(e)
    pass

from neural_bandits.bandits.abstract_bandit import AbstractBandit, ActionInputType
from neural_bandits.bandits.linear_ts_bandit import (
    DiagonalPrecApproxLinearTSBandit,
    LinearTSBandit,
)
from neural_bandits.bandits.linear_ucb_bandit import (
    DiagonalPrecApproxLinearUCBBandit,
    LinearUCBBandit,
)
from neural_bandits.bandits.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.bandits.neural_ucb_bandit import NeuralUCBBandit
from neural_bandits.benchmark.datasets.abstract_dataset import AbstractDataset
from neural_bandits.benchmark.datasets.covertype import CovertypeDataset
from neural_bandits.benchmark.datasets.imdb_reviews import ImdbMovieReviews
from neural_bandits.benchmark.datasets.mnist import MNISTDataset
from neural_bandits.benchmark.datasets.movie_lens import MovieLensDataset
from neural_bandits.benchmark.datasets.statlog import StatlogDataset
from neural_bandits.benchmark.datasets.wheel import WheelBanditDataset
from neural_bandits.benchmark.environment import BanditBenchmarkEnvironment
from neural_bandits.benchmark.logger_decorator import OnlineBanditLoggerDecorator
from neural_bandits.utils.data_storage import (
    AllDataBufferStrategy,
    DataBufferStrategy,
    InMemoryDataBuffer,
    SlidingWindowBufferStrategy,
)
from neural_bandits.utils.selectors import (
    AbstractSelector,
    ArgMaxSelector,
    EpsilonGreedySelector,
    TopKSelector,
)

bandits: dict[str, type[AbstractBandit[Any]]] = {
    "lin_ucb": LinearUCBBandit,
    "approx_lin_ucb": DiagonalPrecApproxLinearUCBBandit,
    "lin_ts": LinearTSBandit,
    "approx_lin_ts": DiagonalPrecApproxLinearTSBandit,
    "neural_linear": NeuralLinearBandit,
    "neural_ucb": NeuralUCBBandit,
}

datasets: dict[str, type[AbstractDataset[Any]]] = {
    "covertype": CovertypeDataset,
    "mnist": MNISTDataset,
    "statlog": StatlogDataset,
    "wheel": WheelBanditDataset,
    "imdb": ImdbMovieReviews,
    "movielens": MovieLensDataset,
}

data_strategies: dict[str, Callable[[dict[str, Any]], DataBufferStrategy]] = {
    "all": lambda params: AllDataBufferStrategy(),
    "sliding_window": lambda params: SlidingWindowBufferStrategy(
        params.get("window_size", params.get("train_batch_size", 1))
    ),
}
selectors: dict[str, Callable[[dict[str, Any]], AbstractSelector]] = {
    "argmax": lambda params: ArgMaxSelector(),
    "epsilon_greedy": lambda params: EpsilonGreedySelector(
        params.get("epsilon", 0.1), seed=params["seed"]
    ),
    "top_k": lambda params: TopKSelector(params.get("k", 1)),
}

networks: dict[str, Callable[[int, int], torch.nn.Module]] = {
    "none": lambda in_size, out_size: torch.nn.Identity(),
    "linear": lambda in_size, out_size: torch.nn.Linear(in_size, out_size),
    "tiny_mlp": lambda in_size, out_size: torch.nn.Sequential(
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, out_size),
    ),
    "small_mlp": lambda in_size, out_size: torch.nn.Sequential(
        torch.nn.Linear(in_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, out_size),
    ),
    "large_mlp": lambda in_size, out_size: torch.nn.Sequential(
        torch.nn.Linear(in_size, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, out_size),
    ),
    "deep_mlp": lambda in_size, out_size: torch.nn.Sequential(
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, out_size),
    ),
    "bert": lambda in_size, out_size: BertModel.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2"
    ),
}


def filter_kwargs(cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only include parameters accepted by cls's constructor.

    Args:
        cls: The class to filter the kwargs for.
        kwargs: The kwargs to filter.

    Returns:
        A dictionary of kwargs that are accepted by cls's constructor.
    """
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


class BanditBenchmark(Generic[ActionInputType]):
    """
    Benchmark class which trains a bandit on a dataset.
    """

    @staticmethod
    def from_config(
        config: dict[str, Any], logger: Optional[Logger] = None
    ) -> "BanditBenchmark[Any]":
        """Initialize a benchmark from a configuration of strings. Will instantiate all necessary classes from given strings for the user.

        Args:
            config: A dictionary of training parameters. These contain any configuration that is not directly passed to the bandit.
                - bandit: The name of the bandit to use.
                - dataset: The name of the dataset to use.
                - bandit_hparams: A dictionary of bandit hyperparameters. These will be filled and passed to the bandit's constructor.
                - selector: The name of the selector to use. For the specific selectors, additional parameters can be passed:
                    - epsilon: For the EpsilonGreedySelector.
                    - k: Number of actions to select for the TopKSelector (Combinatorial Bandits).

                For neural bandits:
                    - network: The name of the network to use.
                    - data_strategy: The name of the data strategy to use.
                    For neural linear:
                        - n_embedding_size: The size of the embedding layer.


            logger: Optional Lightning logger to record metrics.

        Returns:
            An instantiated BanditBenchmark instance.
        """
        bandit_name = config["bandit"]
        dataset = datasets[config["dataset"]]()

        training_params = config
        bandit_hparams: dict[str, Any] = config.get("bandit_hparams", {})
        bandit_hparams["selector"] = selectors[
            bandit_hparams.get("selector", "argmax")
        ](training_params)

        assert dataset.context_size > 0, "Dataset must have a fix context size."
        bandit_hparams["n_features"] = dataset.context_size

        if "neural" in bandit_name:
            bandit_hparams["train_batch_size"] = config.get("train_batch_size", 1)

            network_input_size = dataset.context_size
            network_output_size = (
                bandit_hparams[
                    "n_embedding_size"
                ]  # in neural linear we create an embedding
                if bandit_name == "neural_linear"
                else 1  # in neural ucb/ts we predict the reward directly
            )
            bandit_hparams["network"] = networks[training_params["network"]](
                network_input_size, network_output_size
            )

            data_strategy = data_strategies[training_params["data_strategy"]](
                training_params
            )
            bandit_hparams["buffer"] = InMemoryDataBuffer[torch.Tensor](data_strategy)

        BanditClass = bandits[bandit_name]
        bandit = BanditClass(**filter_kwargs(BanditClass, bandit_hparams))

        return BanditBenchmark(
            bandit,
            dataset,
            training_params,
            logger,
        )

    def __init__(
        self,
        bandit: AbstractBandit[ActionInputType],
        dataset: AbstractDataset[ActionInputType],
        training_params: Dict[str, Any],
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            bandit: A PyTorch Lightning module implementing your bandit.
            dataloader: A DataLoader supplying (contextualized_actions, all_rewards) tuples.
            training_params: Dictionary of parameters for training (e.g. batch_size, etc).
            bandit_hparams: Dictionary of bandit hyperparameters. These will be passed to the bandit's constructor.
            logger: Optional Lightning logger to record metrics.
        """
        self.bandit = bandit

        self.training_params = training_params
        self.training_params["seed"] = training_params.get("seed", 42)
        pl.seed_everything(training_params["seed"])

        self.logger: Optional[OnlineBanditLoggerDecorator] = (
            OnlineBanditLoggerDecorator(logger) if logger is not None else None
        )

        self.dataset = dataset
        self.dataloader: DataLoader[tuple[ActionInputType, torch.Tensor]] = (
            self._initialize_dataloader(dataset)
        )
        # Wrap the dataloader in an environment to simulate delayed feedback.
        self.environment = BanditBenchmarkEnvironment(self.dataloader)

    def _initialize_dataloader(
        self, dataset: AbstractDataset[ActionInputType]
    ) -> DataLoader[tuple[ActionInputType, torch.Tensor]]:
        subset: Dataset[tuple[ActionInputType, torch.Tensor]] = dataset
        if "max_samples" in self.training_params:
            max_samples = self.training_params["max_samples"]
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            subset_indices = indices[:max_samples]
            subset = Subset(dataset, subset_indices)

        # TODO: Add a non-iid data loader as a special setting. Then we need to load a special DataLoader.
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

        train_batch_size = self.training_params.get("train_batch_size", 1)
        # Iterate over one epoch (or limited iterations) from the environment.
        for contextualized_actions in self.environment:
            chosen_actions = self._predict_actions(contextualized_actions)
            # Optional: compute and log regret.
            if self.logger is not None:
                regret = self.environment.compute_regret(chosen_actions)
                self.logger.pre_training_log({"regret": regret})

            # Get feedback dataset for the chosen actions.
            feedback_dataset = self.environment.get_feedback(chosen_actions)
            assert train_batch_size <= chosen_actions.size(
                0
            ), "train_batch_size must be lower than or equal to the data loaders batch_size (feedback_delay)."
            feedback_loader = DataLoader(feedback_dataset, batch_size=train_batch_size)

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

    def _predict_actions(self, contextualized_actions: ActionInputType) -> torch.Tensor:
        """
        Predicts actions for the given contextualized_actions.
        Predictions are made in batches of size 'forward_batch_size'. Therefore, the input batch size must be divisible by 'forward_batch_size'.

        Args:
            contextualized_actions: A tensor of contextualized actions.
        """
        forward_batch_size = self.training_params.get("forward_batch_size", 1)
        contextualized_actions_tensor = (
            contextualized_actions
            if isinstance(contextualized_actions, torch.Tensor)
            else contextualized_actions[0]
        )
        batch_size = contextualized_actions_tensor.size(0)

        if batch_size == forward_batch_size:
            # Forward pass: bandit chooses actions.
            chosen_actions, _ = self.bandit.forward(contextualized_actions)
            return chosen_actions
        elif forward_batch_size < batch_size:
            # Split the batch into smaller forward_batch_size chunks. Process each chunk separately. e.g. we always predict for a single sample but then later train on a batch of samples.
            assert (
                batch_size % forward_batch_size == 0
            ), "data loaders batch_size (feedback_delay) must be divisible by forward_batch_size."
            chosen_actions = torch.tensor(
                [], device=contextualized_actions_tensor.device
            )
            for i in range(0, batch_size, forward_batch_size):
                if isinstance(contextualized_actions, torch.Tensor):
                    actions, _ = self.bandit.forward(
                        contextualized_actions[i : i + forward_batch_size]
                    )
                else:
                    actions, _ = self.bandit.forward(
                        tuple(
                            action[i : i + forward_batch_size]
                            for action in contextualized_actions
                        )
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

    def __init__(
        self,
        log_path: str,
        metrics_file: str = "metrics.csv",
        suppress_plots: bool = False,
    ) -> None:
        """
        Args:
            log_path: Path to the log data.
                Will also be output directory for plots.
                Most likely the log_dir where metrics.csv from your CSV logger is located.
            metrics_file: Name of the metrics file. Default is "metrics.csv".
            suppress_plots: If True, plots will not be automatically shown. Default is False.
        """
        self.log_path = log_path
        self.metrics_file = metrics_file
        self.suppress_plots = suppress_plots
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

        if not self.suppress_plots:
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

        if not self.suppress_plots:
            plt.show()

    def plot_loss(self) -> None:
        # Generate a plot for the loss
        if "loss" not in self.df.columns:
            print("\nNo loss data found in logs.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.df["loss"].dropna())
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss over training steps")

        if not self.suppress_plots:
            plt.show()


def run(
    config: dict[str, Any] = {},
    suppress_plots: bool = False,
) -> None:
    logger = CSVLogger("logs/")
    benchmark = BanditBenchmark.from_config(config, logger)
    print(f"Running benchmark for {config['bandit']} on {config['dataset']} dataset.")
    print(f"Config: {config}")
    print(
        f"Dataset {config['dataset']}: {len(benchmark.dataset)} samples with {benchmark.dataset.context_size} features and {benchmark.dataset.num_actions} actions."
    )
    benchmark.run()

    analyzer = BenchmarkAnalyzer(logger.log_dir, "metrics.csv", suppress_plots)
    analyzer.plot_accumulated_metric("reward")
    analyzer.plot_accumulated_metric("regret")
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()


if __name__ == "__main__":
    run(
        {
            "bandit": "lin_ucb",
            "dataset": "covertype",
            "max_samples": 5000,
            "feedback_delay": 1,
            "train_batch_size": 1,
            "forward_batch_size": 1,
            "bandit_hparams": {
                "alpha": 1.0,
            },
        }
    )
