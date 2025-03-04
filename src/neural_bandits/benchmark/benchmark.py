import argparse
import copy
import inspect
import logging
import os
import random
from typing import Any, Callable, Dict, Generic, Optional
import yaml

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.bandits.action_input_type import ActionInputType
from neural_bandits.bandits.linear_ts_bandit import (
    DiagonalPrecApproxLinearTSBandit,
    LinearTSBandit,
)
from neural_bandits.bandits.linear_ucb_bandit import (
    DiagonalPrecApproxLinearUCBBandit,
    LinearUCBBandit,
)
from neural_bandits.bandits.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.bandits.neural_ts_bandit import NeuralTSBandit
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

try:
    from transformers import BertModel
except Exception as e:
    logging.warning("Importing BertModel failed. Make sure transformers is installed and cuda is set up correctly.")
    logging.warning(e)
    pass

bandits: dict[str, type[AbstractBandit[Any]]] = {
    "lin_ucb": LinearUCBBandit,
    "approx_lin_ucb": DiagonalPrecApproxLinearUCBBandit,
    "lin_ts": LinearTSBandit,
    "approx_lin_ts": DiagonalPrecApproxLinearTSBandit,
    "neural_linear": NeuralLinearBandit,
    "neural_ucb": NeuralUCBBandit,
    "neural_ts": NeuralTSBandit,
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
    "epsilon_greedy": lambda params: EpsilonGreedySelector(params.get("epsilon", 0.1), seed=params["seed"]),
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
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, out_size),
    ),
    "large_mlp": lambda in_size, out_size: torch.nn.Sequential(
        torch.nn.Linear(in_size, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, out_size),
    ),
    "deep_mlp": lambda in_size, out_size: torch.nn.Sequential(
        torch.nn.Linear(in_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, out_size),
    ),
    "bert": lambda in_size, out_size: BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2"),
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
    """Benchmark class which trains a bandit on a dataset."""

    @staticmethod
    def from_config(config: dict[str, Any], logger: Optional[Logger] = None) -> "BanditBenchmark[Any]":
        """Initialize a benchmark from a configuration of strings.

        Will instantiate all necessary classes from given strings for the user.

        Args:
            config: A dictionary of training parameters.
                These contain any configuration that is not directly passed to the bandit.
                - bandit: The name of the bandit to use.
                - dataset: The name of the dataset to use.
                - selector: The name of the selector to use.
                    For the specific selectors, additional parameters can be passed:
                    - epsilon: For the EpsilonGreedySelector.
                    - k: Number of actions to select for the TopKSelector (Combinatorial Bandits).
                - data_strategy: The name of the data strategy to initialize the Buffer with.
                - bandit_hparams: A dictionary of bandit hyperparameters.
                    These will be filled and passed to the bandit's constructor.
                - max_steps: The maximum number of steps to train the bandit. This makes sense in combination
                    with AllDataBufferStrategy.
                For neural bandits:
                    - network: The name of the network to use.
                    - data_strategy: The name of the data strategy to use.
                    - gradient_clip_val: The maximum gradient norm for clipping.
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
        bandit_hparams["selector"] = selectors[bandit_hparams.get("selector", "argmax")](training_params)

        assert dataset.context_size > 0, "Dataset must have a fix context size."
        bandit_hparams["n_features"] = dataset.context_size

        if "neural" in bandit_name:
            bandit_hparams["train_batch_size"] = config.get("train_batch_size", 1)

            network_input_size = dataset.context_size
            network_output_size = (
                bandit_hparams["n_embedding_size"]  # in neural linear we create an embedding
                if bandit_name == "neural_linear"
                else 1  # in neural ucb/ts we predict the reward directly
            )
            bandit_hparams["network"] = networks[training_params["network"]](network_input_size, network_output_size)

            data_strategy = data_strategies[training_params["data_strategy"]](training_params)
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
        """Initializes the benchmark.

        Args:
            bandit: A PyTorch Lightning module implementing your bandit.
            dataset: A dataset supplying (contextualized_actions (type: ActionInputType), all_rewards) tuples.
            training_params: Dictionary of parameters for training (e.g. batch_size, etc).
            logger: Optional Lightning logger to record metrics.
        """
        self.bandit = bandit

        self.training_params = training_params
        self.training_params["seed"] = training_params.get("seed", 42)
        pl.seed_everything(training_params["seed"])

        self.logger: Optional[OnlineBanditLoggerDecorator] = (
            OnlineBanditLoggerDecorator(logger, enable_console_logging=False) if logger is not None else None
        )
        self.log_dir = self.logger.log_dir if self.logger is not None and self.logger.log_dir else "logs"

        self.dataset = dataset
        self.dataloader: DataLoader[tuple[ActionInputType, torch.Tensor]] = self._initialize_dataloader(dataset)
        # Wrap the dataloader in an environment to simulate delayed feedback.
        self.environment = BanditBenchmarkEnvironment(self.dataloader)

        self.regrets = np.array([])
        self.rewards = np.array([])

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
        """Runs the benchmark training.

        For each iteration (or for a set number of runs) the bandit:
            - Samples contextualized_actions from the environment,
            - Chooses actions by calling its forward() method,
            - Obtains feedback via environment.get_feedback(chosen_actions),
            - Updates itself (e.g. via trainer.fit), and
            - Optionally computes and logs regret and other metrics.

        Metrics are logged and can be analyzed later, e.g. using the BenchmarkAnalyzer.
        """
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)

        self.regrets = np.array([])
        self.rewards = np.array([])

        train_batch_size = self.training_params.get("train_batch_size", 1)
        # Iterate over one epoch (or limited iterations) from the environment.
        progress_bar = tqdm(iter(self.environment), total=len(self.environment))
        for contextualized_actions in progress_bar:
            chosen_actions = self._predict_actions(contextualized_actions)

            # Get feedback dataset for the chosen actions.
            chosen_contextualized_actions, realized_rewards = self.environment.get_feedback(chosen_actions)

            regrets = self.environment.compute_regret(chosen_actions)
            self.regrets = np.append(self.regrets, regrets)
            self.rewards = np.append(self.rewards, realized_rewards)
            progress_bar.set_postfix(
                regret=regrets.mean().item(),
                reward=realized_rewards.mean().item(),
                avg_reward=self.rewards.mean(),
                avg_regret=self.regrets.mean(),
                acc_regret=self.regrets.sum(),
            )

            optional_kwargs = {}
            bandit_name = self.bandit.__class__.__name__.lower()
            # Only NeuralUCB and NeuralTS can handle gradient clipping. Others will throw an error!
            if "Neural" in bandit_name and "Linear" not in bandit_name:
                optional_kwargs["gradient_clip_val"] = self.training_params.get("gradient_clip_val", None)

            trainer = pl.Trainer(
                max_epochs=1,
                max_steps=self.training_params.get("max_steps", -1),
                logger=self.logger,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                log_every_n_steps=self.training_params.get("log_every_n_steps", 1),
                **optional_kwargs,
            )

            self.bandit.record_feedback(chosen_contextualized_actions, realized_rewards)
            # Train the bandit on the current feedback.
            trainer.fit(self.bandit)

        df = pd.DataFrame(
            {
                "step": np.arange(len(self.regrets)),
                "regret": self.regrets,
                "reward": self.rewards,
            }
        )
        df.to_csv(os.path.join(self.log_dir, "env_metrics.csv"), index=False)

    def _predict_actions(self, contextualized_actions: ActionInputType) -> torch.Tensor:
        """Predicts actions for the given contextualized_actions.

        Predictions are made in batches of size 'forward_batch_size'.
        Therefore, the input batch size must be divisible by 'forward_batch_size'.

        Args:
            contextualized_actions: A tensor of contextualized actions.
        """
        forward_batch_size = self.training_params.get("forward_batch_size", 1)
        contextualized_actions_tensor = (
            contextualized_actions if isinstance(contextualized_actions, torch.Tensor) else contextualized_actions[0]
        )
        batch_size = contextualized_actions_tensor.size(0)

        if batch_size == forward_batch_size:
            # Forward pass: bandit chooses actions.
            chosen_actions, _ = self.bandit.forward(contextualized_actions)
            return chosen_actions
        elif forward_batch_size < batch_size:
            # Split the batch into smaller forward_batch_size chunks. Process each chunk separately.
            # e.g. we always predict for a single sample but then later train on a batch of samples.
            assert (
                batch_size % forward_batch_size == 0
            ), "data loaders batch_size (feedback_delay) must be divisible by forward_batch_size."
            chosen_actions = torch.tensor([], device=contextualized_actions_tensor.device)
            for i in range(0, batch_size, forward_batch_size):
                if isinstance(contextualized_actions, torch.Tensor):
                    actions, _ = self.bandit.forward(contextualized_actions[i : i + forward_batch_size])
                else:
                    actions, _ = self.bandit.forward(
                        tuple(action[i : i + forward_batch_size] for action in contextualized_actions)
                    )
                chosen_actions = torch.cat((chosen_actions, actions), dim=0)

            return chosen_actions
        else:
            raise ValueError("forward_batch_size must be smaller than the data loaders batch_size (feedback_delay).")


class BenchmarkAnalyzer:
    """Separates out the analysis of CSV logs produced during benchmark training.

    This class reads the CSV logs output by the logger (for example, a CSVLogger)
    and produces metrics, plots, or statistics exactly as you need.

    Keeping analysis separate from training improves modularity.
    """

    def __init__(
        self,
        bandit_logs_file: str = "metrics.csv",
        metrics_file: str = "env_metrics.csv",
        suppress_plots: bool = False,
    ) -> None:
        """Initializes the BenchmarkAnalyzer.

        Args:
            bandit_logs_file: Name of the metrics file of the CSV Logger. Default is "metrics.csv".
            metrics_file: Name of the metrics file. Default is "env_metrics.csv".

            suppress_plots: If True, plots will not be automatically shown. Default is False.
        """
        self.bandit_logs_file = bandit_logs_file
        self.env_metrics_file = metrics_file
        self.suppress_plots = suppress_plots
        self.env_metrics_df = pd.DataFrame()
        self.bandit_logs_df = pd.DataFrame()

    def load_metrics(self, log_path: str, bandit: str = "bandit") -> None:
        """Loads the logs from the log path.

        Args:
            log_path: Path to the log data.
            bandit: A name of the bandit. Default is "bandit".
        """
        new_metrics_df = self._load_df(log_path, self.env_metrics_file)

        if new_metrics_df is not None:
            new_metrics_df["bandit"] = bandit

            self.env_metrics_df = pd.concat([self.env_metrics_df, new_metrics_df], ignore_index=True)

        bandit_metrics_df = self._load_df(log_path, self.bandit_logs_file)
        if bandit_metrics_df is not None:
            bandit_metrics_df["bandit"] = bandit

            self.bandit_logs_df = pd.concat([self.bandit_logs_df, bandit_metrics_df], ignore_index=True)

    def _load_df(self, log_path: str, file_name: str) -> Optional[pd.DataFrame]:
        """Loads the logs from the log path.

        Args:
            log_path: Path to the log data.
            file_name: Name of the file to load.

        Returns:
            A pandas DataFrame containing the logs.
        """
        try:
            return pd.read_csv(os.path.join(log_path, file_name))
        except FileNotFoundError:
            logging.warning(f"Could not find metrics file {file_name} in {log_path}.")
            return None

    def plot_accumulated_metric(self, metric_name: str | list[str]) -> None:
        """Plots the accumulated metric over training steps.

        Args:
            metric_name: The name(s) of the metric(s) to plot.
        """
        if isinstance(metric_name, str):
            metric_name = [metric_name]

        if self.env_metrics_df["bandit"].nunique() > 1 and len(metric_name) > 1:
            raise ValueError("Cannot plot multiple metrics for multiple bandits.")

        if any(name not in self.env_metrics_df.columns for name in metric_name):
            print(f"\nNo {metric_name} data found in logs.")
            return

        plt.figure(figsize=(10, 5))
        if self.env_metrics_df["bandit"].nunique() > 1:
            for _, bandit_df in self.env_metrics_df.groupby("bandit"):
                accumulated_metric = bandit_df[metric_name].fillna(0).cumsum()
                plt.plot(accumulated_metric, label=bandit_df["bandit"].iloc[0])
        else:
            accumulated_metric = self.env_metrics_df[metric_name].fillna(0).cumsum()

            for metric in metric_name:
                accumulated_metric = self.env_metrics_df[metric].fillna(0).cumsum()
                plt.plot(accumulated_metric, label=metric)

        plt.xlabel("Step")
        plt.legend()
        plt.title(f"Accumulated {', '.join(metric_name)} over training steps")

        if not self.suppress_plots:
            plt.show()
            plt.savefig(f"acc_{metric_name}.png")

    def plot_average_metric(self, metric_name: str) -> None:
        """Plots the average metric over training steps.

        Args:
            metric_name: The name of the metric to plot.
        """
        if metric_name not in self.env_metrics_df.columns:
            print(f"\nNo {metric_name} data found in logs.")
            return

        # Print average metrics
        valid_idx = self.env_metrics_df[metric_name].dropna().index
        accumulated_metric = self.env_metrics_df.loc[valid_idx, metric_name].cumsum()
        steps = self.env_metrics_df.loc[valid_idx, "step"]

        # Plot how average changes over time
        plt.figure(figsize=(10, 5))
        for metric in metric_name:
            if metric not in self.env_metrics_df.columns:
                print(f"\nNo {metric} data found in logs.")
                continue

            # Print average metrics
            valid_idx = self.env_metrics_df[metric_name].dropna().index
            accumulated_metric = self.env_metrics_df.loc[valid_idx, metric_name].cumsum()
            steps = self.env_metrics_df.loc[valid_idx, "step"]

            # Plot how average changes over time
            plt.plot(accumulated_metric / steps, label=metric_name)

        plt.xlabel("Step")
        plt.legend()
        plt.title(f"Average {', '.join(metric_name)} over training steps")

        if not self.suppress_plots:
            plt.show()
            plt.savefig(f"avg_{metric_name}.png")

    def plot_loss(self) -> None:
        """Plots the loss over training steps."""
        # Generate a plot for the loss
        if "loss" not in self.bandit_logs_df.columns:
            print("\nNo loss data found in logs.")
            return

        plt.figure(figsize=(10, 5))
        for bandit_name, bandit_df in self.bandit_logs_df.groupby("bandit"):
            loss = bandit_df["loss"].dropna()
            if loss.empty:
                print(f"No loss data found in logs for {bandit_name}")
                continue
            plt.plot(loss, label=bandit_name)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss over training steps")

        if not self.suppress_plots:
            plt.show()
            plt.savefig("loss.png")

    def log_metrics(self, bandit: str = "bandit") -> None:
        """Logs the metrics of the bandits run to the console.

        Args:
            bandit: The name of the bandit. Default is "bandit".
        """
        bandit_df = self.env_metrics_df[self.env_metrics_df["bandit"] == bandit]

        if bandit_df.empty:
            raise ValueError(f"No metrics found for {bandit}.")

        logging.info(f"Metrics of {bandit}:")
        logging.info(f"Avg Regret: {bandit_df['regret'].mean()}")
        logging.info(f"Avg Reward: {bandit_df['reward'].mean()}")
        logging.info(f"Accumulated Regret: {bandit_df['regret'].sum()}")
        logging.info(f"Accumulated Reward: {bandit_df['reward'].sum()}")

        # log avg_regret from first 10, 100, 1000, 10000, ... steps
        while True:
            steps = 10 ** len(str(bandit_df["step"].max()))
            avg_regret = bandit_df[bandit_df["step"] < steps]["regret"].mean()
            logging.info(f"Avg Regret (first {steps} steps): {avg_regret}")

            if steps >= bandit_df["step"].max():
                break

        # log from last steps
        while True:
            steps = 10 ** len(str(bandit_df["step"].max()))
            avg_regret = bandit_df[bandit_df["step"] > bandit_df["step"].max() - steps]["regret"].mean()
            logging.info(f"Avg Regret (last {steps} steps): {avg_regret}")

            if steps >= bandit_df["step"].max():
                break


def run(
    config: dict[str, Any],
    suppress_plots: bool = False,
    log_dir="./logs",
) -> None:
    """Runs the benchmark training on a single given bandit.

    Args:
        config: Contains the `bandit`, `dataset`, `bandit_hparams`
            and other parameters necessary for setting up the benchmark and bandit.
        suppress_plots: If True, plots will not be automatically shown. Default is False.
    """
    logger = CSVLogger(log_dir)
    benchmark = BanditBenchmark.from_config(config, logger)
    print(f"Running benchmark for {config['bandit']} on {config['dataset']} dataset.")
    print(f"Config: {config}")
    print(
        f"Dataset {config['dataset']}:"
        f"{len(benchmark.dataset)} samples with {benchmark.dataset.context_size} features"
        f"and {benchmark.dataset.num_actions} actions."
    )
    benchmark.run()

    analyzer = BenchmarkAnalyzer("metrics.csv", "env_metrics.csv", suppress_plots)
    analyzer.load_metrics(logger.log_dir)
    analyzer.plot_accumulated_metric(["reward", "regret"])
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()


def run_comparison(
    config: dict[str, Any],
    suppress_plots: bool = False,
    log_dir="./logs",
):
    """Runs the benchmark training on multiple bandits.

    Args:
        config: Contains the `bandit`, `dataset`, `bandit_hparams`
            and other parameters necessary for setting up the benchmark and bandit.
            The `bandit` must be a list of bandits to compare.
        suppress_plots: If True, plots will not be automatically shown. Default is False.
    """
    assert isinstance(config["bandit"], list), "Bandit must be a list of bandits to compare."

    analyzer = BenchmarkAnalyzer("metrics.csv", "env_metrics.csv", suppress_plots)

    for bandit in config["bandit"]:
        print("==============================================")
        # deep copy the config to avoid overwriting the original
        bandit_config = copy.deepcopy(config)
        bandit_config["bandit"] = bandit

        logger = CSVLogger(os.path.join(log_dir, bandit))
        benchmark = BanditBenchmark.from_config(bandit_config, logger)
        print(f"Running benchmark for {bandit} on {bandit_config['dataset']} dataset.")
        print(f"Config: {bandit_config}")
        print(
            f"Dataset {bandit_config['dataset']}:"
            f"{len(benchmark.dataset)} samples with {benchmark.dataset.context_size} features"
            f"and {benchmark.dataset.num_actions} actions."
        )
        benchmark.run()

        analyzer.load_metrics(logger.log_dir, bandit)

    analyzer.plot_accumulated_metric("reward")
    analyzer.plot_accumulated_metric("regret")
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a bandit benchmark.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    # Load the configuration from the passed yaml file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    log_dir = os.path.dirname(args.config)

    assert "bandit" in config, "Configuration must contain a 'bandit' key."
    if isinstance(config["bandit"], list):
        run_comparison(config, log_dir)
    elif isinstance(config["bandit"], str):
        run(config, log_dir)
    else:
        raise ValueError("Bandit must be a string or a list of strings.")
