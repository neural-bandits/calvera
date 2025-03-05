import argparse
import copy
import inspect
import logging
import os
import random
from collections.abc import Callable
from typing import Any, Generic

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from calvera.bandits.abstract_bandit import AbstractBandit
from calvera.bandits.action_input_type import ActionInputType
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
from calvera.benchmark.datasets.abstract_dataset import AbstractDataset
from calvera.benchmark.datasets.covertype import CovertypeDataset
from calvera.benchmark.datasets.imdb_reviews import ImdbMovieReviews
from calvera.benchmark.datasets.mnist import MNISTDataset
from calvera.benchmark.datasets.movie_lens import MovieLensDataset
from calvera.benchmark.datasets.statlog import StatlogDataset
from calvera.benchmark.datasets.synthetic import (
    CubicSyntheticDataset,
    LinearCombinationSyntheticDataset,
    LinearSyntheticDataset,
    SinSyntheticDataset,
)
from calvera.benchmark.datasets.wheel import WheelBanditDataset
from calvera.benchmark.environment import BanditBenchmarkEnvironment
from calvera.benchmark.logger_decorator import OnlineBanditLoggerDecorator
from calvera.utils.data_storage import (
    AllDataBufferStrategy,
    DataBufferStrategy,
    InMemoryDataBuffer,
    SlidingWindowBufferStrategy,
)
from calvera.utils.selectors import (
    AbstractSelector,
    ArgMaxSelector,
    EpsilonGreedySelector,
    TopKSelector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from transformers import BertModel
except Exception as e:
    logger.warning("Importing BertModel failed. Make sure transformers is installed and cuda is set up correctly.")
    logger.warning(e)
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
    "synthetic_linear": LinearSyntheticDataset,
    "synthetic_cubic": CubicSyntheticDataset,
    "synthetic_sin": SinSyntheticDataset,
    "synthetic_linear_comb": LinearCombinationSyntheticDataset,
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
    def from_config(config: dict[str, Any], logger: Logger | None = None) -> "BanditBenchmark[Any]":
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
        DatasetClass = datasets[config["dataset"]]
        dataset = DatasetClass(**filter_kwargs(DatasetClass, config.get("dataset_hparams", {})))

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
            bandit_hparams["buffer"] = InMemoryDataBuffer[torch.Tensor](
                data_strategy,
                max_size=training_params.get("max_buffer_size", None),
            )

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
        training_params: dict[str, Any],
        logger: Logger | None = None,
    ) -> None:
        """Initializes the benchmark.

        Args:
            bandit: A PyTorch Lightning module implementing your bandit.
            dataset: A dataset supplying (contextualized_actions (type: ActionInputType), all_rewards) tuples.
            training_params: Dictionary of parameters for training (e.g. batch_size, etc).
            logger: Optional Lightning logger to record metrics.
        """
        self.bandit = bandit
        self.device = training_params.get("device", "cpu")
        bandit.to(self.device)
        print(f"Bandit moved to device: {self.device}")

        self.training_params = training_params
        self.training_params["seed"] = self.training_params.get("seed", 42)
        pl.seed_everything(self.training_params["seed"])

        self.logger: OnlineBanditLoggerDecorator | None = (
            OnlineBanditLoggerDecorator(logger, enable_console_logging=False) if logger is not None else None
        )
        self.log_dir = self.logger.log_dir if self.logger is not None and self.logger.log_dir else "logs"

        self.dataset = dataset
        self.dataloader: DataLoader[tuple[ActionInputType, torch.Tensor]] = self._initialize_dataloader(dataset)
        # Wrap the dataloader in an environment to simulate delayed feedback.
        self.environment = BanditBenchmarkEnvironment(self.dataloader, self.device)

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

        # Iterate over one epoch (or limited iterations) from the environment.
        progress_bar = tqdm(iter(self.environment), total=len(self.environment))
        for contextualized_actions in progress_bar:
            chosen_actions = self._predict_actions(contextualized_actions)

            # Get feedback dataset for the chosen actions.
            chosen_contextualized_actions, realized_rewards = self.environment.get_feedback(chosen_actions)

            regrets = self.environment.compute_regret(chosen_actions)
            self.regrets = np.append(self.regrets, regrets.to(self.regrets.device))
            self.rewards = np.append(self.rewards, realized_rewards.to(self.rewards.device))
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
                accelerator=self.device,
                strategy="fsdp",
                **optional_kwargs,
            )

            self.bandit.record_feedback(chosen_contextualized_actions, realized_rewards)
            # Train the bandit on the current feedback.
            trainer.fit(self.bandit)
            trainer.save_checkpoint(os.path.join(self.log_dir, "checkpoint.ckpt"))

            # Unfortunately, after each training run the model is moved to the CPU by lightning.
            # We need to move it back to the device.
            self.bandit = self.bandit.to(self.device)

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
        log_dir: str = "logs",
        results_dir: str = "results",
        bandit_logs_file: str = "metrics.csv",
        metrics_file: str = "env_metrics.csv",
        save_plots: bool = False,
        suppress_plots: bool = False,
    ) -> None:
        """Initializes the BenchmarkAnalyzer.

        Args:
            log_dir: Directory where the logs are stored/outputted to. Default is "logs".
            results_dir: Subdirectory of log_dir where the results are outputted to. Default is "results".
            bandit_logs_file: Name of the metrics file of the CSV Logger. Default is "metrics.csv".
            metrics_file: Name of the metrics file. Default is "env_metrics.csv".
            save_plots: If True, plots will be saved to the results directory. Default is False.
            suppress_plots: If True, plots will not be automatically shown. Default is False.
        """
        if not save_plots and suppress_plots:
            logging.warning("Suppressing plots and not saving them. Results will not be visible.")

        self.log_dir = log_dir
        self.results_dir = os.path.join(log_dir, results_dir)
        self.bandit_logs_file = bandit_logs_file
        self.env_metrics_file = metrics_file
        self.suppress_plots = suppress_plots
        self.save_plots = save_plots

        self.env_metrics_df = pd.DataFrame()
        self.bandit_logs_df = pd.DataFrame()

    def load_metrics(self, log_path: str | None = None, bandit: str = "bandit") -> None:
        """Loads the logs from the log path.

        Args:
            log_path: Path to the log data.
            bandit: A name of the bandit. Default is "bandit".
        """
        if log_path is None:
            log_path = self.log_dir
        new_metrics_df = self._load_df(log_path, self.env_metrics_file)

        if new_metrics_df is not None:
            new_metrics_df["bandit"] = bandit

            self.env_metrics_df = pd.concat([self.env_metrics_df, new_metrics_df], ignore_index=True)

        bandit_metrics_df = self._load_df(log_path, self.bandit_logs_file)
        if bandit_metrics_df is not None:
            bandit_metrics_df["bandit"] = bandit

            self.bandit_logs_df = pd.concat([self.bandit_logs_df, bandit_metrics_df], ignore_index=True)

    def _load_df(self, log_path: str, file_name: str) -> pd.DataFrame | None:
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
            logger.warning(f"Could not find metrics file {file_name} in {log_path}.")
            return None

    def plot_accumulated_metric(self, metric_name: str | list[str]) -> None:
        """Plots the accumulated metric over training steps.

        If several metrics are passed they are all plotted in the same plot.
        If the analyzer has seen data from several bandits they are plotted in the same plot.

        Args:
            metric_name: The name(s) of the metric(s) to plot.
        """
        if isinstance(metric_name, str):
            metric_name = [metric_name]

        if any(name not in self.env_metrics_df.columns for name in metric_name):
            logger.warning(f"\One of {','.join(metric_name)} data not found in logs.")
            return

        if self.env_metrics_df["bandit"].nunique() > 1 and len(metric_name) > 1:
            raise ValueError("Cannot plot multiple metrics for multiple bandits.")

        plt.figure(figsize=(10, 5))
        if self.env_metrics_df["bandit"].nunique() > 1:
            for bandit_name, bandit_df in self.env_metrics_df.groupby("bandit"):
                accumulated_metric = bandit_df[metric_name[0]].fillna(0).cumsum()
                plt.plot(bandit_df["step"], accumulated_metric, label=bandit_name)
                plt.ylabel(f"Accumulated {metric_name[0]}")
        else:
            for metric in metric_name:
                accumulated_metric = self.env_metrics_df[metric].fillna(0).cumsum()
                plt.plot(self.env_metrics_df["step"], accumulated_metric, label=metric)

        plt.xlabel("Step")
        plt.legend()
        plt.title(f"Accumulated {', '.join(metric_name)} over training steps")

        if self.save_plots:
            path = os.path.join(self.results_dir, f"acc_{'_'.join(metric_name)}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        if not self.suppress_plots:
            plt.show()

    def plot_average_metric(self, metric_name: str) -> None:
        """Plots the average metric over training steps.

        Args:
            metric_name: The name of the metric to plot.
        """
        if metric_name not in self.env_metrics_df.columns:
            logger.warning(f"\nNo {metric_name} data found in logs.")
            return

        # Plot how average changes over time
        plt.figure(figsize=(10, 5))

        for bandit_name, bandit_df in self.env_metrics_df.groupby("bandit"):
            valid_idx = bandit_df[metric_name].dropna().index
            accumulated_metric = bandit_df.loc[valid_idx, metric_name].cumsum()
            steps = bandit_df.loc[valid_idx, "step"]

            # Plot how average changes over time
            plt.plot(steps, accumulated_metric / (steps + 1), label=bandit_name)

        plt.ylabel(f"Average {metric_name}")
        plt.xlabel("Step")
        plt.legend()
        plt.title(f"Average {metric_name} over training steps")

        if self.save_plots:
            path = os.path.join(self.results_dir, f"avg_{metric_name}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        if not self.suppress_plots:
            plt.show()

    def plot_loss(self) -> None:
        """Plots the loss over training steps."""
        # Generate a plot for the loss
        if "loss" not in self.bandit_logs_df.columns:
            logger.warning("\nNo loss data found in logs.")
            return

        plt.figure(figsize=(10, 5))
        for bandit_name, bandit_df in self.bandit_logs_df.groupby("bandit"):
            loss = bandit_df["loss"].dropna()
            if loss.empty:
                logger.warning(f"No loss data found in logs for {bandit_name}")
                continue
            plt.plot(bandit_df["step"], loss, label=bandit_name)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over training steps")

        if self.save_plots:
            path = os.path.join(self.results_dir, "loss.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        if not self.suppress_plots:
            plt.show()

    def log_metrics(self, bandit: str = "bandit") -> None:
        """Logs the metrics of the bandits run to the console.

        Args:
            bandit: The name of the bandit. Default is "bandit".
        """
        bandit_df = self.env_metrics_df[self.env_metrics_df["bandit"] == bandit]

        if bandit_df.empty:
            raise ValueError(f"No metrics found for {bandit}.")

        str = f"Metrics of {bandit}:\n"
        str += f"Avg Regret: {bandit_df['regret'].mean()}\n"
        str += f"Avg Reward: {bandit_df['reward'].mean()}\n"
        str += f"Accumulated Regret: {bandit_df['regret'].sum()}\n"
        str += f"Accumulated Reward: {bandit_df['reward'].sum()}\n"

        # log avg_regret from first 10, 100, 1000, 10000, ... steps
        i = 1
        while True:
            steps = 10**i
            if steps >= bandit_df["step"].max():
                break

            avg_regret = bandit_df[bandit_df["step"] < steps]["regret"].mean()
            str += f"Avg Regret (first {steps} steps): {avg_regret}\n"

            i += 1

        # log from last steps
        i = 1
        while True:
            steps = 10**i
            if steps >= bandit_df["step"].max():
                break

            avg_regret = bandit_df[bandit_df["step"] > bandit_df["step"].max() - steps]["regret"].mean()
            str += f"Avg Regret (last {steps} steps): {avg_regret}\n"

            i += 1

        print(str)

        # Write to file
        if self.save_plots:
            path = os.path.join(self.log_dir, bandit, "metrics.txt")
            with open(path, "w+") as f:
                f.write(str)


def run(
    config: dict[str, Any],
    log_dir: str = "logs",
    save_plots: bool = False,
    suppress_plots: bool = False,
) -> None:
    """Runs the benchmark training on a single given bandit.

    Args:
        config: Contains the `bandit`, `dataset`, `bandit_hparams`
            and other parameters necessary for setting up the benchmark and bandit.
        log_dir: Directory where the logs are stored/outputted to. Default is "logs".
        save_plots: If True, plots be saved on disk. Default is False.
        suppress_plots: If True, plots will not be automatically shown. Default is False.
    """
    logger = CSVLogger(log_dir)
    benchmark = BanditBenchmark.from_config(config, logger)
    print(f"Running benchmark for {config['bandit']} on {config['dataset']} dataset.")
    print(f"Config: {config}")
    print(
        f"Dataset {config['dataset']}: \n"
        f"{len(benchmark.dataset)} samples with {benchmark.dataset.context_size} features "
        f"and {benchmark.dataset.num_actions} actions."
    )
    benchmark.run()

    analyzer = BenchmarkAnalyzer(log_dir, "results", "metrics.csv", "env_metrics.csv", save_plots, suppress_plots)
    analyzer.load_metrics()
    analyzer.log_metrics()
    analyzer.plot_accumulated_metric(["reward", "regret"])
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()


def run_comparison(
    config: dict[str, Any],
    log_dir: str = "logs",
    save_plots: bool = False,
    suppress_plots: bool = False,
) -> None:
    """Runs the benchmark training on multiple bandits.

    Args:
        config: Contains the `bandit`, `dataset`, `bandit_hparams`
            and other parameters necessary for setting up the benchmark and bandit.
            The `bandit` must be a list of bandits to compare.
        log_dir: Directory where the logs are stored/outputted to. Default is "logs".
        save_plots: If True, plots be saved on disk. Default is False.
        suppress_plots: If True, plots will not be automatically shown. Default is False.
    """
    assert isinstance(config["bandit"], list), "Bandit must be a list of bandits to compare."

    analyzer = BenchmarkAnalyzer(log_dir, "results", "metrics.csv", "env_metrics.csv", save_plots, suppress_plots)

    for bandit in config["bandit"]:
        # try:
        print("==============================================")
        # deep copy the config to avoid overwriting the original
        bandit_config = copy.deepcopy(config)
        bandit_config["bandit"] = bandit

        csv_logger = CSVLogger(os.path.join(log_dir, bandit), version=0)
        benchmark = BanditBenchmark.from_config(bandit_config, csv_logger)
        print(f"Running benchmark for {bandit} on {bandit_config['dataset']} dataset.")
        print(f"Config: {bandit_config}")
        print(
            f"Dataset {bandit_config['dataset']}:"
            f"{len(benchmark.dataset)} samples with {benchmark.dataset.context_size} features"
            f"and {benchmark.dataset.num_actions} actions."
        )
        benchmark.run()

        analyzer.load_metrics(csv_logger.log_dir, bandit)
        analyzer.log_metrics(bandit)
        # except Exception as e:
            # print(f"Failed to run benchmark for {bandit}. It might not be part of the final analysis.")
            # print(e)

    for bandit in config.get("load_previous_result", []):
        print("==============================================")
        print(f"Loading previous result for {bandit}.")
        try:
            analyzer.load_metrics(os.path.join(log_dir, bandit), bandit)
            analyzer.log_metrics(bandit)
        except Exception as e:
            print(f"Failed to load previous result for {bandit}.")
            print(e)

    analyzer.plot_accumulated_metric("reward")
    analyzer.plot_accumulated_metric("regret")
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()

    if suppress_plots:
        print("Plots were suppressed. Set suppress_plots to False to show plots.")
    if save_plots:
        print(f"Plots were saved to {analyzer.results_dir}. Set save_plots to False to suppress saving.")
    else:
        print("Plots were not saved. Set save_plots to True to save plots.")


def run_from_yaml(
    config_path: str,
    save_plots: bool = False,
    suppress_plots: bool = False,
) -> None:
    """Runs the benchmark training from a yaml file.

    Args:
        config_path: Path to the configuration file.
        save_plots: If True, plots will be saved to the results directory. Default is False.
        suppress_plots: If True, plots will not be automatically shown. Default is False.
    """
    log_dir = os.path.dirname(config_path)

    # Load the configuration from the passed yaml file
    with open(config_path) as file:
        config = yaml.safe_load(file)

    assert "bandit" in config, "Configuration must contain a 'bandit' key."
    if isinstance(config["bandit"], list):
        run_comparison(config, log_dir, save_plots, suppress_plots)
    elif isinstance(config["bandit"], str):
        run(config, log_dir, save_plots, suppress_plots)
    else:
        raise ValueError("Bandit must be a string or a list of strings.")


"""Runs the benchmark training from the command line.
    
    Args:
        config: Path to the configuration file.

    Usage:
        ``python src/neural_bandits/benchmark/benchmark.py experiments/datasets/covertype.yaml``
"""
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a bandit benchmark.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    run_from_yaml(args.config, save_plots=True, suppress_plots=True)
