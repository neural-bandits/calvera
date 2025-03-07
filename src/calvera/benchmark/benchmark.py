import argparse
import copy
import inspect
import logging
import os
import random
from collections.abc import Callable
from functools import partial, reduce
from typing import Any, Generic

import lightning as pl
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification

from calvera.bandits.abstract_bandit import AbstractBandit, DummyBandit
from calvera.bandits.action_input_type import ActionInputType
from calvera.bandits.linear_ts_bandit import DiagonalPrecApproxLinearTSBandit, LinearTSBandit
from calvera.bandits.linear_ucb_bandit import DiagonalPrecApproxLinearUCBBandit, LinearUCBBandit
from calvera.bandits.neural_linear_bandit import NeuralLinearBandit
from calvera.bandits.neural_ts_bandit import NeuralTSBandit
from calvera.bandits.neural_ucb_bandit import NeuralUCBBandit
from calvera.benchmark.analyzer import BenchmarkAnalyzer
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
    QuadraticSyntheticDataset,
    SinSyntheticDataset,
)
from calvera.benchmark.datasets.synthetic_combinatorial import SyntheticCombinatorialDataset
from calvera.benchmark.datasets.tiny_imagenet import TinyImageNetDataset
from calvera.benchmark.datasets.wheel import WheelBanditDataset
from calvera.benchmark.environment import BanditBenchmarkEnvironment
from calvera.benchmark.logger_decorator import OnlineBanditLoggerDecorator
from calvera.benchmark.network_wrappers import BertWrapper, ResNetWrapper
from calvera.utils.data_sampler import SortedDataSampler
from calvera.utils.data_storage import (
    AllDataBufferStrategy,
    DataBufferStrategy,
    InMemoryDataBuffer,
    ListDataBuffer,
    SlidingWindowBufferStrategy,
)
from calvera.utils.selectors import (
    AbstractSelector,
    ArgMaxSelector,
    EpsilonGreedySelector,
    RandomSelector,
    TopKSelector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    "random": DummyBandit,
}

datasets: dict[str, type[AbstractDataset[Any]]] = {
    "covertype": CovertypeDataset,
    "mnist": MNISTDataset,
    "statlog": StatlogDataset,
    "wheel": WheelBanditDataset,
    "synthetic_linear": LinearSyntheticDataset,
    "synthetic_quadratic": QuadraticSyntheticDataset,
    "synthetic_cubic": CubicSyntheticDataset,
    "synthetic_sin": SinSyntheticDataset,
    "synthetic_linear_comb": LinearCombinationSyntheticDataset,  # not combinatorial!
    "synthetic_combinatorial": SyntheticCombinatorialDataset,
    "imdb": ImdbMovieReviews,
    "movielens": MovieLensDataset,
    "tiny_imagenet": TinyImageNetDataset,
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
    "random": lambda params: RandomSelector(params.get("k", 1), seed=params["seed"]),
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
    "bert": lambda in_size, out_size: BertWrapper(
        BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", output_hidden_states=True)
    ),
    "resnet18": lambda in_size, out_size: ResNetWrapper(
        network=timm.create_model(
            "resnet18.a1_in1k",
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
    ),
}


def transformers_collate(batch: Any, data_collator: DataCollatorForTokenClassification) -> Any:
    """Custom collate function for the DataLoader.

    Args:
        batch: The batch to collate.
        data_collator: The data collator to use.

    Returns:
        The collated batch.
    """
    examples = []
    for item in batch:
        inputs = item[0]
        example = {
            "input_ids": inputs[0],
            "attention_mask": inputs[1],
            "token_type_ids": inputs[2],
        }
        examples.append(example)

    # Let the data collator process the list of individual examples.
    context = data_collator(examples)
    input_ids = context["input_ids"]
    attention_mask = context["attention_mask"]
    token_type_ids = context["token_type_ids"]

    if len(batch[0]) == 2:
        realized_rewards = torch.stack([item[1] for item in batch])
        return (input_ids, attention_mask, token_type_ids), realized_rewards

    embedded_actions = None if batch[0][1] is None else torch.stack([item[1] for item in batch])
    realized_rewards = torch.stack([item[2] for item in batch])
    chosen_actions = None if batch[0][3] is None else torch.stack([item[3] for item in batch])

    return (input_ids, attention_mask, token_type_ids), embedded_actions, realized_rewards, chosen_actions


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
                - data_sampler: The name of the data sampler to use.
                    Currently only "sorted" is supported. Default is None (random).
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
        dataset_hparams = config.get("dataset_hparams", {})
        if "seed" not in dataset_hparams:
            dataset_hparams["seed"] = config.get("seed", 42)
        dataset = DatasetClass(**filter_kwargs(DatasetClass, dataset_hparams))

        training_params = config
        bandit_hparams: dict[str, Any] = config.get("bandit_hparams", {})
        bandit_hparams["selector"] = selectors[bandit_hparams.get("selector", "argmax")](training_params)
        if "k" in training_params and "k" not in bandit_hparams:
            bandit_hparams["k"] = training_params["k"]

        def key_fn(idx: int) -> int:
            return dataset.sort_key(idx)

        training_params["data_sampler"] = (
            SortedDataSampler(
                dataset,
                key_fn=key_fn,
            )
            if training_params.get("data_sampler") == "sorted"
            else None
        )

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

            if "bert" in training_params["network"] or "resnet" in training_params["network"]:
                bandit_hparams["buffer"] = ListDataBuffer(
                    data_strategy,
                    max_size=training_params.get("max_buffer_size", None),
                )
            else:
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

        collate_fn = None
        if isinstance(dataset, ImdbMovieReviews):
            collate_fn = partial(transformers_collate, data_collator=dataset.get_data_collator())  # type: ignore

        if "max_samples" in self.training_params:
            max_samples = self.training_params["max_samples"]
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            subset_indices = indices[:max_samples]
            subset = Subset(dataset, subset_indices)

        return DataLoader(
            subset,
            batch_size=self.training_params.get("feedback_delay", 1),
            collate_fn=collate_fn,
            sampler=self.training_params.get("data_sampler", None),
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
                **optional_kwargs,
            )

            # Only provide the `chosen_action` if necessary.
            chosen_actions_pass = (
                chosen_actions
                if "contextualization_after_network" in self.bandit.hparams
                and self.bandit.hparams["contextualization_after_network"]
                else None
            )

            self.bandit.record_feedback(chosen_contextualized_actions, realized_rewards, chosen_actions_pass)
            # Train the bandit on the current feedback
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
    analyzer.load_metrics(logger.log_dir)
    analyzer.log_metrics()
    analyzer.plot_accumulated_metric(["reward", "regret"])
    analyzer.plot_average_metric("reward")
    analyzer.plot_average_metric("regret")
    analyzer.plot_loss()


def deep_get(dictionary: dict[str, Any], keys: str, default: Any = None) -> Any:
    """Get a value in a nested dictionary.

    Args:
        dictionary: The dictionary to get the value from.
        keys: The keys to traverse the dictionary. Use "/" to separate keys.
        default: The default value to return if the key is not found.

    Returns:
        The value at the given key or the default value if the key is not found.
    """
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("/"), dictionary)


def deep_set(dictionary: dict[str, Any], keys: str, value: Any) -> None:
    """Set a value in a nested dictionary.

    Args:
        dictionary: The dictionary to set the value in.
        keys: The keys to traverse the dictionary. Use "/" to separate keys.
        value: The value to set.
    """
    keys_list = keys.split("/")
    for key in keys_list[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys_list[-1]] = value


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
            Must contain a `comparison_key` which specifies which parameter to run the comparison over.
            This parameter must be a list of values to compare.
        log_dir: Directory where the logs are stored/outputted to. Default is "logs".
        save_plots: If True, plots be saved on disk. Default is False.
        suppress_plots: If True, plots will not be automatically shown. Default is False.
    """
    assert "comparison_key" in config, "To run a comparison a comparison key must be specified."

    if isinstance(config["comparison_key"], list):
        assert (
            len(config["comparison_key"]) == 1
        ), "To run a comparison exactly one valid comparison type must be specified."
        comparison_key = config["comparison_key"][
            0
        ]  # for now only one comparison type is supported. but you could extend it.
    else:
        comparison_key = config["comparison_key"]
    # comparison_values = bandit_config[comparison_type] but comparison_type can be nested by using "/"
    comparison_values = deep_get(config, comparison_key)

    assert comparison_values is not None, f"Could not find comparison values for {comparison_key}."
    assert isinstance(comparison_values, list), f"Comparison values for {comparison_key} must be a list."

    analyzer = BenchmarkAnalyzer(log_dir, "results", "metrics.csv", "env_metrics.csv", save_plots, suppress_plots)

    for comparison_value in comparison_values:
        try:
            experiment_id = str(comparison_value)
            print("==============================================")
            # deep copy the config to avoid overwriting the original but comparison_type can be nested by using "/"
            bandit_config = copy.deepcopy(config)
            # bandit_config[comparison_type] = comparison_value
            deep_set(bandit_config, comparison_key, comparison_value)

            csv_logger = CSVLogger(os.path.join(log_dir, experiment_id), version=0)
            benchmark = BanditBenchmark.from_config(bandit_config, csv_logger)
            print(f"Running benchmark for {bandit_config['bandit']} with {bandit_config['dataset']} dataset.")
            print(f"Setting {comparison_key}={experiment_id}.")
            print(f"Config: {bandit_config}")
            print(
                f"Dataset {bandit_config['dataset']}: "
                f"{len(benchmark.dataset)} samples with {benchmark.dataset.context_size} features "
                f"and {benchmark.dataset.num_actions} actions."
            )
            benchmark.run()

            analyzer.load_metrics(csv_logger.log_dir, experiment_id)
            analyzer.log_metrics(experiment_id)
        except Exception as e:
            print(
                f"Failed to run benchmark for {comparison_key}={comparison_value}. "
                "It might not be part of the final analysis."
            )
            print(e)

    for comparison_value in config.get("load_previous_result", []):
        experiment_id = str(comparison_value)
        print("==============================================")
        print(f"Loading previous result for {comparison_key}={experiment_id}.")
        csv_log_dir = os.path.join(log_dir, experiment_id, "lightning_logs", "version_0")
        try:
            analyzer.load_metrics(csv_log_dir, experiment_id)
            analyzer.log_metrics(experiment_id)
        except Exception as e:
            print(f"Failed to load previous result for {comparison_key}={experiment_id} from {csv_log_dir}.")
            print(e)

    title = comparison_key.replace("bandit_hparams/", "")
    analyzer.plot_accumulated_metric("reward", title)
    analyzer.plot_accumulated_metric("regret", title)
    analyzer.plot_average_metric("reward", title)
    analyzer.plot_average_metric("regret", title)
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
        config: dict[str, Any] = yaml.safe_load(file)

    if config.get("comparison_key") is not None:
        run_comparison(config, log_dir, save_plots, suppress_plots)
    else:
        run(config, log_dir, save_plots, suppress_plots)


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
