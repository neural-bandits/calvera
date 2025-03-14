# Welcome to the documentation of the Calvera library

Calvera is a Python library offering a small collection of multi-armed bandit algorithms.
Currently the following algorithms are implemented:

- (Approximate + Standard) Linear Thompson Sampling

- (Approximate + Standard) Linear UCB

- Neural Linear

- Neural Thompson Sampling

- Neural UCB

_All algorithms can handle combinatorial settings via proper selectors (See [Selectors](#selectors))._

We plan to add the following algorithms in the future:

- Bootstrapped Bandit (See [this paper](https://arxiv.org/abs/2302.07459))

By using different `selectors` these algorithms can be adapted.
Selectors are classes that determine which arm is pulled based on the scores of the arms.
You can provide a selector to the different algorithm classes to modify the selection strategy.
The following selectors are available:

- `ArgMaxSelector`: Selects the arm with the highest score.

- `EpsilonGreedySelector`: Selects the arm with the highest score with probability `1-epsilon` or a random arm with probability `epsilon`.

- `TopKSelector`: Selects the top `k` arms with the highest scores.

- `EpsilonGreedyTopKSelector`: Selects the top `k` arms with probability `1-epsilon` or `k` random arms with probability `epsilon`.

## Installation

Calvera is available on [PyPI](https://pypi.org/project/calvera/).

```bash
pip install calvera
```

## Usage

The general idea of the library is to provide a simple interface for the different bandit algorithms.
A typical workflow looks as follows:

Create or load a bandit.

```python
# Create a bandit for a linear model with 128 features.
N_FEATURES = 128
bandit = LinearTSBandit(n_features=N_FEATURES,)
```

Use it for inference.

```python
# Sample 10 data points with 128 features each.
# Depending on the selector multiple actions can be chosen at once.
data = torch.randn(100, 1, N_FEATURES)
chosen_arms_one_hot, probabilities = bandit(data)
```

Retrieve the rewards of the infered actions (depends on your use case).

```python
rewards = torch.randn(100, 1)
```

Hand the data to the bandit via `record_feedback`.

```python
bandit.record_feedback(chosen_contextualized_actions, rewards)
```

Train the bandit via a PyTorch Lightning Trainer.

```python
trainer = pl.Trainer(
    max_epochs=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    accelerator=accelerator,
)
trainer.fit(bandit)
```

Last but not least, restart the process from the beginning.

This way you can easily integrate the bandit into your existing workflow. You only need to store the data and the rewards.

The combined code looks as follows:

```python
import torch
import lightning as pl

from calvera.bandits import LinearTSBandit, get_linear_ts_trainer

# 1. Create a bandit for a linear model with 128 features.
N_FEATURES = 128
bandit = LinearTSBandit(n_features=N_FEATURES,)

# Sample 10 data points with 128 features each.
# Depending on the selector multiple actions can be chosen at once.
# Therefore the shape is `(batch_size, n_actions, n_features)`.
data = torch.randn(100, 1, N_FEATURES)

# 2. Use the bandit for inference.
chosen_arms_one_hot, probabilities = bandit(data)

# `chosen_arms_one_hot` is one-hot encoded as we could select multiple arms at once.
chosen_arms = chosen_arms_one_hot.argmax(dim=1)

# 3. Retrieve the rewards for the chosen arms.
rewards = torch.randn(100, 1)

# 4. Give the data to the bandit.
chosen_contextualized_actions = data[:, :, chosen_arms]
bandit.record_feedback(chosen_contextualized_actions, rewards)

# 5. Train the bandit.
trainer = pl.Trainer(
    max_epochs=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    accelerator=accelerator,
)
trainer.fit(bandit)

# (6. Repeat the process)
```

As you can see a bandit is a [PyTorch Lightning Module](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) and is designed to be used with a PyTorch Lightning Trainer.

The following sections will give you more information about the different algorithms and selectors. For more information about the usage of the library see the [examples page](./examples/) in the documentation.

## Bandits

A bandit is a PyTorch Lightning Module and therefore implements two major methods `forward()` and `training_step()`.
The `forward()` method is used for inference just like a normal PyTorch `nn.Module`.
The `training_step()` method is a hook that is called by the Lightning Trainer during each training step.

During initialization of a bandit two parameters can be set, a buffer and a selector.
The buffer is used to store and control the data used for training, while the selector is used to select the arms during inference. For more information about buffers and selectors see the corresponding sections ([Buffers](#buffers) and [Selectors](#selectors)).

### Buffers

A buffer is an object that stores the data for the training of the bandit. This is necessary because some algorithms rely on a subset or all the data that was seen in the past.
A buffer should subclass the `AbstractBanditDataBuffer` class and implement the respective abstract methods. See [its documentation](./bandits.md#buffers) for more information.

## Selectors

A selector is an object that selects the arms during inference based on the scores that were computed by the bandit.
A selector should subclass the `AbstractSelector` class and implement the respective abstract methods. The following selectors are available:

- `ArgMaxSelector`: Selects the arm with the highest score.
- `EpsilonGreedySelector`: Selects the arm with the highest score with probability `1-epsilon` or a random arm with probability `epsilon`.
- `TopKSelector`: Selects the top `k` arms with the highest scores.
- `EpsilonGreedyTopKSelector`: Selects the top `k` arms with probability `1-epsilon` or `k` random arms with probability `epsilon`.

The documentation of the selectors can be found [here](./utils/).

## Benchmarks

Calvera provides an extensive benchmark framework via the `calvera[benchmark]` subpackage. With it bandit algorithms can be compared, optimized and tested. For the benchmark setting one or more datasets have to be provided on which the bandits can be tested. Then, the environment simulates the interaction with the bandit. Lastly, the `BanditBenchmark` runs the bandit in a setting specified via a config and outputs the results for evaluation.

### Datasets

Calvera includes several standard bandit datasets for benchmarking purposes:

- `MNISTDataset` (simple image classification, disjoint model)
- `StatlogDataset` (disjoint model)
- `MovieLensDataset` (combinatorial)
- `TinyImageNetDataset` (image classification, disjoint model)
- `WheelBanditDataset` (synthetic)
- Further non-combinatorial and combinatorial synthetic datasets for multi-armed bandits (linear + non-linear)

Users can create their own datasets for benchmarking by implementing the `AbstractDataset` interface. For each item it needs to provide a tupel of a tensor of contextualized actions (i.e. arms) of shape `(num_actions, num_features)` and a tensor of realized rewards per action of shape `(num_actions, )`. The bandit will chose one of the given actions and will receive the reward for that action.

Calvera also provides the option to convert a classification problem into a bandit problem with contextualized actions using the `needs_disjoint_contextualization` option. This converts a single context vector of size `num_features` into a 2d-tensor of `num_arms` context vectors of size `num_arms*num_features` by preserving a specific part of the contextualized action vector space for each action. Given a context $x$ and $m$ number of classes:
$$(x_1, ..., x_m) \to ((x_1, ..., x_m, ..., 0, ..., 0), ..., (0, ..., 0, ..., x_1, ..., x_m))$$

### Environment

The `BanditBenchmarkEnvironment` takes a pytorch `DataLoader` and can be used as an iterator to load batches of samples into the bandit. Once actions have been chosen, they are passed to the environment to receive the reward for those actions. For convencience, the chosen contextualized_actions are also returned, so that they can be easily passed on to the bandit. Finally, it is possible to compute the regret for the set of chosen actions.

```python
from calvera.benchmark import BanditBenchmarkEnvironment

environment = BanditBenchmarkEnvironment(dataloader)
for contextualized_actions in environment:
    chosen_actions, p = bandit.forward(contextualized_actions)  # one-hot tensor
    chosen_contextualized_actions, realized_rewards = environment.get_feedback(chosen_actions)
    bandit.record_feedback(chosen_contextualized_actions, realized_rewards)

    # optional: compute regret
    regret = environment.compute_regret(chosen_actions)
```

### Benchmark

The `BanditBenchmark` handles setting up a training environment from a given config, runs the benchmark and logs the results.

```python
from calvera.benchmark import BanditBenchmark

config = {
    "bandit": "neural_ucb",
    "forward_batch_size": 1,
    "train_batch_size": 32,
    "feedback_delay": 128,  # training every 128 samples
    "max_steps": 16,  # for how many steps to train
    "gradient_clip_val": 20.0,
    "network": "small_mlp",  # or linear, tiny_mlp, small_mlp, large_mlp, bert, resnet18
    "data_strategy": "sliding_window",  # or all
    "window_size": 2048,
    "max_buffer_size": 2048,
    "selector": "epsilon_greedy"  # or argmax (default), top_k, random
    "epsilon": 0.1,
    # "device": "cuda",  # for training on a different device
    "bandit_hparams": {
        "exploration_rate": 0.0001,
        "learning_rate": 0.001,
        # ... further parameters passed to the constructor of the bandit
    }
}
# optional: pass any lightning logger. Note that `logger.log_dir` is used for writing further metrics like the rewards/regret as CSV files.
logger = lightning.pytorch.loggers.CSVLogger(...)
benchmark = BanditBenchmark.from_config(config, logger)

# or directly by passing the classes
benchmark = BanditBenchmark(bandit, dataset, training_params, logger)
```

The `BenchmarkAnalyzer` is used to analyze and log or plot the results. For convenient setup the methods `run` (single bandit), `run_comparison` (several bandits or datasets or other parameters given as a list. The key of the parameter to compare over is specified under the config `comparison_key`) and `run_from_yaml` can be used.

## Our experimental results

## Contributing

Contributions are always welcome! Please refer to the [contribution guidelines](https://github.com/neural-bandits/calvera/blob/main/CONTRIBUTING.md) for more information.

As of 26th February 2025, the library is under active development. Current contributors are:

- [Philipp Kolbe](mailto:philipp.kolbe@student.hpi.uni-potsdam.de)

- [Robert Weeke](mailto:robert.weeke@student.hpi.uni-potsdam.de)

- [Parisa Shahabinejad](mailto:parisa.shahabinejad@student.hpi.uni-potsdam.de)

### License

Calvera is licensed under the MIT license. See the [LICENSE](https://github.com/neural-bandits/calvera/blob/main/LICENSE) file for more details.

### Contact

If you have further questions or feedback, you are welcome to contact one of the authors directly.

- [Philipp Kolbe](mailto:philipp.kolbe@student.hpi.uni-potsdam.de)

- [Robert Weeke](mailto:robert.weeke@student.hpi.uni-potsdam.de)

- [Parisa Shahabinejad](mailto:parisa.shahabinejad@student.hpi.uni-potsdam.de)
