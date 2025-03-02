# Welcome to the documentation of the Calvera library

Calvera is a Python library offering a small collection of multi-armed bandit algorithms.
Currently the following algorithms are implemented:

- Linear Thompson Sampling

- Linear UCB

- Neural Linear

- Neural Thompson Sampling

- Neural UCB

We plan to add the following algorithms in the future:

- Bootstrapped Bandit (See [this paper](https://arxiv.org/abs/2302.07459))


By using different `selectors` these algorithms can be adapted.
Selectors are classes that determine which arm is pulled based on the scores of the arms.
You can provide a selector to the different algorithm classes to modify the selection strategy.
The following selectors are available:

- `ArgMaxSelector`: Selects the arm with the highest score.

- `EpsilonGreedySelector`: Selects the arm with the highest score with probability `1-epsilon` or a random arm with probability `epsilon`.

- `TopKSelector`: Selects the top `k` arms with the highest scores.


## Installation

Calvera is (will be) available on [PyPI](https://pypi.org/).
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

Hand the data to the bandit via `add_data`.
```python
bandit.add_data(chosen_contextualized_actions, rewards)
```

Train the bandit via a PyTorch Lightning Trainer.
```python
trainer = get_linear_ts_trainer(bandit)
trainer.fit(bandit)
```

Last but not least, restart the process from the beginning.

This way you can easily integrate the bandit into your existing workflow. You only need to store the data and the rewards.

The combined code looks as follows:

```python
import torch

from calvera.bandits.linear_ts_bandit import LinearTSBandit, get_linear_ts_trainer

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
bandit.add_data(chosen_contextualized_actions, rewards)

# 5. Train the bandit.
trainer = get_linear_ts_trainer(bandit)
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

The documentation of the selectors can be found [here](./utils/).

## Benchmarks

### Environment

### Datasets

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
