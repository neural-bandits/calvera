# Welcome to the documentation of the Calvera library

Calvera is a Python library for offering a small collection of multi-armed bandit algorithms.
Currently the following algorithms are implemented:

- Linear Thompson Sampling

- Linear UCB

- Neural Linear

- Neural Thompson Sampling

- Neural UCB

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
The following example shows how to use the library to solve a simple multi-armed bandit problem.

```python
import torch

from calvera.bandits.linear_ts_bandit import LinearTSBandit, get_linear_ts_trainer

# Create a bandit for a linear model with 128 features.
N_FEATURES = 128
bandit = LinearTSBandit(n_features=N_FEATURES,)

# Sample 10 data points with 128 features each.
# Depending on the selector multiple actions can be chosen at once.
# Therefore the shape is `(batch_size, n_actions, n_features)`.
data = torch.randn(100, 1, N_FEATURES)

# Use the bandit for inference.
chosen_arms_one_hot, probabilities = bandit(data)

# `chosen_arms_one_hot` is one-hot encoded as we could select multiple arms at once.
chosen_arms = chosen_arms_one_hot.argmax(dim=1) 

# Retrieve the rewards for the chosen arms.
rewards = torch.randn(100, 1)

# Give the data to the bandit.
chosen_contextualized_actions = data[:, :, chosen_arms]
bandit.add_data(chosen_contextualized_actions, rewards)

# Train the bandit.
trainer = get_linear_ts_trainer(bandit)
trainer.fit(bandit)
```

As you can see a bandit is a [PyTorch Lightning Module](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) and is designed to be used with PyTorch Lightning Trainer. 
A typical workflow looks as follows:

1. Create or load a bandit.

2. Use it for inference.

3. Retrieve the rewards of the infered actions (depends on your use case).

4. Hand the data to the bandit via `add_data`.

5. Train the bandit via a PyTorch Lightning Trainer.




## Contributing
Contributions are always welcome! Please refer to the [contribution guidelines](CONTRIBUTING.md) for more information.

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
