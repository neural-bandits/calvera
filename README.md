Below are two markdown files—a polished README and a brief CONTRIBUTING guide—for your Calvera library.

README.md

# Calvera

Calvera is a Python library offering a collection of multi-armed bandit algorithms, designed to integrate seamlessly with PyTorch and PyTorch Lightning. Whether you're exploring contextual bandits or developing new strategies, Calvera provides a flexible, easy-to-use interface.

## Features

- **Multi-Armed Bandit Algorithms:**
  - Linear Thompson Sampling
  - Linear UCB
  - Neural Linear
  - Neural Thompson Sampling
  - Neural UCB

- **Customizable Selectors:**
  - **ArgMaxSelector:** Chooses the arm with the highest score.
  - **EpsilonGreedySelector:** Chooses the best arm with probability `1-epsilon` or a random arm with probability `epsilon`.
  - **TopKSelector:** Selects the top `k` arms with the highest scores.

- **Seamless Integration:**
  - Built on top of [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) for streamlined training and inference.
  - Minimal adjustments needed to plug into your existing workflow.

## Installation

Calvera is available on [PyPI](https://pypi.org/). Install it via pip:

```bash
pip install calvera

Quick Start

Below is a simple example using a Linear Thompson Sampling bandit:

import torch
from calvera.bandits.linear_ts_bandit import LinearTSBandit, get_linear_ts_trainer

# 1. Create a bandit for a linear model with 128 features.
N_FEATURES = 128
bandit = LinearTSBandit(n_features=N_FEATURES)

# 2. Generate sample data (batch_size, n_actions, n_features) and perform inference.
data = torch.randn(100, 1, N_FEATURES)
chosen_arms_one_hot, probabilities = bandit(data)
chosen_arms = chosen_arms_one_hot.argmax(dim=1)

# 3. Retrieve rewards for the chosen arms.
rewards = torch.randn(100, 1)

# 4. Add the data to the bandit.
chosen_contextualized_actions = data[:, :, chosen_arms]
bandit.add_data(chosen_contextualized_actions, rewards)

# 5. Train the bandit.
trainer = get_linear_ts_trainer(bandit)
trainer.fit(bandit)

# (6. Repeat the process as needed)
```

For more detailed examples, see the examples page in the documentation.

## Documentation

- Bandits: Each bandit is implemented as a PyTorch Lightning Module with `forward()` for inference and `training_step()` for training.

- Buffers: Data is managed via buffers that subclass AbstractBanditDataBuffer.

- Selectors: Easily customize your arm selection strategy by using or extending the provided selectors.

## Benchmarks & Experimental Results

Detailed benchmarks, datasets, and experimental results are available in the extended documentation.

## Contributing

Contributions are welcome! For guidelines on how to contribute, please refer to our CONTRIBUTING.md.

License

Calvera is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or feedback, please reach out to one of the authors:
	•	Philipp Kolbe
	•	Robert Weeke
	•	Parisa Shahabinejad

---


[Link to Agreement](https://docs.google.com/document/d/1qs0hDGVd5MHe6PK5uL_GVNjiIePBJscbNkjGotF9-Uk/edit?tab=t.0])
