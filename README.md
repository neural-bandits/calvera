# Calvera

Calvera is a Python library offering a collection of neural multi-armed bandit algorithms, designed to integrate seamlessly with PyTorch and PyTorch Lightning. Whether you're exploring contextual bandits or developing new strategies, Calvera provides a flexible, easy-to-use interface. You can bring your own neural networks and datasets while Calvera focuses on the implementation of the bandit algorithms.

## Features

- **Multi-Armed Bandit Algorithms:**
  - (Approximate + Standard) Linear Thompson Sampling
  - (Approximate + Standard) Linear UCB
  - Neural Linear
  - Neural Thompson Sampling
  - Neural UCB

- **Customizable Selectors:**
  - **ArgMaxSelector:** Chooses the arm with the highest score.
  - **EpsilonGreedySelector:** Chooses the best arm with probability `1-epsilon` or a random arm with probability `epsilon`.
  - **TopKSelector:** Selects the top `k` arms with the highest scores.

- **Integration:**
  - Built on top of [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) for training and inference.
  - Minimal adjustments needed to plug into your existing workflow.

## Installation

Calvera is available on [PyPI](https://pypi.org/). Install it via pip:

```bash
pip install calvera
```
This installs the necessary dependencies for the base library. If you want to use parts of the benchmark subpackage we recommend installing the optional dependencies as well:

```bash
pip install calvera[benchmark]
```
For development you can install the development dependencies via:

```bash
pip install calvera[dev]
```

## Quick Start

Below is a simple example using a Linear Thompson Sampling bandit:
```python
import torch
import lightning as pl
from calvera.bandits import LinearTSBandit

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
trainer = pl.Trainer(
    max_epochs=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    accelerator=accelerator,
)
trainer.fit(bandit)

# (6. Repeat the process as needed)
```

For more detailed examples, see the examples page in [the documentation](http://neural-bandits.github.io/calvera/).

## [Documentation](http://neural-bandits.github.io/calvera/)

- Bandits: Each bandit is implemented as a PyTorch Lightning Module with `forward()` for inference and `training_step()` for training.

- Buffers: Data is managed via buffers that subclass AbstractBanditDataBuffer.

- Selectors: Easily customize your arm selection strategy by using or extending the provided selectors.

## Benchmarks & Experimental Results

Detailed benchmarks, datasets, and experimental results are available in the extended documentation.

## Contributing

Contributions are welcome! For guidelines on how to contribute, please refer to our [CONTRIBUTING.md](https://github.com/neural-bandits/calvera/blob/main/CONTRIBUTING.md).

## License

Calvera is licensed under the MIT License. See the [LICENSE file](https://github.com/neural-bandits/calvera/blob/main/LICENSE) for details.

## Contact

For questions or feedback, please reach out to one of the authors:

- Philipp Kolbe

- Robert Weeke

- Parisa Shahabinejad

---


[Link to Agreement](https://docs.google.com/document/d/1qs0hDGVd5MHe6PK5uL_GVNjiIePBJscbNkjGotF9-Uk/edit?tab=t.0])
