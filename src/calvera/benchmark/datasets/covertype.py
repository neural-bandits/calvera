from typing import cast

import numpy as np
import torch
from sklearn.datasets import fetch_covtype

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset


class CovertypeDataset(AbstractDataset[torch.Tensor]):
    """Loads the Covertype dataset as a PyTorch Dataset from the UCI repository.

    More information can be found at https://archive.ics.uci.edu/ml/datasets/covertype.
    """

    num_actions: int = 7
    context_size: int = 54 * 7  # disjoint model
    num_samples: int = 581012

    def __init__(self, dest_path: str = "./data") -> None:
        """Initialize the Covertype dataset. Downloads the dataset from UCI repository if not found at `dest_path`.

        Args:
            dest_path: Where to store and look for the dataset.
        """
        super().__init__(needs_disjoint_contextualization=True)
        self.data = fetch_covtype(data_home=dest_path)
        X_np = self.data.data.astype(np.float32)
        y_np = self.data.target.astype(np.int64)

        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of contexts / samples in this dataset.

        Returns:
            The number of contexts / samples in this dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the contextualized actions and rewards for a given index.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            contextualized_actions: The contextualized actions for the given index.
            rewards: The rewards for each action. Retrieved via `self.reward`.
        """
        context = self.X[idx].reshape(1, -1)
        contextualized_actions = self.contextualizer(context).squeeze(0)
        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a given index and action.

        1.0 if the action is the correct cover type, 0.0 otherwise.

        Args:
            idx: The index of the context in this dataset.
            action: The action to evaluate.

        Returns:
            1.0 if the action is the correct cover type, 0.0 otherwise.
        """
        return float(self.y[idx] == action + 1)

    def sort_key(self, idx: int) -> int:
        """Return the label for a given index.

        This is only required as the sort key in the  SortedDataSampler in the benchmark.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            The label for the given index.
        """
        return cast(int, self.y[idx].item())
