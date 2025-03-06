from typing import cast

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset


class MNISTDataset(AbstractDataset[torch.Tensor]):
    """Loads the MNIST 784 (version=1) dataset as a PyTorch Dataset.

    More information can be found at https://www.openml.org/search?type=data&status=active&id=554.
    """

    num_actions: int = 10
    context_size: int = 784 * 10  # disjoint model
    num_samples: int = 70000

    def __init__(self, dest_path: str = "./data") -> None:
        """Initialize the MNIST 784 dataset.

        Loads the dataset from OpenML and stores it as PyTorch tensors.

        Args:
            dest_path: Where to store the dataset
        """
        super().__init__(needs_disjoint_contextualization=True)
        self.data: Bunch = fetch_openml(
            name="mnist_784",
            version=1,
            data_home=dest_path,
            as_frame=False,
        )
        self.X = self.data.data.astype(np.float32)
        self.y = self.data.target.astype(np.int64)

    def __len__(self) -> int:
        """Return the number of contexts / samples in this dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the contextualized actions and rewards for a given index.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            contextualized_actions: The contextualized actions for the given index.
            rewards: The rewards for each action. Retrieved via `self.reward`.
        """
        X_item = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        contextualized_actions = self.contextualizer(X_item).squeeze(0)
        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a given index and action.

        1.0 if the action is the same as the label, 0.0 otherwise.

        Args:
            idx: The index of the context in this dataset.
            action: The action for which the reward is requested.
        """
        return float(self.y[idx] == action)

    def sort_key(self, idx: int) -> int:
        """Return the label for a given index.

        This is only required as the sort key in the  SortedDataSampler in the benchmark.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            The label for the given index.
        """
        return cast(int, self.y[idx].item())
