from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from neural_bandits.datasets.abstract_dataset import AbstractDataset


class MNISTDataset(AbstractDataset[torch.Tensor]):
    """Loads the MNIST 784 (version=1) dataset as a PyTorch Dataset.
    See https://www.openml.org/search?type=data&status=active&id=554 for more information of the dataset.

    Args:
        root (str): Where to store the dataset
        download (bool): Whether to download the dataset
    """

    num_actions: int = 10
    context_size: int = 784
    num_samples: int = 70000

    def __init__(self, root: str = "./data", download: bool = True):
        super().__init__(needs_disjoint_contextualization=True)
        self.data: Bunch = fetch_openml(
            name="mnist_784",
            version=1,
            data_home=root,
            as_frame=False,
        )
        self.X = self.data.data.astype(np.float32)
        self.y = self.data.target.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        contextualized_actions = self.contextualizer(X_item).squeeze(0)
        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        return float(self.y[idx] == action)
