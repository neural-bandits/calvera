import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from .abstract_dataset import AbstractDataset


class MNISTDataset(AbstractDataset):
    """Loads the MNIST 784 (version=1) dataset as a pytorch Dataset.

    Args:
        root (str): Where to store the dataset
        download (bool): Whether to download the dataset
    """

    num_actions: int = 10
    context_size: int = 784
    num_samples: int = 70000

    def __init__(self, root: str = "./data", download: bool = True):
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.zeros(self.num_actions, dtype=torch.float32)
        y_item[self.y[idx]] = 1.0

        return X_item

    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(self.y[idx] == action), dtype=torch.float32)
