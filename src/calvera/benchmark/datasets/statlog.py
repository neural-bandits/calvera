from typing import cast

import torch
from ucimlrepo import fetch_ucirepo

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset


class StatlogDataset(AbstractDataset[torch.Tensor]):
    """Loads the Statlog (Shuttle) dataset as a PyTorch Dataset from the UCI repository.

    More information can be found at https://archive.ics.uci.edu/dataset/148/statlog+shuttle.
    """

    num_actions: int = 9
    context_size: int = 7 * 9  # disjoint model
    num_samples: int = 58000

    def __init__(self) -> None:
        """Initialize the Statlog (Shuttle) dataset.

        Loads the dataset from the UCI repository and stores it as PyTorch tensors.
        """
        super().__init__(needs_disjoint_contextualization=True)
        dataset = fetch_ucirepo(id=148)  # id=148 specifies the Statlog (Shuttle) dataset
        X = dataset.data.features
        y = dataset.data.targets

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

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
        contextualized_actions = self.contextualizer(self.X[idx].unsqueeze(0)).squeeze(0)
        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a given index and action.

        Returns 1 if the action is the same as the label, 0 otherwise.

        Args:
            idx: The index of the context in this dataset.
            action: The action for which the reward is requested.
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
