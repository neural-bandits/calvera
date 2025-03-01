from abc import abstractmethod

import torch

from neural_bandits.benchmark.datasets.abstract_dataset import AbstractDataset

class SyntheticDataset(AbstractDataset):
    num_actions = 2
    
    def __init__(self, n_features=2, num_samples=10000, noise_std=0.1) -> None:
        """Initialize the synthetic dataset.
        
        Args:
            n_features: The number of features in the dataset.
            num_samples: The number of samples in the dataset.
            noise_std: The standard deviation of the noise in the dataset.
        """
        super().__init__(needs_disjoint_contextualization=True)
        self.n_features = n_features
        self.context_size = n_features * self.num_actions
        self.num_samples = num_samples

        self.X = torch.randn(num_samples, n_features)
        self.Phi = self.phi(self.X)
        self.theta_opt = torch.randn(self.Phi.shape[1])
        noise = torch.normal(mean=0.0, std=noise_std, size=(num_samples,)) > 0
        self.y = (self.Phi @ self.theta_opt + noise > 0).int()

    def __len__(self) -> int:
        """Return the number of contexts / samples in this dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the contextualized actions and rewards"""
        contextualized_actions = self.contextualizer(self.X[idx].unsqueeze(0)).squeeze(0)
        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )
        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a given index and action."""
        return float(self.y[idx] == action)

    @abstractmethod
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        pass

class LinearSyntheticDataset(SyntheticDataset):
    """A linear synthetic dataset with bias.
    
    In one dimensional case it would be:
    `y = w0 + \sum_i wi*x_i`
    Per added feature, the feature `x_i` is added.
    """
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature matrix for the given input."""
        return torch.cat([
            torch.ones(x.shape[0], 1),
            x,
        ], dim=1)

class CubicSyntheticDataset(SyntheticDataset):
    """A synthetic dataset with bias and cubic features.

    In one dimensional case it would be:
    `y = w0 + \sum_i wi,1*x_i + wi,2*x_i^2 + wi,3*x_i^3`
    Per added feature, the features `x_i`, `x_i^2`, and `x_i^3` are added.
    """
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature matrix for the given input."""
        return torch.cat([
            torch.ones(x.shape[0], 1),
            x,
            x**2,
            x**3,
        ], dim=1)

class SinSyntheticDataset(SyntheticDataset):
    """A non-linear synthetic dataset using `sin(x)`.
    
    y = w1 + \sum_i w_i*x_i + \sum_j w_j*sin(x_j)

    """
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature matrix for the given input."""
        return torch.cat([
            torch.ones(x.shape[0], 1),
            x,
            torch.sin(x),
        ], dim=1)

class LinearCombinationSyntheticDataset(SyntheticDataset):
    """A synthetic dataset of a linear combination of the features.

    We use the outer product of the features as the feature matrix.
    `y = w0 + \sum_i w_i*x_i + \sum_i \sum_{j>i} w_{i,j}*x_i*x_j`
    """
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature matrix for the given input including the outer product."""
        n_samples, n_features = x.shape
        bias = torch.ones(n_samples, 1, device=x.device)
        # Compute outer products: shape (n_samples, n_features, n_features)
        outer = x.unsqueeze(2) * x.unsqueeze(1)
        # Get indices for the upper triangle including the diagonal (offset=0)
        triu_idx = torch.triu_indices(n_features, n_features, offset=0)
        # Index the outer products to get only the upper triangle for each sample
        outer_upper = outer[:, triu_idx[0], triu_idx[1]]
        # Concatenate bias, linear features, and the upper triangle outer product
        return torch.cat([bias, x, outer_upper], dim=1)
