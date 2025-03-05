import torch

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset


class SyntheticCombinatorialDataset(AbstractDataset[torch.Tensor]):
    """Synthetic dataset for Combinatorial Bandit experiments.

    Implements the score functions from Hwang et al. "Combinatorial Neural Bandits" (section 5):
    - Linear: h₁(x) = x^T a
    - Quadratic: h₂(x) = (x^T a)²
    - Cosine: h₃(x) = cos(πx^T a)
    where a is a random vector generated from a unit ball.

    More information can be found at https://arxiv.org/abs/2306.00242.

    The dataset generates contexts and rewards according to the chosen function type.
    Each sample consists of a set of N arms, each with its own context vector.
    The reward for each arm is computed based on the chosen function.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        num_actions: int = 20,
        context_size: int = 80,
        function_type: str = "linear",
        noise_std: float = 0,
        seed: int = 42,
    ):
        """Initialize the synthetic dataset.

        Args:
            n_samples: Number of samples in the dataset
            num_actions: Number of arms in each sample
            context_size: Dimension of context vector
            function_type: Type of score function to use ('linear', 'quadratic', 'cosine')
            noise_std: Standard deviation of Gaussian noise added to rewards
            seed: Random seed for reproducibility
        """
        super().__init__(needs_disjoint_contextualization=False)

        self.n_samples = n_samples
        self.num_actions = num_actions
        self.context_size = context_size
        self.function_type = function_type
        self.noise_std = noise_std

        torch.manual_seed(seed)

        # Generate random vector a for score function
        # This is the parameter vector used in all three function types
        self.a = self._random_unit_ball(1)[0]

        # Generate context vectors for all samples and arms
        # Each context is randomly generated from a normal distribution
        self.contexts = self._random_unit_ball(n_samples * num_actions).reshape(n_samples, num_actions, context_size)

        # Compute rewards based on the chosen function
        self.rewards = self._compute_rewards()

        # Add Gaussian noise to rewards
        if noise_std > 0:
            noise = torch.randn_like(self.rewards) * noise_std
            self.rewards = self.rewards + noise

    def _random_unit_ball(self, n_samples: int) -> torch.Tensor:
        """Generate random vectors from a unit ball of dim=context_size.

        Args:
            n_samples: Number of vectors to generate

        Returns:
            Tensor of random vectors. Shape: (n_samples, dim)
        """
        # Step 1: Sample from standard Gaussian: shape (num_points, d)
        x: torch.Tensor = torch.randn(n_samples, self.context_size)
        # Step 2: Normalize each row to get a direction vector
        directions: torch.Tensor = x / x.norm(dim=1, keepdim=True)
        # Step 3: Sample radii that accounts for the d-dimensional volume distribution
        u = torch.rand(n_samples, 1)
        radii: torch.Tensor = u.pow(1.0 / self.context_size)
        # Scale the directions by the radii to get samples in the unit ball.
        return directions * radii

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards based on the chosen function type.

        Returns:
            Tensor of rewards for all samples and arms. Shape: (n_samples, num_actions)
        """
        # Inner product between contexts and a
        inner_prod = torch.matmul(self.contexts, self.a)

        if self.function_type == "linear":
            # h₁(x) = x^T a
            return inner_prod
        elif self.function_type == "quadratic":
            # h₂(x) = (x^T a)²
            return torch.pow(inner_prod, 2)
        elif self.function_type == "cosine":
            # h₃(x) = cos(πx^T a)
            return torch.cos(torch.pi * inner_prod)
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the contextualized actions and rewards for a given index.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of contextualized actions (shape: (num_actions, context_size)) and
            rewards (shape: (num_actions,))
        """
        return self.contexts[idx], self.rewards[idx]

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a specific index and action.

        Args:
            idx: Index of the sample
            action: Index of the arm

        Returns:
            Reward value
        """
        return self.rewards[idx, action].item()
