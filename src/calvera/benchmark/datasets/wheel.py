from typing import cast

import torch

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset


def _sample_rewards(
    generator: torch.Generator,
    contexts: torch.Tensor,
    actions: torch.Tensor,
    delta: float,
    mu_small: float,
    std_small: float,
    mu_medium: float,
    std_medium: float,
    mu_large: float,
    std_large: float,
) -> torch.Tensor:
    """Sample rewards for each context according to the Wheel Bandit rules.

    Based on: Riquelme et al. "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for
    Thompson Sampling" https://arxiv.org/abs/1802.09127

    Args:
        generator: The random number generator to use. Should already be seeded for reproducibility.
        contexts: A torch.Tensor of shape (num_samples, context_size) representing the sampled contexts.
        actions: A torch.Tensor of shape (num_samples) representing the actions chosen for the corresponding context.
        delta: Exploration parameter: high reward in one region if norm above delta
        mu_small: Mean of the small reward distribution.
        std_small: Standard deviation of the small reward distribution.
        mu_medium: Mean of the medium reward distribution.
        std_medium: Standard deviation of the medium reward distribution.
        mu_large: Mean of the large reward distribution.
        std_large: Standard deviation of the large reward distribution.

    Returns:
        rewards: A torch.Tensor of shape (num_samples, num_actions) with sampled rewards.
    """
    assert len(contexts.shape) == 2, "Contexts should be a 2D tensor of shape (num_samples, context_size)."

    num_samples = contexts.size(0)

    # Initialize rewards with small-reward distribution
    rewards = torch.normal(
        mean=torch.tensor([mu_small], dtype=torch.float32).expand(num_samples),
        std=torch.tensor([std_small], dtype=torch.float32).expand(num_samples),
        generator=generator,
    )
    norms = torch.norm(contexts, dim=1)
    above_delta = norms > delta

    # For contexts above delta, assign the large reward in the correct region
    r_big = torch.normal(
        mean=torch.tensor([mu_large], dtype=torch.float32).expand(num_samples),
        std=torch.tensor([std_large], dtype=torch.float32).expand(num_samples),
        generator=generator,
    )

    r_medium = torch.normal(
        mean=torch.tensor([mu_medium], dtype=torch.float32).expand(num_samples),
        std=torch.tensor([std_medium], dtype=torch.float32).expand(num_samples),
        generator=generator,
    )

    # Determine optimal actions based on context quadrant when norm > delta
    # Quadrants mapping:
    # If contexts[i,0] > 0 and contexts[i,1] > 0 -> action 0 (= 3 - 2 * 1 - 1)
    # If contexts[i,0] > 0 and contexts[i,1] < 0 -> action 1 (= 3 - 2 * 1 - 0)
    # If contexts[i,0] < 0 and contexts[i,1] > 0 -> action 2 (= 3 - 2 * 0 - 1)
    # If contexts[i,0] < 0 and contexts[i,1] < 0 -> action 3 (= 3 - 2 * 0 - 0)
    # Otherwise (norm <= delta) best action is argmax(mean_v).

    # if above delta, assign large reward to optimal action else assign small reward
    idxs_above = torch.where(above_delta)[0]
    for i in idxs_above:
        x, y = contexts[i, 0], contexts[i, 1]
        a = x > 0
        b = y > 0

        if (3 - 2 * a.float() - b.float()).round() == actions[i]:
            rewards[i] = r_big[i]

    # if below delta, assign medium reward when action 4 is taken
    idxs_below_eq = torch.where(~above_delta)[0]
    for i in idxs_below_eq:
        if actions[i] == 4:
            rewards[i] = r_medium[i]

    return rewards


class WheelBanditDataset(AbstractDataset[torch.Tensor]):
    """Generates a dataset for the Wheel Bandit problem.

    Based on: Riquelme et al. "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for
    Thompson Sampling" https://arxiv.org/abs/1802.09127
    """

    num_features: int = 2
    num_actions: int = 5
    context_size: int = 2 * 5

    def __init__(
        self,
        num_samples: int,
        delta: float,
        mu_small: float = 1.0,
        std_small: float = 0.01,
        mu_medium: float = 1.2,
        std_medium: float = 0.01,
        mu_large: float = 50.0,
        std_large: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the Wheel Bandit dataset.

        Args:
            num_samples: Number of samples to generate.
            delta: Exploration parameter: high reward in one region if norm above delta
            mu_small: Mean of the small reward distribution.
            std_small: Standard deviation of the small reward distribution.
            mu_medium: Mean of the medium reward distribution.
            std_medium: Standard deviation of the medium reward distribution.
            mu_large: Mean of the large reward distribution.
            std_large: Standard deviation of the large reward distribution.
            seed: Seed for the random number generator.
        """
        super().__init__(needs_disjoint_contextualization=True)

        self.num_samples = num_samples
        self.delta = delta

        # Reward distributions
        self.mu_small = mu_small
        self.std_small = std_small
        self.mu_medium = mu_medium
        self.std_medium = std_medium
        self.mu_large = mu_large
        self.std_large = std_large

        data, rewards = self._generate_data(seed)
        self.data = data
        self.rewards = rewards

    def _generate_data(self, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Pregenerate the dataset for the Wheel Bandit problem.

        We do this because we need to make the dataset compatible with PyTorch's Dataset.

        Args:
            seed: Seed for the random number generator.

        Returns:
            contexts: A torch.Tensor of shape (num_samples, context_size) representing the sampled contexts.
            rewards: A torch.Tensor of shape (num_samples, num_actions) with sampled rewards
        """
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        # Sample uniform contexts in the unit ball
        # We'll attempt a similar approach: sample more and filter.
        # The original code took a while-loop approach. We'll do the same.

        data_list: list[torch.Tensor] = []
        batch_size = max(int(self.num_samples / 3), 1)
        while len(data_list) < self.num_samples:
            raw_data = (torch.rand(batch_size, self.num_features, generator=generator) * 2.0 - 1.0).float()
            norms = torch.norm(raw_data, dim=1)
            # filter points inside unit norm
            inside = raw_data[norms <= 1]
            data_list.append(inside)

        contexts = torch.cat(data_list, dim=0)
        contexts = contexts[: self.num_samples]

        # sample the rewards for each context-action pair
        contexts_repeat = contexts.repeat_interleave(
            self.num_actions, dim=0
        )  # shape (num_samples * num_actions, context_size)
        actions = torch.arange(self.num_actions).repeat(self.num_samples)  # shape (num_samples * num_actions)
        rewards = _sample_rewards(
            generator,
            contexts_repeat,
            actions,
            self.delta,
            self.mu_small,
            self.std_small,
            self.mu_medium,
            self.std_medium,
            self.mu_large,
            self.std_large,
        ).reshape(self.num_samples, self.num_actions)

        return contexts, rewards

    def __len__(self) -> int:
        """Return the number of contexts / samples in this dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the contextualized actions and rewards for the context at index idx in this dataset.

        Args:
            idx: The index of the context in this dataset.
        """
        contextualized_actions = self.contextualizer(self.data[idx].unsqueeze(0)).squeeze(0)
        rewards = self.rewards[idx]

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward of the given action for the context at index idx in this dataset."""
        return self.rewards[idx, action].item()

    def sort_key(self, idx: int) -> int:
        """Return the action with the highest reward for the context at index idx in this dataset."""
        max_idx = self.rewards[idx].argmax()
        return cast(int, max_idx.item())
