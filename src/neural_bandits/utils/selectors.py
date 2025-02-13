from abc import ABC, abstractmethod

import torch


class AbstractSelector(ABC):
    """Defines the interface for all bandit action selectors.
    Given a tensor of scores per action, the selector chooses the best action (i.e. an arm)
    or the best set of actions (i.e. a super arm in combinatorial bandits). The selector
    returns a one hot encoded tensor of the chosen actions.
    """

    @abstractmethod
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Selects a single best action, or a set of actions in the case of combinatorial bandits.

        Args:
            scores (torch.Tensor): Tensor of shape (batch_size, n_arms).
                This may contain a probability distribution per sample (when used for thompson sampling)
                or simply a score per arm (e.g. for UCB).
                In case of combinatorial bandits, these are the scores per arm from which
                the oracle selects a super arm (e.g. simply top-k).

        Returns:
            chosen_actions (torch.Tensor): One hot encoded actions that were chosen.
                Shape: (batch_size, n_arms).
        """
        pass


class ArgMaxSelector(AbstractSelector):
    """Selects the action with the highest score from a batch of scores."""

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Select the action with the highest score for each sample in the batch.

        Args:
            scores: Tensor of shape (batch_size, n_arms) containing scores for each action.

        Returns:
            Tensor of shape (batch_size, n_arms) containing one-hot encoded selected actions.
        """
        _, n_arms = scores.shape
        return torch.nn.functional.one_hot(
            torch.argmax(scores, dim=1), num_classes=n_arms
        )


class EpsilonGreedySelector(AbstractSelector):
    """Implements an epsilon-greedy action selection strategy."""

    def __init__(self, epsilon: float = 0.1) -> None:
        """Initialize the epsilon-greedy selector.

        Args:
            epsilon: Exploration probability (default: 0.1). Must be between 0 and 1.
        """
        assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1"
        self.epsilon = epsilon

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Select actions using the epsilon-greedy strategy for each sample in the batch.

        Args:
            scores: Tensor of shape (batch_size, n_arms) containing scores for each action.

        Returns:
            Tensor of shape (batch_size, n_arms) containing one-hot encoded selected actions.
        """
        batch_size, n_arms = scores.shape

        random_vals = torch.rand(batch_size)
        explore_mask = random_vals < self.epsilon

        greedy_actions = torch.argmax(scores, dim=1)
        random_actions = torch.randint(0, n_arms, (batch_size,))

        selected_actions = torch.where(explore_mask, random_actions, greedy_actions)

        return torch.nn.functional.one_hot(selected_actions, num_classes=n_arms)


class TopKSelector(AbstractSelector):
    """Selects the top k actions with the highest scores."""

    def __init__(self, k: int):
        """Initialize the top-k selector.

        Args:
            k: Number of actions to select. Must be positive.
        """
        assert k > 0, "k must be positive"
        self.k = k

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Select the top k actions with highest scores for each sample in the batch.

        Args:
            scores: Tensor of shape (batch_size, n_arms) containing scores for each action.

        Returns:
            Tensor of shape (batch_size, n_arms) containing one-hot encoded selected actions,
            where exactly k entries are 1 per sample.
        """
        batch_size, n_arms = scores.shape
        assert (
            self.k <= n_arms
        ), f"k ({self.k}) cannot be larger than number of arms ({n_arms})"

        _, top_k_indices = torch.topk(scores, k=self.k, dim=1)

        selected_actions = torch.zeros(batch_size, n_arms, dtype=torch.int64)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.k)
        selected_actions[batch_indices, top_k_indices] = 1

        return selected_actions
