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
            scores (torch.Tensor): Tensor of shape (batch_size, n_actions).
            This may contain a probability distribution per sample (when used for thompson sampling) or simply a score per arm (e.g. for UCB).
            In case of combinatorial bandits, these are the scores per arm from which the oracle selects a super arm (e.g. simply top-k).

        Returns:
            chosen_actions (torch.Tensor): One hot encoded actions that were chosen.
                Shape: (batch_size, n_actions).
        """
        pass


# TODO: Merge this with the other selectors. This can be removed later but the output shape of this one is different because we need to one-hot encode!!!
# Also I like the "ArgMax" more than the "Argmax" but you choose...
class ArgMaxSelector(AbstractSelector):
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.argmax(scores, dim=1), num_classes=scores.shape[1]
        )  # shape: (batch_size, n_actions)


class ArgmaxSelector:
    """Selects the action with the highest probability from a batch of distributions."""

    def __call__(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Select the action with the highest probability for each distribution in the batch.

        Args:
            probabilities: Tensor of shape [batch, num_actions] (or 1D tensor for a single distribution).
                           Each row must sum to 1.

        Returns:
            Tensor of shape [batch] containing the index of the highest-probability action for each distribution.
        """
        self._validate_probabilities(probabilities)
        return probabilities.argmax(dim=1)

    def _validate_probabilities(self, probabilities: torch.Tensor) -> None:
        """
        Validates that each row in the probabilities tensor sums to 1 and that all values are in [0, 1].

        Args:
            probabilities: A 2D tensor of probabilities.
        """
        row_sums = probabilities.sum(dim=1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-5
        ), f"Each row must sum to 1, got sums {row_sums}"
        assert (
            (probabilities >= 0) & (probabilities <= 1)
        ).all(), "All probabilities must be in the range [0, 1]"


class EpsilonGreedySelector:
    """Implements an epsilon-greedy action selection strategy for a batch of distributions."""

    def __init__(self, epsilon: float = 0.1) -> None:
        """
        Initialize the epsilon-greedy selector.

        Args:
            epsilon: Exploration probability (default: 0.1). Must be between 0 and 1.
        """
        assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1"
        self.epsilon = epsilon

    def __call__(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Select actions using the epsilon-greedy strategy for each distribution in the batch.

        Args:
            probabilities: Tensor of shape [batch, num_actions] (or 1D tensor for a single distribution).
                           Each row must sum to 1.

        Returns:
            Tensor of shape [batch] containing the selected action indices.
        """
        self._validate_probabilities(probabilities)
        batch_size, num_actions = probabilities.shape

        random_vals = torch.rand(batch_size)
        explore_mask = random_vals < self.epsilon

        greedy_actions = probabilities.argmax(dim=1)
        random_actions = torch.randint(0, num_actions, (batch_size,))

        return torch.where(explore_mask, random_actions, greedy_actions)

    def _validate_probabilities(self, probabilities: torch.Tensor) -> None:
        """
        Validates that each row in the probabilities tensor sums to 1 and that all values are in [0, 1].

        Args:
            probabilities: A 2D tensor of probabilities.
        """
        row_sums = probabilities.sum(dim=1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), rtol=1e-5
        ), f"Each row must sum to 1, got sums {row_sums}"
        assert (
            (probabilities >= 0) & (probabilities <= 1)
        ).all(), "All probabilities must be in the range [0, 1]"
