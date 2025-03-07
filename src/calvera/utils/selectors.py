from abc import ABC, abstractmethod
from typing import Any

import torch


class AbstractSelector(ABC):
    """Defines the interface for all bandit action selectors.

    Given a tensor of scores per action, the selector chooses an action (i.e. an arm)
    or a set of actions (i.e. a super arm in combinatorial bandits). The selector
    returns a one hot encoded tensor of the chosen actions.
    """

    @abstractmethod
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Selects a single action, or a set of actions in the case of combinatorial bandits.

        Args:
            scores: Scores for each action. Shape: (batch_size, n_arms).
                This may contain a probability distribution per sample or simply a score per
                arm (e.g. for UCB). In case of combinatorial bandits, these are the scores
                per arm from which the oracle selects a super arm (e.g. simply top-k).

        Returns:
            One hot encoded actions that were chosen. Shape: (batch_size, n_arms).
        """
        pass

    def get_state_dict(self) -> dict[str, Any]:
        """Return a serializable state dictionary for checkpointing.

        Returns:
            A dictionary containing the selector's type information.
        """
        return {"type": self.__class__.__name__}

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> "AbstractSelector":
        """Create a selector from a state dictionary.

        Args:
            state: Dictionary containing the selector's state information.

        Returns:
            A new selector instance initialized with the state.

        Raises:
            ValueError: If the selector type is unknown.
        """
        selector_type = state["type"]
        if selector_type == "EpsilonGreedySelector":
            selector = EpsilonGreedySelector(epsilon=state["epsilon"])
            selector.generator.set_state(state["generator_state"])
            return selector
        elif selector_type == "TopKSelector":
            return TopKSelector(k=state["k"])
        elif selector_type == "ArgMaxSelector":
            return ArgMaxSelector()
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")


class ArgMaxSelector(AbstractSelector):
    """Selects the action with the highest score from a batch of scores."""

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Select the action with the highest score for each sample in the batch.

        Args:
            scores: Scores for each action. Shape: (batch_size, n_arms).

        Returns:
            One-hot encoded selected actions. Shape: (batch_size, n_arms).
        """
        _, n_arms = scores.shape
        return torch.nn.functional.one_hot(torch.argmax(scores, dim=1), num_classes=n_arms)


class EpsilonGreedySelector(AbstractSelector):
    """Implements an epsilon-greedy action selection strategy."""

    def __init__(self, epsilon: float = 0.1, seed: int | None = None) -> None:
        """Initialize the epsilon-greedy selector.

        Args:
            epsilon: Exploration probability. Must be between 0 and 1.
            seed: Random seed for the generator. Defaults to None (explicit seed used).
        """
        assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1"
        self.epsilon = epsilon
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Select actions using the epsilon-greedy strategy for each sample in the batch.

        If the device of the scores
        tensor is different from the device of the generator, the generator is moved to the device of the scores tensor.

        Args:
            scores: Scores for each action. Shape: (batch_size, n_arms).

        Returns:
            One-hot encoded selected actions. Shape: (batch_size, n_arms).
        """
        if scores.device != self.generator.device:
            self.generator = torch.Generator(device=scores.device)
            self.generator.set_state(self.generator.get_state())

        batch_size, n_arms = scores.shape

        random_vals = torch.rand(batch_size, generator=self.generator, device=scores.device)
        explore_mask = random_vals < self.epsilon

        greedy_actions = torch.argmax(scores, dim=1)
        random_actions = torch.randint(0, n_arms, (batch_size,), generator=self.generator, device=scores.device)

        selected_actions = torch.where(explore_mask, random_actions, greedy_actions)

        return torch.nn.functional.one_hot(selected_actions, num_classes=n_arms)

    def get_state_dict(self) -> dict[str, Any]:
        """Return a serializable state dictionary for checkpointing.

        Returns:
            Dictionary containing the selector's state information.
        """
        state = super().get_state_dict()
        state["epsilon"] = self.epsilon
        state["generator_state"] = self.generator.get_state()
        return state


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
            scores: Scores for each action. Shape: (batch_size, n_arms).

        Returns:
            One-hot encoded selected actions where exactly k entries are 1 per sample.
            Shape: (batch_size, n_arms).
        """
        batch_size, n_arms = scores.shape
        assert self.k <= n_arms, f"k ({self.k}) cannot be larger than number of arms ({n_arms})"

        selected_actions = torch.zeros(batch_size, n_arms, dtype=torch.int64, device=scores.device)
        remaining_scores = scores.clone()

        selected_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)

        for _ in range(self.k):
            max_indices = torch.argmax(remaining_scores, dim=1)

            batch_indices = torch.arange(batch_size, device=scores.device)
            selected_actions[batch_indices, max_indices] = 1
            selected_mask[batch_indices, max_indices] = True

            remaining_scores[selected_mask] = float("-inf")

        return selected_actions

    def get_state_dict(self) -> dict[str, Any]:
        """Return a serializable state dictionary for checkpointing.

        Returns:
            Dictionary containing the selector's state information.
        """
        state = super().get_state_dict()
        state["k"] = self.k
        return state


class RandomSelector(AbstractSelector):
    """Selects k random actions from the available actions."""

    def __init__(self, k: int = 1, seed: int | None = None):
        """Initialize the random selector.

        Args:
            k: Number of actions to select. Must be positive.
            seed: Random seed for the generator. Defaults to None.
        """
        self.k = k
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Select k random actions for each sample in the batch.

        Args:
            scores: Scores for each action. Shape: (batch_size, n_arms).

        Returns:
            One-hot encoded selected actions where exactly k entries are 1 per sample.
            Shape: (batch_size, n_arms).
        """
        if scores.device != self.generator.device:
            self.generator = torch.Generator(device=scores.device)
            self.generator.set_state(self.generator.get_state())

        batch_size, n_arms = scores.shape
        selected_actions = torch.zeros(batch_size, n_arms, dtype=torch.int64, device=scores.device)
        for i in range(batch_size):
            perm = torch.randperm(n_arms, generator=self.generator, device=scores.device)
            selected_actions[i, perm[: self.k]] = 1
        return selected_actions
