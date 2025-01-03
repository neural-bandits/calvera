import torch

from ..algorithms.linear_bandits import LinearBandit
from .abstract_trainer import AbstractTrainer


class LinearTrainer(AbstractTrainer[LinearBandit]):
    def __init__(self) -> None:
        pass

    def update(
        self,
        bandit: LinearBandit,
        rewards: torch.Tensor,  # shape: (batch_size,)
        chosen_actions: torch.Tensor,  # shape: (batch_size, features)
    ) -> LinearBandit:
        """Perform an update"""
        batch_size = chosen_actions.shape[0]
        assert rewards.shape == (batch_size,), "Rewards must have shape (batch_size,)"
        assert (
            len(chosen_actions.shape) == 2
            and chosen_actions.shape[1] == bandit.n_features
        ), "Chosen actions must have shape (batch_size, features). Mis-match with bandit features."

        # Update the bandit
        bandit.M += chosen_actions.T @ chosen_actions  # shape: (features, features)
        bandit.b += chosen_actions.T @ rewards  # shape: (features,)
        bandit.theta = torch.linalg.solve(bandit.M, bandit.b)  # shape: (features,)

        return bandit
