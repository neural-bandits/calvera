import torch
from typing import Generic, TypeVar

from ..algorithms.linear_bandits import LinearBandit
from .abstract_trainer import AbstractTrainer

LinearBanditType = TypeVar("LinearBanditType", bound="LinearBandit")


class LinearTrainer(AbstractTrainer[LinearBanditType], Generic[LinearBanditType]):
    def __init__(self) -> None:
        pass

    def update(
        self,
        bandit: LinearBanditType,
        rewards: torch.Tensor,  # shape: (batch_size,)
        chosen_actions: torch.Tensor,  # shape: (batch_size, features)
    ) -> LinearBanditType:
        """Perform an update"""

        # Update the bandit
        bandit.M += chosen_actions.T @ chosen_actions  # shape: (features, features)
        bandit.b += chosen_actions.T @ rewards  # shape: (features,)
        bandit.theta = torch.linalg.solve(bandit.M, bandit.b)  # shape: (features,)

        return bandit
