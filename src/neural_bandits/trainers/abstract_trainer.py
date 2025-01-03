from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from ..algorithms.abstract_bandit import AbstractBandit

BanditType = TypeVar("BanditType", bound="AbstractBandit")


class AbstractTrainer(ABC, Generic[BanditType]):  # this is now
    @abstractmethod
    def update(
        self,
        bandit: BanditType,
        rewards: torch.Tensor,
        chosen_actions: torch.Tensor,
    ) -> BanditType:
        """Perform a single update step"""
        # TODO(rob2u): assert correct shapes (rewards: (batch_size, 1), chosen_actions: (batch_size, dim))
