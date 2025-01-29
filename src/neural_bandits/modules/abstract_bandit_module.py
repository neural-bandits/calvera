from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import lightning as pl
import torch

from neural_bandits.algorithms.abstract_bandit import AbstractBandit

BanditType = TypeVar("BanditType", bound="AbstractBandit")


class AbstractBanditModule(ABC, Generic[BanditType], pl.LightningModule):
    """Abstract class for the training behaviour of a bandit model (BanditType)."""

    bandit: BanditType

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.bandit(x)  # type: ignore

    @abstractmethod
    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single update step. See the documentation for the LightningModule training_step method.

        Args:
            batch: The output of your data iterable, normally a DataLoader.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch. (only if multiple dataloaders used)

        Returns:
            The loss value. In most cases it makes sense to return the negative reward.
        """
        pass
