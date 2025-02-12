from abc import ABC, abstractmethod
from typing import Any

import lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch

class AbstractBandit(ABC, pl.LightningModule):
    """Defines the interface for all Bandit algorithms by implementing pytorch Lightning Module methods."""

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Given the contextualized actions, selects a single best action,
        or a set of actions in the case of combinatorial bandits. This can be computed
        for many samples in one batch.

        Args:
            contextualized_actions (torch.Tensor): Tensor of shape (batch_size, n_actions, n_features).

        Returns:
            tuple:
            - chosen_actions (torch.Tensor): One-hot encoding of which actions were chosen.
                Shape: (batch_size, n_chosen_actions).
            - p (torch.Tensor): The probability of the chosen actions. In the combinatorial case,
                this will be a super set of actions. Non-probabilistic algorithms should always return 1.
                Shape: (batch_size, n_chosen_actions).
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(
        self,
        contextualized_actions: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass is computed batch-wise. Given the contextualized actions, selects a single best action,
        or a set of actions in the case of combinatorial bandits.

        Next to the action(s), the selector also returns the probability of chosing this action. This will allow for logging and Batch Learning from Logged Bandit Feedback (BLBF).
        Deterministic algorithms like UCB will always return 1.

        Args:
            contextualized_actions (torch.Tensor): Tensor of shape (batch_size, n_actions, n_features).

        Returns:
            tuple:
            - chosen_actions (torch.Tensor): One-hot encoding of which actions were chosen.
                Shape: (batch_size, n_chosen_actions).
            - p (torch.Tensor): The probability of the chosen actions. In the combinatorial case,
                this will be one probability for the super set of actions. Deterministic algorithms (like UCB) should always return 1.
                Shape: (batch_size, ).
        """
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform a single update step. See the documentation for the LightningModule's `training_step` method.
        Acts as a wrapper for the `update_step` method in case we want to change something for every bandit or
        use the update independently from lightning, e.g. in tests.

        Args:
            batch (torch.Tensor): The output of your data iterable, usually a DataLoader:
                contextualized_actions (torch.Tensor): shape (batch_size, n_chosen_actions, n_features).
                realized_rewards (torch.Tensor): shape (batch_size, n_chosen_actions).

            batch_idx (int): The index of this batch. Note that if a separate DataLoader is used for each step,
                this will be reset for each new data loader.

            data_loader_idx (int): The index of the data loader. This is useful if you have multiple data loaders
                at once and want to do something different for each one.
        Returns:
            torch.Tensor: The loss value. In most cases, it makes sense to return the negative reward.
                Shape: (1,). Since we do not use the lightning optimizer, this value is only relevant
                for logging/visualization of the training process.
        """
        return self.update_step(*args, **kwargs)

    @abstractmethod
    def update_step(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Abstract method to perform a single update step. Should be implemented by the concrete bandit classes.

        Args:
            batch (torch.Tensor): The output of your data iterable, usually a DataLoader:
                contextualized_actions (torch.Tensor): shape (batch_size, n_chosen_actions, n_features).
                realized_rewards (torch.Tensor): shape (batch_size, n_chosen_actions).

            batch_idx (int): The index of this batch. Note that if a separate DataLoader is used for each step,
                this will be reset for each new data loader.

            data_loader_idx (int): The index of the data loader. This is useful if you have multiple data loaders
                at once and want to do something different for each one.
        Returns:
            torch.Tensor: The loss value. In most cases, it makes sense to return the negative reward.
                Shape: (1,). Since we do not use the lightning optimizer, this value is only relevant
                for logging/visualization of the training process.
        """
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Usually, this method is required for pytorch lightning to define which optimizers are used for training.
        Since we do not use the lightning optimizer and optimize on our own, we return None per default.
        """
        return None
