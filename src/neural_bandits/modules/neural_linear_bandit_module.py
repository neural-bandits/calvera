from typing import Optional

import lightning as pl
import torch
from lightning.pytorch.core.optimizer import LightningOptimizer

from neural_bandits.algorithms.linear_bandits import LinearTSBandit
from neural_bandits.algorithms.neural_linear import NeuralLinearBandit
from neural_bandits.modules.abstract_bandit_module import AbstractBanditModule
from neural_bandits.modules.linear_bandit_module import LinearBanditModule


class NeuralLinearBanditModule(AbstractBanditModule[NeuralLinearBandit]):
    """
    Module for training a Neural Linear bandit model.
    The Neural Linear algorithm is described in the paper Riquelme et al., 2018, Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        n_features: int,
        n_embedding_size: int,
        embedding_update_interval: Optional[int] = 32,
    ) -> None:
        """
        Initializes the NeuralLinearBanditModule.

        Args:
            embedding_model (torch.nn.Module): The embedding model (neural network) to be used.
            n_features (int): The number of features in the input data.
            n_embedding_size (int): The size of the embedding produced by the embedding model.
            embedding_update_interval (Optional[int]): The interval (in steps) at which the embedding model is updated. Default is 32. None means the embedding model is never updated.
        """
        self.linear_head_trainer: LinearBanditModule[LinearTSBandit] = (
            LinearBanditModule(
                linear_bandit_type=LinearTSBandit,  # type: ignore
                n_features=n_embedding_size,
            )
        )

        self.embedding_model = embedding_model
        self.bandit = NeuralLinearBandit(
            n_features=n_features,
            embedding_model=embedding_model,
        )

        self.automatic_optimization = False

        self.n_features = n_features
        self.n_embedding_size = n_embedding_size

        self.embedding_update_interval = embedding_update_interval

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a training step on the neural linear bandit model.
        """
        contextualized_actions: torch.Tensor = batch[0]
        rewards: torch.Tensor = batch[1]

        assert (
            contextualized_actions.shape[2] == self.bandit.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        # TODO: Assert the shape of the rewards tensor

        embedded = self.embedding_model(contextualized_actions)

        assert (
            contextualized_actions.shape[2] == self.n_embedding_size
        ), "Contextualised actions must have shape (batch_size, n_arms, n_embedding_sizes)"

        if batch_idx % self.embedding_update_interval == 0:
            self._update_nn(batch)

        self._update_head(batch)

        return -rewards.mean()

    def _update_head(
        self,
        batch: torch.Tensor,
    ) -> None:
        """Perform an update step on the head of the neural linear bandit"""
        # update using the linear trainer
        self.linear_head_trainer.manual_update()

    def _update_nn(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a full update on the network of the neural linear bandit."""
        opt: LightningOptimizer = self.optimizers()  # type: ignore
        opt.zero_grad()
        loss = self._compute_loss(batch[0], batch[1])
        self.manual_backward(loss)
        opt.step()
        return loss

    def _compute_loss(
        self, embedded: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        # Example MSE loss between embedding and rewards
        # Adjust as needed
        pred = self.bandit.linear_head(embedded)
        return torch.nn.functional.mse_loss(pred, rewards)
