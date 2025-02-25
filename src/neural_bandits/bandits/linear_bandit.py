from abc import ABC, abstractmethod
from typing import Any

import torch

from neural_bandits.bandits.abstract_bandit import AbstractBandit


class LinearBandit(AbstractBandit, ABC):
    # The precision matrix is the inverse of the covariance matrix of the chosen contextualized actions.
    precision_matrix: torch.Tensor
    b: torch.Tensor
    theta: torch.Tensor

    def __init__(
        self,
        n_features: int,
        eps: float = 1e-2,
        lazy_uncertainty_update: bool = False,
        **kw_args: Any,
    ) -> None:
        """Initializes the LinearBanditModule.
        Args:
            n_features: The number of features in the bandit model.
            eps: Small value to ensure invertibility of the precision matrix. Added to the diagonal.
            lazy_uncertainty_update: If True the precision matrix will not be updated during forward, but during the
            update step.
        """
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.lazy_uncertainty_update = lazy_uncertainty_update

        # Disable Lightning's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save Hyperparameters
        hyperparameters = {
            "n_features": n_features,
            "eps": eps,
            "lazy_uncertainty_update": lazy_uncertainty_update,
            **kw_args,
        }
        self.save_hyperparameters(hyperparameters)

        # Model parameters
        self.register_buffer("precision_matrix", torch.eye(n_features))
        self.register_buffer("b", torch.zeros(n_features))
        self.register_buffer("theta", torch.zeros(n_features))

    def _predict_action(
        self, contextualized_actions: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chosen_actions, p = self._predict_action_hook(contextualized_actions, **kwargs)
        if not self.lazy_uncertainty_update:
            chosen_contextualized_actions = contextualized_actions[
                torch.arange(contextualized_actions.shape[0], device=self.device),
                chosen_actions.argmax(dim=1),
            ]
            self._update_precision_matrix(chosen_contextualized_actions)

        return chosen_actions, p

    @abstractmethod
    def _predict_action_hook(
        self, contextualized_actions: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for subclasses to implement the action selection logic."""
        pass

    def _update(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform an update step on the linear bandit model.

        Args:
            batch: The output of your data iterable, normally a DataLoader:
                chosen_contextualized_actions: shape (batch_size, n_chosen_actions, n_features).
                realized_rewards: shape (batch_size, n_chosen_actions).
            batch_idx: The index of this batch. Note that if a separate DataLoader is used for each step,
                this will be reset for each new data loader.

        Returns:
            The loss value as the negative mean of all realized_rewards in this batch.
                Shape: (1,). Since we do not use the lightning optimizer, this value is only relevant
                for logging/visualization of the training process.
        """
        chosen_contextualized_actions: torch.Tensor = batch[0]
        realized_rewards: torch.Tensor = batch[1]

        # Update the self.bandit
        self.update(chosen_contextualized_actions, realized_rewards)

        self.log(
            "reward",
            realized_rewards.sum(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return -realized_rewards.mean()

    def update(
        self,
        chosen_actions: torch.Tensor,
        realized_rewards: torch.Tensor,
    ) -> None:
        """
        Perform an update step on the linear bandit given the actions that were chosen and the rewards that were observed.

        Args:
            chosen_actions: The chosen contextualized actions in this batch. Shape: (batch_size, n_features)
            realized_rewards: The realized rewards of the chosen action in this batch. Shape: (batch_size,)
        """

        assert (
            chosen_actions.ndim == 3
        ), f"Chosen actions must have shape (batch_size, n_chosen_arms, n_features) but got shape {chosen_actions.shape}"

        assert (
            realized_rewards.ndim == 2
        ), f"Realized rewards must have shape (batch_size, n_chosen_actions) but got shape {realized_rewards.shape}"

        assert (
            chosen_actions.shape[0] == realized_rewards.shape[0]
            and chosen_actions.shape[1] == realized_rewards.shape[1]
        ), f"Batch size and num_chosen actions of chosen_actions and realized_rewards must match. Got {chosen_actions.shape[0]} and {realized_rewards.shape[0]}."

        assert (
            chosen_actions.shape[2] == self.hparams["n_features"]
        ), f"Chosen actions must have shape (batch_size, n_chosen_arms, n_features) and n_features must match the bandit's n_features. Got {chosen_actions.shape[1]} but expected {self.hparams['n_features']}."

        assert (
            chosen_actions.shape[1] == 1
        ), f"For now we only support chosing one action at once. Instead got {chosen_actions.shape[1]}. Combinatorial bandits will be implemented in the future."
        chosen_actions = chosen_actions.squeeze(1)
        realized_rewards = realized_rewards.squeeze(1)
        # TODO: Implement linear combinatorial bandits according to Efficient Learning in Large-Scale Combinatorial Semi-Bandits (https://arxiv.org/pdf/1406.7443)

        if self.lazy_uncertainty_update:
            self._update_precision_matrix(chosen_actions)

        self.b.add_(chosen_actions.T @ realized_rewards)  # shape: (features,)
        self.theta.copy_(self.precision_matrix @ self.b)

        assert (
            self.b.ndim == 1 and self.b.shape[0] == self.n_features
        ), "updated b should have shape (n_features,)"

        assert (
            self.theta.ndim == 1 and self.theta.shape[0] == self.n_features
        ), "Theta should have shape (n_features,)"

    def _update_precision_matrix(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        # Calculate new precision matrix using the Sherman-Morrison-Woodbury formula.
        batch_size = chosen_actions.shape[0]
        inverse_term = torch.inverse(
            torch.eye(batch_size, device=self.device)
            + chosen_actions @ self.precision_matrix.clone() @ chosen_actions.T
        )

        self.precision_matrix.add_(
            -self.precision_matrix.clone()
            @ chosen_actions.T
            @ inverse_term
            @ chosen_actions
            @ self.precision_matrix.clone()
        )
        self.precision_matrix.mul_(0.5).add_(self.precision_matrix.T.clone())

        self.precision_matrix.add_(
            torch.eye(self.precision_matrix.shape[0], device=self.device) * self.eps
        )  # add small value to diagonal to ensure invertibility

        # should be symmetric
        assert torch.allclose(
            self.precision_matrix, self.precision_matrix.T
        ), "M must be symmetric"
        return self.precision_matrix
