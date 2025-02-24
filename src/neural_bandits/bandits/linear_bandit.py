from typing import Any

import torch

from neural_bandits.bandits.abstract_bandit import AbstractBandit


class LinearBandit(AbstractBandit[torch.Tensor]):
    def __init__(
        self,
        n_features: int,
        **kw_args: Any,
    ) -> None:
        """Initializes the LinearBanditModule.
        Args:
            n_features: The number of features in the bandit model.
        """
        super().__init__()
        self.n_features = n_features

        # Disable Lightning's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save Hyperparameters
        hyperparameters = {
            "n_features": n_features,
            **kw_args,
        }
        self.save_hyperparameters(hyperparameters)

        # Model parameters
        self.precision_matrix: torch.Tensor = torch.eye(n_features)
        self.b = torch.zeros(n_features)
        self.theta = torch.zeros(n_features)

    def _update(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
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
            realized_rewards.mean(),
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

        # Calculate new precision Matrix M using the Sherman-Morrison formula
        denominator = 1 + (
            (chosen_actions @ self.precision_matrix) * chosen_actions
        ).sum(dim=1).sum(dim=0)
        assert torch.abs(denominator) > 0, "Denominator must not be zero or nan"

        self.precision_matrix = (
            self.precision_matrix
            - (
                self.precision_matrix
                @ torch.einsum("bi,bj->bij", chosen_actions, chosen_actions).sum(dim=0)
                @ self.precision_matrix
            )
            / denominator
        )
        self.precision_matrix = 0.5 * (self.precision_matrix + self.precision_matrix.T)
        # should be symmetric
        assert torch.allclose(
            self.precision_matrix, self.precision_matrix.T
        ), "M must be symmetric"

        self.b += chosen_actions.T @ realized_rewards  # shape: (features,)
        self.theta = self.precision_matrix @ self.b

        assert (
            self.b.ndim == 1 and self.b.shape[0] == self.n_features
        ), "updated b should have shape (n_features,)"

        assert (
            self.theta.ndim == 1 and self.theta.shape[0] == self.n_features
        ), "Theta should have shape (n_features,)"
