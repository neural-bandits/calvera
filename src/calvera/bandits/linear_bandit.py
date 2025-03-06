from abc import ABC, abstractmethod
from typing import Any, cast

import torch

from calvera.bandits.abstract_bandit import AbstractBandit
from calvera.bandits.action_input_type import ActionInputType
from calvera.utils.data_storage import AbstractBanditDataBuffer, BufferDataFormat
from calvera.utils.selectors import AbstractSelector


class LinearBandit(AbstractBandit[ActionInputType], ABC):
    """Baseclass for linear bandit algorithms.

    Implements the update method for linear bandits. Also adds all necesary attributes.
    """

    # The precision matrix is the inverse of the covariance matrix of the chosen contextualized actions.
    precision_matrix: torch.Tensor
    b: torch.Tensor
    theta: torch.Tensor

    def __init__(
        self,
        n_features: int,
        buffer: AbstractBanditDataBuffer[Any, Any] | None = None,
        selector: AbstractSelector | None = None,
        train_batch_size: int = 32,
        eps: float = 1e-2,
        lambda_: float = 1.0,
        lazy_uncertainty_update: bool = False,
        clear_buffer_after_train: bool = True,
        use_sherman_morrison_update: bool = False,
    ) -> None:
        """Initializes the LinearBanditModule.

        Args:
            n_features: The number of features in the bandit model.
            buffer: The buffer used for storing the data for continuously updating the neural network.
                For the linear bandit, it should always be an InMemoryDataBuffer with an AllDataBufferStrategy
                because the buffer is cleared after each update.
            selector: The selector used to choose the best action. Default is ArgMaxSelector (if None).
            train_batch_size: The mini-batch size used for the train loop (started by `trainer.fit()`).
            eps: Small value to ensure invertibility of the precision matrix. Added to the diagonal.
            lambda_: Prior variance for the precision matrix. Acts as a regularization parameter.
            lazy_uncertainty_update: If True the precision matrix will not be updated during forward, but during the
                update step.
            clear_buffer_after_train: If True the buffer will be cleared after training. This is necessary because the
                data is not needed anymore after training.
            use_sherman_morrison_update: If True the precision matrix will be updated using the Sherman-Morrison
                formula. This is a more efficient way to update the precision matrix than the default method. The
                problem however is that the Sherman-Morrison formula can lead to substantial numerical errors.
        """
        super().__init__(
            n_features=n_features,
            buffer=buffer,
            train_batch_size=train_batch_size,
            selector=selector,
        )

        self.save_hyperparameters(
            {
                "lazy_uncertainty_update": lazy_uncertainty_update,
                "eps": eps,
                "lambda_": lambda_,
                "clear_buffer_after_train": clear_buffer_after_train,
                "use_sherman_morrison_update": use_sherman_morrison_update,
            }
        )

        # Disable Lightning's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        self._init_linear_params()

        # Please don't ask. Lightning requires any parameter to be registered in order to train it on cuda.
        self.register_parameter("_", None)

    def _init_linear_params(self) -> None:
        n_features = cast(int, self.hparams["n_features"])
        lambda_ = cast(float, self.hparams["lambda_"])

        # Model parameters
        self.register_buffer(
            "precision_matrix",
            torch.eye(n_features, device=self.device) * lambda_,
        )
        self.register_buffer("b", torch.zeros(n_features, device=self.device))
        self.register_buffer("theta", torch.zeros(n_features, device=self.device))

    def _predict_action(
        self, contextualized_actions: ActionInputType, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chosen_actions, p = self._predict_action_hook(contextualized_actions, **kwargs)
        if not self.hparams["lazy_uncertainty_update"]:
            assert isinstance(contextualized_actions, torch.Tensor), "contextualized_actions must be a torch.Tensor"
            indices = chosen_actions.nonzero(as_tuple=True)
            chosen_contextualized_actions = contextualized_actions[indices[0], indices[1]]
            self._update_precision_matrix(chosen_contextualized_actions)

        return chosen_actions, p

    @abstractmethod
    def _predict_action_hook(
        self, contextualized_actions: ActionInputType, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for subclasses to implement the action selection logic."""
        pass

    def _update(
        self,
        batch: BufferDataFormat[ActionInputType],
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
        assert len(batch) == 4, (
            "Batch must contain four tensors: (contextualized_actions, embedded_actions, rewards, chosen_actions)."
            "`embedded_actions` and `chosen_actions` can be None."
        )

        chosen_contextualized_actions = batch[0]
        assert isinstance(chosen_contextualized_actions, torch.Tensor), "chosen_contextualized_actions must be a tensor"
        realized_rewards: torch.Tensor = batch[2]

        # Update the self.bandit
        self._perform_update(chosen_contextualized_actions, realized_rewards)

        return -realized_rewards.mean()

    def _perform_update(
        self,
        chosen_actions: torch.Tensor,
        realized_rewards: torch.Tensor,
    ) -> None:
        """Perform an update step on the linear bandit.

        Perform an update step on the linear bandit given the actions that were chosen and the rewards that were
        observed. The difference between `_update` and `_perform_update` is that `_update` is the method that is called
        by the lightning training loop and therefore has the signature
        `_update(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor` and is also logging.
        We require `_perform_update` for the NeuralLinearBandit which calls this method to update the parameters of
        its linear head.

        Args:
            chosen_actions: The chosen contextualized actions in this batch. Shape: (batch_size, n_features)
            realized_rewards: The realized rewards of the chosen action in this batch. Shape: (batch_size,)
        """
        # Other assertions are done in the _update method
        assert chosen_actions.shape[2] == self.hparams["n_features"], (
            f"Chosen actions must have shape (batch_size, n_chosen_arms, n_features) and n_features must match the "
            f"bandit's n_features. Got {chosen_actions.shape[1]} but expected {self.hparams['n_features']}."
        )

        assert chosen_actions.shape[1] == 1, (
            f"For now we only support chosing one action at once. Instead got {chosen_actions.shape[1]}."
            "Combinatorial bandits will be implemented in the future."
        )
        chosen_actions = chosen_actions.squeeze(1)
        realized_rewards = realized_rewards.squeeze(1)

        if self.hparams["lazy_uncertainty_update"]:
            self._update_precision_matrix(chosen_actions)

        self.b.add_(chosen_actions.T @ realized_rewards)  # shape: (features,)
        self.theta.copy_(self.precision_matrix @ self.b)

        assert (
            self.b.ndim == 1 and self.b.shape[0] == self.hparams["n_features"]
        ), "updated b should have shape (n_features,)"

        assert (
            self.theta.ndim == 1 and self.theta.shape[0] == self.hparams["n_features"]
        ), "Theta should have shape (n_features,)"

    def _update_precision_matrix(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        # Calculate new precision matrix.

        old_precision = self.precision_matrix.clone()
        if self.hparams["use_sherman_morrison_update"]:
            # Perform the Sherman-Morrison update.
            inverse_term = torch.inverse(
                torch.eye(chosen_actions.shape[0], device=self.device)
                + chosen_actions @ old_precision @ chosen_actions.T
            )
            # Update using the SMW formula: P_new = P - P A^T (I + A P A^T)^{-1} A P
            self.precision_matrix.copy_(
                old_precision - old_precision @ chosen_actions.T @ inverse_term @ chosen_actions @ old_precision
            )
            self.precision_matrix.add_(
                torch.eye(self.precision_matrix.shape[0], device=self.device) * self.hparams["eps"]
            )  # add small value to diagonal to ensure invertibility

        else:
            # Perform the 'standard' update.
            cov_matrix = torch.linalg.inv(old_precision) + chosen_actions.T @ chosen_actions
            self.precision_matrix.copy_(
                torch.linalg.inv(
                    cov_matrix + torch.eye(self.precision_matrix.shape[0], device=self.device) * self.hparams["eps"]
                )
            )

        self.precision_matrix.mul_(0.5).add_(self.precision_matrix.T.clone())

        # should be symmetric
        assert torch.allclose(self.precision_matrix, self.precision_matrix.T), "Precision matrix must be symmetric"

        return self.precision_matrix

    def on_train_end(self) -> None:
        """Clear the buffer after training because the past data is not needed anymore."""
        super().on_train_end()
        if self.hparams["clear_buffer_after_train"]:
            self.buffer.clear()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle saving custom LinearBandit state.

        Args:
            checkpoint: Dictionary to save the state into.
        """
        super().on_save_checkpoint(checkpoint)

        checkpoint["precision_matrix"] = self.precision_matrix
        checkpoint["b"] = self.b
        checkpoint["theta"] = self.theta

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle loading custom LinearBandit state.

        Args:
            checkpoint: Dictionary containing the state to load.
        """
        super().on_load_checkpoint(checkpoint)

        if "precision_matrix" in checkpoint:
            self.register_buffer("precision_matrix", checkpoint["precision_matrix"])

        if "b" in checkpoint:
            self.register_buffer("b", checkpoint["b"])

        if "theta" in checkpoint:
            self.register_buffer("theta", checkpoint["theta"])
