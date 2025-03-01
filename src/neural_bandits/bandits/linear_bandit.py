from abc import ABC, abstractmethod
from typing import Any, Optional, cast

import torch

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.bandits.action_input_type import ActionInputType
from neural_bandits.utils.data_storage import AbstractBanditDataBuffer
from neural_bandits.utils.selectors import ArgMaxSelector, EpsilonGreedySelector, TopKSelector


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
        buffer: Optional[AbstractBanditDataBuffer[Any, Any]] = None,
        train_batch_size: int = 32,
        eps: float = 1e-2,
        lambda_: float = 1.0,
        lazy_uncertainty_update: bool = False,
        clear_buffer_after_train: bool = True,
    ) -> None:
        """Initializes the LinearBanditModule.

        Args:
            n_features: The number of features in the bandit model.
            buffer: The buffer used for storing the data for continuously updating the neural network.
                For the linear bandit, it should always be an InMemoryDataBuffer with an AllDataBufferStrategy
                because the buffer is cleared after each update.
            train_batch_size: The mini-batch size used for the train loop (started by `trainer.fit()`).
            eps: Small value to ensure invertibility of the precision matrix. Added to the diagonal.
            lambda_: Prior variance for the precision matrix. Acts as a regularization parameter.
                Sometimes also called lambda but we already use lambda for the regularization parameter
                of the neural networks in NeuralLinear, NeuralUCB and NeuralTS.
            lazy_uncertainty_update: If True the precision matrix will not be updated during forward, but during the
                update step.
            clear_buffer_after_train: If True the buffer will be cleared after training. This is necessary because the
                data is not needed anymore after training.
        """
        super().__init__(
            n_features=n_features,
            buffer=buffer,
            train_batch_size=train_batch_size,
        )
        self.lazy_uncertainty_update = lazy_uncertainty_update

        self.save_hyperparameters(
            {
                "lazy_uncertainty_update": True,
                "eps": eps,
                "lambda_": lambda_,
                "clear_buffer_after_train": clear_buffer_after_train,
            }
        )

        # Disable Lightning's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        self._init_linear_params()

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
        if not self.lazy_uncertainty_update:
            assert isinstance(contextualized_actions, torch.Tensor), "contextualized_actions must be a torch.Tensor"
            chosen_contextualized_actions = contextualized_actions[
                torch.arange(contextualized_actions.shape[0], device=self.device),
                chosen_actions.argmax(dim=1),
            ]
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
        batch: tuple[ActionInputType, torch.Tensor],
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
        assert len(batch) == 2, "Batch must contain two tensors: (contextualized_actions, rewards)"

        chosen_contextualized_actions = batch[0]
        assert isinstance(chosen_contextualized_actions, torch.Tensor), "chosen_contextualized_actions must be a tensor"
        realized_rewards: torch.Tensor = batch[1]

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
        # TODO: Implement linear combinatorial bandits according to Efficient Learning in Large-Scale Combinatorial
        #   Semi-Bandits (https://arxiv.org/pdf/1406.7443)

        if self.lazy_uncertainty_update:
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
            torch.eye(self.precision_matrix.shape[0], device=self.device) * self.hparams["eps"]
        )  # add small value to diagonal to ensure invertibility

        # should be symmetric
        assert torch.allclose(self.precision_matrix, self.precision_matrix.T), "Precision matrix must be symmetric"
        vals, _ = torch.linalg.eigh(self.precision_matrix)
        assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

        return self.precision_matrix

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle saving custom LinearBandit state."""
        checkpoint["precision_matrix"] = self.precision_matrix
        checkpoint["b"] = self.b
        checkpoint["theta"] = self.theta

        if hasattr(self, "selector"):
            checkpoint["selector_type"] = self.selector.__class__.__name__
            if isinstance(self.selector, EpsilonGreedySelector):
                checkpoint["selector_epsilon"] = self.selector.epsilon
                checkpoint["selector_generator_state"] = self.selector.generator.get_state()
            elif isinstance(self.selector, TopKSelector):
                checkpoint["selector_k"] = self.selector.k

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle loading custom LinearBandit state."""
        if "precision_matrix" in checkpoint:
            self.register_buffer("precision_matrix", checkpoint["precision_matrix"])

        if "b" in checkpoint:
            self.register_buffer("b", checkpoint["b"])

        if "theta" in checkpoint:
            self.register_buffer("theta", checkpoint["theta"])

        if "selector_type" in checkpoint and hasattr(self, "selector"):
            if checkpoint["selector_type"] == "EpsilonGreedySelector":
                self.selector = EpsilonGreedySelector(epsilon=checkpoint["selector_epsilon"])
                if "selector_generator_state" in checkpoint:
                    self.selector.generator.set_state(checkpoint["selector_generator_state"])
            elif checkpoint["selector_type"] == "TopKSelector":
                self.selector = TopKSelector(k=checkpoint["selector_k"])  # type: ignore
            else:
                self.selector = ArgMaxSelector()  # type: ignore

    def on_train_end(self) -> None:
        """Clear the buffer after training because the past data is not needed anymore."""
        super().on_train_end()
        if self.hparams["clear_buffer_after_train"]:
            self.buffer.clear()
