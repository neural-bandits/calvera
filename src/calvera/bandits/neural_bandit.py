import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim

from calvera.bandits.abstract_bandit import AbstractBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import AbstractSelector, ArgMaxSelector

logger = logging.getLogger(__name__)


def get_neural_bandit_trainer(**kwargs: Any) -> pl.Trainer:
    """Instantiates a preconfigured PyTorch Lightning Trainer for Neural Linear.

    The gradient clipping value is set to 20.0.

    Args:
        **kwargs: Additional keyword arguments to pass to the Trainer.
    """
    return pl.Trainer(gradient_clip_val=20.0, **kwargs)


class NeuralBandit(AbstractBandit[torch.Tensor], ABC):
    """Baseclass for both NeuralTS and NeuralUCB.

    Implements most oft the logic except for the `_score` function. This function is
    implemented in the subclasses and is responsible for calculating the scores passed to the selector.
    """

    Z_t: torch.Tensor

    _should_train_network = False

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
        buffer: AbstractBanditDataBuffer[torch.Tensor, Any] | None = None,
        selector: AbstractSelector | None = None,
        exploration_rate: float = 1.0,
        train_batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.0,
        learning_rate_decay: float = 1.0,
        learning_rate_scheduler_step_size: int = 1,
        early_stop_threshold: float | None = 1e-3,
        min_samples_required_for_training: int | None = 64,
        initial_train_steps: int = 1024,
    ) -> None:
        """Initialize the NeuralUCB bandit module.

        Args:
            n_features: Number of input features. Must be greater 0.
            network: Neural network module for function approximation.
            buffer: Buffer for storing bandit interaction data.
            selector: Action selector for the bandit. Defaults to ArgMaxSelector (if None).
            exploration_rate: Exploration parameter for UCB. Called gamma_t=nu in the original paper.
                Defaults to 1. Must be greater 0.
            train_batch_size: Size of mini-batches for training. Defaults to 32. Must be greater 0.
            learning_rate: The learning rate for the optimizer of the neural network.
                Passed to `lr` of `torch.optim.Adam`.
                Default is 1e-3. Must be greater than 0.
            weight_decay: The regularization parameter for the neural network.
                Passed to `weight_decay` of `torch.optim.Adam`.
                Default is 1.0. Must be greater than 0 because the NeuralUCB algorithm is based on this parameter.
            learning_rate_decay: Multiplicative factor for learning rate decay.
                Passed to `gamma` of `torch.optim.lr_scheduler.StepLR`.
                Default is 1.0 (i.e. no decay). Must be greater than 0.
            learning_rate_scheduler_step_size: The step size for the learning rate decay.
                Passed to `step_size` of `torch.optim.lr_scheduler.StepLR`.
                Default is 1. Must be greater than 0.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
                Defaults to 1e-3. Must be greater equal 0.
            min_samples_required_for_training: If less samples have been added via `record_feedback`
                than this value, the network is not trained.
                If None, the network is trained every time `trainer.fit` is called.
                Defaults to 64. Must be greater 0.
            initial_train_steps: For the first `initial_train_steps` samples, the network is always trained even if
                less new data than `min_samples_required_for_training` has been seen. Therefore, this value is only
                required if `min_samples_required_for_training` is set. Set to 0 to disable this feature.
                Defaults to 1024. Must be greater equal 0.
        """
        assert weight_decay >= 0, "Regularization parameter must be greater equal 0."
        assert exploration_rate > 0, "Exploration rate must be greater than 0."
        assert learning_rate > 0, "Learning rate must be greater than 0."
        assert learning_rate_decay >= 0, "The learning rate decay must be greater equal 0."
        assert learning_rate_scheduler_step_size > 0, "Learning rate must be greater than 0."
        assert (
            min_samples_required_for_training is None or min_samples_required_for_training > 0
        ), "Training interval must be greater than 0."
        assert (
            early_stop_threshold is None or early_stop_threshold >= 0
        ), "Early stop threshold must be greater than or equal to 0."
        assert initial_train_steps >= 0, "Initial training steps must be greater than or equal to 0."

        super().__init__(
            n_features=n_features,
            buffer=buffer,
            train_batch_size=train_batch_size,
        )

        self.save_hyperparameters(
            {
                "exploration_rate": exploration_rate,
                "train_batch_size": train_batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "learning_rate_decay": learning_rate_decay,
                "learning_rate_scheduler_step_size": learning_rate_scheduler_step_size,
                "min_samples_required_for_training": min_samples_required_for_training,
                "early_stop_threshold": early_stop_threshold,
                "initial_train_steps": initial_train_steps,
            }
        )

        self.selector = selector if selector is not None else ArgMaxSelector()

        # Model parameters: Initialize θ_t
        self.theta_t = network.to(self.device)
        self.total_params = sum(p.numel() for p in self.theta_t.parameters() if p.requires_grad)

        # Initialize Z_0 = λI
        self.register_buffer(
            "Z_t",
            self.hparams["weight_decay"] * torch.ones((self.total_params,), device=self.device),
        )

    def _predict_action(
        self,
        contextualized_actions: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate UCB scores for each action using diagonal approximation with batch support.

        Args:
            contextualized_actions: Contextualized action tensor. Shape: (batch_size, n_arms, n_features).
            kwargs: Additional keyword arguments. Not used.

        Returns:
            tuple:
            - chosen_actions: One-hot encoding of which actions were chosen. Shape: (batch_size, n_arms).
            - p: Will always return a tensor of ones because UCB does not work on probabilities. Shape: (batch_size, ).
        """
        batch_size, n_arms, n_features = contextualized_actions.shape

        assert (
            n_features == self.hparams["n_features"]
        ), "Contextualized actions must have shape (batch_size, n_arms, n_features)"

        # Reshape input from (batch_size, n_arms, n_features) to (batch_size * n_arms, n_features)
        flattened_actions = contextualized_actions.reshape(-1, n_features)

        # Compute f(x_t,a; θ_t-1) for all arms in batch
        f_t_a: torch.Tensor = self.theta_t(flattened_actions)
        f_t_a = f_t_a.reshape(batch_size, n_arms)

        # Store g(x_t,a; θ_t-1) values
        all_gradients = torch.zeros(batch_size, n_arms, self.total_params, device=self.device)

        for b in range(batch_size):
            for a in range(n_arms):
                # Calculate g(x_t,a; θ_t-1)
                self.theta_t.zero_grad()
                f_t_a[b, a].backward(retain_graph=True)  # type: ignore

                g_t_a = torch.cat([p.grad.flatten().detach() for p in self.theta_t.parameters() if p.grad is not None])
                all_gradients[b, a] = g_t_a

        # Compute uncertainty using diagonal approximation
        # Shape: (batch_size, n_arms)
        exploration_terms = torch.sqrt(
            torch.sum(
                self.hparams["weight_decay"]
                * self.hparams["exploration_rate"]
                * all_gradients
                * all_gradients
                / self.Z_t,
                dim=2,
            )
        )

        # Select a_t = argmax_a U_t,a
        chosen_actions = self.selector(self._score(f_t_a, exploration_terms))

        assert (chosen_actions.sum(dim=1) == 1).all(), "Currently only supports non-combinatorial bandits"
        chosen_actions_idx = chosen_actions.argmax(dim=1)  # TODO: this only works for non-combinatorial bandits!

        # Update Z_t using g(x_t,a_t; θ_t-1)
        for b in range(batch_size):
            a_t = chosen_actions_idx[b]
            self.Z_t += all_gradients[b, a_t] * all_gradients[b, a_t]

        # Return chosen actions and
        return chosen_actions, torch.ones(batch_size, device=self.device)

    @abstractmethod
    def _score(self, f_t_a: torch.Tensor, exploration_terms: torch.Tensor) -> torch.Tensor:
        """Compute a score based on the predicted rewards and exploration terms."""
        pass

    def record_feedback(self, contextualized_actions: torch.Tensor, rewards: torch.Tensor) -> None:
        """Records a pair of chosen actions and rewards in the buffer.

        Also checks if the network should be updated based on the number of samples seen so far
        and sets `should_train_network`.

        Args:
            contextualized_actions: The contextualized actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features).
            rewards: The rewards that were observed for the chosen actions. Size: (batch_size, n_actions).
        """
        super().record_feedback(contextualized_actions, rewards)

        if (
            self.is_initial_training_stage()
            or self._new_samples_count >= self.hparams["min_samples_required_for_training"]
        ):
            self.should_train_network = True
        else:
            self.should_train_network = False

        if (
            self._total_samples_count > cast(int, self.hparams["initial_train_steps"])
            and self._total_samples_count - contextualized_actions.size(0) <= self.hparams["initial_train_steps"]
        ):
            logger.info(
                "\nInitial training stage is over. "
                "The network will now be called only once min_samples_required_for_training samples are recorded."
            )

    def is_initial_training_stage(self) -> bool:
        """Check if the bandit is in the initial training stage.

        Returns:
            True if the total seen samples is smaller or equal to initial_strain_Steps, False otherwise.
        """
        return self._total_samples_count <= cast(int, self.hparams["initial_train_steps"])

    @property
    def should_train_network(self) -> bool:
        """Should the network be updated in the next training epoch?

        If called after `record_action_data`, this property will overwrite the behavior of
        the `min_samples_required_for_training` parameter.
        """
        return self._should_train_network

    @should_train_network.setter
    def should_train_network(self, value: bool) -> None:
        """Should the network be updated in the next training epoch?

        If called after `record_action_data`, this property will overwrite the behavior of
        the `min_samples_required_for_training` parameter.
        """
        self._should_train_network = value

    def on_train_start(self) -> None:
        """Check if enough samples have been recorded to train the network."""
        super().on_train_start()

        assert self.trainer.train_dataloader is not None, "train_dataloader must be set before training starts."
        if self._custom_data_loader_passed:
            logger.warning(
                "You passed a train_dataloader to trainer.fit(). Data from the data buffer will be ignored. "
                "Only the data passed in the train_data_loader is used for training. The data is still added to "
                "the data buffer for future training runs."
            )

            num_samples = len(self.trainer.train_dataloader.dataset)
            required_samples = self.hparams["min_samples_required_for_training"]
            if num_samples <= required_samples and not self.is_initial_training_stage():
                logger.warning(
                    f"The train_dataloader passed to trainer.fit() contains {num_samples}"
                    f"which is less than min_samples_required_for_training={required_samples}."
                    f"Even though the initial training stage is over and not enough data samples were passed, "
                    "the network will still be trained."
                    "Consider passing more data or decreasing min_samples_required_for_training."
                )

            self.should_train_network = True
        elif not self.should_train_network:
            logger.warning(
                "Called trainer.fit but not enough samples have been recorded to train the network. "
                "Therefore, training was cancelled. Consider passing more data, decreasing "
                "min_samples_required_for_training or manually overwriting should_train_network."
            )

            self._skip_training()

        # TODO: warm_start. If should_train_network and not warm_start, reset the network.

    def _update(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute a single training step.

        Args:
            batch: Tuple of (contextualized_actions, rewards) tensors.
            batch_idx: Index of current batch.

        Returns:
            Mean negative reward as loss value.

        Example:
            >>> batch = (context_tensor, reward_tensor)
            >>> loss = model.training_step(batch, 0)
        """
        assert len(batch) == 2, "Batch must contain two tensors: (contextualized_actions, rewards)"

        contextualized_actions: torch.Tensor = batch[0]  # shape: (batch_size, n_arms, n_features)
        realized_rewards: torch.Tensor = batch[1]  # shape: (batch_size, )

        assert (
            contextualized_actions.shape[-1] == self.hparams["n_features"]
        ), "Contextualized actions must have shape (batch_size, n_arms, n_features)"

        loss = self._train_network(contextualized_actions, realized_rewards)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def _train_network(
        self,
        context: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Train the neural network on the given data by computing the loss."""
        # Compute f(x_i,a_i; θ)
        f_theta = self.theta_t(context)
        predicted_reward = f_theta.squeeze(-1)
        L_theta = self._compute_loss(predicted_reward, reward)

        # Compute the average loss
        avg_loss = L_theta.mean()

        if self.hparams["early_stop_threshold"] is not None and avg_loss <= self.hparams["early_stop_threshold"]:
            self.trainer.should_stop = True

        return avg_loss

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss of the neural linear bandit.

        Args:
            y_pred: The predicted rewards. Shape: (batch_size,)
            y: The actual rewards. Shape: (batch_size,)

        Returns:
            The loss.
        """
        return torch.nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Initialize the optimizer and learning rate scheduler for the bandit model."""
        opt = optim.Adam(
            self.theta_t.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=self.hparams["learning_rate_scheduler_step_size"],
            gamma=self.hparams["learning_rate_decay"],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }

    def on_train_end(self) -> None:
        """Reset the training state."""
        super().on_train_end()
        if not self._training_skipped:
            self.should_train_network = False
