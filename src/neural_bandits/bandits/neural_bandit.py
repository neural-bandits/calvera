import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import optim

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.utils.data_storage import AbstractBanditDataBuffer
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector, EpsilonGreedySelector, TopKSelector


class NeuralBandit(AbstractBandit[torch.Tensor], ABC):
    """Baseclass for both NeuralTS and NeuralUCB.

    Implements most oft the logic except for the `_score` function. This function is
    implemented in the subclasses and is responsible for calculating the scores passed to the selector.
    """

    Z_t: torch.Tensor

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
        buffer: AbstractBanditDataBuffer[torch.Tensor, Any],
        selector: Optional[AbstractSelector] = None,
        lambda_: float = 0.00001,
        nu: float = 0.00001,
        train_batch_size: int = 32,
        learning_rate: float = 0.01,
        max_grad_norm: float = 20.0,
        num_grad_steps: int = 1000,
        early_stop_threshold: Optional[float] = 1e-3,
        train_interval: int = 64,
        initial_train_steps: int = 1024,
        **kw_args: Any,
    ) -> None:
        """Initialize the NeuralUCB bandit module.

        Args:
            n_features: Number of input features.
            network: Neural network module for function approximation.
            buffer: Buffer for storing bandit interaction data.
            selector: Action selector for the bandit. Defaults to ArgMaxSelector (if None).
            lambda_: Regularization parameter.
            nu: Exploration parameter for UCB.
            train_batch_size: Size of mini-batches for training. Defaults to 32.
            learning_rate: Learning rate for SGD optimizer.
            max_grad_norm: Maximum gradient norm for gradient clipping.
            num_grad_steps: Maximum number of gradient steps per training iteration.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
            train_interval: Interval between different trainings (in samples).
            initial_train_steps: Number of initial training steps (in samples).
            **kw_args: Additional arguments passed to parent class.
        """
        assert train_batch_size <= train_interval, "Batch size must be less than or equals to train interval."

        assert initial_train_steps % train_batch_size == 0, "initial_train_steps must be divisible by train_batch_size"
        assert train_interval % train_batch_size == 0, "train_interval must be divisible by train_batch_size"

        super().__init__()

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save hyperparameters
        hyperparameters = {
            "n_features": n_features,
            "train_batch_size": train_batch_size,
            "early_stop_threshold": early_stop_threshold,
            "num_grad_steps": num_grad_steps,
            "lambda_": lambda_,
            "nu": nu,
            "learning_rate": learning_rate,
            "train_interval": train_interval,
            "initial_train_steps": initial_train_steps,
            "max_grad_norm": max_grad_norm,
            **kw_args,
        }
        self.save_hyperparameters(hyperparameters)

        self.selector = selector if selector is not None else ArgMaxSelector()

        self._trained_once: bool = False

        # Model parameters

        # Initialize θ_t
        self.theta_t = network.to(self.device)

        # Track {x_i,a_i, r_i,a_i} history
        self.buffer = buffer

        self.total_params = sum(p.numel() for p in self.theta_t.parameters() if p.requires_grad)

        # Initialize Z_0 = λI
        self.register_buffer(
            "Z_t",
            self.hparams["lambda_"] * torch.ones((self.total_params,), device=self.device),
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
                self.hparams["lambda_"] * self.hparams["nu"] * all_gradients * all_gradients / self.Z_t,
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
        contextualized_actions: torch.Tensor = batch[0]  # shape: (batch_size, n_arms, n_features)
        realized_rewards: torch.Tensor = batch[1]  # shape: (batch_size, )

        # Update bandit's history
        self.buffer.add_batch(
            contextualized_actions=contextualized_actions.view(-1, contextualized_actions.size(-1)),
            embedded_actions=None,
            rewards=realized_rewards.squeeze(1),
        )

        # Train network based on schedule
        should_train = len(self.buffer) <= self.hparams["initial_train_steps"] or (
            len(self.buffer) > self.hparams["initial_train_steps"]
            and len(self.buffer) % self.hparams["train_interval"] == 0
        )

        if should_train:
            loss = self._train_network()
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self._trained_once = True

        # Logging
        self.log(
            "reward",
            realized_rewards.sum(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return -realized_rewards.sum()

    def _train_network(self) -> float:
        """Train the neural network using the data stored in the buffer."""
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        L_theta_sum: float = 0.0  # Track cumulative loss L(θ)

        for j in range(self.hparams["num_grad_steps"]):
            context, _, reward = self.buffer.get_batch(batch_size=self.hparams["train_batch_size"])
            context = context.to(self.device)
            reward = reward.to(self.device)

            optimizer.zero_grad()

            # Compute f(x_i,a_i; θ)
            f_theta = self.theta_t(context)
            L_theta = torch.nn.functional.mse_loss(f_theta, reward.unsqueeze(1))
            L_theta = L_theta.sum() / self.hparams["train_batch_size"]
            self.manual_backward(L_theta)

            torch.nn.utils.clip_grad_norm_(
                self.theta_t.parameters(),
                max_norm=self.hparams["max_grad_norm"],
            )

            optimizer.step()

            L_theta_sum += L_theta.item()

            if (
                self.hparams["early_stop_threshold"] is not None
                and (L_theta_sum / (j + 1)) <= self.hparams["early_stop_threshold"]
            ):
                break

        return float(L_theta_sum / (j + 1))

    def configure_optimizers(self) -> optim.Optimizer:
        """Initialize the optimizer for the bandit model."""
        return optim.SGD(
            self.theta_t.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["lambda_"],
        )

    def on_train_epoch_end(self) -> None:
        """Check if the network was trained at least once during the epoch."""
        super().on_train_epoch_end()
        if not self._trained_once:
            logging.warning("Finished the epoch without training the network. Consider decreasing `train_interval`.")

    def on_train_epoch_start(self) -> None:
        """Reset the `_trained_once` flag at the start of each epoch."""
        super().on_train_epoch_start()
        self._trained_once = False

    def on_save_checkpoint(self, checkpoint):
        """Handle saving custom state.

        This ensures all components are properly serialized during checkpoint saving.
        """
        checkpoint["buffer_state"] = self.buffer.state_dict()

        checkpoint["Z_t"] = self.Z_t

        if isinstance(self.selector, EpsilonGreedySelector):
            checkpoint["selector_type"] = "EpsilonGreedySelector"
            checkpoint["selector_epsilon"] = self.selector.epsilon
            checkpoint["selector_generator_state"] = self.selector.generator.get_state()
        elif isinstance(self.selector, TopKSelector):
            checkpoint["selector_type"] = "TopKSelector"
            checkpoint["selector_k"] = self.selector.k
        else:
            checkpoint["selector_type"] = "ArgMaxSelector"

        checkpoint["network_state"] = self.theta_t.state_dict()

        checkpoint["_trained_once"] = self._trained_once

    def on_load_checkpoint(self, checkpoint):
        """Handle loading custom state.

        This ensures all components are properly restored during checkpoint loading.
        """
        if checkpoint.get("buffer_state"):
            self.buffer.load_state_dict(checkpoint["buffer_state"])

        if "Z_t" in checkpoint:
            self.register_buffer("Z_t", checkpoint["Z_t"])

        if "selector_type" in checkpoint:
            if checkpoint["selector_type"] == "EpsilonGreedySelector":
                self.selector = EpsilonGreedySelector(epsilon=checkpoint["selector_epsilon"])
                if "selector_generator_state" in checkpoint:
                    self.selector.generator.set_state(checkpoint["selector_generator_state"])
            elif checkpoint["selector_type"] == "TopKSelector":
                self.selector = TopKSelector(k=checkpoint["selector_k"])
            else:
                self.selector = ArgMaxSelector()

        if "network_state" in checkpoint:
            self.theta_t.load_state_dict(checkpoint["network_state"])

        if "_trained_once" in checkpoint:
            self._trained_once = checkpoint["_trained_once"]
