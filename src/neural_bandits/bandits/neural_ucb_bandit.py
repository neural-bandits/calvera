from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class NeuralUCBBandit(AbstractBandit):
    """NeuralUCB bandit implementation as a PyTorch Lightning module.
    The NeuralUCB algorithm using a neural network for function approximation with diagonal approximation for exploration.
    
    Attributes:
        automatic_optimization: Boolean indicating if Lightning should handle optimization.
        bandit: The underlying NeuralUCBBandit instance.
    """

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
        selector: AbstractSelector = ArgMaxSelector(),
        early_stop_threshold: Optional[float] = 1e-3,
        num_grad_steps: int = 1000,
        lambda_: float = 0.00001,
        nu: float = 0.00001,
        learning_rate: float = 0.01,
        train_freq: int = 100,
        initial_train_steps: int = 1000,
        max_grad_norm: float = 20.0,
        **kw_args: Any,
    ) -> None:
        """Initialize the NeuralUCB bandit module.

        Args:
            n_features: Number of input features.
            network: Neural network module for function approximation.
            selector: Action selector for the bandit. Defaults to ArgMaxSelector.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
            num_grad_steps: Maximum number of gradient steps per training iteration.
            lambda_: Regularization parameter.
            nu: Exploration parameter for UCB.
            learning_rate: Learning rate for SGD optimizer.
            train_freq: Frequency of network training after initial training.
            initial_train_steps: Number of initial training steps.
            max_grad_norm: Maximum gradient norm for gradient clipping.
            **kw_args: Additional arguments passed to parent class.
        """
        super().__init__()

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save hyperparameters
        hyperparameters = {
            "n_features": n_features,
            "early_stop_threshold": early_stop_threshold,
            "num_grad_steps": num_grad_steps,
            "lambda_": lambda_,
            "nu": nu,
            "learning_rate": learning_rate,
            "train_freq": train_freq,
            "initial_train_steps": initial_train_steps,
            "max_grad_norm": max_grad_norm,
            **kw_args,
        }
        self.save_hyperparameters(hyperparameters)

        self.selector = selector

        # self.total_regret: float = 0.0
        self.total_samples: int = 0

        # Model parameters

        # Initialize θ_t
        self.theta_t = network.to(self.device)

        # Track {x_i,a_i, r_i,a_i} history
        self.context_history: list[torch.Tensor] = []
        self.reward_history: list[torch.Tensor] = []

        self.total_params = sum(
            p.numel() for p in self.theta_t.parameters() if p.requires_grad
        )

        # Initialize Z_0 = λI
        self.Z_t = self.hparams["lambda_"] * torch.ones((self.total_params,), device=self.device)


    def predict(
        self,
        contextualized_actions: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate UCB scores for each action using diagonal approximation with batch support.

        Args:
            contextualized_actions (torch.Tensor): Contextualized action tensor. Shape: (batch_size, n_arms, n_features).

        Returns:
            tuple:
            - chosen_actions (torch.Tensor): One-hot encoding of which actions were chosen. Shape: (batch_size, num_actions).
            - p (torch.Tensor): Will always return a tensor of ones because UCB does not work on probabilities. Shape: (batch_size, ).
        """
        contextualized_actions = contextualized_actions.to(self.device)
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
        all_gradients = torch.zeros(
            batch_size, n_arms, self.total_params, device=self.device
        )

        for b in range(batch_size):
            for a in range(n_arms):
                # Calculate g(x_t,a; θ_t-1)
                self.theta_t.zero_grad()
                f_t_a[b, a].backward(retain_graph=True)  # type: ignore

                g_t_a = torch.cat(
                    [
                        p.grad.flatten().detach()
                        for p in self.theta_t.parameters()
                        if p.grad is not None
                    ]
                )
                all_gradients[b, a] = g_t_a

        # Compute uncertainty using diagonal approximation
        # Shape: (batch_size, n_arms)
        exploration_terms = torch.sqrt(
            torch.sum(
                self.hparams["lambda_"] * self.hparams["nu"] * all_gradients * all_gradients / self.Z_t, dim=2
            )
        )

        # UCB score U_t,a
        # Shape: (batch_size, n_arms)
        U_t = f_t_a + exploration_terms

        # Select a_t = argmax_a U_t,a
        chosen_actions = self.selector(U_t)

        # Update Z_t using g(x_t,a_t; θ_t-1)
        for b in range(batch_size):
            a_t = chosen_actions[b]
            self.Z_t += all_gradients[b, a_t] * all_gradients[b, a_t]

        # Return chosen actions and
        return chosen_actions, torch.ones(batch_size, device=self.device)


    def update_step(
        self,
        batch: torch.Tensor,
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
        contextualized_actions: torch.Tensor = batch[
            0
        ]  # shape: (batch_size, n_arms, n_features)
        realized_rewards: torch.Tensor = batch[1]  # shape: (batch_size, n_arms)
        batch_size = realized_rewards.shape[0]

        # Update bandit's history
        self.context_history.append(contextualized_actions)
        self.reward_history.append(realized_rewards)

        # Train network based on schedule
        should_train = batch_idx < self.hparams["initial_train_steps"] or (
            batch_idx >= self.hparams["initial_train_steps"]
            and batch_idx % self.hparams["train_freq"] == 0
        )

        if should_train:
            loss = self._train_network()
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Logging
        self.log(
            "reward",
            realized_rewards.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        # regret = torch.max(rewards, dim=1).values - realized_rewards
        # self.log("regret", regret.mean(), on_step=True, on_epoch=False, prog_bar=True)

        # self.total_regret += regret.sum().item()
        self.total_samples += batch_size

        # self.log(
        #     "average_regret",
        #     self.total_regret / self.total_samples,
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=True,
        # )

        return -realized_rewards.mean()

    def _train_network(self) -> float:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        indices = np.arange(len(self.reward_history))
        np.random.shuffle(indices)

        L_theta_sum: float = 0.0  # Track cumulative loss L(θ)
        j = 0  # Count gradient descent steps

        while True:
            L_theta_batch: float = 0.0  # Track batch loss

            # Compute L(θ) and perform gradient descent
            for i in indices:
                context = self.context_history[i]
                reward = self.reward_history[i]

                batch_size = context.shape[0]

                optimizer.zero_grad()

                # Compute f(x_i,a_i; θ)
                f_theta = self.theta_t(context)
                L_theta = torch.nn.functional.mse_loss(f_theta, reward.unsqueeze(1))
                L_theta = L_theta.sum() / batch_size
                self.manual_backward(L_theta)

                torch.nn.utils.clip_grad_norm_(
                    self.theta_t.parameters(),
                    max_norm=self.hparams["max_grad_norm"],
                )

                optimizer.step()

                L_theta_batch += L_theta.item()
                L_theta_sum += L_theta.item()
                j += 1

                # Return θ⁽ᴶ⁾ after J gradient descent steps
                if j >= self.hparams["num_grad_steps"]:
                    return float(L_theta_sum / self.hparams["num_grad_steps"])

            # Early stopping if threshold is set and loss is small enough
            if (
                self.hparams["early_stop_threshold"] is not None
                and L_theta_batch / len(self.reward_history)
                <= self.hparams["early_stop_threshold"]
            ):
                return float(L_theta_batch / len(self.reward_history))

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(
            self.theta_t.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["lambda_"],
        )
