from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from neural_bandits.algorithms.neural_ucb_bandit import NeuralUCBBandit
from neural_bandits.modules.abstract_bandit_module import AbstractBanditModule


class NeuralUCBBanditModule(AbstractBanditModule[NeuralUCBBandit]):
    """NeuralUCB bandit implementation as a PyTorch Lightning module.

    Attributes:
        automatic_optimization: Boolean indicating if Lightning should handle optimization.
        bandit: The underlying NeuralUCBBandit instance.
    """

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
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

        self.bandit = NeuralUCBBandit(
            network=network,
            n_features=n_features,
            lambda_=lambda_,
            nu=nu,
        )

        self.total_regret: float = 0.0
        self.total_samples: int = 0

    def training_step(
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
        rewards: torch.Tensor = batch[1]  # shape: (batch_size, n_arms)
        contextualized_actions: torch.Tensor = batch[
            0
        ]  # shape: (batch_size, n_arms, n_features)
        batch_size = rewards.shape[0]

        # Get UCB scores and select actions
        ucb_scores = self(contextualized_actions)
        chosen_actions_idx = ucb_scores.argmax(dim=1)  # shape: (batch_size,)
        realized_rewards = rewards[torch.arange(batch_size), chosen_actions_idx]

        # Get chosen features
        chosen_actions = contextualized_actions[
            torch.arange(batch_size), chosen_actions_idx
        ]  # shape: (batch_size, n_features)

        # Update bandit's history
        self.bandit.context_history.append(chosen_actions)
        self.bandit.reward_history.append(realized_rewards)

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
        regret = torch.max(rewards, dim=1).values - realized_rewards
        self.log("regret", regret.mean(), on_step=True, on_epoch=False, prog_bar=True)

        self.total_regret += regret.sum().item()
        self.total_samples += batch_size

        self.log(
            "average_regret",
            self.total_regret / self.total_samples,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return -rewards.mean()

    def _train_network(self) -> float:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        indices = np.arange(len(self.bandit.reward_history))
        np.random.shuffle(indices)

        L_theta_sum: float = 0.0  # Track cumulative loss L(θ)
        j = 0  # Count gradient descent steps

        while True:
            L_theta_batch: float = 0.0  # Track batch loss

            # Compute L(θ) and perform gradient descent
            for i in indices:
                context = self.bandit.context_history[i]
                reward = self.bandit.reward_history[i]

                batch_size = context.shape[0]

                optimizer.zero_grad()

                # Compute f(x_i,a_i; θ)
                f_theta = self.bandit.theta_t(context)
                L_theta = torch.nn.functional.mse_loss(f_theta, reward.unsqueeze(1))
                L_theta = L_theta.sum() / batch_size
                self.manual_backward(L_theta)

                torch.nn.utils.clip_grad_norm_(
                    self.bandit.theta_t.parameters(),
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
                and L_theta_batch / len(self.bandit.reward_history)
                <= self.hparams["early_stop_threshold"]
            ):
                return float(L_theta_batch / len(self.bandit.reward_history))

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(
            self.bandit.theta_t.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.bandit.lambda_,
        )
