from typing import Any, Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ..algorithms.neural_ucb_bandit import NeuralUCB
from .abstract_bandit_module import AbstractBanditModule

NeuralUCBType = TypeVar("NeuralUCBType", bound="NeuralUCB")


class NeuralUCBModule(AbstractBanditModule[NeuralUCBType], Generic[NeuralUCBType]):
    def __init__(
        self,
        neural_bandit_type: NeuralUCBType,
        n_features: int,
        network: nn.Module,
        lambda_: float = 0.00001,
        nu: float = 0.00001,
        learning_rate: float = 0.01,
        train_freq: int = 100,
        initial_train_steps: int = 1000,
        **kw_args: Any,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.train_freq = train_freq
        self.initial_train_steps = initial_train_steps

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save hyperparameters
        # TODO: add network hyperparameters
        hyperparameters = {
            "neural_bandit_type": neural_bandit_type.__name__,
            "n_features": n_features,
            "lambda_": lambda_,
            "nu": nu,
            "learning_rate": learning_rate,
            "train_freq": train_freq,
            "initial_train_steps": initial_train_steps,
            **kw_args,
        }
        self.save_hyperparameters(hyperparameters)

        self.bandit = neural_bandit_type(
            network=network,
            n_features=n_features,
            lambda_=lambda_,
            nu=nu,
        )

        self.step_count = 0

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        rewards: torch.Tensor = batch[1]
        contextualized_actions: torch.Tensor = batch[0]

        # Get UCB scores and select actions
        ucb_scores = self(contextualized_actions)
        chosen_actions_idx = ucb_scores.argmax(dim=1)
        realized_rewards = rewards[torch.arange(rewards.shape[0]), chosen_actions_idx]
        batch_size = chosen_actions_idx.shape[0]

        # Get chosen features
        chosen_actions = contextualized_actions[
            torch.arange(batch_size), chosen_actions_idx
        ]

        # Update bandit's history
        self.bandit.context_history.append(chosen_actions)
        self.bandit.reward_history.append(realized_rewards)

        # Train network based on schedule
        # loss = 0.0
        should_train = self.step_count < self.initial_train_steps or (
            self.step_count >= self.initial_train_steps
            and self.step_count % self.train_freq == 0
        )

        if should_train:
            self._train_network()

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
        # self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Increment step count
        self.step_count += 1

        return -rewards.mean()

    def _train_network(self) -> float:
        # Initialize optimizer with step size η and L2 regularization λ
        optimizer = optim.SGD(
            self.bandit.theta_t.parameters(),
            lr=self.learning_rate,
            weight_decay=self.bandit.lambda_,
        )

        indices = np.arange(len(self.bandit.reward_history))
        np.random.shuffle(indices)

        L_theta_sum = 0  # Track cumulative loss L(θ)
        j = 0  # Count gradient descent steps

        while True:
            L_theta_batch = 0  # Track batch loss

            # Compute L(θ) and perform gradient descent
            for i in indices:
                context = self.bandit.context_history[i]
                reward = self.bandit.reward_history[i]

                optimizer.zero_grad()

                # Compute f(x_i,a_i; θ)
                f_theta = self.bandit.theta_t(context)
                L_theta = (f_theta - reward) ** 2
                L_theta.backward()

                optimizer.step()

                L_theta_batch += L_theta.item()
                L_theta_sum += L_theta.item()
                j += 1

                # Return θ⁽ᴶ⁾ after J gradient descent steps
                if j >= 1000:
                    return L_theta_sum / 1000

            # Early stopping if loss is small enough
            if L_theta_batch / len(self.bandit.reward_history) <= 1e-3:
                return L_theta_batch / len(self.bandit.reward_history)

    def configure_optimizers(self) -> None:
        return None
