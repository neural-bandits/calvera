from typing import Generic, TypeVar

import numpy as np
import torch
from torch import optim

from ..algorithms.neural_ucb_bandit import NeuralUCB
from .abstract_trainer import AbstractTrainer

NeuralUCBType = TypeVar("NeuralUCBType", bound="NeuralUCB")


class NeuralUCBTrainer(AbstractTrainer[NeuralUCBType], Generic[NeuralUCBType]):
    def __init__(self, eta: float = 0.01):
        self.eta = eta

    def update(
        self,
        bandit: NeuralUCBType,
        rewards: torch.Tensor,
        chosen_actions: torch.Tensor,
    ) -> NeuralUCBType:
        chosen_actions = chosen_actions.to(bandit.device)
        rewards = rewards.to(bandit.device)

        bandit.context_history.append(chosen_actions)
        bandit.reward_history.append(rewards)

        # Train neural network
        self._train_network(bandit)

        return bandit

    def _train_network(self, bandit: NeuralUCBType) -> float:
        # Initialize optimizer with step size η and L2 regularization λ
        optimizer = optim.SGD(
            bandit.theta_t.parameters(), lr=self.eta, weight_decay=bandit.lambda_
        )

        indices = np.arange(len(bandit.reward_history))
        np.random.shuffle(indices)

        L_theta_sum = 0  # Track cumulative loss L(θ)
        j = 0  # Count gradient descent steps

        while True:
            L_theta_batch = 0  # Track batch loss

            # Compute L(θ) and perform gradient descent
            for i in indices:
                context = bandit.context_history[i]
                reward = bandit.reward_history[i]

                optimizer.zero_grad()

                # Compute f(x_i,a_i; θ)
                f_theta = bandit.theta_t(context)
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
            if L_theta_batch / len(bandit.reward_history) <= 1e-3:
                return L_theta_batch / len(bandit.reward_history)
