import torch

from neural_bandits.trainers.abstract_trainer import AbstractTrainer
from neural_bandits.algorithms.neural_linear import NeuralLinearBandit
from neural_bandits.trainers.linear_trainer import LinearTrainer
from neural_bandits.algorithms.linear_bandits import LinearTSBandit


class NeuralLinearTrainer(AbstractTrainer[NeuralLinearBandit]):
    def __init__(self) -> None:
        self.linear_trainer: LinearTrainer[LinearTSBandit] = LinearTrainer()

    def update(
        self,
        bandit: NeuralLinearBandit,
        rewards: torch.Tensor,  # shape: (batch_size,)
        chosen_actions: torch.Tensor,  # shape: (batch_size, n_features)
    ) -> NeuralLinearBandit:
        """Perform an update step on the head of the neural linear bandit"""
        # update using the linear trainer
        bandit.linear_head = self.linear_trainer.update(
            bandit.linear_head, rewards, chosen_actions
        )

        return bandit

    def update_nn(self) -> None:
        pass
