from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractBandit(ABC, nn.Module):  # AbstractModel
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features

    @abstractmethod
    def forward(self, contextualized_actions: torch.Tensor) -> torch.Tensor:  # forward
        """Predict a list of multiple sets of contextualized actions

        Args:
            contextualized_actions: A tensor of shape (batch_size, n_actions, n_features)

        Returns:
            A tensor of shape (batch_size, n_actions) of selection probabilities for each action
        """
        pass
