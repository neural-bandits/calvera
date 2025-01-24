from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractBandit(ABC, nn.Module):  # AbstractModel
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features

    @abstractmethod
    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:  # forward
        """Predict a list of multiple sets of contextualised actions

        Args:
            contextualised_actions: A tensor of shape (batch_size, n_actions, n_features)

        Returns:
            A tensor of shape (batch_size, n_actions) of selection probabilities for each action
        """
        pass
