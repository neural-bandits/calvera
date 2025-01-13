import torch
import torch.nn as nn
from neural_bandits.algorithms.abstract_bandit import AbstractBandit
from neural_bandits.algorithms.linear_bandits import LinearTSBandit


class NeuralLinearBandit(AbstractBandit, nn.Module):
    def __init__(self, n_arms: int, n_features: int) -> None:
        super().__init__(n_arms, n_features)

        # TODO(philippkolbe): Could also take one big model create with OrderedDict
        self.embedding_model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )
        self.linear_head = LinearTSBandit(n_arms, n_features)

    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        assert (
            contextualised_actions.shape[1] == self.n_arms
            and contextualised_actions.shape[2] == self.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        embeddings: torch.Tensor = self.embedding_model(
            contextualised_actions
        )  # shape: (batch_size, n_arms, n_features)
        result: torch.Tensor = self.linear_head(
            embeddings
        )  # shape: (batch_size, n_arms)
        return result
