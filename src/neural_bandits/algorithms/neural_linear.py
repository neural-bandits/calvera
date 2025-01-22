import torch
import torch.nn as nn

from neural_bandits.algorithms.abstract_bandit import AbstractBandit
from neural_bandits.algorithms.linear_bandits import LinearTSBandit


class NeuralLinearBandit(AbstractBandit):

    def __init__(self, n_features: int, embedding_model: nn.Module) -> None:
        super().__init__(n_features)

        # TODO(philippkolbe): Could also take one big model create with OrderedDict
        self.embedding_model = embedding_model
        self.linear_head = LinearTSBandit(n_features)

    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = self.embedding_model(
            contextualised_actions
        )  # shape: (batch_size, n_arms, n_embedding_size)

        assert (
            contextualised_actions.shape[2] == self.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        result: torch.Tensor = self.linear_head(
            embeddings
        )  # shape: (batch_size, n_arms)
        return result
