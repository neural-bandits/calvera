import torch
import torch.nn as nn

from neural_bandits.algorithms.abstract_bandit import AbstractBandit


class NeuralLinearBandit(AbstractBandit):
    """Implements the action selection for a neural linear bandit model."""

    def __init__(
        self,
        n_features: int,
        n_embedding_size: int,
        encoder: nn.Module,
    ) -> None:
        """Initializes the NeuralLinearBandit.

        Args:
            n_features: The number of features in the contextualized actions that are fed into the encoder.
            n_embedding_size: The size of the tensors coming out of the encoder.
            encoder: The embedding model (neural network) to be used.
        """
        super().__init__(n_features)

        self.encoder = encoder

        self.hparams = {
            "n_features": n_features,
            "n_embedding_size": n_embedding_size,
        }

        # Initialize the linear head which receives the embeddings
        self.precision_matrix = torch.eye(n_embedding_size)
        self.b = torch.zeros(n_embedding_size)
        self.theta = torch.zeros(n_embedding_size)

    def forward(self, contextualized_actions: torch.Tensor) -> torch.Tensor:
        """Predict the action to take for the given input data according to neural linear.

        Args:
            contextualised_actions: The input data. Shape: (batch_size, n_arms, n_features)
        """

        assert (
            contextualized_actions.ndim == 3
            and contextualized_actions.shape[2] == self.hparams["n_features"]
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        embedded_actions: torch.Tensor = self.encoder(
            contextualized_actions
        )  # shape: (batch_size, n_arms, n_embedding_size)

        assert (
            embedded_actions.ndim == 3
            and embedded_actions.shape[0] == contextualized_actions.shape[0]
            and embedded_actions.shape[1] == contextualized_actions.shape[1]
            and embedded_actions.shape[2] == self.hparams["n_embedding_size"]
        ), "Embedded actions must have shape (batch_size, n_arms, n_features)"

        result = self.linear_ts_head(embedded_actions)  # shape: (batch_size, n_arms)

        assert (
            result.shape[0] == contextualized_actions.shape[0]
            and result.shape[1] == contextualized_actions.shape[1]
        ), "Linear head output misshaped"

        return result

    def linear_ts_head(self, embedded_actions: torch.Tensor) -> torch.Tensor:
        """Linear head for the neural linear bandit model. Uses the Thompson Sampling algorithm to predict the best action.

        Args:
            embedded_actions: The embedded actions. Shape: (batch_size, n_arms, n_embedding_size)
        """
        batch_size, n_arms, _ = embedded_actions.shape

        theta_tilde = torch.distributions.MultivariateNormal(  # type: ignore
            self.theta, self.precision_matrix
        ).sample(
            (batch_size,)
        )  # shape: (batch_size, n_features)

        pred_reward_per_action = torch.einsum(
            "ijk,ik->ij", embedded_actions, theta_tilde
        )  # shape: (batch_size, n_arms)
        result = torch.argmax(pred_reward_per_action, dim=-1)  # shape: (batch_size,)

        return torch.nn.functional.one_hot(result, num_classes=n_arms).reshape(
            -1, n_arms
        )  # shape: (batch_size, n_arms)
