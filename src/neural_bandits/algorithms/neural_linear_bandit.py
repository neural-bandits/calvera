from scipy.stats import invgamma
import torch
import torch.nn as nn

from neural_bandits.algorithms.abstract_bandit import AbstractBandit
from neural_bandits.algorithms.linear_bandits import LinearTSBandit


class NeuralLinearBandit(AbstractBandit):
    """Implements the action selection for a neural linear bandit model."""

    def __init__(
        self,
        n_features: int,
        n_embedding_size: int,
        encoder: nn.Module,
        initial_random_pulls: int = 0,
        eta: float = 6.0,
    ) -> None:
        """
        Initializes the NeuralLinearBandit.

        Args:
        - n_features (int): The number of features in the contextualized actions that are fed into the encoder.
        - n_embedding_size (int): The size of the tensors coming out of the encoder.
        - encoder (torch.nn.Module): The embedding model (neural network) to be used.
        - initial_random_pulls (int): The number of batches to initially select a random action for. Default is 0.
        - eta (float): The hyperparameter for the prior distribution sigma^2 ~ IG(eta, eta). eta > 1. Default is 6.0.
        """
        super().__init__(n_features)

        assert eta > 1, "eta must be greater than 1"

        self.encoder = encoder

        self.hparams = {
            "n_features": n_features,
            "n_embedding_size": n_embedding_size,
            "initial_random_pulls": initial_random_pulls,
            "eta": eta,
        }

        # Initialize the linear head which receives the embeddings
        self.mu = torch.zeros(n_embedding_size)
        self.cov = torch.eye(n_embedding_size)
        self.a: float = self.hparams["eta"]
        self.b: float = self.hparams["eta"]

        self.predictions = 0

    def forward(self, contextualized_actions: torch.Tensor) -> torch.Tensor:
        """
        Predict the action to take for the given input data according to neural linear.

        Args:
        - contextualised_actions (torch.Tensor): The input data. Shape: (batch_size, n_arms, n_features)
        """

        assert (
            contextualized_actions.ndim == 3
            and contextualized_actions.shape[2] == self.hparams["n_features"]
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        # Round robin until we have selected "initial_pulls" actions
        # TODO: Do we really need this?
        self.predictions += 1
        if self.predictions - 1 < self.hparams["initial_random_pulls"]:
            random_selected_actions = torch.randint(
                0, contextualized_actions.shape[1], (contextualized_actions.shape[0],)
            )  # shape: (batch_size,)
            random_result = nn.functional.one_hot(
                random_selected_actions, num_classes=contextualized_actions.shape[1]
            )  # shape: (batch_size, n_arms)
            return random_result

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
        """
        Linear head for the neural linear bandit model. Uses the Thompson Sampling algorithm to predict the best action.

        Args:
        - embedded_actions (torch.Tensor): The embedded actions. Shape: (batch_size, n_arms, n_embedding_size)
        """
        batch_size, n_arms, _ = embedded_actions.shape

        sigma2_tilde: float = self.b * invgamma.rvs(self.a)

        theta_tilde = torch.distributions.MultivariateNormal(  # type: ignore
            self.mu, sigma2_tilde * self.cov
        ).sample(
            (batch_size,)
        )  # shape: (batch_size, n_features)

        pred_reward_per_action = torch.einsum(
            "ijk,ik->ij", embedded_actions, theta_tilde
        )  # shape: (batch_size, n_arms)
        result = torch.argmax(pred_reward_per_action)  # shape: (batch_size,)

        return torch.nn.functional.one_hot(result, num_classes=n_arms).reshape(
            -1, n_arms
        )  # shape: (batch_size, n_arms)
