from typing import Any

import torch

from neural_bandits.bandits.linear_bandit import LinearBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class LinearTSBandit(LinearBandit):
    def __init__(
        self,
        n_features: int,
        selector: AbstractSelector = ArgMaxSelector(),
        **kwargs: Any,
    ) -> None:
        """
        Initializes the LinearTSBandit.

        Args:
            n_features: The number of features in the bandit model.
            selector: The selector used to choose the best action. Default is ArgMaxSelector.
        """
        super().__init__(n_features, **kwargs)
        self.selector = selector

    def _predict_action(
        self, contextualized_actions: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given contextualized actions, predicts the best action using LinTS.

        Args:
            contextualized_actions: The input tensor of shape (batch_size, n_arms, n_features).

        Returns:
            tuple:
            - chosen_actions: The one-hot encoded tensor of the chosen actions.
                Shape: (batch_size, n_arms).
            - p: The probability of the chosen actions. For now we always return 1 but we might return the actual
                probability in the future.
                Shape: (batch_size, ).
        """

        assert (
            contextualized_actions.shape[2] == self.n_features
        ), "contextualized actions must have shape (batch_size, n_arms, n_features)"
        batch_size = contextualized_actions.shape[0]

        theta_tilde = torch.distributions.MultivariateNormal(
            self.theta, precision_matrix=self.precision_matrix
        ).sample(  # type: ignore
            (batch_size,)
        )

        expected_rewards = torch.einsum(
            "ijk,ik->ij", contextualized_actions, theta_tilde
        )

        probabilities = self.compute_probabilities(contextualized_actions, theta_tilde)

        return self.selector(expected_rewards), probabilities

    def compute_probabilities(
        self, contextualized_actions: torch.Tensor, theta_tilde: torch.Tensor
    ) -> torch.Tensor:
        """Compute the probability of the chosen actions.

        Args:
            contextualized_actions: The input tensor of shape (batch_size, n_arms, n_features).
            theta_tilde: The sampled theta from the posterior distribution of the model.
                Shape: (batch_size, n_features).

        Returns:
            The probability of the chosen actions. For now we always return 1 but we might return the actual
            probability in the future.
                Shape: (batch_size, ).
        """
        # TODO: Implement the actual probability computation for Thompson Sampling.
        return torch.ones(
            contextualized_actions.shape[0], device=contextualized_actions.device
        )


class DiagonalPrecApproxLinearTSBandit(LinearTSBandit):
    """LinearUCB but the precision matrix is updated using a diagonal approximation. Instead of doing a full update,
    only diag(Σ⁻¹)⁻¹ = diag(X X^T)⁻¹ is used. For compatibility reasons the precision matrix is still stored as a full
    matrix."""

    def _update_precision_matrix(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        """Update the precision matrix using an diagonal approximation. We use diag(Σ⁻¹)⁻¹.

        Args:
            chosen_actions: The chosen actions in the current batch.
                Shape: (batch_size, n_features).

        Returns:
            The updated precision matrix.
        """

        # Compute the covariance matrix of the chosen actions. Use the diagonal approximation.
        prec_diagonal = chosen_actions.pow(2).sum(dim=0)

        # Update the precision matrix using the diagonal approximation.
        self.precision_matrix = torch.diag_embed(
            torch.diag(self.precision_matrix) + prec_diagonal + self.eps
        )

        return self.precision_matrix
