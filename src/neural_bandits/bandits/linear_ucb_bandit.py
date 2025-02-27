from typing import Any, Optional

import torch

from neural_bandits.bandits.linear_bandit import LinearBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class LinearUCBBandit(LinearBandit):
    """Linear Upper Confidence Bound Bandit.

    Based on: Lattimore et al. "Bandit Algorithms" https://tor-lattimore.com/downloads/book/book.pdf
    """

    def __init__(
        self,
        n_features: int,
        selector: Optional[AbstractSelector] = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initializes the LinearUCBBandit.

        Args:
            n_features: The number of features in the bandit model.
            selector: The selector used to choose the best action. Default is ArgMaxSelector (if None).
            alpha: The exploration parameter for LinUCB.
            kwargs: Additional keyword arguments. Passed to the parent class. See `LinearBandit`.
        """
        super().__init__(n_features, alpha=alpha, **kwargs)
        self.selector = selector if selector is not None else ArgMaxSelector()

    def _predict_action_hook(
        self, contextualized_actions: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given contextualized actions, predicts the best action using LinUCB.

        Args:
            contextualized_actions: The input tensor of shape (batch_size, n_arms, n_features).
            kwargs: Additional keyword arguments. Not used.

        Returns:
            tuple:
            - chosen_actions: The one-hot encoded tensor of the chosen actions.
            Shape: (batch_size, n_arms).
            - p: The probability of the chosen actions. For LinUCB this will always return 1.
            Shape: (batch_size, ).
        """
        assert (
            contextualized_actions.shape[2] == self.hparams["n_features"]
        ), "contextualized actions must have shape (batch_size, n_arms, n_features)"

        result = torch.einsum("ijk,k->ij", contextualized_actions, self.theta) + self.hparams["alpha"] * torch.sqrt(
            torch.einsum(
                "ijk,kl,ijl->ij",
                contextualized_actions,
                self.precision_matrix,
                contextualized_actions,
            )
        )

        return self.selector(result), torch.ones(contextualized_actions.shape[0], device=contextualized_actions.device)


class DiagonalPrecApproxLinearUCBBandit(LinearUCBBandit):
    """LinearUCB but the precision matrix is updated using a diagonal approximation.

    Instead of doing a full update,only diag(Σ⁻¹)⁻¹ = diag(X X^T)⁻¹ is used. For compatibility reasons the precision
    matrix is still stored as a full matrix.
    """

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
        self.precision_matrix = torch.diag_embed(torch.diag(self.precision_matrix) + prec_diagonal + self.eps)

        return self.precision_matrix
