from typing import Any, cast

import torch

from calvera.bandits.linear_bandit import LinearBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import AbstractSelector


class LinearUCBBandit(LinearBandit[torch.Tensor]):
    """Linear Upper Confidence Bound Bandit.

    Based on: Lattimore et al. "Bandit Algorithms" https://tor-lattimore.com/downloads/book/book.pdf
    """

    def __init__(
        self,
        n_features: int,
        selector: AbstractSelector | None = None,
        buffer: AbstractBanditDataBuffer[torch.Tensor, Any] | None = None,
        train_batch_size: int = 32,
        eps: float = 1e-2,
        lambda_: float = 1.0,
        lazy_uncertainty_update: bool = False,
        clear_buffer_after_train: bool = True,
        exploration_rate: float = 1.0,
    ) -> None:
        """Initializes the LinearBanditModule.

        Args:
            n_features: The number of features in the bandit model.
            selector: The selector used to choose the best action. Default is ArgMaxSelector (if None).
            buffer: The buffer used for storing the data for continuously updating the neural network.
            train_batch_size: The mini-batch size used for the train loop (started by `trainer.fit()`).
            eps: Small value to ensure invertibility of the precision matrix. Added to the diagonal.
            lambda_: Prior variance for the precision matrix. Acts as a regularization parameter.
            lazy_uncertainty_update: If True the precision matrix will not be updated during forward, but during the
                update step.
            clear_buffer_after_train: If True the buffer will be cleared after training. This is necessary because the
                data is not needed anymore after training once. Only set it to False if you know what you are doing.
            exploration_rate: The exploration parameter for LinUCB. In the original paper this is denoted as alpha.
                Must be greater than 0.
        """
        super().__init__(
            n_features,
            buffer=buffer,
            train_batch_size=train_batch_size,
            eps=eps,
            lambda_=lambda_,
            lazy_uncertainty_update=lazy_uncertainty_update,
            clear_buffer_after_train=clear_buffer_after_train,
            selector=selector,
        )

        self.save_hyperparameters({"exploration_rate": exploration_rate})

        assert exploration_rate > 0, "exploration_rate must be greater than 0"

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

        result = torch.einsum("ijk,k->ij", contextualized_actions, self.theta) + self.hparams[
            "exploration_rate"
        ] * torch.sqrt(
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
        self.precision_matrix = torch.diag_embed(
            torch.diag(self.precision_matrix) + prec_diagonal + cast(float, self.hparams["eps"])
        )

        return self.precision_matrix
