from typing import Any, cast

import torch

from calvera.bandits.action_input_type import ActionInputType
from calvera.bandits.linear_bandit import LinearBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import AbstractSelector


class LinearTSBandit(LinearBandit[ActionInputType]):
    """Linear Thompson Sampling Bandit.

    Based on: Agrawal et al. "Thompson Sampling for Contextual Bandits with Linear Payoffs" https://arxiv.org/abs/1209.3352
    """

    def __init__(
        self,
        n_features: int,
        selector: AbstractSelector | None = None,
        buffer: AbstractBanditDataBuffer[ActionInputType, Any] | None = None,
        train_batch_size: int = 32,
        eps: float = 1e-2,
        lambda_: float = 1.0,
        lazy_uncertainty_update: bool = False,
        clear_buffer_after_train: bool = True,
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
                data is not needed anymore after training. Only set it to False if you know what you are doing.
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

    def _predict_action_hook(
        self, contextualized_actions: ActionInputType, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given contextualized actions, predicts the best action using LinTS.

        Args:
            contextualized_actions: The input tensor of shape (batch_size, n_arms, n_features).
            kwargs: Additional keyword arguments. Not used.

        Returns:
            A tuple containing:
            - chosen_actions: The one-hot encoded tensor of the chosen actions.
                Shape: (batch_size, n_arms).
            - p: The probability of the chosen actions. For now we always return 1 but we might return the actual
                probability in the future. Shape: (batch_size, ).
        """
        assert isinstance(contextualized_actions, torch.Tensor), "contextualized_actions must be a torch.Tensor"
        assert contextualized_actions.shape[2] == self.hparams["n_features"], (
            "contextualized actions must have shape (batch_size, n_arms, n_features), "
            f"Got {contextualized_actions.shape}"
        )

        batch_size = contextualized_actions.shape[0]

        try:
            theta_tilde = torch.distributions.MultivariateNormal(self.theta, self.precision_matrix).sample(  # type: ignore
                (batch_size,)
            )
        except ValueError as e:
            # TODO: Could improve this case. See issue #158.
            raise ValueError(
                "The precision_matrix is not invertible anymore because it is not positive definite. "
                "This can happen due to numerical imprecisions. Try to increase the `eps` hyperparameter."
            ) from e

        expected_rewards = torch.einsum("ijk,ik->ij", contextualized_actions, theta_tilde)

        probabilities = self.compute_probabilities(contextualized_actions, theta_tilde)

        return self.selector(expected_rewards), probabilities

    def compute_probabilities(self, contextualized_actions: torch.Tensor, theta_tilde: torch.Tensor) -> torch.Tensor:
        """Compute the probability of the chosen actions.

        Args:
            contextualized_actions: The input tensor of shape (batch_size, n_arms, n_features).
            theta_tilde: The sampled theta from the posterior distribution of the model.
                Shape: (batch_size, n_features).

        Returns:
            The probability of the chosen actions. For now we always return 1 but we might return the actual probability
                in the future. Shape: (batch_size, ).
        """
        # TODO: Implement the actual probability computation for Thompson Sampling. See issue #72.
        return torch.ones(contextualized_actions.shape[0], device=contextualized_actions.device)


class DiagonalPrecApproxLinearTSBandit(LinearTSBandit[torch.Tensor]):
    """LinearTS but the precision matrix is updated using a diagonal approximation.

    Instead of doing a full update,
    only diag(Σ⁻¹)⁻¹ = diag(X X^T)⁻¹ is used. For compatibility reasons the precision matrix is still stored as a full
    matrix.
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
        self.precision_matrix.add_(torch.diag_embed(prec_diagonal) + cast(float, self.hparams["eps"]))

        return self.precision_matrix
