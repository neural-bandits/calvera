from typing import Any, cast

import torch

from calvera.bandits.linear_bandit import LinearBandit
from calvera.utils.action_input_type import ActionInputType
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import AbstractSelector


class LinearTSBandit(LinearBandit[ActionInputType]):
    r"""Linear Thompson Sampling Bandit.

    This implementation supports both standard and combinatorial bandit settings.

    Implementation details:
        Standard setting:

        - Initialize: $M^{-1} = I \cdot \lambda$, $b = 0$, $\theta = 0$

        - Sample: $\tilde{\theta}_t \sim \mathcal{N}(\theta_t, M^{-1})$

        - Score: $S_k(t) = x_k^T \tilde{\theta}_t$

        - Update:

            $b = b + r_t x_{a_t}$

            $M^{-1} = \left(M + x_{a_t}^T x_{a_t} + \varepsilon I \right)^{-1}$

            $M^{-1} = \frac{M^{-1} + \left( M^{-1} \right)^T}{2}$

            $\theta_t = M^{-1} b$

        (We store $M^{-1}$, the precision matrix, because we also allow the Sherman-Morrison update.)

        Combinatorial setting:

        - Uses the same initialization and sampling as the standard setting

        - Action selection: Use a selector (oracle $\mathcal{O}_S$) to select multiple arms as a super-action $S_t$

        - Updates $M$ and $\theta$ for each chosen arm $i \in S_t$

    References:
        - [Agrawal et al. "Thompson Sampling for Contextual Bandits with Linear Payoffs"](https://arxiv.org/abs/1209.3352)

        - [Wen et al. "Efficient Learning in Large-Scale Combinatorial Semi-Bandits"](https://arxiv.org/abs/1406.7443)
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
            lambda_: Prior precision for the precision matrix. Acts as a regularization parameter.
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
    r"""LinearTS but the precision matrix is updated using a diagonal approximation.

    Instead of doing a full update, only $\text{diag}(\Sigma^{-1})^{-1} = \text{diag}(X X^T)^{-1}$ is used.
    For compatibility reasons the precision matrix is still stored as a full matrix.
    """

    def _update_precision_matrix(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        """Update the precision matrix using an diagonal approximation. We use diag(Σ⁻¹)⁻¹.

        Args:
            chosen_actions: The chosen actions in the current batch.
                Shape: (batch_size, n_features).

        Returns:
            The updated precision matrix.
        """
        # Use the diagonal approximation.
        prec_diagonal = 1 / (torch.clamp(chosen_actions.pow(2).sum(dim=0), min=self.hparams["eps"]))

        # Update the precision matrix using the diagonal approximation. We use 1/(a+b) = 1/a * 1/b * 1/(1/a + 1/b) here.
        self.precision_matrix.copy_(
            torch.diag_embed(prec_diagonal)
            * self.precision_matrix.diag()
            / (torch.diag_embed(prec_diagonal) + self.precision_matrix.diag())
            + torch.diag_embed(torch.ones_like(prec_diagonal) * (cast(float, self.hparams["eps"])))
        )

        return self.precision_matrix
