from typing import Any, cast

import torch

from calvera.bandits.linear_bandit import LinearBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import AbstractSelector


class LinearUCBBandit(LinearBandit[torch.Tensor]):
    r"""Linear Upper Confidence Bound Bandit.

    This implementation supports both standard and combinatorial bandit settings.

    Implementation details:
        Standard setting:

        - Initialize: $M^{-1} = I \cdot \lambda$, $b = 0$, $\theta = 0$

        - Compute UCB scores: $U_k(t) = x_k^T \hat{\theta}_t + \alpha \sqrt{x_k^T M^{-1} x_k}$
          where $\alpha$ is the exploration parameter

        - Update:

            $b = b + r_t x_{a_t}$

            $M^{-1} = \left( M + x_{a_t}^T x_{a_t} + \varepsilon I \right)^{-1}$

            $M^{-1} = \frac{M^{-1} + \left( M^{-1} \right)^T}{2}$

            $\theta_t = M^{-1} b$

        (We store $M^{-1}$, the precision matrix, because we also allow the Sherman-Morrison update.)

        Combinatorial setting:

        - Uses the same initialization and UCB formula as the standard setting

        - Action selection: Use a selector (oracle $\mathcal{O}_S$) to select multiple arms as a super-action $S_t$

        - Updates $M$ and $\theta$ for each chosen arm $i \in S_t$

    References:
        - [Lattimore et al. "Bandit Algorithms", Chapter 19](https://tor-lattimore.com/downloads/book/book.pdf)

        - [Wen et al. "Efficient Learning in Large-Scale Combinatorial Semi-Bandits"](https://arxiv.org/abs/1406.7443)
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
            lambda_: Prior precision for the precision matrix. Acts as a regularization parameter.
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
    r"""LinearUCB but the precision matrix is updated using a diagonal approximation.

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
