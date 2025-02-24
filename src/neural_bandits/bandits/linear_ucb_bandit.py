from typing import Any

import torch

from neural_bandits.bandits.linear_bandit import LinearBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class LinearUCBBandit(LinearBandit):
    """Linear Upper Confidence Bound Bandit."""
    def __init__(
        self,
        n_features: int,
        selector: AbstractSelector = ArgMaxSelector(),
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initializes the LinearUCBBandit.
        
        Args:
            n_features: The number of features in the bandit model.
            selector: The selector used to choose the best action. Default is ArgMaxSelector.
            alpha: The exploration parameter for LinUCB.
            kwargs: Additional keyword arguments. Passed to the parent class. See `LinearBandit`.
        """
        super().__init__(n_features, alpha=alpha, **kwargs)
        self.selector = selector

    def _predict_action(
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

        result = torch.einsum(
            "ijk,k->ij", contextualized_actions, self.theta
        ) + self.hparams["alpha"] * torch.sqrt(
            torch.einsum(
                "ijk,kl,ijl->ij",
                contextualized_actions,
                self.precision_matrix,
                contextualized_actions,
            )
        )

        return self.selector(result), torch.ones(
            contextualized_actions.shape[0], device=contextualized_actions.device
        )
