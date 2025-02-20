from typing import Any

import torch

from neural_bandits.bandits.linear_bandit import LinearBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class LinearUCBBandit(LinearBandit):
    def __init__(
        self,
        n_features: int,
        selector: AbstractSelector = ArgMaxSelector(),
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_features, alpha=alpha, **kwargs)
        self.selector = selector

    def _predict_action(
        self, contextualized_actions: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given contextualized actions, predicts the best action using LinUCB.

        Args:
            contextualized_actions (torch.Tensor): The input tensor of shape (batch_size, n_arms, n_features).

        Returns:
            tuple:
            - chosen_actions (torch.Tensor): The one-hot encoded tensor of the chosen actions.
            Shape: (batch_size, n_arms).
            - p (torch.Tensor): The probability of the chosen actions. For LinUCB this will always return 1.
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
