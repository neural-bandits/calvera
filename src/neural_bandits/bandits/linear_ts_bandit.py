from typing import Any, Optional

import torch

from neural_bandits.bandits.linear_bandit import LinearBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class LinearTSBandit(LinearBandit):
    """Linear Thompson Sampling Bandit.

    See https://arxiv.org/abs/1802.09127 for more details.
    """

    def __init__(
        self,
        n_features: int,
        selector: Optional[AbstractSelector] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the LinearTSBandit.

        Args:
            n_features: The number of features in the bandit model.
            selector: The selector used to choose the best action. Default is ArgMaxSelector (if None).
            kwargs: Additional keyword arguments. Passed to the parent class. See `LinearBandit`.
        """
        super().__init__(n_features, **kwargs)
        self.selector = selector if selector is not None else ArgMaxSelector()

    def _predict_action(self, contextualized_actions: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
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
        assert (
            contextualized_actions.shape[2] == self.n_features
        ), "contextualized actions must have shape (batch_size, n_arms, n_features)"
        batch_size = contextualized_actions.shape[0]

        theta_tilde = torch.distributions.MultivariateNormal(self.theta, self.precision_matrix).sample(  # type: ignore
            (batch_size,)
        )

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
        # TODO: Implement the actual probability computation for Thompson Sampling.
        return torch.ones(contextualized_actions.shape[0], device=contextualized_actions.device)
