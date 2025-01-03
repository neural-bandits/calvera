import torch

from .abtract_contextualiser import AbstractContextualiser


class MultiClassContextualiser(AbstractContextualiser):
    def __init__(
        self,
        n_arms: int,
    ) -> None:
        super().__init__()
        self.n_arms = n_arms

    def contextualise(
        self,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the disjoint model contextualisation.
        Example: [[1, 0]] with 2 arms becomes [[1, 0, 0, 0], [0, 0, 1, 0]]

        Args:
            feature_vector (torch.Tensor): Input feature vector of shape (batch_size, n_features)

        Returns:
            torch.Tensor: Contextualised actions of shape (batch_size, n_arms, n_features * n_arms)
        """
        n_features = feature_vector.shape[1]
        contextualised_actions = torch.einsum(
            "ij,bk->bijk", torch.eye(self.n_arms), feature_vector
        )
        contextualised_actions = contextualised_actions.reshape(
            -1, self.n_arms, n_features * self.n_arms
        )

        return contextualised_actions
