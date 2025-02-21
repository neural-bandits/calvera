import torch


class MultiClassContextualizer:
    def __init__(
        self,
        n_arms: int,
    ) -> None:
        super().__init__()
        self.n_arms = n_arms

    def __call__(
        self,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the disjoint model contextualisation.
        Example: [[1, 0]] with 2 arms becomes [[1, 0, 0, 0], [0, 0, 1, 0]]

        Args:
            feature_vector: Input feature vector of shape (batch_size, n_features)

        Returns:
            contextualized actions of shape (batch_size, n_arms, n_features * n_arms)
        """
        assert (
            len(feature_vector.shape) == 2
        ), "Feature vector must have shape (batch_size, n_features)"

        n_features = feature_vector.shape[1]
        contextualized_actions = torch.einsum(
            "ij,bk->bijk", torch.eye(self.n_arms), feature_vector
        )
        contextualized_actions = contextualized_actions.reshape(
            -1, self.n_arms, n_features * self.n_arms
        )

        return contextualized_actions
