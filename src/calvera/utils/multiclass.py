import torch


class MultiClassContextualizer:
    """Applies disjoint model contextualization to the input feature vector.

    Example:
        ```python
        contextualizer = MultiClassContextualizer(n_arms=2)
        feature_vector = torch.tensor([[1, 0]])
        contextualizer(feature_vector)

        tensor([[[1, 0, 0, 0],
             [0, 0, 1, 0]]])
        ```
    """

    def __init__(
        self,
        n_arms: int,
    ) -> None:
        """Initializes the MultiClassContextualizer.

        Args:
            n_arms: The number of arms in the bandit model.
        """
        super().__init__()
        self.n_arms = n_arms

    def __call__(
        self,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the disjoint model contextualisation.

        Args:
            feature_vector: Input feature vector of shape (batch_size, n_features)

        Returns:
            contextualized actions of shape (batch_size, n_arms, n_features * n_arms)
        """
        assert len(feature_vector.shape) == 2, "Feature vector must have shape (batch_size, n_features)"

        n_features = feature_vector.shape[1]
        contextualized_actions = torch.einsum(
            "ij,bk->bijk", torch.eye(self.n_arms, device=feature_vector.device), feature_vector
        )
        contextualized_actions = contextualized_actions.reshape(-1, self.n_arms, n_features * self.n_arms)

        return contextualized_actions
