import torch
from neural_bandits.algorithms.linear_bandits import LinearBandit


class LinearBanditTrainer:
    """
    Trainer class for the linear bandit model.
    Implements the Sherman-Morrison formula for updating the precision matrix M and the vector b of the given bandit.
    """

    def __init__(self, bandit: LinearBandit):
        """
        Initializes the LinearBanditTrainer with the given bandit.

        Args:
            bandit: The linear bandit model to train.
        """
        self.bandit = bandit

    def training_step(
        self,
        chosen_actions: torch.Tensor,
        realized_rewards: torch.Tensor,
    ) -> None:
        """
        Perform an update step on the linear bandit given the actions that were chosen and the rewards that were observed.

        Args:
            chosen_actions: The chosen contextualized actions in this batch. Shape: (batch_size, n_features)
            realized_rewards: The realized rewards of the chosen action in this batch. Shape: (batch_size,)
        """

        assert (
            chosen_actions.shape[0] == realized_rewards.shape[0]
        ), "Batch size of chosen actions and realized_rewards must match"

        assert (
            chosen_actions.shape[1] == self.bandit.n_features
        ), "Chosen actions must have shape (batch_size, n_features) and n_features must match the bandit's n_features"

        assert (
            realized_rewards.ndim == 1
        ), "Realized rewards must have shape (batch_size,)"

        # Calculate new precision Matrix M using the Sherman-Morrison formula
        denominator = 1 + (
            (chosen_actions @ self.bandit.precision_matrix) * chosen_actions
        ).sum(dim=1).sum(dim=0)
        assert torch.abs(denominator) > 0, "Denominator must not be zero or nan"

        self.bandit.precision_matrix = (
            self.bandit.precision_matrix
            - (
                self.bandit.precision_matrix
                @ torch.einsum("bi,bj->bij", chosen_actions, chosen_actions).sum(dim=0)
                @ self.bandit.precision_matrix
            )
            / denominator
        )
        self.bandit.precision_matrix = 0.5 * (
            self.bandit.precision_matrix + self.bandit.precision_matrix.T
        )
        # should be symmetric
        assert torch.allclose(
            self.bandit.precision_matrix, self.bandit.precision_matrix.T
        ), "M must be symmetric"

        self.bandit.b += chosen_actions.T @ realized_rewards  # shape: (features,)
        self.bandit.theta = self.bandit.precision_matrix @ self.bandit.b

        assert (
            self.bandit.b.ndim == 1 and self.bandit.b.shape[0] == self.bandit.n_features
        ), "updated b should have shape (n_features,)"

        assert (
            self.bandit.theta.ndim == 1
            and self.bandit.theta.shape[0] == self.bandit.n_features
        ), "Theta should have shape (n_features,)"
