import torch

from neural_bandits.bandits.neural_bandit import NeuralBandit


class NeuralUCBBandit(NeuralBandit):
    """NeuralUCB bandit implementation as a PyTorch Lightning module.
    The NeuralUCB algorithm using a neural network for function approximation with diagonal approximation for exploration.

    Attributes:
        automatic_optimization: Boolean indicating if Lightning should handle optimization.
        bandit: The underlying NeuralUCBBandit instance.
    """

    def _score(
        self, f_t_a: torch.Tensor, exploration_terms: torch.Tensor
    ) -> torch.Tensor:
        """Compute a score based on the predicted rewards and exploration terms."""
        # UCB score U_t,a
        U_t = f_t_a + exploration_terms  # Shape: (batch_size, n_arms)

        return U_t
