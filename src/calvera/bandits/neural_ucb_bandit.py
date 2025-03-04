import torch

from calvera.bandits.neural_bandit import NeuralBandit


class NeuralUCBBandit(NeuralBandit):
    """NeuralUCB bandit implementation as a PyTorch Lightning module.

    Based on: Zhou et al. "Neural Contextual Bandits with UCB-based Exploration" https://arxiv.org/abs/1911.04462

    The NeuralUCB algorithm using a neural network for function approximation with diagonal approximation for
    exploration.
    """

    def _score(self, f_t_a: torch.Tensor, exploration_terms: torch.Tensor) -> torch.Tensor:
        """Compute a score based on the predicted rewards and exploration terms."""
        # UCB score U_t,a
        U_t = f_t_a + exploration_terms  # Shape: (batch_size, n_arms)

        return U_t
