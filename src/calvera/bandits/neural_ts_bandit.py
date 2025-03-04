import torch

from calvera.bandits.neural_bandit import NeuralBandit


class NeuralTSBandit(NeuralBandit):
    """Neural Thompson Sampling (TS) bandit implementation as a PyTorch Lightning module.

    Based on: Zhang et al. "Neural Thompson Sampling" https://arxiv.org/abs/2010.00827

    Implements the NeuralTS algorithm using a neural network for function approximation
    with a diagonal approximation. The module maintains a history of contexts and rewards,
    and periodically updates the network parameters via gradient descent.
    """

    def _score(self, f_t_a: torch.Tensor, exploration_terms: torch.Tensor) -> torch.Tensor:
        # For TS, draw samples from Normal distributions:
        # For each arm: sample ~ N(mean = f_t_a, std = sigma)
        ts_samples = torch.normal(mean=f_t_a, std=exploration_terms)  # shape: (batch_size, n_arms)

        return ts_samples
