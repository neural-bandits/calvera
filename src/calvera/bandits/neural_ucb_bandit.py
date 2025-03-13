import torch

from calvera.bandits.neural_bandit import NeuralBandit


class NeuralUCBBandit(NeuralBandit):
    r"""NeuralUCB bandit implementation as a PyTorch Lightning module.

    The NeuralUCB algorithm using a neural network for function approximation with diagonal approximation for
    exploration. This implementation supports both standard and combinatorial bandit settings.

    Implementation details:
        Standard setting:

        - UCB: $u_{t,a} = f(x_{t,a}; \theta_{t-1}) + \sqrt{\lambda \nu \cdot g(x_{t,a}; \theta_{t-1})^T
        Z_{t-1}^{-1} g(x_{t,a}; \theta_{t-1})}$

        - Update: $Z_t = Z_{t-1} + g(x_{t,a_t}; \theta_{t-1})g(x_{t,a_t}; \theta_{t-1})^T$

        Combinatorial setting:

        - Same UCB formula for each arm

        - Select super arm: $S_t = \mathcal{O}_S(u_t)$

        - Update includes gradients from all chosen arms:
        $Z_t = Z_{t-1} + \sum_{a \in S_t} g(x_{t,a_t}; \theta_{t-1})g(x_{t,a_t}; \theta_{t-1})^T$


    References:
        - [Zhou et al. "Neural Contextual Bandits with UCB-based Exploration" (2020)](https://arxiv.org/abs/1911.04462)

        - [Hwang et al. "Combinatorial Neural Bandits" (2023)](https://arxiv.org/abs/2306.00242)
    """

    def _score(self, f_t_a: torch.Tensor, exploration_terms: torch.Tensor) -> torch.Tensor:
        """Compute a score based on the predicted rewards and exploration terms."""
        # UCB score U_t,a
        U_t = f_t_a + exploration_terms  # Shape: (batch_size, n_arms)

        return U_t
