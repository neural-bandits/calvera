import torch
import torch.nn as nn

from .abstract_bandit import AbstractBandit


class NeuralTSBandit(AbstractBandit):
    """NeuralUCB bandit algorithm implementation.

    Implements the NeuralUCB algorithm using a neural network
    for function approximation with diagonal approximation for exploration.

    Attributes:
        device: PyTorch device (CPU/GPU).
        theta_t: Neural network for function approximation.
        context_history: List of context tensors.
        reward_history: List of reward tensors.
        lambda_: Regularization parameter.
        nu: Exploration parameter.
        Z_t: Diagonal approximation of covariance matrix.
        n_features: Number of input features.
    """

    def __init__(
        self,
        network: nn.Module,
        n_features: int,
        lambda_: float = 1.0,
        nu: float = 1.0,
    ) -> None:
        """Initialize NeuralUCB bandit.

        Args:
            network: Neural network for function approximation.
            n_features: Number of input features.
            lambda_: Regularization parameter.
            nu: Exploration parameter.
        """
        super().__init__(n_features)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize θ_t
        self.theta_t = network.to(self.device)

        # Track {x_i,a_i, r_i,a_i} history
        self.context_history: list[torch.Tensor] = []
        self.reward_history: list[torch.Tensor] = []

        self.lambda_ = lambda_
        self.nu = nu
        self.total_params = sum(
            p.numel() for p in self.theta_t.parameters() if p.requires_grad
        )

        # Initialize Z_0 = λI
        self.Z_t = lambda_ * torch.ones((self.total_params,), device=self.device)

        self.n_features = n_features

    def forward(self, contextualized_actions: torch.Tensor) -> torch.Tensor:
        """Calculate UCB scores for each action using diagonal approximation with batch support.

        Args:
            contextualized_actions: Contextualized action tensor of shape
                (batch_size, n_arms, n_features).

        Returns:
            Tensor of softmax probabilities over UCB scores with shape (batch_size, n_arms).

        Raises:
            AssertionError: If input tensor shape doesn't match n_features.
        """
        contextualized_actions = contextualized_actions.to(self.device)
        batch_size, n_arms, n_features = contextualized_actions.shape

        assert (
            n_features == self.n_features
        ), "Contextualized actions must have shape (batch_size, n_arms, n_features)"

        # Reshape input from (batch_size, n_arms, n_features) to (batch_size * n_arms, n_features)
        flattened_actions = contextualized_actions.reshape(-1, n_features)

        # Compute f(x_t,a; θ_t-1) for all arms in batch
        f_t_a = self.theta_t(flattened_actions)
        f_t_a = f_t_a.reshape(batch_size, n_arms)

        # Store g(x_t,a; θ_t-1) values
        all_gradients = torch.zeros(
            batch_size, n_arms, self.total_params, device=self.device
        )

        for b in range(batch_size):
            for a in range(n_arms):
                # Calculate g(x_t,a; θ_t-1)
                self.theta_t.zero_grad()
                f_t_a[b, a].backward(retain_graph=True)

                g_t_a = torch.cat(
                    [
                        p.grad.flatten().detach()
                        for p in self.theta_t.parameters()
                        if p.grad is not None
                    ]
                )
                all_gradients[b, a] = g_t_a

        # Compute uncertainty using diagonal approximation
        # Shape: (batch_size, n_arms)
        exploration_terms = torch.sqrt(
            torch.sum(
                self.lambda_ * self.nu * all_gradients * all_gradients / self.Z_t, dim=2
            )
        )

        # For TS, draw samples from Normal distributions:
        # For each arm: sample ~ N(mean = f_t_a, std = sigma)
        ts_samples = torch.normal(
            mean=f_t_a, std=exploration_terms
        )  # shape: (batch_size, n_arms)

        # Select a_t = argmax_a U_t,a
        chosen_actions = torch.argmax(ts_samples, dim=1)

        # Update Z_t using g(x_t,a_t; θ_t-1)
        for b in range(batch_size):
            a_t = chosen_actions[b]
            self.Z_t += all_gradients[b, a_t] * all_gradients[b, a_t]

        probabilities = torch.softmax(ts_samples, dim=1)
        return probabilities
