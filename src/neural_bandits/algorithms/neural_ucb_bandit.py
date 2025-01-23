import numpy as np
import torch
import torch.nn as nn

from .abstract_bandit import AbstractBandit


class NeuralUCBBandit(AbstractBandit):
    def __init__(
        self,
        network: nn.Module,
        n_features: int,
        lambda_: float = 1.0,
        nu: float = 1.0,
    ) -> None:
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

    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass using diagonal approximation
        """
        contextualised_actions = contextualised_actions.to(self.device)

        assert (
            contextualised_actions.shape[2] == self.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        contextualised_actions = contextualised_actions.squeeze(0)

        U_t_a_list = []  # Store U_t,a values
        g_t_a_list = []  # Store g(x_t,a; θ_t-1) values

        # Compute f(x_t,a; θ_t-1) for each arm
        f_t_a_list = self.theta_t(contextualised_actions)

        for f_t_a in f_t_a_list:
            # Calculate g(x_t,a; θ_t-1)
            self.theta_t.zero_grad()
            f_t_a.backward(retain_graph=True)
            g_t_a = torch.cat(
                [
                    p.grad.flatten().detach()
                    for p in self.theta_t.parameters()
                    if p.grad is not None
                ]
            )
            g_t_a_list.append(g_t_a)

            # Compute uncertainty using diagonal approximation
            exploration_term = torch.sqrt(
                torch.sum(self.lambda_ * self.nu * g_t_a * g_t_a / self.Z_t)
            )

            # UCB score U_t,a
            U_t_a = f_t_a.item() + exploration_term.item()
            U_t_a_list.append(U_t_a)

        # Select a_t = argmax_a U_t,a
        a_t = np.argmax(U_t_a_list)

        # Update Z_t using g(x_t,a_t; θ_t-1)
        self.Z_t += g_t_a_list[a_t] * g_t_a_list[a_t]

        U_t_tensor = torch.tensor(U_t_a_list).reshape(1, -1)
        probabilities = torch.softmax(U_t_tensor, dim=1)
        return probabilities
