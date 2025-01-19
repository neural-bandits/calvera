import torch
import torch.nn as nn
import numpy as np

from .abstract_bandit import AbstractBandit

class NeuralUCB(AbstractBandit):
    def __init__(self, network: nn.Module, n_arms: int, n_features: int, lambda_: float = 1.0, 
                 nu: float = 1.0, hidden_size: int = 100):
        super().__init__(n_arms, n_features)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.theta = network.to(self.device)

        self.context_list = []
        self.reward_list = []

        self.lambda_ = lambda_
        self.total_params = sum(p.numel() for p in self.theta.parameters() if p.requires_grad)
        self.U = lambda_ * torch.ones((self.total_params,), device=self.device)
        self.nu = nu
        
        self.n_arms = n_arms
        self.n_features = n_features
        self.hidden_size = hidden_size

    
    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass using diagonal approximation
        """
        contextualised_actions = contextualised_actions.to(self.device)
        
        assert (
            contextualised_actions.shape[1] == self.n_arms
            and contextualised_actions.shape[2] == self.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"
        
        # batch_size, n_arms, total_features = contextualised_actions.shape
        contextualised_actions = contextualised_actions.squeeze(0)
        
        sampled_values = []
        g_list = []

        mu_list = self.theta(contextualised_actions)
        
        for mu in mu_list:
            self.theta.zero_grad()
            mu.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.theta.parameters()])
            g_list.append(g)
            
            # Compute uncertainty using diagonal approximation
            sigma2 = self.lambda_ * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            
            # UCB score
            sampled_value = mu.item() + sigma.item()
            sampled_values.append(sampled_value)

        arm = np.argmax(sampled_values)
        self.U += g_list[arm] * g_list[arm]
        
        sampled_values_tensor = torch.tensor(sampled_values).reshape(1, -1)
        probabilities = torch.softmax(sampled_values_tensor, dim=1)
        return probabilities