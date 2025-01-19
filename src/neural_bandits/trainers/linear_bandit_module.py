from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn

from ..algorithms.linear_bandits import LinearBandit, LinearTSBandit, LinearUCBBandit
from ..utils.multiclass import MultiClassContextualizer
from .abstract_bandit_module import AbstractBanditModule, RewardFunction

LinearBanditType = TypeVar("LinearBanditType", bound="LinearBandit")


class LinearBanditModule(
    AbstractBanditModule[LinearBanditType], Generic[LinearBanditType]
):
    def __init__(
        self,
        linear_bandit_type: LinearBanditType,
        n_features: int,
        **kw_args: Any,
    ) -> None:
        """Initializes the LinearBanditModule.
        Args:
            n_features: The number of features in the bandit model.
        """
        super().__init__()
        self.n_features = n_features

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save Hyperparameters
        hyperparameters = {
            "linear_bandit_type": linear_bandit_type.__name__,
            "n_features": n_features,
            **kw_args,
        }
        

        self.bandit = linear_bandit_type(n_features=n_features, **kw_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bandit(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform an update step on the linear bandit model."""
        contextualized_actions, rewards = batch
        chosen_actions_idx = self(contextualized_actions).argmax(dim=1)
        realized_rewards = rewards[torch.arange(rewards.shape[0]), chosen_actions_idx]
        batch_size = chosen_actions_idx.shape[0]
        
        # Update the self.bandit
        chosen_actions = contextualized_actions[torch.arange(batch_size), chosen_actions_idx]
        
        # Calculate new precision Matrix M using the Sherman-Morrison formula
        denominator = 1 + ((chosen_actions @ self.bandit.M) * chosen_actions).sum(dim=1).sum(dim=0)
        assert torch.abs(denominator - 0) > 0, "Denominator must not be zero"
        
        self.bandit.M = self.bandit.M - (self.bandit.M @ torch.einsum('bi,bj->bij', chosen_actions, chosen_actions).sum(dim=0) @ self.bandit.M) / denominator
        self.bandit.M = 0.5 * (self.bandit.M + self.bandit.M.T)
        # should be symmetric
        assert torch.allclose(self.bandit.M, self.bandit.M.T), "M must be symmetric"
        
        self.bandit.b += chosen_actions.T @ realized_rewards  # shape: (features,)
        self.bandit.theta = self.bandit.M @ self.bandit.b

        
        self.log("reward", realized_rewards.mean(), on_step=True, on_epoch=False, prog_bar=True)
        regret = torch.max(rewards, dim=1).values - realized_rewards
        self.log("regret", regret.mean(), on_step=True, on_epoch=False, prog_bar=True)
        
        
        
        return -rewards.mean()
    
    def configure_optimizers(self):
        return None
