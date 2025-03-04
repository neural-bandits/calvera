import logging
from typing import Any, Optional, cast

import torch
import torch.nn as nn

from calvera.bandits.neural_ucb_bandit import NeuralUCBBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import TopKSelector

logger = logging.getLogger(__name__)


class CombinatorialNeuralUCBBandit(NeuralUCBBandit):
    """Combinatorial Neural UCB bandit implementation as a PyTorch Lightning module.

    Based on: Hwang et al. "Combinatorial Neural Bandits" https://arxiv.org/abs/2306.00534

    This implementation follows the CN-UCB algorithm from the paper:
    1. Uses a neural network to approximate the score function h(x)
    2. Computes UCB scores based on the neural network outputs and gradients
    3. Selects a subset of K arms (combinatorial action) that maximizes the UCB score
    4. Updates the model based on the observed rewards
    """

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
        k: int,
        buffer: Optional[AbstractBanditDataBuffer[torch.Tensor, Any]] = None,
        exploration_rate: float = 1.0,
        train_batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.0,
        learning_rate_decay: float = 1.0,
        learning_rate_scheduler_step_size: int = 1,
        early_stop_threshold: Optional[float] = 1e-3,
        min_samples_required_for_training: Optional[int] = 64,
        initial_train_steps: int = 1024,
    ) -> None:
        """Initialize the Combinatorial Neural UCB bandit module.

        Args:
            n_features: Number of input features. Must be greater 0.
            network: Neural network module for function approximation.
            k: Number of arms to select in each round. Must be greater 0.
            buffer: Buffer for storing bandit interaction data.
            exploration_rate: Exploration parameter for UCB. Called gamma_t=nu in the original paper.
                Defaults to 1. Must be greater 0.
            train_batch_size: Size of mini-batches for training. Defaults to 32. Must be greater 0.
            learning_rate: The learning rate for the optimizer of the neural network.
                Default is 1e-3. Must be greater than 0.
            weight_decay: The regularization parameter for the neural network.
                Default is 1.0. Must be greater than 0.
            learning_rate_decay: Multiplicative factor for learning rate decay.
                Default is 1.0 (i.e. no decay). Must be greater than 0.
            learning_rate_scheduler_step_size: The step size for the learning rate decay.
                Default is 1. Must be greater than 0.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
                Defaults to 1e-3. Must be greater equal 0.
            min_samples_required_for_training: If less samples have been added via `record_feedback`
                than this value, the network is not trained.
                If None, the network is trained every time `trainer.fit` is called.
                Defaults to 64. Must be greater 0.
            initial_train_steps: For the first `initial_train_steps` samples, the network is always trained even if
                less new data than `min_samples_required_for_training` has been seen.
                Defaults to 1024. Must be greater equal 0.
        """
        assert k > 0, "Number of arms to select must be greater than 0"

        selector = TopKSelector(k)

        super().__init__(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=selector,
            exploration_rate=exploration_rate,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            learning_rate_decay=learning_rate_decay,
            learning_rate_scheduler_step_size=learning_rate_scheduler_step_size,
            early_stop_threshold=early_stop_threshold,
            min_samples_required_for_training=min_samples_required_for_training,
            initial_train_steps=initial_train_steps,
        )

        self.save_hyperparameters({"k": k})

    def _predict_action(
        self,
        contextualized_actions: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate UCB scores for each action and select k arms.

        This override ensures that Z_t is updated correctly for all chosen arms in combinatorial setting.
        Based on Algorithm 1 from the paper, with diagonal approximation for computational efficiency.

        Args:
            contextualized_actions: Contextualized action tensor. Shape: (batch_size, n_arms, n_features).
            kwargs: Additional keyword arguments. Not used.

        Returns:
            tuple:
            - chosen_actions: One-hot encoding of which actions were chosen. Shape: (batch_size, n_arms).
              For combinatorial bandit, this will have k ones per row.
            - p: Will always return a tensor of ones because UCB does not work on probabilities. Shape: (batch_size, ).
        """
        batch_size, n_arms, n_features = contextualized_actions.shape

        assert (
            n_features == self.hparams["n_features"]
        ), "Contextualized actions must have shape (batch_size, n_arms, n_features)"

        # Reshape input from (batch_size, n_arms, n_features) to (batch_size * n_arms, n_features)
        flattened_actions = contextualized_actions.reshape(-1, n_features)

        # Compute f(x_t,a; θ_t-1) for all arms in batch - predicted scores
        f_t_a: torch.Tensor = self.theta_t(flattened_actions)
        f_t_a = f_t_a.reshape(batch_size, n_arms)

        # Store g(x_t,a; θ_t-1) values - gradients for all arms
        all_gradients = torch.zeros(batch_size, n_arms, self.total_params, device=self.device)

        for b in range(batch_size):
            for a in range(n_arms):
                # Calculate g(x_t,a; θ_t-1)
                self.theta_t.zero_grad()
                f_t_a[b, a].backward(retain_graph=True)  # type: ignore

                g_t_a = torch.cat([p.grad.flatten().detach() for p in self.theta_t.parameters() if p.grad is not None])
                all_gradients[b, a] = g_t_a

        # Compute uncertainty using diagonal approximation - UCB exploration term
        # This is an approximation of the NTK-based confidence bound from the paper
        # Shape: (batch_size, n_arms)
        exploration_terms = torch.sqrt(
            torch.sum(
                self.hparams["weight_decay"]
                * self.hparams["exploration_rate"]
                * all_gradients
                * all_gradients
                / self.Z_t,
                dim=2,
            )
        )

        # Calculate UCB scores and select top k arms using the selector
        chosen_actions = self.selector(self._score(f_t_a, exploration_terms))

        # Verify that k actions are chosen per sample
        assert (
            chosen_actions.sum(dim=1) == self.hparams["k"]
        ).all(), f"Selector should choose {self.hparams['k']} actions"

        # Update Z_t using gradients of all chosen arms
        # In combinatorial setting, we update for each arm in the super arm
        for b in range(batch_size):
            chosen_arms = chosen_actions[b].nonzero().squeeze(1)
            for arm_idx in chosen_arms:
                self.Z_t += all_gradients[b, arm_idx] * all_gradients[b, arm_idx]

        # Return chosen actions and probability (always 1 for UCB)
        return chosen_actions, torch.ones(batch_size, device=self.device)

    def _add_data_to_buffer(
        self,
        contextualized_actions: torch.Tensor,
        realized_rewards: torch.Tensor,
        embedded_actions: Optional[torch.Tensor] = None,
    ) -> None:
        """Override to handle combinatorial feedback.

        This is a workaround for the current buffer that doesn't fully support combinatorial bandits.
        For each chosen arm, we add a separate entry to the buffer.

        Args:
            contextualized_actions: The contextualized actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features).
            realized_rewards: The rewards that were observed for the chosen actions.
                Size: (batch_size, n_actions).
            embedded_actions: The embedded actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features). Optional because not every model uses embedded actions.
        """
        assert realized_rewards.ndim == 2, "Realized rewards must have shape (batch_size, n_chosen_actions)."

        batch_size = realized_rewards.shape[0]
        n_chosen_actions = realized_rewards.shape[1]

        for b in range(batch_size):
            for a in range(n_chosen_actions):
                single_action = contextualized_actions[b, a].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_features)
                single_reward = realized_rewards[b, a].unsqueeze(0).unsqueeze(1)  # Shape: (1, 1)

                super()._add_data_to_buffer(cast(torch.Tensor, single_action), single_reward, None)
