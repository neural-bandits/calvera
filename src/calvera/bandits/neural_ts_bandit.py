from typing import Any

import torch
import torch.nn as nn

from calvera.bandits.neural_bandit import NeuralBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer
from calvera.utils.selectors import AbstractSelector


class NeuralTSBandit(NeuralBandit):
    """Neural Thompson Sampling (TS) bandit implementation as a PyTorch Lightning module.

    Based on: Zhang et al. "Neural Thompson Sampling" https://arxiv.org/abs/2010.00827

    Implements the NeuralTS algorithm using a neural network for function approximation
    with a diagonal approximation. The module maintains a history of contexts and rewards,
    and periodically updates the network parameters via gradient descent.
    """

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
        buffer: AbstractBanditDataBuffer[torch.Tensor, Any] | None = None,
        selector: AbstractSelector | None = None,
        exploration_rate: float = 1.0,
        train_batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.0,
        learning_rate_decay: float = 1.0,
        learning_rate_scheduler_step_size: int = 1,
        early_stop_threshold: float | None = 1e-3,
        min_samples_required_for_training: int = 64,
        initial_train_steps: int = 1024,
        num_samples_per_arm: int = 1,
    ) -> None:
        """Initialize the NeuralTS bandit module.

        Args:
            n_features: Number of input features. Must be greater 0.
            network: Neural network module for function approximation.
            buffer: Buffer for storing bandit interaction data.
            selector: Action selector for the bandit. Defaults to ArgMaxSelector (if None).
            exploration_rate: Exploration parameter for UCB. Called gamma_t=nu in the original paper.
                Defaults to 1. Must be greater 0.
            train_batch_size: Size of mini-batches for training. Defaults to 32. Must be greater 0.
            learning_rate: The learning rate for the optimizer of the neural network.
                Passed to `lr` of `torch.optim.Adam`.
                Default is 1e-3. Must be greater than 0.
            weight_decay: The regularization parameter for the neural network.
                Passed to `weight_decay` of `torch.optim.Adam`.
                Default is 1.0. Must be greater than 0 because the NeuralUCB algorithm is based on this parameter.
            learning_rate_decay: Multiplicative factor for learning rate decay.
                Passed to `gamma` of `torch.optim.lr_scheduler.StepLR`.
                Default is 1.0 (i.e. no decay). Must be greater than 0.
            learning_rate_scheduler_step_size: The step size for the learning rate decay.
                Passed to `step_size` of `torch.optim.lr_scheduler.StepLR`.
                Default is 1. Must be greater than 0.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
                Defaults to 1e-3. Must be greater equal 0.
            min_samples_required_for_training: If less samples have been added via `record_feedback`
                than this value, the network is not trained.
                Defaults to 64. Must be greater 0.
            initial_train_steps: For the first `initial_train_steps` samples, the network is always trained even if
                less new data than `min_samples_required_for_training` has been seen. Therefore, this value is only
                required if `min_samples_required_for_training` is set. Set to 0 to disable this feature.
                Defaults to 1024. Must be greater equal 0.
            num_samples_per_arm: Number of samples to draw from each Normal distribution in Thompson Sampling.
                Defaults to 1. Must be greater than 0.
        """
        assert num_samples_per_arm > 0, "Number of samples must be greater than 0."

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

        self.save_hyperparameters({"num_samples_per_arm": num_samples_per_arm})

    def _score(self, f_t_a: torch.Tensor, exploration_terms: torch.Tensor) -> torch.Tensor:
        # For Thompson Sampling, draw M samples from Normal distributions for each arm
        # For each arm: sample ~ N(mean = f_t_a, std = sigma)

        batch_size, n_arms = f_t_a.shape
        all_samples = torch.zeros(batch_size, n_arms, self.hparams["num_samples_per_arm"], device=self.device)

        for m in range(self.hparams["num_samples_per_arm"]):
            samples = torch.normal(mean=f_t_a, std=exploration_terms)  # shape: (batch_size, n_arms)
            all_samples[:, :, m] = samples

        # Take the maximum across all samples for each arm (optimistic sampling)
        optimistic_samples, _ = torch.max(all_samples, dim=2)

        return optimistic_samples
