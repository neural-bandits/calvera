from typing import Any, Optional

import torch
import torch.nn as nn
from torch import optim

from neural_bandits.bandits.abstract_bandit import AbstractBandit
from neural_bandits.utils.data_storage import AbstractBanditDataBuffer
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class NeuralUCBBandit(AbstractBandit):
    """NeuralUCB bandit implementation as a PyTorch Lightning module.
    The NeuralUCB algorithm using a neural network for function approximation with diagonal approximation for exploration.

    Attributes:
        automatic_optimization: Boolean indicating if Lightning should handle optimization.
        bandit: The underlying NeuralUCBBandit instance.
    """

    def __init__(
        self,
        n_features: int,
        network: nn.Module,
        buffer: AbstractBanditDataBuffer,
        batch_size: int = 32,
        selector: AbstractSelector = ArgMaxSelector(),
        early_stop_threshold: Optional[float] = 1e-3,
        num_grad_steps: int = 1000,
        lambda_: float = 0.00001,
        nu: float = 0.00001,
        learning_rate: float = 0.01,
        train_freq: int = 100,
        initial_train_steps: int = 1000,
        max_grad_norm: float = 20.0,
        **kw_args: Any,
    ) -> None:
        """Initialize the NeuralUCB bandit module.

        Args:
            n_features: Number of input features.
            network: Neural network module for function approximation.
            buffer: Buffer for storing bandit interaction data.
            batch_size: Size of mini-batches for training. Defaults to 32.
            selector: Action selector for the bandit. Defaults to ArgMaxSelector.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
            num_grad_steps: Maximum number of gradient steps per training iteration.
            lambda_: Regularization parameter.
            nu: Exploration parameter for UCB.
            learning_rate: Learning rate for SGD optimizer.
            train_freq: Frequency of network training after initial training.
            initial_train_steps: Number of initial training steps.
            max_grad_norm: Maximum gradient norm for gradient clipping.
            **kw_args: Additional arguments passed to parent class.
        """
        super().__init__()

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

        # Save hyperparameters
        hyperparameters = {
            "n_features": n_features,
            "batch_size": batch_size,
            "early_stop_threshold": early_stop_threshold,
            "num_grad_steps": num_grad_steps,
            "lambda_": lambda_,
            "nu": nu,
            "learning_rate": learning_rate,
            "train_freq": train_freq,
            "initial_train_steps": initial_train_steps,
            "max_grad_norm": max_grad_norm,
            **kw_args,
        }
        self.save_hyperparameters(hyperparameters)

        self.selector = selector

        # Model parameters

        # Initialize θ_t
        self.theta_t = network.to(self.device)

        # Track {x_i,a_i, r_i,a_i} history
        self.buffer = buffer

        self.total_params = sum(
            p.numel() for p in self.theta_t.parameters() if p.requires_grad
        )

        # Initialize Z_0 = λI
        self.Z_t = self.hparams["lambda_"] * torch.ones(
            (self.total_params,), device=self.device
        )

        self._samples_after_initial = 0

    def _predict_action(
        self,
        contextualized_actions: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate UCB scores for each action using diagonal approximation with batch support.

        Args:
            contextualized_actions: Contextualized action tensor. Shape: (batch_size, n_arms, n_features).

        Returns:
            tuple:
            - chosen_actions: One-hot encoding of which actions were chosen. Shape: (batch_size, num_actions).
            - p: Will always return a tensor of ones because UCB does not work on probabilities. Shape: (batch_size, ).
        """
        contextualized_actions = contextualized_actions.to(self.device)
        batch_size, n_arms, n_features = contextualized_actions.shape

        assert (
            n_features == self.hparams["n_features"]
        ), "Contextualized actions must have shape (batch_size, n_arms, n_features)"

        # Reshape input from (batch_size, n_arms, n_features) to (batch_size * n_arms, n_features)
        flattened_actions = contextualized_actions.reshape(-1, n_features)

        # Compute f(x_t,a; θ_t-1) for all arms in batch
        f_t_a: torch.Tensor = self.theta_t(flattened_actions)
        f_t_a = f_t_a.reshape(batch_size, n_arms)

        # Store g(x_t,a; θ_t-1) values
        all_gradients = torch.zeros(
            batch_size, n_arms, self.total_params, device=self.device
        )

        for b in range(batch_size):
            for a in range(n_arms):
                # Calculate g(x_t,a; θ_t-1)
                self.theta_t.zero_grad()
                f_t_a[b, a].backward(retain_graph=True)  # type: ignore

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
                self.hparams["lambda_"]
                * self.hparams["nu"]
                * all_gradients
                * all_gradients
                / self.Z_t,
                dim=2,
            )
        )

        # UCB score U_t,a
        # Shape: (batch_size, n_arms)
        U_t = f_t_a + exploration_terms

        # Select a_t = argmax_a U_t,a
        chosen_actions = self.selector(U_t)

        assert (
            chosen_actions.sum(dim=1) == 1
        ).all(), "Currently only supports non-combinatorial bandits"
        chosen_actions_idx = chosen_actions.argmax(
            dim=1
        )  # TODO: this only works for non-combinatorial bandits!

        # Update Z_t using g(x_t,a_t; θ_t-1)
        for b in range(batch_size):
            a_t = chosen_actions_idx[b]
            self.Z_t += all_gradients[b, a_t] * all_gradients[b, a_t]

        # Return chosen actions and
        return chosen_actions, torch.ones(batch_size, device=self.device)

    def _update(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute a single training step.

        Args:
            batch: Tuple of (contextualized_actions, rewards) tensors.
            batch_idx: Index of current batch.

        Returns:
            Mean negative reward as loss value.

        Example:
            >>> batch = (context_tensor, reward_tensor)
            >>> loss = model.training_step(batch, 0)
        """
        contextualized_actions: torch.Tensor = batch[
            0
        ]  # shape: (batch_size, n_arms, n_features)
        realized_rewards: torch.Tensor = batch[1]  # shape: (batch_size, n_arms)

        assert (
            self.hparams["batch_size"] <= self.hparams["train_freq"]
        ), f"Batch size ({self.hparams['batch_size']}) must be less than or equal to train frequency ({self.hparams['train_freq']})"

        # Update bandit's history
        self.buffer.add_batch(
            contextualized_actions=contextualized_actions.view(
                -1, contextualized_actions.size(-1)
            ),
            embedded_actions=None,
            rewards=realized_rewards.squeeze(1),
        )

        # Train network based on schedule
        should_train = self._should_train_network()

        if should_train:
            loss = self._train_network()
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Logging
        self.log(
            "reward",
            realized_rewards.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return -realized_rewards.mean()

    def _train_network(self) -> float:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        L_theta_sum: float = 0.0  # Track cumulative loss L(θ)

        for j in range(self.hparams["num_grad_steps"]):
            context, _, reward = self.buffer.get_batch(
                batch_size=self.hparams["batch_size"]
            )

            optimizer.zero_grad()

            # Compute f(x_i,a_i; θ)
            f_theta = self.theta_t(context)
            L_theta = torch.nn.functional.mse_loss(f_theta, reward.unsqueeze(1))
            L_theta = L_theta.sum() / self.hparams["batch_size"]
            self.manual_backward(L_theta)

            torch.nn.utils.clip_grad_norm_(
                self.theta_t.parameters(),
                max_norm=self.hparams["max_grad_norm"],
            )

            optimizer.step()

            L_theta_sum += L_theta.item()

            if (
                self.hparams["early_stop_threshold"] is not None
                and (L_theta_sum / (j + 1)) <= self.hparams["early_stop_threshold"]
            ):
                break

        return float(L_theta_sum / (j + 1))

    def _should_train_network(self) -> bool:
        """
        Determine if the network should be trained based on buffer size,
        initial training steps, and training frequency.
        """
        if len(self.buffer) <= self.hparams["initial_train_steps"]:
            return True

        if (
            len(self.buffer) - self.hparams["batch_size"]
            <= self.hparams["initial_train_steps"]
        ):
            self._samples_after_initial = (
                len(self.buffer) - self.hparams["initial_train_steps"]
            )
        else:
            self._samples_after_initial += self.hparams["batch_size"]

        if self._samples_after_initial >= self.hparams["train_freq"]:
            self._samples_after_initial -= self.hparams["train_freq"]
            return True

        return False

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(
            self.theta_t.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["lambda_"],
        )
