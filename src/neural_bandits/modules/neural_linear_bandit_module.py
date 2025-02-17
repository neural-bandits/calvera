from typing import Optional

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from neural_bandits.algorithms.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.modules.abstract_bandit_module import AbstractBanditModule
from neural_bandits.utils.data_storage import AbstractBanditDataBuffer


class NeuralLinearBanditModule(AbstractBanditModule[NeuralLinearBandit]):
    """
    Module for training a Neural Linear bandit model.
    The Neural Linear algorithm is described in the paper Riquelme et al., 2018, Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling.
    A Neural Linear bandit model consists of a neural network that produces embeddings of the input data and a linear head that is trained on the embeddings.
    Since updating the neural network (encoder) is computationally expensive, the neural network is only updated every `embedding_update_interval` steps.
    On the other hand, the linear head is updated every `head_update_freq` steps which should be much lower.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        n_features: int,
        buffer: AbstractBanditDataBuffer,
        n_embedding_size: Optional[int],
        encoder_update_freq: int = 32,
        encoder_update_batch_size: int = 32,
        head_update_freq: int = 1,
        lr: float = 1e-3,
        max_grad_norm: float = 5.0,
    ) -> None:
        """
        Initializes the NeuralLinearBanditModule.

        Args:
            encoder: The encoder model (neural network) to be used.
            n_features: The number of features in the input data.
            n_embedding_size: The size of the embedding produced by the encoder model. Defaults to n_features.
            encoder_update_freq: The interval (in steps) at which the encoder model is updated. Default is 32. None means the encoder model is never updated.
            encoder_update_batch_size: The batch size for the encoder model update. Default is 32.
            head_update_freq: The interval (in steps) at which the encoder model is updated. Default is 1. None means the linear head is never updated independently.
            lr: The learning rate for the optimizer of the encoder model. Default is 1e-3.
            max_grad_norm: The maximum norm of the gradients for the encoder model. Default is 5.0.
            eta: The hyperparameter for the prior distribution sigma^2 ~ IG(eta, eta). Default is 6.0.
        """
        super().__init__()

        if n_embedding_size is None:
            n_embedding_size = n_features

        assert n_features > 0, "The number of features must be greater than 0."
        assert n_embedding_size > 0, "The embedding size must be greater than 0."
        assert (
            encoder_update_freq is None or encoder_update_freq > 0
        ), "The encoder_update_freq must be greater than 0. Set it to None to never update the neural network."
        assert (
            head_update_freq is None or head_update_freq > 0
        ), "The head_update_freq must be greater than 0. Set it to None to never update the head independently."

        hyperparameters = {
            "n_features": n_features,
            "n_embedding_size": n_embedding_size,
            "encoder_update_freq": encoder_update_freq,
            "encoder_update_batch_size": encoder_update_batch_size,
            "head_update_freq": head_update_freq,
            "lr": lr,
            "max_grad_norm": max_grad_norm,
        }

        self.save_hyperparameters(hyperparameters)

        # The bandit contains all of the code and models for making predictions
        self.bandit = NeuralLinearBandit(
            encoder=encoder,
            n_features=n_features,
            n_embedding_size=n_embedding_size,
        )

        # We use this network to train the encoder model. We mock a linear head with the final layer of the encoder, hence the single output dimension.
        # TODO: it would be cleaner if this was a lightning module?
        self.net = torch.nn.Sequential(
            self.bandit.encoder,
            torch.nn.Linear(self.hparams["n_embedding_size"], 1),
        )

        self.buffer = buffer

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a training step on the neural linear bandit model.
        """
        contextualized_actions: torch.Tensor = batch[0]
        rewards: torch.Tensor = batch[1]

        assert (
            contextualized_actions.shape[2] == self.hparams["n_features"]
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"

        assert (
            rewards.shape[0] == contextualized_actions.shape[0]
            and rewards.shape[1] == contextualized_actions.shape[1]
        ), "Rewards must have shape (batch_size, n_arms) same as contextualized actions."

        # retrieve an action
        embedded_actions: torch.Tensor = self.bandit.encoder(
            contextualized_actions
        )  # shape: (batch_size, n_arms, n_embedding_size)

        chosen_actions_idx = self(contextualized_actions).argmax(dim=1)

        batch_size = contextualized_actions.shape[0]
        chosen_contextualized_actions = contextualized_actions[
            torch.arange(batch_size), chosen_actions_idx
        ]  # shape: (batch_size, n_features)
        chosen_embedded_actions = embedded_actions[
            torch.arange(batch_size), chosen_actions_idx
        ]  # shape: (batch_size, n_embedding_size)
        realized_rewards = rewards[
            torch.arange(batch_size), chosen_actions_idx
        ]  # shape: (batch_size,)

        # Log the reward and regret
        self.log(
            "reward",
            realized_rewards.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        regret = torch.max(rewards, dim=1).values - realized_rewards
        self.log("regret", regret.mean(), on_step=True, on_epoch=False, prog_bar=True)

        # Update the replay buffer
        self.buffer.add_batch(
            contextualized_actions=chosen_contextualized_actions,
            embedded_actions=chosen_embedded_actions,
            rewards=realized_rewards,
        )

        assert (
            embedded_actions.shape[0] == contextualized_actions.shape[0]
            and embedded_actions.shape[1] == contextualized_actions.shape[1]
            and embedded_actions.shape[2] == self.hparams["n_embedding_size"]
        ), "The embedding produced by the encoder must have the specified size (batch_size, n_arms, n_embedding_size)."

        # Update the neural network and the linear head
        should_update_encoder = (
            self.hparams["encoder_update_freq"] is not None
            and len(self.buffer) % self.hparams["encoder_update_freq"] == 0
        )
        if should_update_encoder:
            self._train_nn()
            self._update_embeddings()

        should_update_head = (
            self.hparams["head_update_freq"] is not None
            and len(self.buffer) % self.hparams["head_update_freq"] == 0
        )
        if should_update_head or should_update_encoder:
            self._update_head()

        return -rewards.mean()

    def _train_nn(
        self,
        num_steps: int = 32,
    ) -> None:
        """Perform a full update on the network of the neural linear bandit."""
        # TODO: How can we use a Lightning trainer here? Possibly extract into a separate BanditNeuralNetwork module?

        # We train the encoder so that it produces embeddings that are useful for a linear head.
        # The actual linear head is trained in a seperate step but we "mock" a linear head with the final layer of the encoder.

        # TODO: optimize by not passing Z since X and Y are enough

        self.bandit.encoder.train()
        for _ in range(num_steps):
            x, z, y = self.buffer.get_batch(self.hparams["encoder_update_batch_size"])
            self.optimizers().zero_grad()  # type: ignore

            # x  # shape: (batch_size, n_features)
            # z  # shape: (batch_size, n_embedding_size)
            # y  # shape: (batch_size,)

            y_pred: torch.Tensor = self.net(x)  # shape: (batch_size,)

            loss = self._compute_loss(y_pred, y)

            cost = loss.sum() / self.hparams["encoder_update_batch_size"]
            cost.backward()  # type: ignore

            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.hparams["max_grad_norm"]
            )

            self.optimizers().step()  # type: ignore

            self.log("nn_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        self.lr_schedulers().step()  # type: ignore

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss of the neural linear bandit.

        Args:
            y_pred (torch.Tensor): The predicted rewards. Shape: (batch_size,)
            y (torch.Tensor): The actual rewards. Shape: (batch_size,)

        Returns:
            torch.Tensor: The loss.
        """
        # TODO: Should this be configurable?
        return torch.nn.functional.mse_loss(y_pred, y)

    def _update_embeddings(self) -> None:
        """Update the embeddings of the neural linear bandit"""
        # TODO: possibly do lazy updates of the embeddings as computing all at once is gonna take for ever
        self.bandit.encoder.eval()
        contexts, _, _ = self.buffer.get_batch(len(self.buffer))

        new_embedded_actions = torch.empty(
            len(contexts), self.hparams["n_embedding_size"]
        )

        with torch.no_grad():
            for i, x in enumerate(contexts):
                new_embedded_actions[i] = self.bandit.encoder(x)

        self.buffer.update_embeddings(new_embedded_actions)

    def _update_head(self) -> None:
        """Perform an update step on the head of the neural linear bandit. Currently, it recomputes the linear head from scratch."""
        # TODO: make this sequential! Then we don't need to reset the parameters on every update (+ update the method comment).
        # TODO: But when we recompute after training the encoder, we need to actually reset these parameters. And we need to only load the latest data from the replay buffer.
        # TODO: We could actually make this recompute configurable and not force a recompute but just continue using the old head.

        # Reset the parameters
        self.bandit.precision_matrix = torch.eye(self.hparams["n_embedding_size"])
        self.bandit.b = torch.zeros(self.hparams["n_embedding_size"])
        self.bandit.theta = torch.zeros(self.hparams["n_embedding_size"])

        # Update the linear head
        _, z, y = self.buffer.get_batch(len(self.buffer))

        if z is None:
            raise ValueError("Embedded actions required for updating linear head")

        data_size, n_embedding_size = z.shape

        assert (
            z.dim() == 2 and n_embedding_size == self.hparams["n_embedding_size"]
        ), "Expected embedded_actions (z) to be 2D (batch_size, n_embedding_size)"
        assert (
            y.dim() == 1 and y.shape[0] == data_size
        ), "Expected rewards (y) to be 1D (batch_size,) and of same length as embedded_actions (z)"

        denominator = 1 + ((z @ self.bandit.precision_matrix) * z).sum(dim=1).sum(dim=0)
        assert torch.abs(denominator - 0) > 0, "Denominator must not be zero"

        # Update the precision matrix M using the Sherman-Morrison formula
        self.bandit.precision_matrix = (
            self.bandit.precision_matrix
            - (
                self.bandit.precision_matrix
                @ torch.einsum("bi,bj->bij", z, z).sum(dim=0)
                @ self.bandit.precision_matrix
            )
            / denominator
        )
        self.bandit.precision_matrix = 0.5 * (
            self.bandit.precision_matrix + self.bandit.precision_matrix.T
        )

        # should be symmetric
        assert torch.allclose(
            self.bandit.precision_matrix, self.bandit.precision_matrix.T
        ), "M must be symmetric"

        # Finally, update the rest of the parameters of the linear head
        self.bandit.b += z.T @ y  # shape: (features,)
        self.bandit.theta = self.bandit.precision_matrix @ self.bandit.b

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig:
        # TODO: make it more configurable?
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }
