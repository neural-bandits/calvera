from typing import Any, Optional

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from neural_bandits.bandits.linear_ts_bandit import LinearTSBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class NeuralLinearBandit(LinearTSBandit):
    """
    Lightning Module implementing a Neural Linear bandit.
    The Neural Linear algorithm is described in the paper Riquelme et al., 2018, Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling.
    A Neural Linear bandit model consists of a neural network that produces embeddings of the input data and a linear head that is trained on the embeddings.
    Since updating the neural network (encoder) is computationally expensive, the neural network is only updated every `embedding_update_interval` steps.
    On the other hand, the linear head is updated every `head_update_freq` steps which should be much lower.
    """

    contextualized_actions: torch.Tensor
    embedded_actions: torch.Tensor
    rewards: torch.Tensor

    def __init__(
        self,
        encoder: torch.nn.Module,
        n_encoder_input_size: int,
        n_embedding_size: Optional[int],
        selector: AbstractSelector = ArgMaxSelector(),
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
            n_encoder_input_size: The number of features in the input data.
            n_embedding_size: The size of the embedding produced by the encoder model. Defaults to n_encoder_input_size.
            selector: The selector used to choose the best action. Default is ArgMaxSelector.
            encoder_update_freq: The interval (in steps) at which the encoder model is updated. Default is 32. None means the encoder model is never updated.
            encoder_update_batch_size: The batch size for the encoder model update. Default is 32.
            head_update_freq: The interval (in steps) at which the encoder model is updated. Default is 1. None means the linear head is never updated independently.
            lr: The learning rate for the optimizer of the encoder model. Default is 1e-3.
            max_grad_norm: The maximum norm of the gradients for the encoder model. Default is 5.0.
            eta: The hyperparameter for the prior distribution sigma^2 ~ IG(eta, eta). Default is 6.0.
        """
        if n_embedding_size is None:
            n_embedding_size = n_encoder_input_size

        super().__init__(n_features=n_embedding_size, selector=selector)

        assert (
            n_encoder_input_size > 0
        ), "The number of features must be greater than 0."
        assert n_embedding_size > 0, "The embedding size must be greater than 0."
        assert (
            encoder_update_freq is None or encoder_update_freq > 0
        ), "The encoder_update_freq must be greater than 0. Set it to None to never update the neural network."
        assert (
            head_update_freq is None or head_update_freq > 0
        ), "The head_update_freq must be greater than 0. Set it to None to never update the head independently."

        self.hparams.update(
            {
                "n_encoder_input_size": n_encoder_input_size,
                "n_embedding_size": n_embedding_size,  # same as n_features
                "encoder_update_freq": encoder_update_freq,
                "encoder_update_batch_size": encoder_update_batch_size,
                "head_update_freq": head_update_freq,
                "lr": lr,
                "max_grad_norm": max_grad_norm,
            }
        )

        self.encoder = encoder.to(self.device)

        # We use this network to train the encoder model. We mock a linear head with the final layer of the encoder, hence the single output dimension.
        # TODO: it would be cleaner if this was a lightning module?
        self.net = torch.nn.Sequential(
            self.encoder,
            torch.nn.Linear(self.hparams["n_embedding_size"], 1, device=self.device),
        )

        self.register_buffer(
            "contextualized_actions", torch.empty(0, device=self.device)
        )  # shape: (buffer_size, n_encoder_input_size)
        self.register_buffer(
            "embedded_actions", torch.empty(0, device=self.device)
        )  # shape: (buffer_size, n_encoder_input_size)
        self.register_buffer(
            "rewards", torch.empty(0, device=self.device)
        )  # shape: (buffer_size,)

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

    def _predict_action(
        self, contextualized_actions: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the action to take for the given input data according to neural linear.

        Args:
            contextualized_actions: The input data. Shape: (batch_size, n_arms, n_encoder_input_size)

        Returns:
            tuple:
            - chosen_actions: The one-hot encoded tensor of the chosen actions. Shape: (batch_size, n_arms).
            - p: The probability of the chosen actions. For now we always return 1 but we might return the actual probability in the future. Shape: (batch_size, ).
        """

        assert (
            contextualized_actions.ndim == 3
            and contextualized_actions.shape[2] == self.hparams["n_encoder_input_size"]
        ), f"Contextualized actions must have shape (batch_size, n_arms, n_encoder_input_size). Expected shape {(contextualized_actions.shape)} but got shape {contextualized_actions.shape}"

        embedded_actions: torch.Tensor = self.encoder(
            contextualized_actions
        )  # shape: (batch_size, n_arms, n_embedding_size)

        assert (
            embedded_actions.ndim == 3
            and embedded_actions.shape[0] == contextualized_actions.shape[0]
            and embedded_actions.shape[1] == contextualized_actions.shape[1]
            and embedded_actions.shape[2] == self.hparams["n_embedding_size"]
        ), f"Embedded actions must have shape (batch_size, n_arms, n_encoder_input_size). Expected shape {(contextualized_actions.shape[0], contextualized_actions.shape[1], self.hparams['n_embedding_size'])} but got shape {embedded_actions.shape}"

        # Call the linear bandit to get the best action via Thompson Sampling. Unfortunately, we can't use its forward method here: because of inheriting it would call our forward and _predict_action method again.
        result, p = super()._predict_action(
            embedded_actions
        )  # shape: (batch_size, n_arms)

        assert (
            result.shape[0] == contextualized_actions.shape[0]
            and result.shape[1] == contextualized_actions.shape[1]
        ), f"Linear head output must have shape (batch_size, n_arms). Expected shape {(contextualized_actions.shape[0], contextualized_actions.shape[1])} but got shape {result.shape}"

        assert (
            p.ndim == 1
            and p.shape[0] == contextualized_actions.shape[0]
            and torch.all(p >= 0)
            and torch.all(p <= 1)
        ), f"The probabilities must be between 0 and 1 and have shape ({contextualized_actions.shape[0]}, ) but got shape {p.shape}"

        return result, p

    def _update(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a training step on the neural linear bandit model.
        """
        chosen_contextualized_actions: torch.Tensor = batch[0]
        realized_rewards: torch.Tensor = batch[1]

        assert (
            chosen_contextualized_actions.ndim == 3
            and chosen_contextualized_actions.shape[2]
            == self.hparams["n_encoder_input_size"]
        ), "Contextualized actions must have shape (batch_size, n_chosen_arms, n_encoder_input_size)"

        assert (
            realized_rewards.shape[0] == chosen_contextualized_actions.shape[0]
            and realized_rewards.shape[1] == chosen_contextualized_actions.shape[1]
        ), "Rewards must have shape (batch_size, n_chosen_arms) same as contextualized actions."

        assert (
            chosen_contextualized_actions.shape[1] == 1
        ), "The neural linear bandit can only choose one action at a time. Combinatorial Neural Linear is not supported at the moment."

        # retrieve an action
        chosen_embedded_actions: torch.Tensor = self.encoder(
            chosen_contextualized_actions
        )  # shape: (batch_size, n_arms, n_embedding_size)

        # Log the reward
        self.log(
            "reward",
            realized_rewards.sum(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        # Update the replay buffer
        self.contextualized_actions = torch.cat(
            [
                self.contextualized_actions,
                chosen_contextualized_actions.view(
                    -1, chosen_contextualized_actions.size(-1)
                ),
            ],
            dim=0,
        )
        self.embedded_actions = torch.cat(
            [
                self.embedded_actions,
                chosen_embedded_actions.view(-1, chosen_embedded_actions.size(-1)),
            ],
            dim=0,
        )
        self.rewards = torch.cat([self.rewards, realized_rewards.squeeze(1)], dim=0)

        assert (
            chosen_embedded_actions.shape[0] == chosen_contextualized_actions.shape[0]
            and chosen_embedded_actions.shape[1]
            == chosen_contextualized_actions.shape[1]
            and chosen_embedded_actions.shape[2] == self.hparams["n_embedding_size"]
        ), "The embeddings produced by the encoder must have the specified size (batch_size, n_chosen_arms, n_embedding_size)."

        # Update the neural network and the linear head
        should_update_encoder = (
            self.hparams["encoder_update_freq"] is not None
            and self.embedded_actions.shape[0] % self.hparams["encoder_update_freq"]
            == 0
        )
        if should_update_encoder:
            self._train_nn()
            self._update_embeddings()
            print("trained nn")

        should_update_head = (
            self.hparams["head_update_freq"] is not None
            and self.embedded_actions.shape[0] % self.hparams["head_update_freq"] == 0
        )
        if should_update_head or should_update_encoder:
            self._update_head()
            print("updated head")

        return -realized_rewards.sum()

    def _train_nn(
        self,
        num_steps: int = 32,
    ) -> None:
        """Perform a full update on the network of the neural linear bandit."""
        # TODO: How can we use a Lightning trainer here? Possibly extract into a separate BanditNeuralNetwork module?

        # We train the encoder so that it produces embeddings that are useful for a linear head.
        # The actual linear head is trained in a seperate step but we "mock" a linear head with the final layer of the encoder.

        # TODO: optimize by not passing Z since X and Y are enough
        batch_size: int = self.hparams["encoder_update_batch_size"]
        X, Z, Y = self.get_batches(num_steps, batch_size)

        self.encoder.train()
        for x, z, y in zip(X, Z, Y):
            self.optimizers().zero_grad()  # type: ignore

            # x  # shape: (batch_size, n_encoder_input_size)
            # z  # shape: (batch_size, n_embedding_size)
            # y  # shape: (batch_size,)

            y_pred: torch.Tensor = self.net(x)  # shape: (batch_size,)

            loss = self._compute_loss(y_pred, y)

            cost = loss.sum() / batch_size
            cost.backward()  # type: ignore

            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), self.hparams["max_grad_norm"]
            )

            self.optimizers().step()  # type: ignore

            self.log("nn_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        self.lr_schedulers().step()  # type: ignore

    def get_batches(
        self, num_batches: int, batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Get a random batch of data from the replay buffer.

        Args:
            num_batches: The number of batches to return.
            batch_size: The size of each batch.

        Returns: tuple
            - X: The contextualized actions. Shape: (num_batches, batch_size, n_encoder_input_size)
            - Z: The embedded actions. Shape: (num_batches, batch_size, n_embedding_size)
            - Y: The rewards. Shape: (num_batches, batch_size)
        """
        # TODO: Implement buffer that returns random samples from the n most recent data
        # TODO: possibly faster if those are stored in a tensor and then indexed
        # TODO: possibly use a DataLoader instead of this method?
        X = []
        Z = []
        Y = []

        # create num_batch mini batches of size batch_size
        # TODO: make sure that mini batches are not overlapping?
        # TODO: make this reproducible? (seeded generator)
        random_indices = torch.randint(
            0, self.contextualized_actions.shape[0], (num_batches, batch_size)
        )

        for i in range(num_batches):
            idx = random_indices[i]
            X.append(self.contextualized_actions[idx])
            Z.append(self.embedded_actions[idx])
            Y.append(self.rewards[idx])

        return X, Z, Y

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss of the neural linear bandit.

        Args:
            y_pred: The predicted rewards. Shape: (batch_size,)
            y: The actual rewards. Shape: (batch_size,)

        Returns:
            The loss.
        """
        # TODO: Should this be configurable?
        return torch.nn.functional.mse_loss(y_pred, y)

    def _update_embeddings(self) -> None:
        """Update the embeddings of the neural linear bandit"""
        # TODO: possibly do lazy updates of the embeddings as computing all at once is gonna take for ever
        self.encoder.eval()
        with torch.no_grad():
            for i, x in enumerate(self.contextualized_actions):
                # TODO: Do batched inference
                self.embedded_actions[i] = self.encoder(x)
        self.encoder.train()

    def _update_head(self) -> None:
        """Perform an update step on the head of the neural linear bandit. Currently, it recomputes the linear head from scratch."""
        # TODO: make this sequential! Then we don't need to reset the parameters on every update (+ update the method comment).
        # TODO: But when we recompute after training the encoder, we need to actually reset these parameters. And we need to only load the latest data from the replay buffer.
        # TODO: We could actually make this recompute configurable and not force a recompute but just continue using the old head.

        # Reset the parameters
        self.precision_matrix.copy_(
            torch.eye(self.hparams["n_embedding_size"], device=self.device)
        )
        self.b.copy_(torch.zeros(self.hparams["n_embedding_size"], device=self.device))
        self.theta.copy_(
            torch.zeros(self.hparams["n_embedding_size"], device=self.device)
        )

        # Update the linear head
        z = self.embedded_actions  # shape: (buffer_size, n_embedding_size)
        y = self.rewards  # shape: (buffer_size,)

        super().update(z.unsqueeze(1), y.unsqueeze(1))

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
