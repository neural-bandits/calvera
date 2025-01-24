from typing import Optional

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from neural_bandits.algorithms.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.modules.abstract_bandit_module import AbstractBanditModule

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
        n_embedding_size: Optional[int],
        encoder_update_freq: int = 32,
        encoder_update_batch_size: int = 32,
        head_update_freq: int = 1,
        lr: float = 1e-3,
        max_grad_norm: float = 5.0,
        lambda_prior: float = 0.25,
        eta: float = 6.0,
    ) -> None:
        """
        Initializes the NeuralLinearBanditModule.

        Args:
            encoder (torch.nn.Module): The encoder model (neural network) to be used.
            n_features (int): The number of features in the input data.
            n_embedding_size (Optional[int]): The size of the embedding produced by the encoder model. Defaults to n_features.
            encoder_update_freq (int): The interval (in steps) at which the encoder model is updated. Default is 32. None means the encoder model is never updated.
            head_update_freq (int): The interval (in steps) at which the encoder model is updated. Default is 1. None means the linear head is never updated independently.
            lr (float): The learning rate for the optimizer of the encoder model. Default is 1e-3.
            max_grad_norm (float): The maximum norm of the gradients for the encoder model. Default is 5.0.
            lambda_prior (float): The regularization hyperparameter for the prior distribution of linear head theta | sigma^2 ~ N(0, sigma^2/lambda * I). Must be >= 0. Default is 0.25.
            eta (float): The hyperparameter for the prior distribution sigma^2 ~ IG(eta, eta). Default is 6.0.
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

        assert eta > 1, "eta must be greater than 1"
        assert lambda_prior >= 0, "lambda_prior must be greater than or equal to 0."

        hyperparameters = {
            "n_features": n_features,
            "n_embedding_size": n_embedding_size,
            "encoder_update_freq": encoder_update_freq,
            "encoder_update_batch_size": encoder_update_batch_size,
            "head_update_freq": head_update_freq,
            "lr": lr,
            "max_grad_norm": max_grad_norm,
            "eta": eta,
            "lambda_prior": lambda_prior,
        }

        self.save_hyperparameters(hyperparameters)

        # The bandit contains all of the code and models for making predictions
        self.bandit = NeuralLinearBandit(
            encoder=encoder,
            n_features=n_features,
            n_embedding_size=n_embedding_size,
            eta=eta,
        )

        # We use this network to train the encoder model. We mock a linear head with the final layer of the encoder, hence the single output dimension.
        # TODO: it would be cleaner if this was a lightning module?
        self.net = torch.nn.Sequential(
            self.bandit.encoder,
            torch.nn.Linear(self.hparams["n_embedding_size"], 1),
        )

        # TODO: Unsure if these should be np.arrays or torch.Tensors
        self.contextualized_actions: torch.Tensor = torch.empty(
            0
        )  # shape: (buffer_size, n_features)
        self.embedded_actions: torch.Tensor = torch.empty(
            0
        )  # shape: (buffer_size, n_embedding_size)
        self.rewards: torch.Tensor = torch.empty(0)  # shape: (buffer_size,)

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
        self.contextualized_actions = torch.cat(
            [self.contextualized_actions, chosen_contextualized_actions], dim=0
        )
        self.embedded_actions = torch.cat(
            [self.embedded_actions, chosen_embedded_actions], dim=0
        )
        self.rewards = torch.cat([self.rewards, realized_rewards], dim=0)

        assert (
            embedded_actions.shape[0] == contextualized_actions.shape[0]
            and embedded_actions.shape[1] == contextualized_actions.shape[1]
            and embedded_actions.shape[2] == self.hparams["n_embedding_size"]
        ), "The embedding produced by the encoder must have the specified size (batch_size, n_arms, n_embedding_size)."

        # Update the neural network and the linear head
        if (
            self.hparams["encoder_update_freq"] is not None
            and batch_idx % self.hparams["encoder_update_freq"] == 0
        ):
            self._train_nn()
            self._update_embeddings()

        if (
            self.hparams["head_update_freq"] is not None
            and batch_idx % self.hparams["head_update_freq"] == 0
        ):
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
        batch_size: int = self.hparams["encoder_update_batch_size"]
        X, Z, Y = self.get_batches(num_steps, batch_size)

        self.bandit.encoder.train()
        for x, z, y in zip(X, Z, Y):
            self.optimizers().zero_grad()  # type: ignore

            # x  # shape: (batch_size, n_features)
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
            num_batches (int): The number of batches to return.
            batch_size (int): The size of each batch.

        Returns:
            tuple of lists X, Z, Y:
            - X (list[torch.Tensor]): The contextualized actions. Shape: (num_batches, batch_size, n_features)
            - Z (list[torch.Tensor]): The embedded actions. Shape: (num_batches, batch_size, n_embedding_size)
            - Y (list[torch.Tensor]): The rewards. Shape: (num_batches, batch_size)
        """
        # TODO: Implement buffer that returns random samples from the n most recent data
        # TODO: possibly faster if those are stored in a tensor and then indexed
        # TODO: possibly use a DataLoader instead of this method?
        X = []
        Z = []
        Y = []

        # create num_batch mini batches of size batch_size
        # TODO: make sure that mini batches are not overlapping?
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
        with torch.no_grad():
            for i, x in enumerate(self.contextualized_actions):
                # TODO: Do batched inference
                self.embedded_actions[i] = self.bandit.encoder(x)

    def _update_head(self) -> None:
        """Perform an update step on the head of the neural linear bandit.
        It is implemented as a Bayesian linear regression with a normal-inverse-gamma prior:
            - y ~ theta^T * x + epsilon
            - theta | sigma^2 ~ N(mu, sigma^2/lambda * I)
            - sigma^2 ~ IG(a, b)
        Priors:
            - mu = 0
            - lambda = "lambda_prior" = 0.25
            - a = b = "eta" = 6
        """
        # TODO: algorithm could be improved with sequential formula

        z = self.embedded_actions  # shape: (buffer_size, n_embedding_size)
        y = self.rewards  # shape: (buffer_size,)

        data_size, n_embedding_size = z.shape

        assert (
            z.dim() == 2 and n_embedding_size == self.hparams["n_embedding_size"]
        ), "Expected embedded_actions (z) to be 2D (batch_size, n_embedding_size)"
        assert (
            y.dim() == 1 and y.shape[0] == data_size
        ), "Expected rewards (y) to be 1D (batch_size,) and of same length as embedded_actions (z)"

        s = z.T @ z  # shape (n_embedding_size, n_embedding_size)

        # Some terms are removed as we assume prior mu_0 = 0.
        precision_a = s + self.hparams["lambda_prior"] * torch.eye(
            n_embedding_size
        )  # shape: (n_embedding_size, n_embedding_size)

        assert torch.allclose(
            precision_a, precision_a.T
        ), "Precision matrix must be symmetric"
        # Check positive definiteness by ensuring all real eigenvalues > 0
        p_eigvals, _ = torch.linalg.eig(precision_a)
        assert torch.all(
            p_eigvals.real > 0
        ), "Precision matrix must be positive definite"

        cov_post = torch.inverse(
            precision_a
        )  # shape: (n_embedding_size, n_embedding_size)
        mu_post = cov_post @ (z.T @ y)  # shape: (n_embedding_size,)

        # Inverse Gamma posterior update
        a0 = b0 = self.hparams["eta"]
        a_post = a0 + data_size / 2
        b_upd = (y @ y) - (mu_post @ (precision_a @ mu_post))
        b_post = b0 + b_upd / 2

        assert a_post > 0, "a_post must be positive"
        assert b_post > 0, "b_post must be positive"

        assert mu_post.shape == (n_embedding_size,)
        assert cov_post.shape == (n_embedding_size, n_embedding_size)

        cov_eigvals, _ = torch.linalg.eig(cov_post)
        assert torch.all(
            cov_eigvals.real > 0
        ), "Covariance matrix must be positive definite"

        print(cov_eigvals)
        print(cov_post)

        # Store the new posterior parameters
        self.bandit.mu = mu_post
        self.bandit.cov = cov_post
        self.bandit.a = a_post
        self.bandit.b = b_post

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
