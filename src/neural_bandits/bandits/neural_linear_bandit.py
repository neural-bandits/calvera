import math
from typing import Any, Generic, Optional, cast

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from neural_bandits.bandits.abstract_bandit import ActionInputType
from neural_bandits.bandits.linear_ts_bandit import LinearTSBandit
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class HelperNetwork(torch.nn.Module):
    """A helper network that is used to train the neural network of the NeuralLinearBandit.
    It adds a linear head to the neural network which mocks the linear head of the NeuralLinearBandit,
    hence the single output dimension of the linear layer.
    This allows for training an embedding which is useful for the linear head of the NeuralLinearBandit.
    """

    def __init__(self, network: torch.nn.Module, output_size: int):
        """
        Args:
            network: The neural network to be used to encode the input data into an embedding.
            output_size: The size of the output of the neural network.
        """
        super().__init__()
        self.network = network
        self.linear_head = torch.nn.Linear(
            output_size, 1
        )  # mock linear head so we can learn an embedding that is useful for the linear head

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        z = self.network.forward(*x)
        return self.linear_head.forward(z)

    def reset_linear_head(self) -> None:
        """Reset the parameters of the linear head."""
        self.linear_head.reset_parameters()


# That we have to inherit from Generic[ActionInputType] again here is a little unfortunate. LinearTSBandit fixes the ActionInputType to torch.Tensor but we want to keep it open here.
# It would be cleaner to implement NeuralLinear by having a variable containing the LinearTSBandit.
class NeuralLinearBandit(LinearTSBandit, Generic[ActionInputType]):
    """
    Lightning Module implementing a Neural Linear bandit.
    The Neural Linear algorithm is described in the paper Riquelme et al., 2018, Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling.
    A Neural Linear bandit model consists of a neural network that produces embeddings of the input data and a linear head that is trained on the embeddings.
    Since updating the neural network which encodes the inputs into embeddings is computationally expensive, the neural network is only updated every `embedding_update_interval` steps.
    On the other hand, the linear head is updated every `head_update_freq` steps which should be much lower.

    ActionInputType:
        The type of the input data to the neural network. Can be a single tensor or a tuple of tensors.
    """

    contextualized_actions: (
        torch.Tensor
    )  # shape: (buffer_size, n_parts, n_network_input_size)
    embedded_actions: torch.Tensor  # shape: (buffer_size, n_network_input_size)
    rewards: torch.Tensor  # shape: (buffer_size,)

    def __init__(
        self,
        network: torch.nn.Module,
        n_embedding_size: int,
        selector: AbstractSelector = ArgMaxSelector(),
        network_update_freq: int = 32,
        network_update_batch_size: int = 32,
        head_update_freq: int = 1,
        lr: float = 1e-3,
        max_grad_norm: float = 5.0,
    ) -> None:
        """
        Initializes the NeuralLinearBanditModule.

        Args:
            network: The neural network to be used to encode the input data into an embedding.
            n_embedding_size: The size of the embedding produced by the neural network.
            selector: The selector used to choose the best action. Default is ArgMaxSelector.
            network_update_freq: The interval (in steps) at which the neural network is updated. Default is 32. None means the neural network is never updated.
            network_update_batch_size: The batch size for the neural network update. Default is 32.
            head_update_freq: The interval (in steps) at which the neural network is updated. Default is 1. None means the linear head is never updated independently.
            lr: The learning rate for the optimizer of the neural network. Default is 1e-3.
            max_grad_norm: The maximum norm of the gradients for the neural network. Default is 5.0.
            eta: The hyperparameter for the prior distribution sigma^2 ~ IG(eta, eta). Default is 6.0.
        """
        super().__init__(n_features=n_embedding_size, selector=selector)

        assert n_embedding_size > 0, "The embedding size must be greater than 0."
        assert (
            network_update_freq is None or network_update_freq > 0
        ), "The network_update_freq must be greater than 0. Set it to None to never update the neural network."
        assert (
            head_update_freq is None or head_update_freq > 0
        ), "The head_update_freq must be greater than 0. Set it to None to never update the head independently."

        self.save_hyperparameters(
            {
                "n_embedding_size": n_embedding_size,  # same as n_features
                "network_update_freq": network_update_freq,
                "network_update_batch_size": network_update_batch_size,
                "head_update_freq": head_update_freq,
                "lr": lr,
                "max_grad_norm": max_grad_norm,
            }
        )

        self.network = network

        # Initialize the linear head which receives the embeddings
        self.precision_matrix = torch.eye(n_embedding_size)
        self.b = torch.zeros(n_embedding_size)
        self.theta = torch.zeros(n_embedding_size)

        # We use this network to train the neural network. We mock a linear head with the final layer of the neural network.
        self.helper_network = HelperNetwork(
            self.network, self.hparams["n_embedding_size"]
        )

        self.contextualized_actions: torch.Tensor = torch.empty(
            0
        )  # shape: (buffer_size, n_parts, n_network_input_size)
        self.embedded_actions: torch.Tensor = torch.empty(
            0
        )  # shape: (buffer_size, n_network_input_size)
        self.rewards: torch.Tensor = torch.empty(0)  # shape: (buffer_size,)

        # Disable Lightnight's automatic optimization. We handle the update in the `training_step` method.
        self.automatic_optimization = False

    # TODO: Here we have a big problem! Inheriting from LinearTSBandit does not work because the type of the input data is fixed to torch.Tensor. We need to keep it open here.
    def _predict_action(
        self, contextualized_actions: ActionInputType, **kwargs: Any  # type: ignore
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the action to take for the given input data according to neural linear.

        Args:
            contextualized_actions: The input data. Shape: (batch_size, n_arms, n_network_input_size)
                or a tuple of tensors of shape (batch_size, n_arms, n_network_input_size) if there are several inputs to the model.

        Returns:
            tuple:
            - chosen_actions: The one-hot encoded tensor of the chosen actions. Shape: (batch_size, n_arms).
            - p: The probability of the chosen actions. For now we always return 1 but we might return the actual probability in the future. Shape: (batch_size, ).
        """

        embedded_actions = self._embed_contextualized_actions(
            contextualized_actions
        )  # shape: (batch_size, n_arms, n_embedding_size)

        # Call the linear bandit to get the best action via Thompson Sampling. Unfortunately, we can't use its forward method here: because of inheriting it would call our forward and _predict_action method again.
        result, p = super()._predict_action(
            embedded_actions
        )  # shape: (batch_size, n_arms)

        assert (
            result.shape[0] == embedded_actions.shape[0]
            and result.shape[1] == embedded_actions.shape[1]
        ), f"Linear head output must have shape (batch_size, n_arms). Expected shape {(embedded_actions.shape[0], embedded_actions.shape[1])} but got shape {result.shape}"

        assert (
            p.ndim == 1
            and p.shape[0] == embedded_actions.shape[0]
            and torch.all(p >= 0)
            and torch.all(p <= 1)
        ), f"The probabilities must be between 0 and 1 and have shape ({embedded_actions.shape[0]}, ) but got shape {p.shape}"

        return result, p

    # TODO: Same problem here as in _predict_action
    def _update(
        self,
        batch: tuple[ActionInputType, torch.Tensor],  # type: ignore
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a training step on the neural linear bandit model.
        """
        chosen_contextualized_actions: ActionInputType = batch[
            0
        ]  # shape: (batch_size, n_chosen_arms, n_features)
        realized_rewards: torch.Tensor = batch[1]

        if isinstance(chosen_contextualized_actions, torch.Tensor):
            assert (
                chosen_contextualized_actions.ndim == 3
            ), f"Chosen actions must have shape (batch_size, n_chosen_arms, n_features) but got shape {chosen_contextualized_actions.shape}"
            batch_size, n_chosen_arms, n_network_input_size = (
                chosen_contextualized_actions.shape
            )
            assert (
                n_chosen_arms == 1
            ), "The neural linear bandit can only choose one action at a time. Combinatorial Neural Linear is not supported."

            self.contextualized_actions = torch.cat(
                [
                    self.contextualized_actions,
                    chosen_contextualized_actions,  # shape: (batch_size, 1, n_features)
                ],
                dim=0,
            )  # shape: (batch_size, 1, n_embedding_size)
        elif isinstance(chosen_contextualized_actions, tuple) or isinstance(
            chosen_contextualized_actions, list
        ):
            assert (
                len(chosen_contextualized_actions) > 1
                and chosen_contextualized_actions[0].ndim == 3
            ), "The tuple of contextualized_actions must contain more than one element and be of of shape (batch_size, n_chosen_arms, n_features)."
            batch_size, n_chosen_arms, n_network_input_size = (
                chosen_contextualized_actions[0].shape
            )
            assert (
                n_chosen_arms == 1
            ), "The neural linear bandit can only choose one action at a time. Combinatorial Neural Linear is not supported."

            # Dimensions of other parts are validated in the _embed_contextualized_actions method. Otherwise, we would need to validate them here.

            # Update the replay buffer
            self.contextualized_actions = torch.cat(
                [
                    self.contextualized_actions,
                    torch.cat(
                        chosen_contextualized_actions,  # each part is of shape (batch_size, 1, n_features) so
                        dim=1,
                    ),
                ],
                dim=0,
            )  # shape: (batch_size, num_parts, n_embedding_size)
        else:
            raise ValueError(
                "The chosen_contextualized_actions must be either a torch.Tensor or a tuple of torch.Tensors."
            )

        assert (
            realized_rewards.shape[0] == batch_size
            and realized_rewards.shape[1] == n_chosen_arms
        ), f"Rewards must have shape (batch_size, n_chosen_arms) same as contextualized actions. Expected shape {(batch_size, n_chosen_arms)} but got shape {realized_rewards.shape}"
        self.rewards = torch.cat(
            [self.rewards, realized_rewards.squeeze(1)], dim=0
        )  # shape: (buffer_size,)

        # Log the reward
        self.log(
            "reward",
            realized_rewards.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        # Update the neural network and the linear head
        num_samples = self.contextualized_actions.size(0)
        should_update_network = (
            self.hparams["network_update_freq"] is not None
            and num_samples % self.hparams["network_update_freq"] == 0
        )
        if should_update_network:
            # To ensure consistancy in the data structures we add a dummy embedded_action of zeros. It will be overwritten in the _update_embeddings method anyways.
            # We do this to avoid unnecessary computations of embeddings.
            self.embedded_actions = torch.cat(
                [
                    self.embedded_actions,
                    torch.zeros(batch_size, self.hparams["n_embedding_size"]),
                ],
                dim=0,
            )  # shape: (buffer_size, n_embedding_size)

            # Now we can update the neural network
            self._train_nn()
            self._update_embeddings()
        else:  # only if we don't update the network, we need to embed and store the embeddings. Otherwise they are invalid anyways because the network was updated.
            # TODO: This is still inefficient because we are already embedding those actions in the forward pass. One could store the embeddings in the forward pass and reuse them here.
            chosen_embedded_actions = self._embed_contextualized_actions(
                chosen_contextualized_actions
            )

            assert (
                chosen_embedded_actions.shape[0] == batch_size
                and chosen_embedded_actions.shape[1] == n_chosen_arms
                and chosen_embedded_actions.shape[2] == self.hparams["n_embedding_size"]
            ), f"The embeddings produced by the neural network must have the specified size (batch_size, n_chosen_arms, n_embedding_size). Expected shape {(batch_size, n_chosen_arms, self.hparams['n_embedding_size'])} but got shape {chosen_embedded_actions.shape}."

            self.embedded_actions = torch.cat(
                [
                    self.embedded_actions,
                    chosen_embedded_actions.view(-1, chosen_embedded_actions.size(-1)),
                ],
                dim=0,
            )  # shape: (buffer_size, n_embedding_size)

        should_update_head = (
            self.hparams["head_update_freq"] is not None
            and num_samples % self.hparams["head_update_freq"] == 0
        )
        if should_update_head or should_update_network:
            self._update_head()

        return -realized_rewards.mean()

    def _embed_contextualized_actions(
        self,
        contextualized_actions: ActionInputType,
    ) -> torch.Tensor:
        """Embed the actions using the neural network.

        Args:
            contextualized_actions: The input data. Shape: (batch_size, n_arms, n_network_input_size)
                or a tuple of tensors of shape (batch_size, n_arms, n_network_input_size) if there are several inputs to the model.
        """
        if isinstance(contextualized_actions, torch.Tensor):
            assert (
                contextualized_actions.ndim == 3
            ), f"Contextualized actions must have shape (batch_size, n_chosen_arms, n_network_input_size) but got shape {contextualized_actions.shape}"

            batch_size, n_arms, n_network_input_size = contextualized_actions.shape

            # We flatten the input to pass a two-dimensional tensor to the network
            flattened_actions = contextualized_actions.view(
                -1, contextualized_actions.size(-1)
            )  # shape: (batch_size * n_arms, n_network_input_size)

            # TODO: One could optimize by splitting this input into several parts of size forward_batch_size (or the given batch_size) and passing them to the network separately
            # TODO: We should probably pass the kwargs here but then we would need to pass them in the update method as well.
            embedded_actions: torch.Tensor = self.network.forward(
                flattened_actions,
            )  # shape: (batch_size * n_arms, n_embedding_size)
        elif isinstance(contextualized_actions, tuple) or isinstance(
            contextualized_actions, list
        ):
            # assert shape of all tensors
            assert (
                len(contextualized_actions) > 1 and contextualized_actions[0].ndim == 3
            ), "The tuple of contextualized_actions must contain more than one element and be of shape (batch_size, n_chosen_arms, n_network_input_size)."

            batch_size, n_arms, n_network_input_size = contextualized_actions[0].shape

            flattened_actions_list: list[torch.Tensor] = []
            for i, input_part in enumerate(contextualized_actions):
                assert (
                    input_part.ndim == 3
                    and input_part.shape[0] == batch_size
                    and input_part.shape[1] == n_arms
                    and input_part.shape[2] == n_network_input_size
                ), f"All parts of the contextualized actions inputs must have shape (batch_size, n_chosen_arms, n_network_input_size). Expected shape {(batch_size, n_arms, n_network_input_size)} but got shape {input_part.shape} for the {i}-th part."
                # We flatten the input because e.g. BERT expects a tensor of shape (batch_size, sequence_length) and not (batch_size, sequence_length, hidden_size)
                flattened_actions_list.append(input_part.view(-1, n_network_input_size))

            # TODO: One could optimize by splitting this input into several parts of size forward_batch_size (or the given batch_size) and passing them to the network separately
            embedded_actions = self.network.forward(
                *flattened_actions_list,
            )  # shape: (batch_size * n_arms, n_embedding_size)
        else:
            raise ValueError(
                "The contextualized_actions must be either a torch.Tensor or a tuple of torch.Tensors."
            )

        assert (
            embedded_actions.ndim == 2
            and embedded_actions.shape[0] == batch_size * n_arms
            and embedded_actions.shape[1] == self.hparams["n_embedding_size"]
        ), f"Embedded actions must have shape (batch_size * n_arms, n_embedding_size). Expected shape {(batch_size * n_arms, self.hparams['n_embedding_size'])} but got shape {embedded_actions.shape}"

        embedded_actions = embedded_actions.view(
            batch_size, n_arms, -1
        )  # shape: (batch_size, n_arms, n_embedding_size)

        return embedded_actions

    def _train_nn(
        self,
    ) -> None:
        """Perform a full update on the network of the neural linear bandit."""
        # TODO: How can we use a Lightning trainer here? Possibly extract into a separate BanditNeuralNetwork module?

        # We train the neural network so that it produces embeddings that are useful for a linear head.
        # The actual linear head is trained in a seperate step but we "mock" a linear head with the final layer of the network.

        # Retrain on the whole buffer
        batch_size: int = self.hparams["network_update_batch_size"]
        num_steps = math.ceil(self.contextualized_actions.size(0) / batch_size)
        # TODO: optimize by not passing Z since X and Y are enough
        X, _, Y = self.get_batches(num_steps, batch_size)

        self.helper_network.reset_linear_head()
        # helper network should always be in training mode
        self.helper_network.train()
        for x, y in zip(X, Y):
            self.optimizers().zero_grad()  # type: ignore

            # x  # shape: (batch_size, n_network_input_size)
            # z  # shape: (batch_size, n_embedding_size)
            # y  # shape: (batch_size,)

            if isinstance(x, torch.Tensor):
                y_pred: torch.Tensor = self.helper_network.forward(
                    x
                )  # shape: (batch_size,)
            else:
                y_pred = self.helper_network.forward(*x)  # shape: (batch_size,)

            loss = self._compute_loss(y_pred, y)

            cost = loss.sum() / batch_size
            cost.backward()  # type: ignore

            torch.nn.utils.clip_grad_norm_(
                self.helper_network.parameters(), self.hparams["max_grad_norm"]
            )

            self.optimizers().step()  # type: ignore

            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        self.lr_schedulers().step()  # type: ignore

    def get_batches(
        self, num_batches: int, batch_size: int
    ) -> tuple[list[ActionInputType], list[torch.Tensor], list[torch.Tensor]]:
        """Get a random batch of data from the replay buffer.

        Args:
            num_batches: The number of batches to return.
            batch_size: The size of each batch.

        Returns: tuple
            - X: The contextualized actions. Shape: (num_batches, batch_size, n_network_input_size)
            - Z: The embedded actions. Shape: (num_batches, batch_size, n_embedding_size)
            - Y: The rewards. Shape: (num_batches, batch_size)
        """
        # TODO: Implement buffer that returns random samples from the n most recent data
        # TODO: possibly faster if those are stored in a tensor and then indexed
        # TODO: possibly use a DataLoader instead of this method?
        X: list[ActionInputType] = []
        Z: list[torch.Tensor] = []
        Y: list[torch.Tensor] = []

        # create num_batch mini batches of size batch_size
        # TODO: make sure that mini batches are not overlapping?
        random_indices = torch.randint(
            0, self.contextualized_actions.shape[0], (num_batches, batch_size)
        )

        for i in range(num_batches):
            batch_indices = random_indices[i]

            contextualized_actions_tensor = self.contextualized_actions[
                batch_indices
            ]  # shape: (batch_size, n_parts, n_network_input_size)
            if contextualized_actions_tensor.size(1) == 1:  # single input
                contextualized_actions = cast(
                    ActionInputType, contextualized_actions_tensor.squeeze(1)
                )  # shape: (batch_size, n_network_input_size)
            else:  # multiple inputs -> input as tuple
                contextualized_actions_tuple = tuple(
                    torch.unbind(contextualized_actions_tensor, dim=1)
                )  # n_parts tuples of tensors of shape (batch_size, n_network_input_size)

                contextualized_actions = cast(
                    ActionInputType, contextualized_actions_tuple
                )

            X.append(contextualized_actions)
            Z.append(self.embedded_actions[batch_indices])
            Y.append(self.rewards[batch_indices])

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
        num_samples, num_parts, _ = self.contextualized_actions.shape
        assert num_samples == self.embedded_actions.size(
            0
        ), "Expected the number of contextualized actions and embedded actions to be the same."

        batch_size = self.hparams["network_update_batch_size"]
        self.network.eval()
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_of_contextualized_actions = self.contextualized_actions[
                    i : i + batch_size
                ]  # shape: (batch_size, n_parts, n_network_input_size)
                if num_parts == 1:
                    self.embedded_actions[i : i + batch_size] = self.network.forward(
                        batch_of_contextualized_actions.squeeze(
                            1
                        )  # shape: (batch_size, n_network_input_size)
                    )  # shape: (batch_size, n_embedding_size)
                else:
                    contextualized_actions = tuple(
                        [
                            batch_of_contextualized_actions[
                                :, j
                            ]  # shape: (batch_size, n_network_input_size)
                            for j in range(num_parts)
                        ]
                    )
                    self.embedded_actions[i : i + batch_size] = self.network.forward(
                        *contextualized_actions
                    )
        self.network.train()

    def _update_head(self) -> None:
        """Perform an update step on the head of the neural linear bandit. Currently, it recomputes the linear head from scratch."""
        # TODO: make this sequential! Then we don't need to reset the parameters on every update (+ update the method comment).
        # TODO: But when we recompute after training the neural network, we need to actually reset these parameters. And we need to only load the latest data from the replay buffer.
        # TODO: We could actually make this recompute configurable and not force a recompute but just continue using the old head.

        # Reset the parameters
        self.precision_matrix = torch.eye(self.hparams["n_embedding_size"])
        self.b = torch.zeros(self.hparams["n_embedding_size"])
        self.theta = torch.zeros(self.hparams["n_embedding_size"])

        # Update the linear head
        z = self.embedded_actions  # shape: (buffer_size, n_embedding_size)
        y = self.rewards  # shape: (buffer_size,)

        data_size, n_embedding_size = z.shape

        assert (
            z.dim() == 2 and n_embedding_size == self.hparams["n_embedding_size"]
        ), "Expected embedded_actions (z) to be 2D (batch_size, n_embedding_size)"
        assert (
            y.dim() == 1 and y.shape[0] == data_size
        ), "Expected rewards (y) to be 1D (batch_size,) and of same length as embedded_actions (z)"

        denominator = 1 + ((z @ self.precision_matrix) * z).sum(dim=1).sum(dim=0)
        assert torch.abs(denominator - 0) > 0, "Denominator must not be zero"

        # Update the precision matrix M using the Sherman-Morrison formula
        self.precision_matrix = (
            self.precision_matrix
            - (
                self.precision_matrix
                @ torch.einsum("bi,bj->bij", z, z).sum(dim=0)
                @ self.precision_matrix
            )
            / denominator
        )
        self.precision_matrix = 0.5 * (self.precision_matrix + self.precision_matrix.T)

        # should be symmetric
        assert torch.allclose(
            self.precision_matrix, self.precision_matrix.T
        ), "M must be symmetric"

        # Finally, update the rest of the parameters of the linear head
        self.b += z.T @ y  # shape: (features,)
        self.theta = self.precision_matrix @ self.b

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig:
        opt = torch.optim.Adam(self.helper_network.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }
