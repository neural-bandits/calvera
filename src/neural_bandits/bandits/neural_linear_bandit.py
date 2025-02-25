import math
from typing import Any, Generic, cast

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from neural_bandits.bandits.abstract_bandit import ActionInputType
from neural_bandits.bandits.linear_ts_bandit import LinearTSBandit
from neural_bandits.utils.data_storage import AbstractBanditDataBuffer
from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector


class HelperNetwork(torch.nn.Module):
    """A helper network that is used to train the neural network of the NeuralLinearBandit.
    It adds a linear head to the neural network which mocks the linear head of the NeuralLinearBandit,
    hence the single output dimension of the linear layer.
    This allows for training an embedding which is useful for the linear head of the NeuralLinearBandit.
    """

    def __init__(self, network: torch.nn.Module, output_size: int) -> None:
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
        buffer: AbstractBanditDataBuffer[ActionInputType, Any],
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

        self.network = network.to(self.device)

        # We use this network to train the encoder model. We mock a linear head with the final layer of the encoder, hence the single output dimension.
        # TODO: it would be cleaner if this was a lightning module?
        self.helper_network = HelperNetwork(
            self.network,
            self.hparams["n_embedding_size"],
        ).to(self.device)

        self.register_buffer(
            "contextualized_actions", torch.empty(0, device=self.device)
        )  # shape: (buffer_size, n_parts, n_network_input_size)
        self.register_buffer(
            "embedded_actions", torch.empty(0, device=self.device)
        )  # shape: (buffer_size, n_network_input_size)
        self.register_buffer(
            "rewards", torch.empty(0, device=self.device)
        )  # shape: (buffer_size,)
        self.buffer = buffer
        self.num_samples = 0

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

        # Asserting shapes of the input data
        if isinstance(chosen_contextualized_actions, torch.Tensor):
            assert (
                chosen_contextualized_actions.ndim == 3
            ), f"Chosen actions must have shape (batch_size, n_chosen_arms, n_features) but got shape {chosen_contextualized_actions.shape}"
            batch_size, n_chosen_arms, n_network_input_size = (
                chosen_contextualized_actions.shape
            )
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

        assert (
            realized_rewards.shape[0] == batch_size
            and realized_rewards.shape[1] == n_chosen_arms
        ), f"Rewards must have shape (batch_size, n_chosen_arms) same as contextualized actions. Expected shape {(batch_size, n_chosen_arms)} but got shape {realized_rewards.shape}"

        # Log the reward
        self.log(
            "reward",
            realized_rewards.sum(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.num_samples += batch_size

        # Decide if we should update the neural network and the linear head
        should_update_network = (
            self.hparams["network_update_freq"] is not None
            and self.num_samples % self.hparams["network_update_freq"] == 0
        )

        self._update_replay_buffer(
            chosen_contextualized_actions,
            realized_rewards,
            should_update_network,
        )

        if should_update_network:
            self._train_nn()
            self._update_embeddings()

        should_update_head = (
            self.hparams["head_update_freq"] is not None
            and self.num_samples % self.hparams["head_update_freq"] == 0
        )
        if should_update_head or should_update_network:
            self._update_head()
            print("updated head")

        return -realized_rewards.sum()

    def _update_replay_buffer(
        self,
        chosen_contextualized_actions: ActionInputType,  # shape: (batch_size, n_chosen_arms, n_network_input_size)
        realized_rewards: torch.Tensor,  # shape: (batch_size, n_chosen_arms)
        should_update_network: bool,
    ) -> None:
        batch_size, n_chosen_arms = realized_rewards.shape

        # Compute (or mock) the embeddings of the chosen actions to update the replay buffer
        if should_update_network:
            # To ensure consistancy in the data structures we add a dummy embedded_action of zeros. It will be overwritten in the _update_embeddings method anyways.
            # We do this to avoid unnecessary computations of embeddings.
            chosen_embedded_actions = torch.zeros(
                batch_size, n_chosen_arms, self.hparams["n_embedding_size"]
            )
        else:
            # only if we don't update the network, we need to embed and store the embeddings. Otherwise they are invalid anyways because the network was updated.

            # TODO: This is still inefficient because we are already embedding those actions in the forward pass. One could store the embeddings in the forward pass and reuse them here.
            chosen_embedded_actions = self._embed_contextualized_actions(
                chosen_contextualized_actions
            )

            assert (
                chosen_embedded_actions.shape[0] == batch_size
                and chosen_embedded_actions.shape[1] == n_chosen_arms
                and chosen_embedded_actions.shape[2] == self.hparams["n_embedding_size"]
            ), f"The embeddings produced by the neural network must have the specified size (batch_size, n_chosen_arms, n_embedding_size). Expected shape {(batch_size, n_chosen_arms, self.hparams['n_embedding_size'])} but got shape {chosen_embedded_actions.shape}."

        # Because we assume that only a single action was chosen we can safely squeeze the tensors
        if isinstance(chosen_contextualized_actions, torch.Tensor):
            chosen_contextualized_actions = cast(
                ActionInputType, chosen_contextualized_actions.squeeze(1)
            )  # shape: (batch_size, n_network_input_size)
        else:
            chosen_contextualized_actions = cast(
                ActionInputType,
                tuple(
                    input_part.squeeze(1)
                    for input_part in chosen_contextualized_actions
                ),
            )  # shape: (batch_size, n_network_input_size)

        # Update the replay buffer
        self.buffer.add_batch(
            contextualized_actions=chosen_contextualized_actions,
            embedded_actions=chosen_embedded_actions.squeeze(1),
            rewards=realized_rewards.squeeze(1),
        )

    def _embed_contextualized_actions(
        self,
        contextualized_actions: ActionInputType,
    ) -> torch.Tensor:
        """Embed the actions using the neural network.

        Args:
            contextualized_actions: The input data. Shape: (batch_size, n_arms, n_network_input_size)
                or a tuple of tensors of shape (batch_size, n_arms, n_network_input_size) if there are several inputs to the model.

        Returns:
            out: The embedded actions. Shape: (batch_size, n_arms, n_embedding_size)
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
        batch_size: int = cast(int, self.hparams["network_update_batch_size"])
        num_steps = math.ceil(self.num_samples / batch_size)

        self.network.train()
        self.helper_network.reset_linear_head()
        for _ in range(num_steps):
            x, _, y = self.buffer.get_batch(batch_size)
            self.optimizers().zero_grad()  # type: ignore

            # x  # shape: (batch_size, n_network_input_size)
            # y  # shape: (batch_size,)

            if isinstance(x, torch.Tensor):
                y_pred: torch.Tensor = self.helper_network.forward(
                    x.to(self.device)
                )  # shape: (batch_size,)
            else:
                y_pred = self.helper_network.forward(
                    *tuple(input_part.to(self.device) for input_part in x)
                )  # shape: (batch_size,)

            y = y.to(self.device)
            loss = self._compute_loss(y_pred, y)

            cost = loss.sum() / batch_size
            cost.backward()  # type: ignore

            torch.nn.utils.clip_grad_norm_(
                self.helper_network.parameters(), self.hparams["max_grad_norm"]
            )

            self.optimizers().step()  # type: ignore

            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        self.lr_schedulers().step()  # type: ignore

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
        contexts, _, _ = self.buffer.get_batch(
            self.num_samples
        )  # shape: (num_samples, n_network_input_size)

        new_embedded_actions = torch.empty(
            self.num_samples, self.hparams["n_embedding_size"], device=self.device
        )

        self.network.eval()

        batch_size = cast(int, self.hparams["network_update_batch_size"])
        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                if isinstance(contexts, torch.Tensor):
                    batch_input = cast(
                        ActionInputType,
                        contexts[i : i + batch_size].unsqueeze(1).to(self.device),
                    )
                elif isinstance(contexts, tuple) or isinstance(contexts, list):
                    batch_input = cast(
                        ActionInputType,
                        tuple(
                            input_part[i : i + batch_size].unsqueeze(1).to(self.device)
                            for input_part in contexts
                        ),
                    )
                else:
                    raise ValueError(
                        "The contextualized_actions must be either a torch.Tensor or a tuple of torch.Tensors."
                    )

                new_embedded_actions[
                    i : i + batch_size
                ] = self._embed_contextualized_actions(
                    batch_input  # shape: (batch_size, 1, n_network_input_size)
                ).squeeze(
                    1
                )  # shape: (batch_size, n_embedding_size)

        self.network.train()

        self.buffer.update_embeddings(new_embedded_actions)

    def _update_head(self) -> None:
        """Perform an update step on the head of the neural linear bandit. Currently, it recomputes the linear head from scratch."""
        # TODO: make this sequential! Then we don't need to reset the parameters on every update (+ update the method comment).
        # TODO: But when we recompute after training the neural network, we need to actually reset these parameters. And we need to only load the latest data from the replay buffer.
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
        _, z, y = self.buffer.get_batch(self.num_samples)
        z = z.to(self.device) if z is not None else None
        y = y.to(self.device)

        if z is None:
            raise ValueError("Embedded actions required for updating linear head")

        super()._perform_update(z.unsqueeze(1), y.unsqueeze(1))

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig:
        opt = torch.optim.Adam(self.helper_network.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }
