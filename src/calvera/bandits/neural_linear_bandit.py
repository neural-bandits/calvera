import logging
from typing import Any, cast

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from calvera.bandits.action_input_type import ActionInputType
from calvera.bandits.linear_ts_bandit import LinearTSBandit
from calvera.utils.data_storage import AbstractBanditDataBuffer, BufferDataFormat, ListDataBuffer
from calvera.utils.multiclass import MultiClassContextualizer
from calvera.utils.selectors import AbstractSelector

logger = logging.getLogger(__name__)


class HelperNetwork(torch.nn.Module):
    """A helper network that is used to train the neural network of the NeuralLinearBandit.

    It adds a linear head to the neural network which mocks the linear head of the NeuralLinearBandit,
    hence the single output dimension of the linear layer.
    This allows for training an embedding which is useful for the linear head of the NeuralLinearBandit.
    """

    def __init__(
        self, network: torch.nn.Module, output_size: int, contextualizer: MultiClassContextualizer | None = None
    ) -> None:
        """Initialize the HelperNetwork.

        Args:
            network: The neural network to be used to encode the input data into an embedding.
            output_size: The size of the output of the neural network.
            contextualizer: If provided disjoint model contextualization will be applied to the embeddings
        """
        super().__init__()
        self.network = network
        self.linear_head = torch.nn.Linear(
            output_size, 1
        )  # mock linear head so we can learn an embedding that is useful for the linear head
        self.contextualizer = contextualizer

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the HelperNetwork.

        Args:
            *x: The input data. Can be a single tensor or a tuple of tensors.
                Must be of shape (batch_size, n_network_input_size).

        Returns:
            The output of the linear head.
        """
        # TODO: Do we pass kwargs to the network? See issue #148.
        z = self.network.forward(*x)
        if self.contextualizer is not None:
            z = self.contextualizer(z)
        return self.linear_head.forward(z)

    def reset_linear_head(self) -> None:
        """Reset the parameters of the linear head."""
        self.linear_head.reset_parameters()


# That we have to inherit from Generic[ActionInputType] again here is a little unfortunate. LinearTSBandit fixes the
# ActionInputType to torch.Tensor but we want to keep it open here.
# It would be cleaner to implement NeuralLinear by having a variable containing the LinearTSBandit.
class NeuralLinearBandit(LinearTSBandit[ActionInputType]):
    """Lightning Module implementing a Neural Linear bandit.

    Based on: Riquelme et al. "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for
    Thompson Sampling" https://arxiv.org/abs/1802.09127

    A Neural Linear bandit model consists of a neural network that produces embeddings of the input data and a linear
    head that is trained on the embeddings. Since updating the neural network which encodes the inputs into embeddings
    is computationally expensive, the neural network is only updated once more than `min_samples_required_for_training`
    samples have been collected. Otherwise, only the linear head is updated.

    ActionInputType:
        The type of the input data to the neural network. Can be a single tensor or a tuple of tensors.
    """

    _should_train_network = False
    _samples_without_training_network = 0

    def __init__(
        self,
        network: torch.nn.Module,
        buffer: AbstractBanditDataBuffer[ActionInputType, Any] | None,
        n_embedding_size: int,
        min_samples_required_for_training: int | None = 1024,
        selector: AbstractSelector | None = None,
        train_batch_size: int = 32,
        lazy_uncertainty_update: bool = False,
        lambda_: float = 1.0,
        eps: float = 1e-2,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 1.0,
        learning_rate_scheduler_step_size: int = 1,
        early_stop_threshold: float | None = 1e-3,
        initial_train_steps: int = 1024,
        contextualization_after_network: bool = False,
        n_arms: int | None = None,
        warm_start: bool = True,
    ) -> None:
        """Initializes the NeuralLinearBanditModule.

        Args:
            network: The neural network to be used to encode the input data into an embedding.
            buffer: The buffer used for storing the data for continuously updating the neural network and
                storing the embeddings for the linear head.
            n_embedding_size: The size of the embedding produced by the neural network. Must be greater than 0.
                IF `contextualization_after_network` is True, `n_embedding_size` is the size of the output of the
                network * n_arms (Using disjoint contextualization).
            selector: The selector used to choose the best action. Default is ArgMaxSelector (if None).
            train_batch_size: The batch size for the neural network update. Must be greater than 0.
            min_samples_required_for_training: The interval (in steps) at which the neural network is updated.
                None means the neural network is never updated. If not None, it must be greater than 0.
                Must Default is 1024.
            lazy_uncertainty_update: If True the precision matrix will not be updated during forward, but during the
                update step.
            lambda_: The regularization parameter for the linear head. Must be greater than 0.
            eps: Small value to ensure invertibility of the precision matrix. Added to the diagonal.
                Must be greater than 0.
            learning_rate: The learning rate for the optimizer of the neural network.
                Passed to `lr` of `torch.optim.Adam`.
                Must be greater than 0.
            weight_decay: The regularization parameter for the neural network.
                Passed to `weight_decay` of `torch.optim.Adam`.
                Must be greater equal 0.
            learning_rate_decay: Multiplicative factor for learning rate decay.
                Passed to `gamma` of `torch.optim.lr_scheduler.StepLR`.
                Default is 1.0 (i.e. no decay). Must be greater than 0.
            learning_rate_scheduler_step_size: The step size for the learning rate decay.
                Passed to `step_size` of `torch.optim.lr_scheduler.StepLR`.
                Must be greater than 0.
                The learning rate scheduler is called every time the neural network is updated.
            early_stop_threshold: Loss threshold for early stopping. None to disable.
                Must be greater equal 0.
            initial_train_steps: Number of initial training steps (in samples).
                Defaults to 1024. Must be greater equal 0.
            contextualization_after_network: If True, the contextualization is applied after the network. Useful for
                situations where you want to use the model for retrieving an embedding then use this single embedding
                for multiple actions.
            n_arms: The number of arms to contextualize after the network. Only needed if
                contextualization_after_network is True. Else the number of arms is determined by the input data.
                Must be greater equal 0.
            warm_start: If `False` the parameters of the network are reset in order to be retrained from scratch using
                `network.reset_parameters()` everytime a retraining of the network occurs. If `True` the network is
                trained from the current state.
        """
        assert n_embedding_size > 0, "The embedding size must be greater than 0."
        assert min_samples_required_for_training is None or min_samples_required_for_training > 0, (
            "The min_samples_required_for_training must be greater than 0."
            "Set it to None to never update the neural network."
        )
        assert lambda_ > 0, "The lambda_ must be greater than 0."
        assert eps > 0, "The eps must be greater than 0."
        assert weight_decay >= 0, "The weight_decay must be greater equal 0."
        assert learning_rate > 0, "The learning rate must be greater than 0."
        assert learning_rate_decay >= 0, "The learning rate decay must be greater equal 0."
        assert learning_rate_scheduler_step_size > 0, "The learning rate decay step size must be greater than 0."
        assert (
            early_stop_threshold is None or early_stop_threshold >= 0
        ), "Early stop threshold must be greater than or equal to 0."
        assert initial_train_steps >= 0, "Initial training steps must be greater than or equal to 0."

        if contextualization_after_network:
            assert n_arms is not None, "`n_arms` need to be provided when performing `contextualization_after_network`."
            n_embedding_size *= n_arms

        super().__init__(
            n_features=n_embedding_size,
            selector=selector,
            buffer=buffer,
            train_batch_size=train_batch_size,
            eps=eps,
            lambda_=lambda_,
            lazy_uncertainty_update=lazy_uncertainty_update,
            clear_buffer_after_train=False,
        )

        self.save_hyperparameters(
            {
                "n_embedding_size": n_embedding_size,
                "min_samples_required_for_training": min_samples_required_for_training,
                "train_batch_size": train_batch_size,
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
                "learning_rate_decay": learning_rate_decay,
                "learning_rate_scheduler_step_size": learning_rate_scheduler_step_size,
                "early_stop_threshold": early_stop_threshold,
                "initial_train_steps": initial_train_steps,
                "contextualization_after_network": contextualization_after_network,
                "n_arms": n_arms,
                "warm_start": warm_start,
            }
        )

        self.network = network.to(self.device)

        self.register_buffer(
            "contextualized_actions", torch.empty(0, device=self.device)
        )  # shape: (buffer_size, n_parts, n_network_input_size)
        self.register_buffer(
            "embedded_actions", torch.empty(0, device=self.device)
        )  # shape: (buffer_size, n_network_input_size)
        self.register_buffer("rewards", torch.empty(0, device=self.device))  # shape: (buffer_size,)

        # Disable Lightning's automatic optimization. Has to be kept in sync with should_train_network.
        self.automatic_optimization = False

        self.contextualizer: MultiClassContextualizer | None = None
        if self.hparams["contextualization_after_network"]:
            assert n_arms is not None, "The number of arms must be provided if contextualization_after_network is True."

            assert n_embedding_size % n_arms == 0, (
                "If `contextualization_after_network` is True, `n_embedding_size` is the size of the output of the "
                "network * n_arms (Using disjoint contextualization)."
                "Therefore, `n_embedding_size` must be divisible by `n_arms`."
            )

            assert isinstance(buffer, ListDataBuffer), (
                "Currently only the `ListDataBuffer` supports" "`contextualization_after_network`."
            )

            self.contextualizer = MultiClassContextualizer(n_arms=n_arms)

        # We use this network to train the encoder model. We mock a linear head with the final layer of the encoder,
        # hence the single output dimension.
        self._helper_network = HelperNetwork(
            self.network,
            self.hparams["n_embedding_size"],
            self.contextualizer,
        ).to(self.device)

        self._helper_network_init = self._helper_network.state_dict().copy() if not self.hparams["warm_start"] else None

    def _predict_action(
        self, contextualized_actions: ActionInputType, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts the action to take for the given input data according to neural linear.

        Args:
            contextualized_actions: The input data. Shape: (batch_size, n_arms, n_network_input_size)
                or a tuple of tensors of shape (batch_size, n_arms, n_network_input_size) if there are several inputs to
                the model.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple:
            - chosen_actions: The one-hot encoded tensor of the chosen actions. Shape: (batch_size, n_arms).
            - p: The probability of the chosen actions. For now we always return 1 but we might return the actual
                probability in the future. Shape: (batch_size, ).
        """
        # MyPy forces us to call it `contextualized_actions` but this would be misleading here.
        input_data = contextualized_actions

        embedded_actions = self._get_contextualized_actions(input_data)  # shape: (batch_size, n_arms, n_embedding_size)

        # Call the linear bandit to get the best action via Thompson Sampling. Unfortunately, we can't use its forward
        # method here: because of inheriting it would call our forward and _predict_action method again.
        result, p = super()._predict_action(cast(ActionInputType, embedded_actions))  # shape: (batch_size, n_arms)

        return result, p

    def _get_contextualized_actions(self, input_data: ActionInputType) -> torch.Tensor:
        """Contextualize the input data.

        Args:
            input_data: A batch of input data. See `ActionInputType`. Shape: (batch_size, n_arms, n_network_input_size)
                or a tuple of tensors of shape (batch_size, n_arms, n_network_input_size) if there are several inputs to
                the model.

        Returns:
            A tensor of shape (batch_size, n_arms, n_embedding_size)
        """
        if self.hparams["contextualization_after_network"]:
            assert self.contextualizer is not None, "Missing contextualizer."
            # The network should output (batchsize * (n_arms = 1), n_)
            sample: torch.Tensor
            if isinstance(input_data, torch.Tensor):
                sample = input_data
            elif isinstance(input_data, tuple | list):
                sample = input_data[0]
            else:
                raise ValueError("The contextualized_actions must be either a torch.Tensor or a tuple of torch.Tensor.")

            assert sample.ndim >= 2, "The input data must have shape (batch_size, n_arms, ...)."

            assert sample.shape[1] == 1, (
                "If `contextualization_after_network` is True, `n_arms` must be 1. " f"But shape is {sample.shape}."
            )

            contextualized_embeddings: torch.Tensor = (
                self.contextualizer(self.network.forward(input_data))
                if isinstance(input_data, torch.Tensor)
                else self.contextualizer(self.network.forward(*input_data))
            )
            # Shape (batch_size, n_arms, model_output_size * n_arms)
            assert (
                contextualized_embeddings.ndim == 3
            ), "The contextualized embeddings must have shape (batch_size, n_arms, model_output_size * n_arms)."

            return contextualized_embeddings
        else:
            return self._embed_contextualized_actions(input_data)

    def _embed_contextualized_actions(
        self,
        contextualized_actions: ActionInputType,
    ) -> torch.Tensor:
        """Embed the actions using the neural network.

        Args:
            contextualized_actions: The input data. Shape: (batch_size, n_arms, n_network_input_size)
                or a tuple of tensors of shape (batch_size, n_arms, n_network_input_size) if there are several inputs to
                the model.

        Returns:
            out: The embedded actions. Shape: (batch_size, n_arms, n_embedding_size)
        """
        if isinstance(contextualized_actions, torch.Tensor):
            assert contextualized_actions.ndim >= 3, (
                f"Contextualized actions must have shape (batch_size, n_chosen_arms, n_network_input_size)"
                f"but got shape {contextualized_actions.shape}"
            )

            batch_size, n_arms = contextualized_actions.shape[:2]

            # # We flatten the input to pass a two-dimensional tensor to the network
            flattened_actions = contextualized_actions.view(
                batch_size * n_arms, *contextualized_actions.shape[2:]
            )  # shape: (batch_size * n_arms, n_network_input_size)

            # TODO: Do we pass kwargs to the network? See issue #148.
            embedded_actions: torch.Tensor = self.network.forward(
                flattened_actions,
            )  # shape: (batch_size * n_arms, n_embedding_size)
        elif isinstance(contextualized_actions, tuple | list):
            # assert shape of all tensors
            assert len(contextualized_actions) > 1 and contextualized_actions[0].ndim >= 3, (
                "The tuple of contextualized_actions must contain more than one element and be of shape"
                "(batch_size, n_chosen_arms, ...). "
                f"Encountered shape {contextualized_actions[0].shape}"
            )

            batch_size, n_arms = contextualized_actions[0].shape[:2]

            flattened_actions_list: list[torch.Tensor] = []
            for i, input_part in enumerate(contextualized_actions):
                assert input_part.ndim >= 3 and input_part.shape[0] == batch_size and input_part.shape[1] == n_arms, (
                    f"All parts of the contextualized actions inputs must have shape"
                    f"(batch_size, n_chosen_arms, n_network_input_size)."
                    f"Expected shape {(batch_size, n_arms, ...)}"
                    f"but got shape {input_part.shape} for the {i}-th part."
                )
                # We flatten the input because e.g. BERT expects a tensor of shape (batch_size, sequence_length)
                # and not (batch_size, sequence_length, hidden_size)
                flattened_actions_list.append(input_part.view(batch_size * n_arms, *input_part.shape[2:]))

            # TODO: Do we pass kwargs to the network? See issue #148.
            embedded_actions = self.network.forward(
                *tuple(flattened_actions_list),
            )  # shape: (batch_size * n_arms, n_embedding_size)
        else:
            raise ValueError("The contextualized_actions must be either a torch.Tensor or a tuple of torch.Tensors.")

        assert (
            embedded_actions.ndim == 2
            and embedded_actions.shape[0] == batch_size * n_arms
            and embedded_actions.shape[1] == self.hparams["n_embedding_size"]
        ), (
            f"Embedded actions must have shape (batch_size * n_arms, n_embedding_size)."
            f"Expected shape {(batch_size * n_arms, self.hparams['n_embedding_size'])}"
            f"but got shape {embedded_actions.shape}"
        )

        embedded_actions = embedded_actions.view(
            batch_size, n_arms, -1
        )  # shape: (batch_size, n_arms, n_embedding_size)

        return embedded_actions

    def record_feedback(
        self,
        contextualized_actions: ActionInputType,
        rewards: torch.Tensor,
        chosen_actions: torch.Tensor | None = None,
    ) -> None:
        """Record a pair of actions and rewards for the bandit.

        Args:
            contextualized_actions: The contextualized actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features).
            rewards: The rewards that were observed for the chosen actions. Size: (batch_size, n_actions).
            chosen_actions: The chosen actions one-hot encoded. Size: (batch_size, n_actions). Should only be provided
                if there is only a single contextualized action (see `contextualization_after_network`).
        """
        # TODO: embedding actions is unnecessary if we will update the network later anyways. See issue #149.
        embedded_actions = self._get_contextualized_actions(
            contextualized_actions
        )  # shape: (batch_size, n_actions, n_embedding_size)

        chosen_actions_idx = torch.argmax(chosen_actions, dim=1) if chosen_actions is not None else None
        assert (
            chosen_actions_idx is None or embedded_actions.shape[1] != 1
        ), "If there is only a single action, the chosen_actions_idx must be provided."

        embedded_actions = (
            embedded_actions[torch.arange(embedded_actions.shape[0]), (chosen_actions_idx)].unsqueeze(1)
            if chosen_actions_idx is not None
            else embedded_actions
        )

        self._add_data_to_buffer(contextualized_actions, rewards, embedded_actions, chosen_actions)

        # _total_samples_count is managed by the AbstractBandit
        self._samples_without_training_network += rewards.shape[0]

        if self.is_initial_training_stage() or (
            self.hparams["min_samples_required_for_training"] is not None
            and self._samples_without_training_network >= self.hparams["min_samples_required_for_training"]
        ):
            self.should_train_network = True
        else:
            self.should_train_network = False

        if (
            cast(int, self.hparams["initial_train_steps"] > 0)
            and self._total_samples_count > self.hparams["initial_train_steps"]
            and self._total_samples_count - rewards.size(0) <= self.hparams["initial_train_steps"]
        ):
            logger.info(
                "\nInitial training stage is over. "
                "The network will now be called only once min_samples_required_for_training samples are recorded."
            )

    @property
    def should_train_network(self) -> bool:
        """Should the network be updated in the next training epoch?

        If called after `record_action_data`, this property will overwrite the behavior of
        the `min_samples_required_for_training` parameter.
        """
        return self._should_train_network

    @should_train_network.setter
    def should_train_network(self, value: bool) -> None:
        """Should the network be updated in the next training epoch?

        If called after `record_action_data`, this property will overwrite the behavior of
        the `min_samples_required_for_training` parameter.

        Also sets lightning `automatic_optimization` to `should_train_network`.
        This is necessary to allow for correct updates of the neural network.
        It also needs to be False when we only train the head.
        """
        self.automatic_optimization = value
        self._should_train_network = value

    def on_train_start(self) -> None:
        """Lightning hook. Log a warning if a custom data loader was passed to `trainer.fit`."""
        super().on_train_start()

        assert self.trainer.train_dataloader is not None, "train_dataloader must be set before training starts."
        if self._custom_data_loader_passed:
            logger.warning(
                "You passed a train_dataloader to trainer.fit(). Data from the data buffer will be ignored. "
                "Only the data passed in the train_data_loader is used for training. The data is still added to "
                "the data buffer for future training runs."
            )

            num_samples = len(self.trainer.train_dataloader.dataset)
            required_samples = self.hparams["min_samples_required_for_training"]
            if (
                required_samples is not None
                and num_samples <= required_samples
                and not self.is_initial_training_stage()
            ):
                logger.warning(
                    f"The train_dataloader passed to trainer.fit() contains {num_samples} "
                    f"which is less than min_samples_required_for_training={required_samples}. "
                    f"Even though the initial training stage is over and not enough data samples were passed, "
                    "the network will still be trained, only on this data (no data from buffer). "
                    "Consider passing more data or decreasing min_samples_required_for_training."
                )

            self.should_train_network = True

        if not self.hparams["warm_start"] and self.should_train_network and self._helper_network_init is not None:
            self._helper_network.load_state_dict(self._helper_network_init)

    def is_initial_training_stage(self) -> bool:
        """Check if the bandit is in the initial training stage.

        Returns:
            True if the total seen samples is smaller or equal to initial_strain_Steps, False otherwise.
        """
        return self._total_samples_count <= cast(int, self.hparams["initial_train_steps"])

    def _update(
        self,
        batch: BufferDataFormat[ActionInputType],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a training step on the neural linear bandit model.

        Args:
            batch: The batch of data to train on. Contains either
                the chosen contextualized actions (shape: batch_size, n_chosen_arms, n_features)
                and rewards (shape: batch_size, n_chosen_arms)
                or the embedded actions (batch_size, n_chosen_arms, n_embedding_size) and rewards
                or all three.
                When the head is updated, we always need all 3
                For Neural Linear n_chosen_arms must always be 1 because combinatorial bandits are
                not supported.
            batch_idx: The index of the batch.
        """
        assert len(batch) == 4, (
            "For head updates, batch must be three tensors: "
            "(contextualized_actions, embedded_actions, rewards, chosen_actions)."
            "Either use the `record_feedback` method and do not pass a `train_dataloader`"
            "to `trainer.fit` or make sure that enough data exists to train the network "
            "or manually set `should_train_network` to `True`."
        )

        realized_rewards: torch.Tensor = batch[2]  # shape: (batch_size, n_chosen_arms)

        assert realized_rewards.shape[1] == 1, (
            "The neural linear bandit can only choose one action at a time."
            "Combinatorial Neural Linear is not supported."
        )

        assert self.automatic_optimization == self.should_train_network, (
            "Automatic optimization needs to be True if the neural network should be trained."
            "Automatic optimization needs to be False if only the head should be trained."
            f"Currently, automatic_optimization={self.automatic_optimization} and"
            f"should_train_network={self.should_train_network}."
        )

        if self.should_train_network:
            chosen_contextualized_actions: ActionInputType = batch[0]  # shape: (batch_size, n_chosen_arms, n_features)
            chosen_actions = batch[3]

            # Asserting shapes of the input data.
            loss = self._train_network(chosen_contextualized_actions, realized_rewards, chosen_actions=chosen_actions)
            self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=False)
            return loss
        else:  # update the head
            assert batch[1] is not None, "The embedded actions must be provided for updating the head."
            embedded_actions: torch.Tensor = batch[1]  # shape: (batch_size, n_chosen_arms, n_embedding_size)

            self._train_head(embedded_actions, realized_rewards)
            # Since we are not training the network, we return a dummy loss.
            return torch.tensor(0.0, device=self.device)

    def _train_network(
        self,
        context: ActionInputType,
        reward: torch.Tensor,
        chosen_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Train the neural network on the given data by computing the loss."""
        assert self.automatic_optimization, "Automatic optimization must be enabled for training the network."

        assert reward.size(1) == 1, (
            "The neural linear bandit can only choose one action at a time."
            "Combinatorial Neural Linear is not supported."
        )
        if isinstance(context, torch.Tensor):
            assert context.size(1) == 1, (
                "The neural linear bandit can only choose one action at a time."
                "Combinatorial Neural Linear is not supported."
            )
            predicted_reward: torch.Tensor = self._helper_network.forward(
                context.squeeze(1).to(self.device)
            )  # shape: (batch_size,)
        else:
            assert all(
                input_part.size(1) == 1 for input_part in context
            ), "The neural linear bandit can only choose one action at a time."
            predicted_reward = self._helper_network.forward(
                *tuple(input_part.squeeze(1).to(self.device) for input_part in context)
            )  # shape: (batch_size,)

        if self.hparams["contextualization_after_network"]:
            assert (
                chosen_actions is not None
            ), "The `chosen_actions` are necessary if `contextualization_after_network` is True"

            chosen_actions_idx = chosen_actions.argmax(dim=1)
            predicted_reward = predicted_reward[torch.arange(predicted_reward.shape[0]), chosen_actions_idx]

        loss = self._compute_loss(predicted_reward, reward)

        # Compute the average loss
        avg_loss = loss.mean()

        if self.hparams["early_stop_threshold"] is not None and avg_loss <= self.hparams["early_stop_threshold"]:
            self.trainer.should_stop = True

        return avg_loss

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
        return torch.nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig:
        """Configure the optimizers and learning rate scheduler for the network training."""
        opt = torch.optim.Adam(
            self._helper_network.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=self.hparams["learning_rate_scheduler_step_size"],
            gamma=self.hparams["learning_rate_decay"],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }

    def _train_head(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """Perform an update step on the head of the neural linear bandit.

        Does not recompute the linear model from scratch and instead updates the existing linear model.

        Args:
            z: The embedded actions. Shape: (batch_size, n_actions, n_embedding_size)
            y: The rewards. Shape: (batch_size, n_actions)
        """
        super()._perform_update(z, y)

    def on_train_end(self) -> None:
        """Lightning hook. Reset the flags after training."""
        super().on_train_end()

        if self.should_train_network:
            self._samples_without_training_network = 0
            self.update_embeddings()
            self.retrain_head()

        if not self._training_skipped:
            self.should_train_network = False

    def update_embeddings(self) -> None:
        """Update all of the embeddings stored in the replay buffer."""
        # TODO: recomputing all embeddings at once takes forever. See issue #149.
        contexts, _, _, chosen_actions = self.buffer.get_all_data()  # shape: (num_samples, n_network_input_size)

        num_samples = contexts.shape[0] if isinstance(contexts, torch.Tensor) else contexts[0].shape[0]
        if num_samples == 0:
            return

        new_embedded_actions = torch.empty(num_samples, self.hparams["n_embedding_size"], device=self.device)

        self.network.eval()

        batch_size = cast(int, self.hparams["train_batch_size"])
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                if isinstance(contexts, torch.Tensor):
                    batch_input = cast(
                        ActionInputType,
                        contexts[i : i + batch_size].unsqueeze(1).to(self.device),
                    )
                elif isinstance(contexts, tuple | list):
                    batch_input = cast(
                        ActionInputType,
                        tuple(input_part[i : i + batch_size].unsqueeze(1).to(self.device) for input_part in contexts),
                    )
                else:
                    raise ValueError(
                        "The contextualized_actions must be either a torch.Tensor or a tuple of torch.Tensors."
                    )

                embedded_actions = self._get_contextualized_actions(
                    batch_input  # shape: (batch_size, 1, n_network_input_size)
                )  # shape: (batch_size, n_embedding_size)

                if not self.hparams["contextualization_after_network"]:
                    embedded_actions = embedded_actions.squeeze(1)
                else:
                    assert chosen_actions is not None, (
                        "Chosen actions are needed when " "`contextualization_after_network is True"
                    )
                    chosen_actions_idx = chosen_actions[i : i + batch_size].argmax(dim=1)
                    embedded_actions = embedded_actions[
                        torch.arange(embedded_actions.shape[0], device=embedded_actions.device), chosen_actions_idx
                    ]

                new_embedded_actions[i : i + batch_size] = embedded_actions

        self.buffer.update_embeddings(new_embedded_actions)

    def retrain_head(self) -> None:
        """Retrain the linear head of the neural linear bandit.

        Recomputes the linear model from scratch.
        """
        # Retrieve training data.
        # We would like to retrain the head on the whole buffer.
        # We have to min with len(self.buffer) because the buffer might have deleted some of the old samples.
        _, z, y, _ = self.buffer.get_all_data()

        if y.shape[0] == 0:
            return

        if z is None:
            raise ValueError("Embedded actions required for updating linear head")

        # We need to unsqueeze the tensors because the buffer might not store the action dimension
        if z.ndim == 2 and y.ndim == 1:
            z = z.unsqueeze(1)
            y = y.unsqueeze(1)

        assert z.ndim == 3 and y.ndim == 2, (
            "The retrieved embedded actions must have shape (num_samples, n_actions, n_embedding_size "
            "and the rewards must have shape (num_samples, n_actions)."
        )
        assert z.shape[0] == y.shape[0], "The number of samples in the embedded actions and rewards must be the same."
        assert z.shape[1] == 1 and y.shape[1] == 1, "The number of actions in the embedded actions must be 1."
        assert (
            z.shape[2] == self.hparams["n_embedding_size"]
        ), f"The number of features in the embedded actions must be {self.hparams['n_embedding_size']}."

        # Reset the parameters
        self._init_linear_params()

        dataset = torch.utils.data.TensorDataset(z, y)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams["train_batch_size"],
            shuffle=False,  # no need to shuffle because the update algorithm is assosicative
        )
        for z_batch, y_batch in train_loader:
            super()._perform_update(
                z_batch.to(self.device),  # shape: (num_samples, 1, n_embedding_size)
                y_batch.to(self.device),  # shape: (num_samples, 1)
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle saving custom NeuralLinearBandit state.

        Args:
            checkpoint: Dictionary to save the state into.
        """
        super().on_save_checkpoint(checkpoint)

        checkpoint["network_state"] = self.network.state_dict()
        checkpoint["helper_network_state"] = self._helper_network.state_dict()
        if self._helper_network_init is not None:
            checkpoint["init_helper_network_state"] = self._helper_network_init

        checkpoint["_should_train_network"] = self._should_train_network
        checkpoint["_samples_without_training_network"] = self._samples_without_training_network

        checkpoint["contextualized_actions"] = self.contextualized_actions
        checkpoint["embedded_actions"] = self.embedded_actions
        checkpoint["rewards"] = self.rewards

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle loading custom NeuralLinearBandit state.

        Args:
            checkpoint: Dictionary containing the state to load.
        """
        super().on_load_checkpoint(checkpoint)

        if "network_state" in checkpoint:
            self.network.load_state_dict(checkpoint["network_state"])

        if "helper_network_state" in checkpoint:
            self._helper_network.load_state_dict(checkpoint["helper_network_state"])

        if "init_helper_network_state" in checkpoint and not self.hparams["warm_start"]:
            self._helper_network_init = checkpoint["init_helper_network_state"]

        if "_should_train_network" in checkpoint:
            self._should_train_network = checkpoint["_should_train_network"]
            self.automatic_optimization = self._should_train_network

        if "_samples_without_training_network" in checkpoint:
            self._samples_without_training_network = checkpoint["_samples_without_training_network"]

        if "contextualized_actions" in checkpoint:
            self.register_buffer("contextualized_actions", checkpoint["contextualized_actions"])

        if "embedded_actions" in checkpoint:
            self.register_buffer("embedded_actions", checkpoint["embedded_actions"])

        if "rewards" in checkpoint:
            self.register_buffer("rewards", checkpoint["rewards"])
