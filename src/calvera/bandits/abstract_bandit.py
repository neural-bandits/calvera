import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, cast

import lightning as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader

from calvera.bandits.action_input_type import ActionInputType
from calvera.utils.data_storage import (
    AbstractBanditDataBuffer,
    AllDataBufferStrategy,
    BufferDataFormat,
    InMemoryDataBuffer,
    ListDataBuffer,
)
from calvera.utils.selectors import AbstractSelector, ArgMaxSelector, RandomSelector


def _collate_fn(
    batch: list[tuple[ActionInputType, torch.Tensor | None, torch.Tensor, torch.Tensor | None]],
) -> tuple[ActionInputType, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    batch_contexts: ActionInputType
    if isinstance(batch[0][0], torch.Tensor):
        batch_contexts = cast(ActionInputType, torch.stack([cast(torch.Tensor, sample[0]) for sample in batch], dim=0))
    else:
        batch_contexts = cast(
            ActionInputType, tuple([torch.stack([sample[0][i] for sample in batch]) for i in range(len(batch[0][0]))])
        )

    batch_embeddings = (
        torch.stack(cast(list[torch.Tensor], [sample[1] for sample in batch]), dim=0)
        if batch[0][1] is not None
        else None
    )
    batch_reward = torch.stack([sample[2] for sample in batch], dim=0)
    batch_chosen_actions = (
        torch.stack(cast(list[torch.Tensor], [sample[3] for sample in batch]), dim=0)
        if batch[0][3] is not None
        else None
    )

    return batch_contexts, batch_embeddings, batch_reward, batch_chosen_actions


logger = logging.getLogger(__name__)


class AbstractBandit(ABC, pl.LightningModule, Generic[ActionInputType]):
    """Defines the interface for all Bandit algorithms by implementing pytorch Lightning Module methods."""

    selector: AbstractSelector
    buffer: AbstractBanditDataBuffer[ActionInputType, Any]
    _custom_data_loader_passed = (
        True  # If no train_dataloader is passed on trainer.fit(bandit), then this will be set to False.
    )
    _training_skipped = False  # Was training was skipped before starting because of not enough data?
    _new_samples_count = 0  # tracks the number of new samples added to the buffer in the current epoch.
    _total_samples_count = 0  # tracks the total number of samples seen by the bandit.

    def __init__(
        self,
        n_features: int,
        buffer: AbstractBanditDataBuffer[ActionInputType, Any] | None = None,
        train_batch_size: int = 32,
        selector: AbstractSelector | None = None,
    ):
        """Initializes the Bandit.

        Args:
            n_features: The number of features in the contextualized actions.
            buffer: The buffer used for storing the data for continuously updating the neural network.
            train_batch_size: The mini-batch size used for the train loop (started by `trainer.fit()`).
            selector: The selector used to choose the best action. Default is ArgMaxSelector (if None).
        """
        assert n_features > 0, "The number of features must be greater than 0."
        assert train_batch_size > 0, "The batch_size for training must be greater than 0."

        super().__init__()

        if buffer is None:
            self.buffer = InMemoryDataBuffer(
                buffer_strategy=AllDataBufferStrategy(),
                max_size=None,
                device=self.device,
            )
        else:
            self.buffer = buffer

        self.selector = selector if selector is not None else ArgMaxSelector()

        self.save_hyperparameters(
            {
                "n_features": n_features,
                "train_batch_size": train_batch_size,
            }
        )

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Given the contextualized actions, selects a single best action, or a set of actions in the case of combinatorial
        bandits. This can be computed for many samples in one batch.

        Args:
            contextualized_actions: Tensor of shape (batch_size, n_actions, n_features).
            *args: Additional arguments. Passed to the `_predict_action` method
            **kwargs: Additional keyword arguments. Passed to the `_predict_action` method.

        Returns:
            chosen_actions: One-hot encoding of which actions were chosen.
                Shape: (batch_size, n_actions).
            p: The probability of the chosen actions. In the combinatorial case,
                this will be a super set of actions. Non-probabilistic algorithms should always return 1.
                Shape: (batch_size, ).
        """
        contextualized_actions = kwargs.get(
            "contextualized_actions", args[0]
        )  # shape: (batch_size, n_actions, n_features)
        assert contextualized_actions is not None, "contextualized_actions must be passed."

        if isinstance(contextualized_actions, torch.Tensor):
            assert contextualized_actions.ndim >= 3, (
                "Chosen actions must have shape (batch_size, num_actions, ...) "
                f"but got shape {contextualized_actions.shape}"
            )
            batch_size = contextualized_actions.shape[0]
        elif isinstance(contextualized_actions, tuple | list):
            assert len(contextualized_actions) > 1, "Tuple must contain at least 2 tensors"
            assert contextualized_actions[0].ndim >= 3, (
                "Chosen actions must have shape (batch_size, num_actions, ...) "
                f"but got shape {contextualized_actions[0].shape}"
            )
            batch_size = contextualized_actions[0].shape[0]
            assert all(
                action_item.ndim >= 3 for action_item in contextualized_actions
            ), "All tensors in tuple must have shape (batch_size, num_actions, ...)"
        else:
            raise ValueError(
                f"Contextualized actions must be a torch.Tensor or a tuple of torch.Tensors."
                f"Received {type(contextualized_actions)}."
            )

        result, p = self._predict_action(*args, **kwargs)

        # assert result.shape[0] == batch_size (
        #     f"Batch size mismatch. Expected shape {batch_size} but got {result.shape[0]}"
        # )

        assert (
            p.ndim == 1 and p.shape[0] == batch_size and torch.all(p >= 0) and torch.all(p <= 1)
        ), f"The probabilities must be between 0 and 1 and have shape {batch_size} but got shape {p.shape}"

        return result, p

    @abstractmethod
    def _predict_action(
        self,
        contextualized_actions: ActionInputType,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, computed batch-wise.

        Given the contextualized actions, selects a single best action, or a set of actions in the case of combinatorial
        bandits. Next to the action(s), the selector also returns the probability of chosing this action. This will
        allow for logging and Batch Learning from Logged Bandit Feedback (BLBF). Deterministic algorithms like UCB will
        always return 1.

        Args:
            contextualized_actions: Input into bandit or network containing all actions. Either Tensor of shape
                (batch_size, n_actions, n_features) or a tuple of tensors of shape (batch_size, n_actions, n_features)
                if there are several inputs to the model.
            **kwargs: Additional keyword arguments.

        Returns:
            chosen_actions: One-hot encoding of which actions were chosen.
                Shape: (batch_size, n_actions).
            p: The probability of the chosen actions. In the combinatorial case,
                this will be one probability for the super set of actions. Deterministic algorithms (like UCB) should
                always return 1. Shape: (batch_size, ).
        """
        pass

    def record_feedback(
        self,
        contextualized_actions: ActionInputType,
        rewards: torch.Tensor,
        chosen_actions: torch.Tensor | None = None,
    ) -> None:
        """Records a pair of chosen actions and rewards in the buffer.

        Args:
            contextualized_actions: The contextualized actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features).
            rewards: The rewards that were observed for the chosen actions.
                Size: (batch_size, n_actions).
            chosen_actions: The chosen actions one-hot encoded. Size: (batch_size, n_actions). Should only be provided
                if there is only a single contextualized action
                (see NeuralLinearBandit -> `contextualization_after_network`).
        """
        self._add_data_to_buffer(
            contextualized_actions=contextualized_actions, realized_rewards=rewards, chosen_actions=chosen_actions
        )

    def _add_data_to_buffer(
        self,
        contextualized_actions: ActionInputType,
        realized_rewards: torch.Tensor,
        embedded_actions: torch.Tensor | None = None,
        chosen_actions: torch.Tensor | None = None,
    ) -> None:
        """Records a pair of chosen actions and rewards in the buffer.

        Args:
            contextualized_actions: The contextualized actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features).
            realized_rewards: The rewards that were observed for the chosen actions. Size: (batch_size, n_actions).
            embedded_actions: The embedded actions that were chosen by the bandit.
                Size: (batch_size, n_actions, n_features). Optional because not every model uses embedded actions.
            chosen_actions: The chosen actions one-hot encoded. Size: (batch_size, n_actions). Should only be provided
                if there is only a single contextualized action
                (see NeuralLinearBandit -> `contextualization_after_network`).
        """
        assert realized_rewards.ndim == 2, "Realized rewards must have shape (batch_size, n_chosen_actions)."

        batch_size = realized_rewards.shape[0]

        assert realized_rewards.shape[0] == batch_size and (
            embedded_actions is None or embedded_actions.shape[0] == batch_size
        ), "The batch sizes of the input tensors must match."

        assert chosen_actions is None or (
            len(chosen_actions.shape) == 2 and chosen_actions.shape[0] == batch_size
        ), "`chosen_actions` should have shape (batch_size, n_actions) if provided."

        if embedded_actions is not None:
            assert embedded_actions.shape[1] == realized_rewards.shape[1], (
                "The number of chosen_actions in the embedded actions must match the number of chosen_actions in the "
                f"rewards. Encountered shapes: {embedded_actions.shape} and {realized_rewards.shape}"
            )
            embedded_actions_reshaped = embedded_actions.reshape(-1, *embedded_actions.shape[2:])
        else:
            embedded_actions_reshaped = None

        if isinstance(contextualized_actions, torch.Tensor):
            assert contextualized_actions.ndim >= 3, (
                "Chosen actions must have shape (batch_size, num_actions, ...) "
                f"but got shape {contextualized_actions.shape}"
            )
            assert (
                contextualized_actions.shape[0] == realized_rewards.shape[0]
                and contextualized_actions.shape[1] == realized_rewards.shape[1]
            ), "Batch size and number of actions must match number of rewards"
            # For now the data buffer only supports non-combinatorial bandits. so we have to reshape.
            contextualized_actions_reshaped = cast(
                ActionInputType,
                contextualized_actions.reshape(-1, *contextualized_actions.shape[2:]),  # remove the action dimension
            )
        elif isinstance(contextualized_actions, tuple | list):
            assert len(contextualized_actions) > 1, "Tuple must contain at least 2 tensors"
            assert (
                contextualized_actions[0].ndim >= 3 and contextualized_actions[0].shape[0] == realized_rewards.shape[0]
            ), (
                "Chosen actions must have shape (batch_size, num_actions, ...) "
                f"but got shape {contextualized_actions[0].shape}"
            )
            assert all(
                action_item.ndim >= 3 for action_item in contextualized_actions
            ), "All tensors in tuple must have shape (batch_size, num_actions, ...)"

            # For now the data buffer only supports non-combinatorial bandits. so we have to reshape.
            contextualized_actions_reshaped = cast(
                ActionInputType,
                tuple(action_item.reshape(-1, *action_item.shape[2:]) for action_item in contextualized_actions),
            )
        else:
            raise ValueError(
                f"Contextualized actions must be a torch.Tensor or a tuple of torch.Tensors. "
                f"Received {type(contextualized_actions)}."
            )

        realized_rewards_reshaped = realized_rewards.reshape(-1)

        self.buffer.add_batch(
            contextualized_actions=contextualized_actions_reshaped,
            embedded_actions=embedded_actions_reshaped,
            rewards=realized_rewards_reshaped,
            chosen_actions=chosen_actions,
        )

        self._new_samples_count += batch_size
        self._total_samples_count += batch_size

    def train_dataloader(
        self, custom_collate_fn: Callable[[Any], BufferDataFormat[ActionInputType]] | None = None
    ) -> DataLoader[BufferDataFormat[ActionInputType]]:
        """Dataloader used by PyTorch Lightning if none is passed via `trainer.fit(..., dataloader)`."""
        if custom_collate_fn is None:
            custom_collate_fn = _collate_fn

        if len(self.buffer) > 0:
            self._custom_data_loader_passed = False
            return DataLoader(
                self.buffer,
                self.hparams["train_batch_size"],
                shuffle=True,
                collate_fn=custom_collate_fn,
            )
        else:
            raise ValueError("The buffer is empty. Please add data to the buffer before calling trainer.fit().")

    def on_train_start(self) -> None:
        """Hook called by PyTorch Lightning.

        Prints a warning if the trainer is set to run for more than one epoch.
        """
        super().on_train_start()
        if self.trainer.max_epochs is None or self.trainer.max_epochs > 1:
            logger.warning(
                "The trainer will run for more than one epoch. This is not recommended for bandit algorithms."
            )

    def _skip_training(self) -> None:
        """Skip training if there is not enough data."""
        self._training_skipped = True
        self.trainer.should_stop = True

    def training_step(self, batch: BufferDataFormat[ActionInputType], batch_idx: int) -> torch.Tensor:
        """Perform a single update step.

        See the documentation for the LightningModule's `training_step` method.
        Acts as a wrapper for the `_update` method in case we want to change something for every bandit or use the
        update independently from lightning, e.g. in tests.

        Args:
            batch: The output of your data iterable, usually a DataLoader. It may contain 2 or 3 elements:
                contextualized_actions: shape (batch_size, n_chosen_actions, n_features).
                [Optional: embedded_actions: shape (batch_size, n_chosen_actions, n_features).]
                realized_rewards: shape (batch_size, n_chosen_actions).
                The embedded_actions are only passed and required for certain bandits like the NeuralLinearBandit.
            batch_idx: The index of this batch. Note that if a separate DataLoader is used for each step,
                this will be reset for each new data loader.
            data_loader_idx: The index of the data loader. This is useful if you have multiple data loaders
                at once and want to do something different for each one.
            *args: Additional arguments. Passed to the `_update` method.
            **kwargs: Additional keyword arguments. Passed to the `_update` method.

        Returns:
            The loss value. In most cases, it makes sense to return the negative reward.
                Shape: (1,). Since we do not use the lightning optimizer, this value is only relevant
                for logging/visualization of the training process.
        """
        assert len(batch) == 4, (
            "Batch must contain four tensors: (contextualized_actions, embedded_actions, rewards, chosen_actions)."
            "`embedded_actions` and `chosen_actions` can be None."
        )

        realized_rewards: torch.Tensor = batch[2]  # shape: (batch_size, n_chosen_arms)

        assert realized_rewards.ndim == 2, "Rewards must have shape (batch_size, n_chosen_arms)"
        assert realized_rewards.device == self.device, "Realized reward must be on the same device as the model."

        batch_size, n_chosen_arms = realized_rewards.shape

        (
            contextualized_actions,
            embedded_actions,
        ) = batch[:2]

        if self._custom_data_loader_passed:
            self.record_feedback(contextualized_actions, realized_rewards)

        if isinstance(contextualized_actions, torch.Tensor):
            assert (
                contextualized_actions.device == self.device
            ), "Contextualized actions must be on the same device as the model."

            assert contextualized_actions.ndim >= 3, (
                f"Chosen actions must have shape (batch_size, n_chosen_arms, ...) "
                f"but got shape {contextualized_actions.shape}"
            )
            assert contextualized_actions.shape[0] == batch_size and contextualized_actions.shape[1] == n_chosen_arms, (
                "Chosen contextualized actions must have shape (batch_size, n_chosen_arms, ...) "
                f"same as reward. Expected shape ({(batch_size, n_chosen_arms)}, ...) "
                f"but got shape {contextualized_actions.shape}"
            )
        elif isinstance(contextualized_actions, tuple | list):
            assert all(
                action.device == self.device for action in contextualized_actions
            ), "Contextualized actions must be on the same device as the model."

            assert len(contextualized_actions) > 1 and contextualized_actions[0].ndim >= 3, (
                "The tuple of contextualized_actions must contain more than one element and be of shape "
                "(batch_size, n_chosen_arms, ...)."
            )
            assert (
                contextualized_actions[0].shape[0] == batch_size and contextualized_actions[0].shape[1] == n_chosen_arms
            ), (
                "Chosen contextualized actions must have shape (batch_size, n_chosen_arms, ...) "
                f"same as reward. Expected shape ({(batch_size, n_chosen_arms)}, ...) "
                f"but got shape {contextualized_actions[0].shape}"
            )
        else:
            raise ValueError(
                f"Contextualized actions must be a torch.Tensor or a tuple of torch.Tensors. "
                f"Received {type(contextualized_actions)}."
            )

        if embedded_actions is not None:
            assert embedded_actions.device == self.device, "Embedded actions must be on the same device as the model."
            assert (
                embedded_actions.ndim == 3
            ), "Embedded actions must have shape (batch_size, n_chosen_arms, n_features)"
            assert embedded_actions.shape[0] == batch_size and embedded_actions.shape[1] == n_chosen_arms, (
                "Chosen embedded actions must have shape (batch_size, n_chosen_arms, n_features) "
                f"same as reward. Expected shape ({(batch_size, n_chosen_arms)}, n_features) "
                f"but got shape {embedded_actions[0].shape}"
            )

        loss = self._update(
            batch,
            batch_idx,
        )

        assert loss.ndim == 0, "Loss must be a scalar value."

        return loss

    @abstractmethod
    def _update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Abstract method to perform a single update step. Should be implemented by the concrete bandit classes.

        Args:
            batch: The output of your data iterable, usually a DataLoader. It contains 4 elements:
                contextualized_actions: shape (batch_size, n_chosen_actions, n_features).
                [Optional: embedded_actions: shape (batch_size, n_chosen_actions, n_features).]
                embedded_actions: only passed and required for certain bandits like the NeuralLinearBandit.
                realized_rewards: shape (batch_size, n_chosen_actions).
                chosen_actions: only passed and required for certain bandits like the NeuralLinearBandit.
            batch_idx: The index of this batch. Note that if a separate DataLoader is used for each step,
                this will be reset for each new data loader.
            data_loader_idx: The index of the data loader. This is useful if you have multiple data loaders
                at once and want to do something different for each one.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The loss value. In most cases, it makes sense to return the negative reward.
                Shape: (1,). If we do not use the lightning optimizer, this value is only relevant
                for logging/visualization of the training process.
        """
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizers and learning rate schedulers.

        This method is required by LightningModule.Can be overwritten by the concrete bandit classes.
        """
        return None

    def on_train_end(self) -> None:
        """Hook called by PyTorch Lightning."""
        super().on_train_end()
        if not self._training_skipped:
            self._new_samples_count = 0

        self._custom_data_loader_passed = True
        self._training_skipped = False

    def on_validation_start(self) -> None:
        """Hook called by PyTorch Lightning."""
        raise ValueError("Validating the bandit via the lightning Trainer is not supported.")

    def on_test_start(self) -> None:
        """Hook called by PyTorch Lightning."""
        raise ValueError("Testing the bandit via the lightning Trainer is not supported.")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle saving AbstractBandit state.

        Args:
            checkpoint: Dictionary to save the state into.
        """
        if not isinstance(self.buffer, ListDataBuffer):
            # Currently, the ListDataBuffer is not supported for saving the state.
            # Therefore after saving and loading the bandit, the buffer will be empty.
            checkpoint["buffer_state"] = self.buffer.state_dict()

        checkpoint["_new_samples_count"] = self._new_samples_count
        checkpoint["_total_samples_count"] = self._total_samples_count

        checkpoint["_custom_data_loader_passed"] = self._custom_data_loader_passed
        checkpoint["_training_skipped"] = self._training_skipped

        checkpoint["selector_state"] = self.selector.get_state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Handle loading AbstractBandit state.

        Args:
            checkpoint: Dictionary containing the state to load.
        """
        if "buffer_state" in checkpoint:
            self.buffer.load_state_dict(checkpoint["buffer_state"])

        if "_new_samples_count" in checkpoint:
            self._new_samples_count = checkpoint["_new_samples_count"]

        if "_total_samples_count" in checkpoint:
            self._total_samples_count = checkpoint["_total_samples_count"]

        if "_custom_data_loader_passed" in checkpoint:
            self._custom_data_loader_passed = checkpoint["_custom_data_loader_passed"]

        if "_training_skipped" in checkpoint:
            self._training_skipped = checkpoint["_training_skipped"]

        if "selector_state" in checkpoint:
            self.selector = AbstractSelector.from_state_dict(checkpoint["selector_state"])


class DummyBandit(AbstractBandit[ActionInputType]):
    """A dummy bandit that always selects random actions."""

    def __init__(self, n_features: int, k: int = 1) -> None:
        """Initializes a DummyBandit with a RandomSelector.

        Args:
            n_features: The number of features in the bandit model. Must be positive.
            k: Number of actions to select. Must be positive. Default is 1.
        """
        super().__init__(
            selector=RandomSelector(k=k),
            n_features=n_features,
        )
        self.automatic_optimization = False
        # Please don't ask. Lightning requires any parameter to be registered in order to train it on cuda.
        self.register_parameter("_", None)

    def _predict_action(
        self,
        contextualized_actions: ActionInputType,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, computed batch-wise. Does nothing but call the selector."""
        context_tensor = (
            contextualized_actions if isinstance(contextualized_actions, torch.Tensor) else contextualized_actions[0]
        )
        batch_size = context_tensor.shape[0]
        n_arms = context_tensor.shape[1]

        selected_actions_one_hot = self.selector(torch.ones((batch_size, n_arms), device=context_tensor.device))

        if isinstance(self.selector, RandomSelector):
            p = torch.ones((batch_size,), device=context_tensor.device) / n_arms
        else:
            selected_actions_indices = selected_actions_one_hot.argmax(dim=1)
            p = torch.zeros((batch_size,), device=context_tensor.device)
            p[selected_actions_indices] = 1.0

        return selected_actions_one_hot, p

    def _update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Dummy implementation of the update method."""
        return torch.tensor(0.0)
