from abc import ABC, abstractmethod
from collections.abc import Mapping, Sized
from typing import (
    Any,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
)

import torch

from calvera.bandits.action_input_type import ActionInputType

StateDictType = TypeVar("StateDictType", bound=Mapping[str, Any])
BufferDataFormat = tuple[ActionInputType, torch.Tensor] | tuple[ActionInputType, torch.Tensor, torch.Tensor]


# TODO: Fix DocStrings
# TODO: Move some classes to a separate file
class BanditStateDict(TypedDict):
    """Type definition for bandit state dictionary.

    This TypedDict defines the structure and types for the state dictionary used in checkpointing.
    Each key corresponds to a specific piece of state data with its expected type.

    Attributes:
        contextualized_actions: Tensor storing all contextualized actions in buffer.
            Shape: (buffer_size, num_items, n_features).
        embedded_actions: Tensor storing all embedded action representations. Shape: (buffer_size, n_embedding_size).
        rewards: Tensor storing all received rewards. Shape: (buffer_size,).
        buffer_strategy: Strategy object controlling how data is managed
        max_size: Optional maximum size limit of the buffer. None means no size limit.
    """

    contextualized_actions: torch.Tensor
    embedded_actions: torch.Tensor
    rewards: torch.Tensor
    buffer_strategy: "DataBufferStrategy"
    max_size: int | None


class DataBufferStrategy(Protocol):
    """Protocol defining how training data should be managed in the buffer."""

    def get_training_indices(self, total_samples: int) -> torch.Tensor:
        """Get indices of data points to use for training.

        For the `InMemoryDataBuffer` this has to be deterministic.

        Args:
            total_samples: Total number of samples in the buffer

        Returns:
            Tensor of indices to use for training, shape: (n_selected_samples,)
        """
        ...


class AllDataBufferStrategy(DataBufferStrategy):
    """Strategy that uses all available data points in the buffer for training."""

    def get_training_indices(self, total_samples: int) -> torch.Tensor:
        """Returns indices for all samples in the buffer.

        Args:
            total_samples: Total number of samples in the buffer

        Returns:
            Tensor containing indices [0, ..., total_samples-1]
        """
        return torch.arange(total_samples)


class SlidingWindowBufferStrategy(DataBufferStrategy):
    """Strategy that uses only the last n data points from the buffer for training."""

    def __init__(self, window_size: int):
        """Initialize the sliding window strategy.

        Args:
            window_size: Number of most recent samples to use for training
        """
        self.window_size = window_size

    def get_training_indices(self, total_samples: int) -> torch.Tensor:
        """Returns indices for the last window_size samples.

        Args:
            total_samples: Total number of samples in the buffer

        Returns:
            Tensor containing the last window_size indices
        """
        start_idx = max(0, total_samples - self.window_size)
        return torch.arange(start_idx, total_samples)


class AbstractBanditDataBuffer(
    ABC,
    torch.utils.data.Dataset[BufferDataFormat[ActionInputType]],
    Generic[ActionInputType, StateDictType],
    Sized,
):
    """Abstract base class for bandit data buffer management."""

    def __init__(self, buffer_strategy: DataBufferStrategy):
        """Initialize the data buffer.

        Args:
            buffer_strategy: Strategy for managing training data selection
        """
        self.buffer_strategy = buffer_strategy

    @abstractmethod
    def add_batch(
        self,
        contextualized_actions: ActionInputType,
        embedded_actions: torch.Tensor | None,
        rewards: torch.Tensor,
    ) -> None:
        """Add a batch of data points to the buffer.

        Args:
            contextualized_actions: Tensor of contextualized actions of shape: (buffer_size, n_features) or n_items
                tuple of tensors of shape (buffer_size, n_features)
            embedded_actions: Optional tensor of embedded actions. Shape: (buffer_size, n_embedding_size).
            rewards: Tensor of rewards received for each action. Shape: (buffer_size,).
        """
        pass

    @abstractmethod
    def get_all_data(
        self,
    ) -> tuple[ActionInputType, torch.Tensor | None, torch.Tensor]:
        """Get all available data from the buffer.

        Note that data which may have been deleted due to buffer size limits is not included.

        Returns:
            Tuple of (contextualized_actions, embedded_actions, rewards) for all available data in the buffer.
        """
        pass

    @abstractmethod
    def get_batch(
        self,
        batch_size: int,
    ) -> tuple[ActionInputType, torch.Tensor | None, torch.Tensor]:
        """Get batches of training data according to buffer strategy.

        Args:
            batch_size: Size of each batch to return

        Returns:
            Iterator yielding tuples of (contextualized_actions, embedded_actions, rewards)
            batches where total samples across all batches equals num_samples

        Raises:
            ValueError: If requested batch_size is larger than available data
        """
        pass

    @abstractmethod
    def update_embeddings(self, embedded_actions: torch.Tensor) -> None:
        """Update the embedded actions in the buffer.

        Args:
            embedded_actions: New embeddings for all contexts in buffer
        """
        pass

    @abstractmethod
    def state_dict(
        self,
    ) -> StateDictType:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing all necessary state information
        """
        pass

    @abstractmethod
    def load_state_dict(
        self,
        state_dict: StateDictType,
    ) -> None:
        """Load state from checkpoint dictionary.

        Args:
            state_dict: Dictionary containing state information
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the complete buffer."""
        pass


class InMemoryDataBuffer(AbstractBanditDataBuffer[ActionInputType, BanditStateDict]):
    """In-memory implementation of bandit data buffer.

    Known limitations:
    - It can't handle a varying amount of actions over time.
    - It can't handle multiple actions (i.e. combinatorial bandits).
    """

    def __init__(
        self,
        buffer_strategy: DataBufferStrategy,
        max_size: int | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the in-memory buffer.

        Args:
            buffer_strategy: Strategy for managing training data selection
            max_size: Optional maximum number of samples to store
            device: Device to store data on (default: CPU)
        """
        super().__init__(buffer_strategy)

        self.max_size = max_size
        self.device = device if device is not None else torch.device("cpu")

        self.contextualized_actions = torch.empty(0, 0, 0, device=device)  # shape: (n, input_items, n_features)
        self.embedded_actions = torch.empty(0, 0, device=device)  # shape: (n, n_embedding_size)
        self.rewards = torch.empty(0, device=device)  # shape: (n,)

    def add_batch(
        self,
        contextualized_actions: ActionInputType,
        embedded_actions: torch.Tensor | None,
        rewards: torch.Tensor,
    ) -> None:
        """Add each point from the batch to the buffer.

        Args:
            contextualized_actions: Tensor of contextualized actions of shape: (batch_size, n_features) or n_items tuple
                of tensors of shape (batch_size, n_features)
            rewards: Tensor of rewards, shape: (batch_size,)
            embedded_actions: Optional tensor of embedded actions, shape: (batch_size, n_embedding_size)
        """
        assert (
            embedded_actions is None or embedded_actions.shape[0] == rewards.shape[0]
        ), "Number of embeddings must match number of rewards"
        assert rewards.ndim == 1, "Rewards must have shape (batch_size,)"

        if isinstance(contextualized_actions, torch.Tensor):
            assert (
                contextualized_actions.ndim == 2
            ), f"Chosen actions must have shape (batch_size, n_features) but got shape {contextualized_actions.shape}"
            assert (
                contextualized_actions.shape[0] == rewards.shape[0]
            ), "Number of contextualized actions must match number of rewards"

            contextualized_actions_tensor = contextualized_actions.unsqueeze(1)  # shape: (batch_size, 1, n_features)
        elif isinstance(contextualized_actions, tuple | list):
            assert len(contextualized_actions) > 1, "Tuple must contain at least 2 tensors"
            assert contextualized_actions[0].ndim == 2 and contextualized_actions[0].shape[0] == rewards.shape[0], (
                f"Chosen actions must have shape (batch_size, n_features)"
                f"but got shape {contextualized_actions[0].shape}"
            )
            assert all(
                action_item.ndim == 2 and action_item.shape == contextualized_actions[0].shape
                for action_item in contextualized_actions
            ), "All tensors in tuple must have shape (batch_size, n_features)"

            contextualized_actions_tensor = torch.stack(
                contextualized_actions, dim=1
            )  # shape: (batch_size, n_parts, n_features)
        else:
            raise ValueError(
                f"Contextualized actions must be a torch.Tensor or a tuple of torch.Tensors."
                f"Received {type(contextualized_actions)}."
            )

        # Move data to device
        contextualized_actions_tensor = contextualized_actions_tensor.to(self.device)
        if embedded_actions is not None:
            embedded_actions = embedded_actions.to(self.device)
        rewards = rewards.to(self.device)

        # Initialize buffer with proper shapes if empty
        if self.contextualized_actions.shape[2] == 0:
            self.contextualized_actions = torch.empty(
                0,
                contextualized_actions_tensor.shape[1],
                contextualized_actions_tensor.shape[2],
                device=self.device,
            )  # shape: (n, input_items, n_features)
        if embedded_actions is not None and self.embedded_actions.shape[1] == 0:
            self.embedded_actions = torch.empty(
                0, embedded_actions.shape[1], device=self.device
            )  # shape: (n, n_embedding_size)

        assert (
            contextualized_actions_tensor.shape[1:] == self.contextualized_actions.shape[1:]
        ), f"Input shape does not match buffer shape. Expected {self.contextualized_actions.shape[1:]},\
            got {contextualized_actions_tensor.shape[1:]}"

        self.contextualized_actions = torch.cat([self.contextualized_actions, contextualized_actions_tensor], dim=0)
        if embedded_actions is not None:
            assert (
                embedded_actions.shape[1] == self.embedded_actions.shape[1]
            ), f"Embedding size does not match embeddings in buffer. Expected {self.embedded_actions.shape[1]},\
                got {embedded_actions.shape[1]}"

            self.embedded_actions = torch.cat([self.embedded_actions, embedded_actions], dim=0)

        self.rewards = torch.cat([self.rewards, rewards])

        # Handle max size limit by keeping only the most recent data
        if self.max_size and self.contextualized_actions.shape[0] > self.max_size:
            self.contextualized_actions = self.contextualized_actions[-self.max_size :]
            if embedded_actions is not None:
                self.embedded_actions = self.embedded_actions[-self.max_size :]
            self.rewards = self.rewards[-self.max_size :]

    def __getitem__(self, index: int) -> BufferDataFormat[ActionInputType]:
        """Get contextualized actions and rewards for a specific index.

        Implements the torch Dataset protocol.

        Args:
            index: Index of the data point to retrieve

        Returns:
            Tuple of (contextualized_actions, rewards) for the given index.
        """
        available_indices = self._get_available_indices()
        available_index = available_indices[index]
        actions, embeddings, rewards = self._get_data(torch.tensor([available_index]))

        if embeddings is None:
            return actions, rewards
        else:
            return actions, embeddings, rewards

    def get_all_data(
        self,
    ) -> tuple[ActionInputType, torch.Tensor | None, torch.Tensor]:
        """Get all available data from the buffer.

        Note that data which may have been deleted due to buffer size limits is not included.

        Returns:
            Tuple of (contextualized_actions, embedded_actions, rewards) for all available data in the buffer.
        """
        num_items = self.contextualized_actions.shape[0]
        if num_items > 0:
            return self._get_data(torch.arange(num_items, device=self.device))
        else:
            return (
                torch.empty(0, 0, device=self.device),
                None,
                torch.empty(0, device=self.device),
            )  # type: ignore

    def get_batch(
        self,
        batch_size: int,
    ) -> tuple[ActionInputType, torch.Tensor | None, torch.Tensor]:
        """Get a random batch of training data from the buffer. Uses the buffer strategy to select data.

        Args:
            batch_size: Number of samples in each batch

        Returns:
            Iterator of (contextualized_actions, embedded_actions, rewards) batches. Contextualized actions can be a
            single tensor or a tuple of tensors.

        Raises:
            ValueError: If batch_size exceeds available data.
        """
        available_indices = self._get_available_indices()

        if len(available_indices) < batch_size:
            raise ValueError(
                f"Requested batch size {batch_size} is larger than data retrieved by BufferStrategy."
                f"BufferStrategy retrieved {len(available_indices)} data point(s)."
                f"To retrieve all data, use get_all_data()."
            )

        perm = torch.randperm(len(available_indices), device=self.device)
        batch_indices = available_indices[perm[:batch_size]]

        return self._get_data(batch_indices)

    def _get_data(self, indices: torch.Tensor) -> tuple[ActionInputType, torch.Tensor | None, torch.Tensor]:
        """Get data for the given indices.

        Args:
            indices: Indices of data points to retrieve.

        Returns:
            Tuple of (contextualized_actions, embedded_actions, rewards) for the given indices.
        """
        assert indices.ndim == 1, "Indices must be a 1D tensor"
        assert indices.size(0) > 0, "Indices must not be empty"

        contextualized_actions_tensor = self.contextualized_actions[
            indices
        ]  # shape: (batch_size, n_parts, n_network_input_size)
        if contextualized_actions_tensor.size(1) == 1:  # single input
            contextualized_actions_batch = cast(
                ActionInputType, contextualized_actions_tensor.squeeze(1)
            )  # shape: (batch_size, n_network_input_size)
        else:  # multiple inputs -> input as tuple
            contextualized_actions_tuple = tuple(
                torch.unbind(contextualized_actions_tensor, dim=1)
            )  # n_parts tuples of tensors of shape (batch_size, n_network_input_size)

            contextualized_actions_batch = cast(ActionInputType, contextualized_actions_tuple)

        rewards_batch = self.rewards[indices]

        embedded_actions_batch = None
        if self.embedded_actions.numel() > 0:
            embedded_actions_batch = self.embedded_actions[indices]

        return contextualized_actions_batch, embedded_actions_batch, rewards_batch

    def update_embeddings(self, embedded_actions: torch.Tensor) -> None:
        """Update the embedded actions in the buffer.

        Args:
            embedded_actions: New embeddings for all contexts in buffer. Shape: (buffer_size, n_embedding_size)
        """
        assert embedded_actions.shape[0] == self.embedded_actions.shape[0], (
            f"Number of embeddings to update must match buffer size. "
            f"Expected {self.embedded_actions.shape[0]}, got {embedded_actions.shape[0]}"
        )

        if embedded_actions.shape[0] > 0:
            assert embedded_actions.ndim == 2 and embedded_actions.shape[1] == self.embedded_actions.shape[1], (
                f"Embedding size does not match embeddings in buffer. "
                f"Expected {self.embedded_actions.shape[1]}, got {embedded_actions.shape[1]}"
            )

            self.embedded_actions = embedded_actions.to(self.device)

    def __len__(self) -> int:
        """Get number of samples that the buffer strategy considers for training.

        Returns:
            Number of samples for the buffer strategy.
        """
        available_indices = self._get_available_indices()
        return len(available_indices)

    def _get_available_indices(self) -> torch.Tensor:
        return self.buffer_strategy.get_training_indices(len(self.contextualized_actions)).to(self.device)

    def len_of_all_data(self) -> int:
        """Get number of samples in the buffer.

        Returns:
            Number of samples in the buffer.
        """
        return len(self.contextualized_actions)

    def state_dict(
        self,
    ) -> BanditStateDict:
        """Create a state dictionary for checkpointing.

        Returns:
            Dictionary containing all necessary state information.
        """
        return {
            "contextualized_actions": self.contextualized_actions,
            "embedded_actions": self.embedded_actions,
            "rewards": self.rewards,
            "buffer_strategy": self.buffer_strategy,
            "max_size": self.max_size,
        }

    def load_state_dict(
        self,
        state_dict: BanditStateDict,
    ) -> None:
        """Load state from a checkpoint dictionary.

        Args:
            state_dict: Dictionary containing state information.

        Raises:
            ValueError: If the state dictionary is missing required keys.
        """
        self.contextualized_actions = state_dict["contextualized_actions"].to(device=self.device)
        self.embedded_actions = state_dict["embedded_actions"].to(device=self.device)
        self.rewards = state_dict["rewards"].to(device=self.device)
        self.buffer_strategy = state_dict["buffer_strategy"]
        self.max_size = state_dict["max_size"]

    def clear(self) -> None:
        """Clear the complete buffer."""
        self.contextualized_actions = torch.empty(0, 0, 0, device=self.device)  # shape: (n, input_items, n_features)
        self.embedded_actions = torch.empty(0, 0, device=self.device)  # shape: (n, n_embedding_size)
        self.rewards = torch.empty(0, device=self.device)  # shape: (n,)
