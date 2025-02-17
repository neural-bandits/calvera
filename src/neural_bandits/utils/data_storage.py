from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, TypedDict
import torch


# TODO: Fix DocStrings
# TODO: Move some classes to a separate file
class BanditStateDict(TypedDict):
    """Type definition for bandit state dictionary.

    This TypedDict defines the structure and types for the state dictionary used in checkpointing.
    Each key corresponds to a specific piece of state data with its expected type.

    Attributes:
        contextualized_actions: Tensor storing all contextualized actions in buffer
        embedded_actions: Tensor storing all embedded action representations
        rewards: Tensor storing all received rewards
        buffer_strategy: Strategy object controlling how data is managed
        max_size: Optional maximum size limit of the buffer
    """

    contextualized_actions: torch.Tensor  # shape: (buffer_size, n_features)
    embedded_actions: torch.Tensor  # shape: (buffer_size, n_embedding_size)
    rewards: torch.Tensor  # shape: (buffer_size,)
    buffer_strategy: "DataBufferStrategy"  # Forward reference since class defined later
    max_size: Optional[int]  # None means no size limit


class DataBufferStrategy(Protocol):
    """Protocol defining how training data should be managed in the buffer."""

    def get_training_indices(self, total_samples: int) -> torch.Tensor:
        """Get indices of data points to use for training.

        Args:
            total_samples: Total number of samples in the buffer

        Returns:
            Tensor of indices to use for training, shape: (n_selected_samples,)
        """
        ...


@dataclass
class AllDataBufferStrategy:
    """Strategy that uses all available data points in the buffer for training."""

    def get_training_indices(self, total_samples: int) -> torch.Tensor:
        """Returns indices for all samples in the buffer.

        Args:
            total_samples: Total number of samples in the buffer

        Returns:
            Tensor containing indices [0, ..., total_samples-1]
        """
        return torch.arange(total_samples)


@dataclass
class SlidingWindowBufferStrategy:
    """Strategy that uses only the last n data points from the buffer for training."""

    window_size: int

    def get_training_indices(self, total_samples: int) -> torch.Tensor:
        """Returns indices for the last window_size samples.

        Args:
            total_samples: Total number of samples in the buffer

        Returns:
            Tensor containing the last window_size indices
        """
        start_idx = max(0, total_samples - self.window_size)
        return torch.arange(start_idx, total_samples)


# TODO: Add __init__ method to AbstractBanditDataBuffer
class AbstractBanditDataBuffer(ABC):
    """Abstract base class for bandit data buffer management."""

    @abstractmethod
    def add_batch(
        self,
        contextualized_actions: torch.Tensor,  # shape: (batch_size, n_features)
        embedded_actions: Optional[
            torch.Tensor
        ],  # shape: (batch_size, n_embedding_size)
        rewards: torch.Tensor,  # shape: (batch_size,)
    ) -> None:
        """Add a batch of data points to the buffer.

        Args:
            contextualized_actions: Tensor of contextualized actions
            embedded_actions: Optional tensor of embedded actions
            rewards: Tensor of rewards received for each action
        """
        pass

    @abstractmethod
    def get_batch(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
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
    def __len__(self) -> int:
        """Get total number of data points in buffer."""
        pass

    @abstractmethod
    def state_dict(
        self,
    ) -> BanditStateDict:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing all necessary state information
        """
        pass

    @abstractmethod
    def load_state_dict(
        self,
        state_dict: BanditStateDict,
    ) -> None:
        """Load state from checkpoint dictionary.

        Args:
            state_dict: Dictionary containing state information
        """
        pass


class InMemoryDataBuffer(AbstractBanditDataBuffer):
    """In-memory implementation of bandit data buffer."""

    def __init__(
        self, buffer_strategy: DataBufferStrategy, max_size: Optional[int] = None
    ):
        """Initialize the in-memory buffer.

        Args:
            buffer_strategy: Strategy for managing training data selection
            max_size: Optional maximum number of samples to store
        """
        self.buffer_strategy = buffer_strategy
        self.max_size = max_size

        self.contextualized_actions: torch.Tensor = torch.empty(
            0, 0
        )  # shape: (buffer_size, n_features)
        self.embedded_actions: torch.Tensor = torch.empty(
            0, 0
        )  # shape: (buffer_size, n_embedding_size)
        self.rewards: torch.Tensor = torch.empty(0)  # shape: (buffer_size,)

    def add_batch(
        self,
        contextualized_actions: torch.Tensor,
        embedded_actions: Optional[torch.Tensor],
        rewards: torch.Tensor,
    ) -> None:
        """Add each point from the batch to the buffer.

        Args:
            contextualized_actions: Tensor of contextualized actions, shape: (batch_size, n_features)
            embedded_actions: Optional tensor of embedded actions, shape: (batch_size, n_embedding_size)
            rewards: Tensor of rewards, shape: (batch_size,)
        """
        # TODO: n_feature as an input?
        if self.contextualized_actions.shape[1] == 0:
            self.contextualized_actions = torch.empty(
                0, contextualized_actions.shape[1]
            )
        if embedded_actions is not None and self.embedded_actions.shape[1] == 0:
            self.embedded_actions = torch.empty(0, embedded_actions.shape[1])

        self.contextualized_actions = torch.cat(
            [self.contextualized_actions, contextualized_actions], dim=0
        )
        if embedded_actions is not None:
            self.embedded_actions = torch.cat(
                [self.embedded_actions, embedded_actions], dim=0
            )
        self.rewards = torch.cat([self.rewards, rewards])

        # Handle max size limit by keeping only the most recent data
        if self.max_size and len(self) > self.max_size:
            self.contextualized_actions = self.contextualized_actions[-self.max_size :]
            if embedded_actions is not None:
                self.embedded_actions = self.embedded_actions[-self.max_size :]
            self.rewards = self.rewards[-self.max_size :]

    def get_batch(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Get a random batch of training data from the buffer.

        Args:
            batch_size: Number of samples in each batch

        Returns:
            Iterator of (contextualized_actions, embedded_actions, rewards) batches

        Raises:
            ValueError: If batch_size exceeds available data
        """
        available_indices = self.buffer_strategy.get_training_indices(len(self))

        if len(available_indices) < batch_size:
            raise ValueError(
                f"Requested batch size {batch_size} is larger than available data {len(available_indices)}"
            )

        batch_indices = available_indices[
            torch.randint(0, len(available_indices), (batch_size,))
        ]

        contextualized_actions_batch = self.contextualized_actions[batch_indices]
        rewards_batch = self.rewards[batch_indices]

        embedded_actions_batch = None
        if self.embedded_actions.numel() > 0:
            embedded_actions_batch = self.embedded_actions[batch_indices]

        return contextualized_actions_batch, embedded_actions_batch, rewards_batch

    def update_embeddings(self, embedded_actions: torch.Tensor) -> None:
        """Update the embedded actions in the buffer."""
        assert len(embedded_actions) == len(
            self
        ), "Number of embeddings must match buffer size"
        self.embedded_actions = embedded_actions

    def __len__(self) -> int:
        """Get total number of stored data points in buffer."""
        return len(self.contextualized_actions)

    def state_dict(
        self,
    ) -> BanditStateDict:
        """Create a state dictionary for checkpointing."""
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
        """Load state from a checkpoint dictionary."""
        self.contextualized_actions = state_dict["contextualized_actions"]
        self.embedded_actions = state_dict["embedded_actions"]
        self.rewards = state_dict["rewards"]
        self.buffer_strategy = state_dict["buffer_strategy"]
        self.max_size = state_dict["max_size"]
