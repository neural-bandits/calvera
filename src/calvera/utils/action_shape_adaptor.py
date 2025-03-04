import torch
import torch.nn as nn


class ImageActionAdaptor(nn.Module):
    """Adapts a shape of (batch_size, actions, c * h * w) to (batch_size * actions, c, h, w) for a model.

    A bandit always expects an input shape of (batch_size, actions, features). However for image models, the input
    shape is typically (usually batchsize but we can have multiple actions) (batch_size * actions, c, h, w). This
    module adapts the shape of the action to the expected.
    """

    def __init__(self, network: nn.Module, c: int, h: int, w: int) -> None:
        """Initializes the ImageActionAdaptor.

        Args:
            network: The model that expects the adapted action.
            n_arms: The number of arms.
            c: The number of channels.
            h: The height of the image.
            w: The width of the image.
        """
        super().__init__()

        self.network = network
        assert c > 0, "c must be greater than 0"
        assert h > 0, "h must be greater than 0"
        assert w > 0, "w must be greater than 0"

        self.c = c
        self.h = h
        self.w = w

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Adapt the shape of the action to the expected shape of the model.

        Args:
            action: The action to adapt.

        Returns:
            The adapted action.
        """
        batch_size = action.shape[0]
        action = action.view(-1, self.c, self.h, self.w)
        return self.network(action).view(batch_size, -1)  # type: ignore


class TextActionAdaptor(nn.Module):
    """Adapts a shape of (batch_size, actions, seq_length * features) to (batch_size * actions, seq_length, features).

    A bandit always expects an input shape of (batch_size, actions, features). However for text models (from the
    `transformer` library), the input shape is typically (batch_size * actions, features). This module adapts the shape
    of the action to the expected.
    """

    def __init__(self, network: nn.Module, embedding_size: int) -> None:
        """Initializes the TextActionAdaptor.

        Args:
            network: The model that expects the adapted action.
            embedding_size: The size of a single embedded token
        """
        super().__init__()

        self.network = network
        assert embedding_size > 0, "features must be greater than 0"

        self.embedding_size = embedding_size

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Adapt the shape of the action to the expected shape of the model.

        Args:
            action: The action to adapt.

        Returns:
            The adapted action.
        """
        batch_size = action.shape[0]
        action = action.view(batch_size, -1, self.embedding_size)

        return self.network(action).view(batch_size, -1)  # type: ignore
