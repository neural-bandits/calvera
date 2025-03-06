import math
from typing import Any, cast

import torch
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertWrapper(nn.Module):
    """Wrapper for BERT-like models from the `transformers` library."""

    def __init__(self, network: nn.Module, *args: Any, **kwargs: Any) -> None:
        """Initializes the BertWrapper."""
        super().__init__(*args, **kwargs)

        self.network = network

    def forward(self, *x: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Forward pass through the network."""
        assert isinstance(x, list | tuple), "Input must be a list or tuple"
        assert isinstance(x[0], torch.Tensor), "Input must be a list or tuple of tensor"

        input = [x.squeeze(1) for x in x]
        input[0] = input[0].long()
        input[1] = input[1].float()
        input[2] = input[2].long()
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.network(*input)

        return output.last_hidden_state[:, 0, :]


class ResNetWrapper(nn.Module):
    """Wrapper for ResNet-like models from the `timm` library."""

    def __init__(self, network: nn.Module, *args: Any, **kwargs: Any) -> None:
        """Initializes the ResNetWrapper."""
        super().__init__(*args, **kwargs)

        self.network = network.eval()

        # See timm: https://huggingface.co/timm/resnet18.a1_in1k
        # We excluded the normalization layer, as it is already performed in the dataset.
        self.transforms = Compose(
            [
                Resize(size=235, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
                CenterCrop(size=(224, 224)),
            ]
        )

        self.dim_reduction = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        assert isinstance(x, torch.Tensor), "Input must be a tensor"

        w = int(math.sqrt(x.shape[-1] / 3))
        x = x.reshape(-1, 3, w, w)

        x = self.transforms(x)

        output = self.network.forward_features(x)  # type: ignore
        # output is unpooled, a (1, 512, 7, 7) shaped tensor

        output = self.network.forward_head(output, pre_logits=True)  # type: ignore
        output = self.dim_reduction(output)

        return cast(torch.Tensor, output)
