from typing import TypeVar

import torch

"""
    The inputs for some models is just a tensor and for others a tuple of several torch tensors.
    For example, the input to a model from the `transformers` library is a tuple of three tensors
    corresponding to the `input_ids`, `attention_mask`, and `token_type_ids`.
    On the other hand, a neural network only takes a single tensor as input.
"""
ActionInputType = TypeVar("ActionInputType", bound=torch.Tensor | (tuple[torch.Tensor, ...] | list[torch.Tensor]))
