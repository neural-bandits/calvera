import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from calvera.benchmark.network_wrappers import BertWrapper, ResNetWrapper


class DummyBertNetwork(nn.Module):
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        batch_size = input_ids.shape[0]
        seq_len = 10
        hidden_size = 768

        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=torch.ones(batch_size, seq_len, hidden_size)
        )
        return output


class DummyResNetNetwork(nn.Module):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.ones(batch_size, 512, 7, 7)

    def forward_head(self, x: torch.Tensor, pre_logits: bool) -> torch.Tensor:
        return x.mean(dim=[2, 3])


def test_bert_wrapper() -> None:
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, 1, seq_len))
    attention_mask = torch.rand(batch_size, 1, seq_len)
    token_type_ids = torch.randint(0, 2, (batch_size, 1, seq_len))

    dummy_network = DummyBertNetwork()
    model = BertWrapper(dummy_network)
    output = model(input_ids, attention_mask, token_type_ids)

    assert output.shape == (batch_size, 768)


def test_resnet_wrapper() -> None:
    batch_size = 1
    # For a 224x224 image with 3 channels, the flattened length is 3*224*224.
    dummy_input = torch.rand(batch_size, 3 * 224 * 224)

    dummy_network = DummyResNetNetwork()
    model = ResNetWrapper(dummy_network)
    # Call the forward pass
    output = model(dummy_input)
    # The output should have shape (batch_size, 128) after the dim_reduction layer.
    assert output.shape == (batch_size, 128)
