import torch
from torch import nn

from neural_bandits.utils.action_shape_adaptor import ImageActionAdaptor, TextActionAdaptor

def test_image_action_adaptor():
    embedding_size = 10
    batch_size = 2
    c = 3
    h = 32
    w = 32
    
    
    network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(h * w * c, embedding_size)
    )
    action_adaptor = ImageActionAdaptor(network, c, h, w)
    action = torch.randn(batch_size, c, h, w)
    output = action_adaptor(action)
    assert output.shape == (batch_size, embedding_size)
    
def test_test_action_adaptor():
    batch_size = 2
    embedding_size = 10
    seq_length = 32
    token_embedding_size = 8
    
    network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_length * token_embedding_size, embedding_size)
    )
    action_adaptor = TextActionAdaptor(network, token_embedding_size)
    action = torch.randn(batch_size, seq_length, token_embedding_size)
    output = action_adaptor(action)
    assert output.shape == (batch_size, embedding_size)