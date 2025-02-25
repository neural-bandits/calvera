import lightning as pl
import pytest
import torch
import torch.nn as nn

from neural_bandits.bandits.neural_linear_bandit import NeuralLinearBandit
from neural_bandits.utils.data_storage import AllDataBufferStrategy, InMemoryDataBuffer


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


# ------------------------------------------------------------------------------
# 1) Tests for NeuralLinearBandit
# ------------------------------------------------------------------------------
def test_neural_linear_bandit_forward_shape() -> None:
    """
    Verify forward() returns a one-hot action (batch_size, n_arms) with correct shape.
    """
    batch_size, n_arms, n_features, n_embeddings = 2, 3, 4, 5

    # Simple encoder: embed from 4 to 5 dimensions
    encoder = nn.Sequential(
        nn.Linear(n_features, n_embeddings, bias=False),
        # don't add a ReLU here because its the final layer
    )
    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    # Create bandit
    bandit = NeuralLinearBandit(
        n_encoder_input_size=n_features,
        n_embedding_size=n_embeddings,  # same as input if encoder is identity
        encoder=encoder,
        buffer=buffer,
    )

    contextualized_actions = torch.randn(batch_size, n_arms, n_features)
    output, p = bandit.forward(contextualized_actions)

    # Check shape
    assert output.shape == (
        batch_size,
        n_arms,
    ), f"Expected shape {(batch_size, n_arms)}, got {output.shape}"

    assert p.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {p.shape}"
    assert torch.all(0 <= p) and torch.all(p <= 1), "Probabilities should be in [0, 1]"


def test_neural_linear_bandit_forward_no_network_small_sample() -> None:
    """
    Test forward with a small sample data we can reason about:
    If the bandit is random or identity, we just confirm shape & no errors.
    """
    n_features = 2
    encoder = nn.Identity()
    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())
    bandit = NeuralLinearBandit(
        n_encoder_input_size=n_features,
        n_embedding_size=n_features,
        encoder=encoder,
        buffer=buffer,
    )

    # Provide a simple known input
    contextualized_actions = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    output, p = bandit(contextualized_actions)
    # The bandit returns an argmax one-hot. Just confirm shape & no error
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    assert torch.sum(output).item() == 1, "One-hot vector in the row"

    assert p.shape == (1,), f"Expected shape (1,), got {p.shape}"
    assert 0 <= p.item() <= 1, "Probability should be in [0, 1]"


def test_neural_linear_bandit_forward_small_sample_correct() -> None:
    """
    Test forward with a small sample data we can reason about:
    Actually confirm the correct output.
    """
    n_features = 2
    encoder = nn.Sequential(
        nn.Linear(n_features, n_features, bias=False),
        # don't add a ReLU here because its the final layer
    )

    # fix the weights of the encoder to only regard the first feature, and the second one a litple bit
    encoder[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 0.1]])
    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralLinearBandit(
        n_encoder_input_size=n_features,
        n_embedding_size=n_features,
        encoder=encoder,
        buffer=buffer,
    )

    # Provide a simple known input
    contextualized_actions = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    # Set the bandit's theta to select the first feature (x1)
    bandit.theta = torch.tensor([1.0, 0.0])
    # Decrease the precision matrix to make the selection more deterministic
    bandit.precision_matrix = torch.tensor([[0.01, 0.0], [0.0, 0.01]])

    output, _ = bandit(contextualized_actions)
    assert output.shape == (1, 2)
    # assert that the correct action is selected
    assert torch.all(output == torch.tensor([[0, 1]]))

    # now change the weights of the head to only regard the second feature (x2)
    bandit.theta = torch.tensor([0.0, 1.0])
    bandit.precision_matrix = torch.tensor([[0.01, 0.0], [0.0, 0.01]])

    output, _ = bandit(contextualized_actions)
    assert output.shape == (1, 2)
    # assert that the correct action is selected
    assert torch.all(output == torch.tensor([[1, 0]]))

    # TODO: test output probabilities are correct


# ------------------------------------------------------------------------------
# 2) Tests for updating the NeuralLinear bandit
# ------------------------------------------------------------------------------
@pytest.fixture
def small_context_reward_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
]:
    """
    Returns (chosen_contextualized_actions, rewards):
      chosen_contextualized_actions shape: (batch_size=2, n_chosen_arms=1, n_features=4)
      rewards shape: (2,1)
    """
    batch_size, n_chosen_arms, n_features = 2, 1, 4
    contextualized_actions = torch.randn(batch_size, n_chosen_arms, n_features)
    # e.g., random rewards
    rewards = torch.randn(batch_size, n_chosen_arms)

    class RandomDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __init__(self, actions: torch.Tensor, rewards: torch.Tensor):
            self.actions = actions
            self.rewards = rewards

        def __len__(self) -> int:
            return len(self.actions)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.actions[idx], self.rewards[idx]

    dataset = RandomDataset(contextualized_actions, rewards)
    return contextualized_actions, rewards, dataset


def test_neural_linear_bandit_training_step(
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
) -> None:
    """
    Test that a training step runs without error on a small dataset and updates the replay buffer.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features = actions.shape[2]
    n_embedding_size = 4

    encoder = nn.Sequential(
        nn.Linear(n_features, n_embedding_size, bias=False),
        # don't add a ReLU because its the final layer
    )

    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralLinearBandit(
        encoder=encoder,
        n_encoder_input_size=n_features,
        n_embedding_size=n_embedding_size,
        encoder_update_batch_size=2,
        encoder_update_freq=4,
        head_update_freq=2,
        lr=1e-3,
        buffer=buffer,
    )

    theta_1 = bandit.theta.clone()
    precision_matrix_1 = bandit.precision_matrix.clone()
    b_1 = bandit.b.clone()
    nn_before = encoder[0].weight.clone()

    # Initially empty buffer
    assert bandit.buffer.contextualized_actions.numel() == 0
    assert bandit.buffer.embedded_actions.numel() == 0
    assert bandit.buffer.rewards.numel() == 0

    # Run training step
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        ),
    )

    # After training step, buffer should have newly appended rows
    assert bandit.buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert bandit.buffer.embedded_actions.shape[0] == actions.shape[0]
    assert bandit.buffer.rewards.shape[0] == actions.shape[0]

    # The head should have been updated
    assert not torch.allclose(bandit.theta, theta_1)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_1)
    assert not torch.allclose(bandit.b, b_1)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(
        vals > 0
    ), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # But the encoder should not have been updated
    assert torch.allclose(nn_before, encoder[0].weight)

    # Store the updated values
    theta_2 = bandit.theta.clone()
    precision_matrix_2 = bandit.precision_matrix.clone()
    b_2 = bandit.b.clone()

    # Now run another training step
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        ),
    )

    # The buffer should have grown
    assert bandit.buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert bandit.buffer.embedded_actions.shape[0] == 2 * actions.shape[0]
    assert bandit.buffer.rewards.shape[0] == 2 * actions.shape[0]

    # The head should have been updated again
    assert not torch.allclose(bandit.theta, theta_2)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_2)
    assert not torch.allclose(bandit.b, b_2)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(
        vals > 0
    ), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # And the encoder should have been updated
    assert not torch.allclose(nn_before, encoder[0].weight)


def test_neural_linear_bandit_hparams_effect() -> None:
    """
    Verify hyperparameters are saved and affect the bandit:
    E.g., different embedding size changes bandit.precision_matrix shape, etc.
    """
    n_features = 4
    n_embedding_size = 10

    # Dummy encoder
    encoder = nn.Linear(n_features, n_embedding_size, bias=False)

    buffer = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralLinearBandit(
        encoder=encoder,
        n_encoder_input_size=n_features,
        n_embedding_size=n_embedding_size,
        encoder_update_freq=10,
        head_update_freq=5,
        lr=1e-2,
        buffer=buffer,
    )

    # Check hparams
    assert bandit.hparams["n_encoder_input_size"] == n_features
    assert (
        bandit.hparams["n_features"] == n_embedding_size
    )  # these are the features after embedding that are input into the linear head... this is a little ugly but it comes from the inheritance of the LinearTSBandit
    assert bandit.hparams["n_embedding_size"] == n_embedding_size
    assert bandit.hparams["encoder_update_freq"] == 10
    assert bandit.hparams["head_update_freq"] == 5
    assert bandit.hparams["lr"] == 1e-2

    # Check that NeuralLinearBandit was configured accordingly
    assert bandit.precision_matrix.shape == (
        n_embedding_size,
        n_embedding_size,
    ), "Precision matrix should match n_embedding_size."
