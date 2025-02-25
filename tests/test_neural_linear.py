from unittest.mock import MagicMock

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
    """Verify forward() returns a one-hot action (batch_size, n_arms) with correct shape."""
    batch_size, n_arms, n_features, n_embeddings = 2, 3, 4, 5

    # Simple network: embed from 4 to 5 dimensions
    network = nn.Sequential(
        nn.Linear(n_features, n_embeddings, bias=False),
        # don't add a ReLU here because its the final layer
    )
    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    # Create bandit
    bandit = NeuralLinearBandit[torch.Tensor](
        n_embedding_size=n_embeddings,  # same as input if encoder is identity
        network=network,
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
    assert torch.all(p >= 0) and torch.all(p <= 1), "Probabilities should be in [0, 1]"


def test_neural_linear_bandit_forward_no_network_small_sample() -> None:
    """Test forward with a small sample data we can reason about:
    If the bandit is random or identity, we just confirm shape & no errors.
    """
    n_features = 2
    network = nn.Identity()
    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())
    bandit = NeuralLinearBandit[torch.Tensor](
        n_embedding_size=n_features,
        network=network,
        buffer=buffer,
    )

    # Provide a simple known input
    contextualized_actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)  # shape (1, 2, 2)

    output, p = bandit.forward(contextualized_actions)
    # The bandit returns an argmax one-hot. Just confirm shape & no error
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    assert torch.sum(output).item() == 1, "One-hot vector in the row"

    assert p.shape == (1,), f"Expected shape (1,), got {p.shape}"
    assert 0 <= p.item() <= 1, "Probability should be in [0, 1]"


def test_neural_linear_bandit_forward_small_sample_correct() -> None:
    """Test forward with a small sample data we can reason about:
    Actually confirm the correct output.
    """
    n_features = 2
    network = nn.Sequential(
        nn.Linear(n_features, n_features, bias=False),
        # don't add a ReLU here because its the final layer
    )

    # fix the weights of the encoder to only regard the first feature, and the second one a little bit
    network[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 0.1]])
    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    bandit: NeuralLinearBandit[torch.Tensor] = NeuralLinearBandit(
        n_embedding_size=n_features,
        network=network,
        buffer=buffer,
    )

    # Provide a simple known input
    contextualized_actions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)  # shape (1, 2, 2)

    # Set the bandit's theta to select the first feature (x1)
    bandit.theta = torch.tensor([1.0, 0.0])
    # Decrease the precision matrix to make the selection more deterministic
    bandit.precision_matrix = torch.tensor([[0.01, 0.0], [0.0, 0.01]])

    output, _ = bandit(contextualized_actions)
    assert output.shape == (1, 2)
    # assert that the correct action is selected
    assert torch.all(output == torch.tensor([[1, 0]]))

    # now change the weights of the head to only regard the second feature (x2)
    bandit.theta = torch.tensor([0.0, 1.0])
    bandit.precision_matrix = torch.tensor([[0.01, 0.0], [0.0, 0.01]])

    output, _ = bandit(contextualized_actions)
    assert output.shape == (1, 2)
    # assert that the correct action is selected
    assert torch.all(output == torch.tensor([[0, 1]]))

    # TODO: test output probabilities are correct


# ------------------------------------------------------------------------------
# 2) Tests for updating the NeuralLinear bandit
# ------------------------------------------------------------------------------
@pytest.fixture
def small_context_reward_batch() -> (
    tuple[torch.Tensor, torch.Tensor, torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]]
):
    """
    Generates synthetic test data for training steps.

    Returns:
        tuple: A tuple containing:
            - chosen_contextualized_actions: A tensor with shape (2, 1, 4) representing the selected contextualized
                actions.
            - rewards: A tensor with shape (2, 1) representing the corresponding rewards.
            - dataset: A dataset containing 2 samples for testing purposes.
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
    """Test that a training step runs without error on a small dataset and updates the replay buffer."""
    actions, rewards, dataset = small_context_reward_batch
    n_features = actions.shape[2]
    n_embedding_size = 4

    network = nn.Sequential(
        nn.Linear(n_features, n_embedding_size, bias=False),
        # don't add a ReLU because its the final layer
    )

    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralLinearBandit[torch.Tensor](
        network=network,
        n_embedding_size=n_embedding_size,
        network_update_freq=4,
        head_update_freq=2,
        network_update_batch_size=2,
        lr=1e-3,
        buffer=buffer,
    )
    bandit.lazy_uncertainty_update = True

    theta_1 = bandit.theta.clone()
    precision_matrix_1 = bandit.precision_matrix.clone()
    b_1 = bandit.b.clone()
    nn_before = network[0].weight.clone()

    # Initially empty buffer
    assert buffer.contextualized_actions.numel() == 0
    assert buffer.embedded_actions.numel() == 0
    assert buffer.rewards.numel() == 0

    # Run training step
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0),
    )

    # After training step, buffer should have newly appended rows
    assert buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert buffer.embedded_actions.shape[0] == actions.shape[0]
    assert buffer.rewards.shape[0] == actions.shape[0]

    # The head should have been updated
    assert not torch.allclose(bandit.theta, theta_1)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_1)
    assert not torch.allclose(bandit.b, b_1)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # But the network should not have been updated
    assert torch.allclose(nn_before, network[0].weight)

    # Store the updated values
    theta_2 = bandit.theta.clone()
    precision_matrix_2 = bandit.precision_matrix.clone()
    b_2 = bandit.b.clone()

    # Now run another training step
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0),
    )

    # The buffer should have grown
    assert buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.embedded_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.rewards.shape[0] == 2 * actions.shape[0]

    # The head should have been updated again
    assert not torch.allclose(bandit.theta, theta_2)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_2)
    assert not torch.allclose(bandit.b, b_2)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # And the network should have been updated
    assert not torch.allclose(nn_before, network[0].weight)


def test_neural_linear_bandit_hparams_effect() -> None:
    """Verify hyperparameters are saved and affect the bandit:
    E.g., different embedding size changes bandit.precision_matrix shape, etc.
    """
    n_features = 4
    n_embedding_size = 10

    # Dummy network
    network = nn.Linear(n_features, n_embedding_size, bias=False)

    buffer: InMemoryDataBuffer[torch.Tensor] = InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy())

    bandit = NeuralLinearBandit[torch.Tensor](
        network=network,
        n_embedding_size=n_embedding_size,
        network_update_freq=10,
        head_update_freq=5,
        lr=1e-2,
        buffer=buffer,
    )

    # Check hparams
    # these are the features after embedding that are input into the linear head... this is a little ugly but it
    # comes from the inheritance of the LinearTSBandit
    assert bandit.hparams["n_features"] == n_embedding_size
    assert bandit.hparams["n_embedding_size"] == n_embedding_size
    assert bandit.hparams["network_update_freq"] == 10
    assert bandit.hparams["head_update_freq"] == 5
    assert bandit.hparams["lr"] == 1e-2

    # Check that NeuralLinearBandit was configured accordingly
    assert bandit.precision_matrix.shape == (
        n_embedding_size,
        n_embedding_size,
    ), "Precision matrix should match n_embedding_size."


# ------------------------------------------------------------------------------
# 3) Tests for NeuralLinearBandit with tuple input
# ------------------------------------------------------------------------------


@pytest.fixture
def small_context_reward_tupled_batch() -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.utils.data.Dataset[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
]:
    """
    Returns (chosen_contextualized_actions, rewards):
      chosen_contextualized_actions shape: (batch_size=2, n_chosen_arms=1, n_features=4)
      rewards shape: (2,1)
    """
    batch_size, n_arms, n_features = 2, 3, 4
    n_chosen_arms = 1
    contextualized_actions = torch.randn(batch_size, n_arms, n_features)
    other_input_tensor = torch.randn(batch_size, n_arms, n_features)
    # e.g., random rewards
    realized_rewards = torch.randn(batch_size, n_chosen_arms)

    class RandomTupleDataset(torch.utils.data.Dataset[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]):
        def __init__(
            self,
            actions: tuple[torch.Tensor, torch.Tensor],
            realized_rewards: torch.Tensor,
        ):
            self.actions = actions
            self.realized_rewards = realized_rewards

        def __len__(self) -> int:
            return len(self.actions)

        def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            return (self.actions[0][idx], self.actions[1][idx]), self.realized_rewards[idx]

    chosen_idx = 0
    chosen_contextualized_actions = (
        contextualized_actions[:, chosen_idx, :].unsqueeze(1),
        other_input_tensor[:, chosen_idx, :].unsqueeze(1),
    )

    dataset = RandomTupleDataset(chosen_contextualized_actions, realized_rewards)
    return (
        (contextualized_actions, other_input_tensor),
        chosen_contextualized_actions,
        realized_rewards,
        dataset,
    )


def test_neural_linear_bandit_tuple_input(
    small_context_reward_tupled_batch: tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
) -> None:
    """
    Test that the neural linear bandit can handle tuple input.
    """
    batch_size, n_arms, n_features = 2, 3, 4
    n_chosen_arms = 1
    n_embedding_size = 10

    contextualized_actions, chosen_contextualized_actions, _, dataset = small_context_reward_tupled_batch

    # Dummy network
    network = nn.Linear(n_features, n_embedding_size, bias=False)
    network.forward = MagicMock()  # type: ignore
    network.forward.return_value = torch.randn(batch_size * n_arms, n_embedding_size)

    bandit: NeuralLinearBandit[tuple[torch.Tensor, torch.Tensor]] = NeuralLinearBandit(
        network=network,
        buffer=InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy()),
        n_embedding_size=n_embedding_size,
        network_update_freq=batch_size,
        head_update_freq=1,
        network_update_batch_size=batch_size,
        lr=1e-2,
    )

    output, p = bandit.forward(contextualized_actions)
    assert output.shape == (batch_size, n_arms)
    assert torch.sum(output).item() == batch_size, "One-hot vector in each row"

    assert p.shape == (batch_size,)
    assert torch.all(p >= 0) and torch.all(p <= 1), "Probabilities should be in [0, 1]"

    assert network.forward.call_count == 1
    call_args = network.forward.call_args
    assert call_args is not None, "network.forward was not called."
    args, kwargs = call_args

    # Check that the expected tensors are in args and compare them using torch.allclose.
    # We expect to receive tensors of shapes (batch_size * n_arms, n_features)
    expected_arg1 = contextualized_actions[0].reshape(batch_size * n_arms, n_features)
    expected_arg2 = contextualized_actions[1].reshape(batch_size * n_arms, n_features)

    assert torch.allclose(args[0], expected_arg1), "First argument of forward mismatches."
    assert torch.allclose(args[1], expected_arg2), "Second argument of forward mismatches."

    network.forward = MagicMock()  # type: ignore
    network.forward.return_value = torch.randn(batch_size * n_chosen_arms, n_embedding_size)

    ###################### now for the training step ######################
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(
        bandit,
        torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0),
    )
    assert (
        network.forward.call_count == 2 * n_chosen_arms
    )  # batch_size * n_chosen_arms / batch_size. Once forward and once backward.

    call_args = network.forward.call_args
    assert call_args is not None, "network.forward was not called."
    args, kwargs = call_args

    # Check that the expected tensors are in args and compare them using torch.allclose.
    # We expect to receive tensors of shapes (batch_size * n_chosen_arms, n_features)
    expected_arg1 = chosen_contextualized_actions[0].reshape(batch_size * n_chosen_arms, n_features)
    expected_arg2 = chosen_contextualized_actions[1].reshape(batch_size * n_chosen_arms, n_features)

    assert torch.allclose(args[0], expected_arg1), "First argument of forward mismatches."
    assert torch.allclose(args[1], expected_arg2), "Second argument of forward mismatches."
