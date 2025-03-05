from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import lightning as pl
import pytest
import torch
import torch.nn as nn
from torch.nn import Sequential

from calvera.bandits.neural_linear_bandit import NeuralLinearBandit
from calvera.utils.data_storage import (
    AllDataBufferStrategy,
    InMemoryDataBuffer,
    SlidingWindowBufferStrategy,
)


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

    # TODO: test output probabilities are correct. See issue #72.


def test_neural_linear_bandit_checkpoint_save_load(
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    tmp_path: Path,
) -> None:
    """
    Test saving and loading a NeuralLinearBandit checkpoint.

    Verifies that all components (network, helper network, buffer, linear head)
    are properly serialized and restored.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features = actions.shape[2]
    n_embedding_size = 6

    network = nn.Sequential(
        nn.Linear(n_features, n_embedding_size, bias=False),
    )
    nn.init.normal_(network[0].weight, mean=0.5, std=0.1)  # Specific weights for testing

    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    original_bandit = NeuralLinearBandit[torch.Tensor](
        network=network,
        n_embedding_size=n_embedding_size,
        buffer=buffer,
        train_batch_size=2,
        learning_rate=0.02,
        min_samples_required_for_training=4,
        initial_train_steps=2,
    )

    with torch.no_grad():
        original_bandit.precision_matrix = torch.eye(n_embedding_size) * 2.0
        original_bandit.b = torch.ones(n_embedding_size) * 0.5
        original_bandit.theta = torch.ones(n_embedding_size) * 0.25

    test_context = torch.randn(1, 3, n_features)

    with torch.no_grad():
        torch.manual_seed(42)
        original_output, _ = original_bandit(test_context)

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        enable_checkpointing=True,
    )
    trainer.fit(original_bandit, torch.utils.data.DataLoader(dataset, batch_size=2))

    checkpoint_path = tmp_path / "neural_linear_bandit.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    new_network = nn.Sequential(
        nn.Linear(n_features, n_embedding_size, bias=False),
    )
    nn.init.zeros_(new_network[0].weight)  # Different weights

    new_buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    loaded_bandit = NeuralLinearBandit[torch.Tensor].load_from_checkpoint(
        checkpoint_path,
        network=new_network,
        buffer=new_buffer,
        n_embedding_size=n_embedding_size,
    )

    # Verify linear head parameters are preserved
    assert torch.allclose(original_bandit.precision_matrix, loaded_bandit.precision_matrix)
    assert torch.allclose(original_bandit.b, loaded_bandit.b)
    assert torch.allclose(original_bandit.theta, loaded_bandit.theta)

    # Verify neural network weights are preserved
    assert torch.allclose(
        cast(Sequential, original_bandit.network)[0].weight, cast(Sequential, loaded_bandit.network)[0].weight
    )

    # Verify helper network weights are preserved
    assert torch.allclose(
        cast(Sequential, original_bandit._helper_network.network)[0].weight,
        cast(Sequential, loaded_bandit._helper_network.network)[0].weight,
    )

    # Verify buffer content is preserved
    assert len(loaded_bandit.buffer) == len(original_bandit.buffer)
    assert torch.allclose(original_bandit.buffer.contextualized_actions, loaded_bandit.buffer.contextualized_actions)  # type: ignore
    assert torch.allclose(original_bandit.buffer.rewards, loaded_bandit.buffer.rewards)  # type: ignore

    # Verify hyperparameters are preserved
    assert loaded_bandit.hparams["n_embedding_size"] == n_embedding_size
    assert loaded_bandit.hparams["min_samples_required_for_training"] == 4
    assert loaded_bandit.hparams["learning_rate"] == 0.02
    assert loaded_bandit.hparams["initial_train_steps"] == 2

    # Verify training state is preserved
    assert loaded_bandit._should_train_network == original_bandit._should_train_network
    assert loaded_bandit._samples_without_training_network == original_bandit._samples_without_training_network
    assert loaded_bandit.automatic_optimization == original_bandit.automatic_optimization

    # Verify model produces identical predictions with Thompson Sampling
    n_samples = 50
    original_choices = []
    loaded_choices = []

    for i in range(n_samples):
        seed = 1000 + i
        torch.manual_seed(seed)
        orig_action, _ = original_bandit(test_context)
        torch.manual_seed(seed)
        loaded_action, _ = loaded_bandit(test_context)

        original_choices.append(orig_action.argmax(dim=1).item())
        loaded_choices.append(loaded_action.argmax(dim=1).item())

    # Verify the pattern of selections is similar (agreement rate > 0.7)
    agreement_rate = (
        sum(
            original_choice == loaded_choice
            for original_choice, loaded_choice in zip(original_choices, loaded_choices, strict=False)
        )
        / n_samples
    )
    assert agreement_rate > 0.7, f"Models only agreed on {agreement_rate:.1%} of selections"


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
        min_samples_required_for_training=4,
        initial_train_steps=0,
        train_batch_size=2,
        learning_rate=1e-3,
        buffer=buffer,
    )
    # If True the uncertainty will be updated in the training and not the forward step.
    bandit.save_hyperparameters({"lazy_uncertainty_update": True})

    theta_1 = bandit.theta.clone()
    precision_matrix_1 = bandit.precision_matrix.clone()
    b_1 = bandit.b.clone()
    nn_weights_before = network[0].weight.clone()

    # Initially empty buffer
    assert buffer.contextualized_actions.numel() == 0
    assert buffer.embedded_actions.numel() == 0
    assert buffer.rewards.numel() == 0

    # Run training step
    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    # buffer should have newly appended rows
    assert buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert buffer.embedded_actions.shape[0] == actions.shape[0]
    assert buffer.rewards.shape[0] == actions.shape[0]

    assert not bandit.should_train_network, "Not enough data to train yet."

    trainer.fit(bandit)

    # The head should have been updated
    assert not torch.allclose(bandit.theta, theta_1)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_1)
    assert not torch.allclose(bandit.b, b_1)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # But the network should not have been updated
    assert network is bandit.network, "Network reference should not have been changed."
    assert torch.allclose(nn_weights_before, network[0].weight)

    # Store the updated values
    theta_2 = bandit.theta.clone()
    precision_matrix_2 = bandit.precision_matrix.clone()
    b_2 = bandit.b.clone()

    # Now run another training step
    trainer = pl.Trainer(fast_dev_run=True)

    bandit.record_feedback(actions, rewards)
    # The buffer should have grown
    assert buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.embedded_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.rewards.shape[0] == 2 * actions.shape[0]

    assert bandit.should_train_network, "Should train the network."

    trainer.fit(bandit)

    # The head should have been updated again
    assert not torch.allclose(bandit.theta, theta_2)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_2)
    assert not torch.allclose(bandit.b, b_2)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # And the network should have been updated
    assert network is bandit.network, "Network should be updated in place."
    assert not torch.allclose(network[0].weight, nn_weights_before)

    # Also test that the helper network has been updated. Necessary for correct future updates.
    assert not torch.allclose(cast(nn.Sequential, bandit._helper_network.network)[0].weight, nn_weights_before)

    # Store the updated values
    theta_3 = bandit.theta.clone()
    precision_matrix_3 = bandit.precision_matrix.clone()
    b_3 = bandit.b.clone()

    nn_weights_before = network[0].weight.clone()

    # Set the should_train_network to True manually
    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert not bandit.should_train_network, "Not enough data to train yet."
    bandit.should_train_network = True
    assert bandit.should_train_network, "Just set it."
    assert bandit.automatic_optimization, "Required to train the network."
    trainer.fit(bandit)

    # The head should have been updated again
    assert not torch.allclose(bandit.theta, theta_3)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_3)
    assert not torch.allclose(bandit.b, b_3)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # And the network should have been updated
    assert not torch.allclose(network[0].weight, nn_weights_before)

    # Now try the same with setting should_train_network to False
    theta_4 = bandit.theta.clone()
    precision_matrix_4 = bandit.precision_matrix.clone()
    b_4 = bandit.b.clone()

    nn_weights_before = network[0].weight.clone()

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert not bandit.should_train_network, "Not enough data to train yet."
    # Add a second batch!
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Has enough data now."
    bandit.should_train_network = False
    assert not bandit.should_train_network, "Just set it."
    assert not bandit.automatic_optimization, "Required to train the head without error."

    trainer.fit(bandit)

    assert not torch.allclose(bandit.theta, theta_4)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_4)
    assert not torch.allclose(bandit.b, b_4)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # And the network should NOT have been updated
    assert torch.allclose(network[0].weight, nn_weights_before)

    # Try training with a custom data loader
    theta_5 = bandit.theta.clone()
    precision_matrix_5 = bandit.precision_matrix.clone()
    b_5 = bandit.b.clone()

    nn_weights_before = network[0].weight.clone()

    assert not bandit.should_train_network, "Not enough data to train yet."
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(bandit, torch.utils.data.DataLoader(dataset, batch_size=2))

    # The buffer should have grown
    assert buffer.contextualized_actions.shape[0] == 6 * actions.shape[0]
    assert buffer.embedded_actions.shape[0] == 6 * actions.shape[0]
    assert buffer.rewards.shape[0] == 6 * actions.shape[0]

    # The head should have been updated again
    assert not torch.allclose(bandit.theta, theta_5)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_5)
    assert not torch.allclose(bandit.b, b_5)

    # Check that the precision matrix is symmetric and positive definite
    assert torch.allclose(bandit.precision_matrix, bandit.precision_matrix.T)
    vals, _ = torch.linalg.eigh(bandit.precision_matrix)
    assert torch.all(vals > 0), "Precision matrix must be positive definite, but eigenvalues are not all positive."

    # And the network should have been updated
    assert not torch.allclose(network[0].weight, nn_weights_before)


def test_neural_linear_sliding_window(
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
) -> None:
    """
    Verify that the sliding window buffer strategy works as expected with neural linear.
    Basically, only testing that we are able to use the sliding window buffer strategy without errors.
    But its not that easy to test that the buffer technique is used correctly.
    """
    actions, rewards, _ = small_context_reward_batch
    n_features = actions.shape[2]
    n_embedding_size = 4

    network = nn.Sequential(
        nn.Linear(n_features, n_embedding_size, bias=False),
        # don't add a ReLU because its the final layer
    )

    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=SlidingWindowBufferStrategy(window_size=1))

    bandit = NeuralLinearBandit[torch.Tensor](
        network=network,
        n_embedding_size=n_embedding_size,
        min_samples_required_for_training=4,
        initial_train_steps=0,
        train_batch_size=1,  # must be because window_size=1
        learning_rate=1e-3,
        buffer=buffer,
    )
    # If True the uncertainty will be updated in the training and not the forward step.
    bandit.save_hyperparameters({"lazy_uncertainty_update": True})

    theta_1 = bandit.theta.clone()
    precision_matrix_1 = bandit.precision_matrix.clone()
    b_1 = bandit.b.clone()
    nn_weights_before = network[0].weight.clone()

    # Run training step
    trainer = pl.Trainer(fast_dev_run=True)

    bandit.record_feedback(actions, rewards)
    trainer.fit(bandit)

    # After training step, buffer should have newly appended rows
    assert buffer.contextualized_actions.shape[0] == 2
    assert buffer.embedded_actions.shape[0] == 2
    assert buffer.rewards.shape[0] == 2

    # The head should have been updated
    assert not torch.allclose(bandit.theta, theta_1)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_1)
    assert not torch.allclose(bandit.b, b_1)

    # But the network should not have been updated
    assert torch.allclose(nn_weights_before, network[0].weight)

    # Store the updated values
    theta_2 = bandit.theta.clone()
    precision_matrix_2 = bandit.precision_matrix.clone()
    b_2 = bandit.b.clone()

    # Now run another training step
    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    trainer.fit(bandit)

    # The buffer should have grown
    assert buffer.contextualized_actions.shape[0] == 4
    assert buffer.embedded_actions.shape[0] == 4
    assert buffer.rewards.shape[0] == 4

    # The head should have been updated again
    assert not torch.allclose(bandit.theta, theta_2)
    assert not torch.allclose(bandit.precision_matrix, precision_matrix_2)
    assert not torch.allclose(bandit.b, b_2)

    # And the network should have been updated
    assert not torch.allclose(nn_weights_before, network[0].weight)


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
        min_samples_required_for_training=10,
        learning_rate=1e-2,
        buffer=buffer,
    )

    # Check hparams
    # these are the features after embedding that are input into the linear head... this is a little ugly but it
    # comes from the inheritance of the LinearTSBandit
    assert bandit.hparams["n_features"] == n_embedding_size
    assert bandit.hparams["n_embedding_size"] == n_embedding_size
    assert bandit.hparams["min_samples_required_for_training"] == 10
    assert bandit.hparams["learning_rate"] == 1e-2

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

    contextualized_actions, chosen_contextualized_actions, rewards, _ = small_context_reward_tupled_batch

    # Dummy network
    network = nn.Linear(n_features, n_embedding_size, bias=False)
    network.forward = MagicMock()  # type: ignore
    network.forward.return_value = torch.randn(batch_size * n_arms, n_embedding_size, device="cpu")

    bandit: NeuralLinearBandit[tuple[torch.Tensor, torch.Tensor]] = NeuralLinearBandit(
        network=network.to("cpu"),
        buffer=InMemoryDataBuffer(buffer_strategy=AllDataBufferStrategy()),
        n_embedding_size=n_embedding_size,
        min_samples_required_for_training=batch_size,
        train_batch_size=batch_size,
        learning_rate=1e-2,
    ).to("cpu")

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
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu")
    bandit.record_feedback(
        chosen_contextualized_actions,
        rewards,
    )
    # Call it once in record. This is unnecessary if we know that we will update the network but
    # required if we want to update the head.
    # TODO: We could refactor to only compute the embeddings in the update step of the head. See issue #149.
    assert network.forward.call_count == 1
    trainer.fit(bandit)
    # Call it two more times in the training loop. Once during training and once after training
    # to update the embeddings in the buffer. Both calls are required to update the head.
    assert network.forward.call_count == 3

    call_args = network.forward.call_args
    assert call_args is not None, "network.forward was not called."
    args, kwargs = call_args

    # Check that the expected tensors are in args and compare them using torch.allclose.
    # We expect to receive tensors of shapes (batch_size * n_chosen_arms, n_features)
    # Swapping the order of the chosen_contextualized_actions because the training batches are randomized
    expected_arg1 = chosen_contextualized_actions[0].reshape(batch_size * n_chosen_arms, n_features)
    expected_arg2 = chosen_contextualized_actions[1].reshape(batch_size * n_chosen_arms, n_features)

    assert torch.allclose(args[0], expected_arg1), "First argument of forward mismatches."
    assert torch.allclose(args[1], expected_arg2), "Second argument of forward mismatches."
