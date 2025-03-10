from pathlib import Path
from typing import Any, cast

import lightning as pl
import pytest
import torch
import torch.nn as nn

from calvera.bandits.abstract_bandit import _collate_fn
from calvera.bandits.neural_bandit import NeuralBandit
from calvera.bandits.neural_ts_bandit import NeuralTSBandit
from calvera.bandits.neural_ucb_bandit import NeuralUCBBandit
from calvera.utils.data_storage import AllDataBufferStrategy, InMemoryDataBuffer, SlidingWindowBufferStrategy
from calvera.utils.selectors import EpsilonGreedySelector


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


@pytest.fixture
def network_and_buffer() -> tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]]:
    """
    Create a simple network and buffer for bandit testing
    """
    n_features: int = 4
    network: nn.Module = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())
    return n_features, network, buffer


@pytest.fixture
def small_context_reward_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]],
]:
    """
    Returns (chosen_contextualized_actions, rewards, dataset):
      chosen_contextualized_actions shape: (batch_size=2, n_chosen_arms=1, n_features=4)
      rewards shape: (2,1)
    """
    batch_size, n_chosen_arms, n_features = 2, 1, 4
    contextualized_actions = torch.randn(batch_size, n_chosen_arms, n_features)
    rewards = torch.randn(batch_size, n_chosen_arms)

    class RandomDataset(
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]]
    ):
        def __init__(self, actions: torch.Tensor, rewards: torch.Tensor):
            self.actions = actions
            self.rewards = rewards

        def __len__(self) -> int:
            return len(self.actions)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, None, torch.Tensor, None]:
            return self.actions[idx], None, self.rewards[idx], None

    dataset = RandomDataset(contextualized_actions, rewards)
    return contextualized_actions, rewards, dataset


# ------------------------------------------------------------------------------
# 1) Common Tests for NeuralBandit Base Class
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_forward_shape(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    bandit_type: str,
) -> None:
    """
    Verify forward() returns a one-hot action (batch_size, n_arms) with correct shape
    for both UCB and TS bandits.
    """
    batch_size, n_arms = 2, 3
    n_features, network, buffer = network_and_buffer

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(n_features=n_features, network=network, buffer=buffer)
    else:
        bandit = NeuralTSBandit(n_features=n_features, network=network, buffer=buffer)

    contextualized_actions: torch.Tensor = torch.randn(batch_size, n_arms, n_features)
    output, p = bandit.forward(contextualized_actions)

    assert output.shape == (
        batch_size,
        n_arms,
    ), f"Expected shape {(batch_size, n_arms)}, got {output.shape}"
    assert p.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {p.shape}"
    assert torch.all(p >= 0) and torch.all(p <= 1), "Probabilities should be in [0, 1]"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_training_step(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    bandit_type: str,
) -> None:
    """
    Test that a training step runs without error and updates parameters correctly
    for both UCB and TS bandits.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features, network, buffer = network_and_buffer

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            min_samples_required_for_training=4,
            initial_train_steps=4,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            min_samples_required_for_training=4,
            initial_train_steps=4,
        )

    params_1 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    assert buffer.contextualized_actions.numel() == 0
    assert buffer.rewards.numel() == 0
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    trainer = pl.Trainer(fast_dev_run=True, max_steps=10)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    trainer.fit(bandit)

    assert buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert buffer.rewards.shape[0] == rewards.shape[0]
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    # Training should happen because we're within `initial_train_steps` (buffer size = 2 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_1[name]), f"Parameter {name} was not updated"

    params_2 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    trainer.fit(bandit)

    assert buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.rewards.shape[0] == 2 * rewards.shape[0]
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    # Training should happen because we're within `initial_train_steps` (buffer size = 4 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_2[name]), f"Parameter {name} was not updated"

    params_3 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)

    assert not bandit.should_train_network, "Not enough samples to train"

    trainer.fit(bandit)

    assert buffer.contextualized_actions.shape[0] == 3 * actions.shape[0]
    assert buffer.rewards.shape[0] == 3 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should not be in initial training stage"

    # Training should NOT happen here because new samples since last train = 2 = (6 - 4) <= 4
    # Network parameters should remain unchanged
    for name, param in bandit.theta_t.named_parameters():
        assert torch.allclose(param, params_3[name])

    params_4 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)

    assert bandit.should_train_network, "Should train network after new samples"

    trainer.fit(bandit)

    assert buffer.contextualized_actions.shape[0] == 4 * actions.shape[0]
    assert buffer.rewards.shape[0] == 4 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should not be in initial training stage"

    # Training SHOULD happen here because new samples since last train = 4 = (8 - 4) <= 4
    # Network parameters should be updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_4[name]), f"Parameter {name} was not updated"

    params_5 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert not bandit.should_train_network, "Not enough samples to train"

    assert buffer.contextualized_actions.shape[0] == 5 * actions.shape[0]
    assert buffer.rewards.shape[0] == 5 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should not be in initial training stage"

    bandit.should_train_network = True
    assert bandit.should_train_network, "Just set it to True"

    trainer.fit(bandit)

    # Training should happen because explicitly set to True
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_5[name]), f"Parameter {name} was not updated"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_training_step_custom_dataloader(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]],
    ],
    bandit_type: str,
) -> None:
    """
    Test that a training step runs without error and updates parameters correctly
    for both UCB and TS bandits.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features, network, buffer = network_and_buffer

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            min_samples_required_for_training=4,
            initial_train_steps=4,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            min_samples_required_for_training=4,
            initial_train_steps=4,
        )

    params_1 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    assert buffer.contextualized_actions.numel() == 0
    assert buffer.rewards.numel() == 0

    trainer = pl.Trainer(fast_dev_run=True, max_steps=10)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    trainer.fit(bandit)

    assert buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert buffer.rewards.shape[0] == rewards.shape[0]
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    # Training should happen because we're within `initial_train_steps` (buffer size = 2 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_1[name]), f"Parameter {name} was not updated"

    params_2 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    trainer.fit(bandit)

    assert buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.rewards.shape[0] == 2 * rewards.shape[0]
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    # Training should happen because we're within `initial_train_steps` (buffer size = 4 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_2[name]), f"Parameter {name} was not updated"

    params_3 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)

    assert not bandit.should_train_network, "Not enough samples to train"

    trainer.fit(
        bandit, torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=_collate_fn)
    )

    assert buffer.contextualized_actions.shape[0] == 3 * actions.shape[0]
    assert buffer.rewards.shape[0] == 3 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should be in initial training stage"

    # Training should happen here because we called trainer.fit with a dataloader.
    # even if not enough new samples since last train = 2 = (6 - 4) <= 4
    # we still force the update.
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_3[name])


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_training_step_sliding_window(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    bandit_type: str,
) -> None:
    """
    Test that a training step runs with SlidingWindowBufferStrategy runs without error and
    updates parameters correctly for both UCB and TS bandits.

    It's hard to test that we actually trained on the correct data though.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features, network, _ = network_and_buffer
    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=SlidingWindowBufferStrategy(window_size=4))

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            min_samples_required_for_training=4,
            initial_train_steps=4,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=2,
            min_samples_required_for_training=4,
            initial_train_steps=4,
        )

    params_1 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    assert buffer.contextualized_actions.numel() == 0
    assert buffer.rewards.numel() == 0
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    trainer = pl.Trainer(fast_dev_run=True, max_steps=10)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    assert buffer.contextualized_actions.shape[0] == actions.shape[0]
    assert buffer.rewards.shape[0] == rewards.shape[0]
    trainer.fit(bandit)

    # Training should happen because we're within `initial_train_steps` (buffer size = 2 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_1[name]), f"Parameter {name} was not updated"

    params_2 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    assert buffer.contextualized_actions.shape[0] == 2 * actions.shape[0]
    assert buffer.rewards.shape[0] == 2 * rewards.shape[0]
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"
    trainer.fit(bandit)

    # Training should happen because we're within `initial_train_steps` (buffer size = 4 <= 4)
    # Network parameters should have been updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_2[name]), f"Parameter {name} was not updated"

    params_3 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)

    bandit.record_feedback(actions, rewards)

    assert buffer.contextualized_actions.shape[0] == 3 * actions.shape[0]
    assert buffer.rewards.shape[0] == 3 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should not be in initial training stage"
    assert not bandit.should_train_network, "Not enough samples to train"

    trainer.fit(bandit)

    # Training should NOT happen here because new samples since last train = 2 = (6 - 4) <= 4
    # Network parameters should remain unchanged
    for name, param in bandit.theta_t.named_parameters():
        assert torch.allclose(param, params_3[name])

    params_4 = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)

    assert buffer.contextualized_actions.shape[0] == 4 * actions.shape[0]
    assert buffer.rewards.shape[0] == 4 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should not be in initial training stage"
    assert bandit.should_train_network, "Should train network after new samples"

    trainer.fit(bandit)

    # Training SHOULD happen here because new samples since last train = 4 = (8 - 4) <= 4
    # Network parameters should be updated
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_4[name]), f"Parameter {name} was not updated"

    trainer = pl.Trainer(fast_dev_run=True)
    bandit.record_feedback(actions, rewards)
    assert buffer.contextualized_actions.shape[0] == 5 * actions.shape[0]
    assert buffer.rewards.shape[0] == 5 * rewards.shape[0]
    assert not bandit.is_initial_training_stage(), "Should not be in initial training stage"
    assert not bandit.should_train_network, "Not enough samples to train"

    bandit.should_train_network = True
    assert bandit.should_train_network, "Just set it to True"

    trainer.fit(bandit)

    # Training should happen because explicitly set to True
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_4[name]), f"Parameter {name} was not updated"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_hparams_effect(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    bandit_type: str,
) -> None:
    """
    Verify hyperparameters are saved and affect the bandit behavior
    for both UCB and TS bandits.
    """
    n_features, network, buffer = network_and_buffer

    weight_decay: float = 0.1
    exploration_rate: float = 0.2
    learning_rate: float = 0.01
    train_batch_size: int = 25
    min_samples_required_for_training: int = 50
    initial_train_steps: int = 100

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            weight_decay=weight_decay,
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            learning_rate_decay=0.2,
            learning_rate_scheduler_step_size=2,
            train_batch_size=train_batch_size,
            min_samples_required_for_training=min_samples_required_for_training,
            initial_train_steps=initial_train_steps,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            weight_decay=weight_decay,
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            learning_rate_decay=0.2,
            learning_rate_scheduler_step_size=2,
            train_batch_size=train_batch_size,
            min_samples_required_for_training=min_samples_required_for_training,
            initial_train_steps=initial_train_steps,
        )

    assert bandit.hparams["n_features"] == n_features
    assert bandit.hparams["weight_decay"] == weight_decay
    assert bandit.hparams["exploration_rate"] == exploration_rate
    assert bandit.hparams["train_batch_size"] == train_batch_size
    assert bandit.hparams["learning_rate"] == learning_rate
    assert bandit.hparams["min_samples_required_for_training"] == min_samples_required_for_training
    assert bandit.hparams["initial_train_steps"] == initial_train_steps

    assert torch.all(bandit.Z_t == weight_decay), "Z_t should be initialized with weight_decay"

    opt_lr = cast(dict[str, Any], bandit.configure_optimizers())
    assert isinstance(opt_lr["optimizer"], torch.optim.Adam)
    assert opt_lr["optimizer"].param_groups[0]["lr"] == learning_rate
    assert opt_lr["optimizer"].param_groups[0]["weight_decay"] == weight_decay

    assert isinstance(opt_lr["lr_scheduler"], torch.optim.lr_scheduler.StepLR)
    assert opt_lr["lr_scheduler"].step_size == 2
    assert opt_lr["lr_scheduler"].gamma == 0.2


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_parameter_validation(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    bandit_type: str,
) -> None:
    """
    Test that the bandit properly validates parameter relationships.
    """
    n_features, network, buffer = network_and_buffer

    BanditClass: type[NeuralUCBBandit | NeuralTSBandit]

    BanditClass = NeuralUCBBandit if bandit_type == "ucb" else NeuralTSBandit

    # This should work fine
    BanditClass(
        n_features=n_features,
        network=network,
        buffer=buffer,
        train_batch_size=16,
        min_samples_required_for_training=32,
        initial_train_steps=48,
    )

    # Invalid: negative parameters
    with pytest.raises(
        AssertionError,
    ):
        BanditClass(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=15,
            min_samples_required_for_training=-40,
            initial_train_steps=45,
        )

    # Invalid: negative parameters
    with pytest.raises(AssertionError):
        BanditClass(
            n_features=n_features,
            network=network,
            buffer=buffer,
            train_batch_size=-17,
            min_samples_required_for_training=34,
            initial_train_steps=50,
        )


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_save_load_checkpoint(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    tmp_path: Path,
    bandit_type: str,
) -> None:
    """
    Test saving and loading a Neural bandit model checkpoint.
    Verifies that the loaded model preserves state and produces identical predictions.
    """
    n_features, network, buffer = network_and_buffer
    actions, rewards, dataset = small_context_reward_batch

    if bandit_type == "ucb":
        original_bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            weight_decay=0.05,
            exploration_rate=0.1,
            learning_rate=0.02,
            train_batch_size=1,
            min_samples_required_for_training=2,
            initial_train_steps=1,
        )
    else:
        original_bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            weight_decay=0.05,
            exploration_rate=0.1,
            learning_rate=0.02,
            train_batch_size=1,
            min_samples_required_for_training=2,
            initial_train_steps=1,
        )

    test_context = torch.randn(1, 3, n_features)

    original_predictions, _ = original_bandit(test_context)

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=3,
        enable_checkpointing=True,
    )
    trainer.fit(original_bandit, torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=_collate_fn))

    checkpoint_path = tmp_path / f"neural_{bandit_type}.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    new_network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    new_buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    bandit_class = NeuralUCBBandit if bandit_type == "ucb" else NeuralTSBandit
    loaded_bandit = bandit_class.load_from_checkpoint(
        checkpoint_path,
        n_features=n_features,
        network=new_network,
        buffer=new_buffer,
    )

    # Verify hyperparameters are preserved
    assert loaded_bandit.hparams["weight_decay"] == original_bandit.hparams["weight_decay"]
    assert loaded_bandit.hparams["exploration_rate"] == original_bandit.hparams["exploration_rate"]
    assert loaded_bandit.hparams["learning_rate"] == original_bandit.hparams["learning_rate"]

    # Verify Z_t tensor is preserved
    assert torch.allclose(original_bandit.Z_t, loaded_bandit.Z_t)

    # Verify network weights are preserved
    for (orig_name, orig_param), (loaded_name, loaded_param) in zip(
        original_bandit.theta_t.named_parameters(), loaded_bandit.theta_t.named_parameters(), strict=True
    ):
        assert orig_name == loaded_name
        assert torch.allclose(orig_param, loaded_param)

    # Verify the model produces identical predictions after loading
    if bandit_type == "ucb":
        assert torch.equal(original_bandit(test_context)[0], loaded_bandit(test_context)[0])
    else:
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

        # Verify the pattern of selections is similar (correlation > 0.8)
        agreement_rate = (
            sum(
                original_choice == loaded_choice
                for original_choice, loaded_choice in zip(original_choices, loaded_choices, strict=False)
            )
            / n_samples
        )
        assert agreement_rate > 0.8, f"Models only agreed on {agreement_rate:.1%} of selections"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_save_load_with_epsilon_greedy(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    tmp_path: Path,
    bandit_type: str,
) -> None:
    """
    Test saving and loading a Neural bandit with EpsilonGreedySelector.
    Ensures the selector type and state are preserved.
    """
    n_features, network, buffer = network_and_buffer
    _, _, dataset = small_context_reward_batch

    epsilon = 0.15

    if bandit_type == "ucb":
        original_bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=EpsilonGreedySelector(epsilon=epsilon, seed=42),
            train_batch_size=1,
            initial_train_steps=1,
        )
    else:
        original_bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=EpsilonGreedySelector(epsilon=epsilon, seed=42),
            train_batch_size=1,
            initial_train_steps=1,
        )

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        enable_checkpointing=True,
    )
    trainer.fit(original_bandit, torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=_collate_fn))

    checkpoint_path = tmp_path / f"neural_{bandit_type}_eps_greedy.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    new_network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    new_buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    bandit_class = NeuralUCBBandit if bandit_type == "ucb" else NeuralTSBandit
    loaded_bandit = bandit_class.load_from_checkpoint(
        checkpoint_path,
        n_features=n_features,
        network=new_network,
        buffer=new_buffer,
    )

    # Verify selector was properly restored
    assert isinstance(loaded_bandit.selector, EpsilonGreedySelector)
    assert loaded_bandit.selector.epsilon == epsilon

    # Verify selector produces the same outputs with the same seeds
    scores = torch.tensor([[0.9, 0.8, 0.7]])

    original_bandit.selector.generator.manual_seed(123)  # type: ignore
    loaded_bandit.selector.generator.manual_seed(123)

    original_selection = original_bandit.selector(scores)
    loaded_selection = loaded_bandit.selector(scores)

    assert torch.equal(original_selection, loaded_selection), "Selector states don't match after loading"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_neural_bandit_buffer_state_preserved(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    tmp_path: Path,
    bandit_type: str,
) -> None:
    """
    Test that buffer contents are preserved when saving/loading a checkpoint.
    """
    n_features, network, buffer = network_and_buffer
    actions, rewards, dataset = small_context_reward_batch

    if bandit_type == "ucb":
        original_bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            min_samples_required_for_training=2,
            train_batch_size=2,
            initial_train_steps=2,
        )
    else:
        original_bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            min_samples_required_for_training=2,
            train_batch_size=2,
            initial_train_steps=2,
        )

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        enable_checkpointing=True,
    )
    trainer.fit(original_bandit, torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=_collate_fn))

    checkpoint_path = tmp_path / f"neural_{bandit_type}_buffer.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    new_network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    new_buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())

    bandit_class = NeuralUCBBandit if bandit_type == "ucb" else NeuralTSBandit
    loaded_bandit = bandit_class.load_from_checkpoint(
        checkpoint_path,
        n_features=n_features,
        network=new_network,
        buffer=new_buffer,
    )

    assert len(loaded_bandit.buffer) == 2

    assert torch.equal(actions, loaded_bandit.buffer.contextualized_actions)  # type: ignore
    assert torch.equal(rewards.squeeze(1), loaded_bandit.buffer.rewards)  # type: ignore

    # Verify counters were properly saved/loaded
    assert loaded_bandit._new_samples_count == original_bandit._new_samples_count
    assert loaded_bandit._total_samples_count == original_bandit._total_samples_count
    assert loaded_bandit._should_train_network == original_bandit._should_train_network


# ------------------------------------------------------------------------------
# 2) Specific Tests for each Bandit Type
# ------------------------------------------------------------------------------
def test_ucb_score_method(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
) -> None:
    """
    Test that NeuralUCBBandit._score method correctly implements UCB scoring.
    """
    f_t_a: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]])
    exploration_terms: torch.Tensor = torch.tensor([[0.5, 0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1, 0.05]])

    n_features, network, buffer = network_and_buffer
    bandit: NeuralUCBBandit = NeuralUCBBandit(n_features=n_features, network=network, buffer=buffer)

    scores: torch.Tensor = bandit._score(f_t_a, exploration_terms)

    expected_scores: torch.Tensor = f_t_a + exploration_terms
    assert torch.allclose(scores, expected_scores), "UCB scoring should be f_t_a + exploration_terms"


def test_ts_score_method(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
) -> None:
    """
    Test that NeuralTSBandit._score method correctly implements TS scoring.
    """
    batch_size, n_arms = 3, 4
    f_t_a: torch.Tensor = torch.ones((batch_size, n_arms))  # means all set to 1.0
    exploration_terms: torch.Tensor = torch.ones((batch_size, n_arms)) * 0.1  # std devs all set to 0.1

    n_features, network, buffer = network_and_buffer
    bandit: NeuralTSBandit = NeuralTSBandit(n_features=n_features, network=network, buffer=buffer)

    scores: torch.Tensor = bandit._score(f_t_a, exploration_terms)

    # We can't test exact values due to randomness, but we can verify:
    # 1. Shape is correct
    assert scores.shape == f_t_a.shape, f"Expected shape {f_t_a.shape}, got {scores.shape}"

    # 2. Values differ from means (extremely unlikely to be exactly equal)
    assert not torch.allclose(scores, f_t_a), "TS scores should differ from means due to sampling"

    # 3. Most values should be within 3 standard deviations (99.7% statistically)
    within_bounds: float = ((scores - f_t_a).abs() <= 3 * exploration_terms).float().mean().item()
    assert within_bounds > 0.95, f"Expected >95% of samples within 3σ, got {within_bounds * 100:.2f}%"


def test_neural_ucb_forward_deterministic() -> None:
    """
    Test that NeuralUCBBandit forward pass is deterministic with fixed parameters.
    """
    n_features: int = 2
    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    network[0].weight.data = torch.tensor([[1.0, 0.1]])

    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())
    bandit: NeuralUCBBandit = NeuralUCBBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        weight_decay=1.0,
        exploration_rate=0.1,
    )

    contextualized_actions: torch.Tensor = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    # Forward pass should be deterministic for UCB
    output1, _ = bandit(contextualized_actions)
    output2, _ = bandit(contextualized_actions)

    assert torch.allclose(output1, output2), "UCB forward pass should be deterministic"

    # With the given weights [1.0, 0.1], the first action should be chosen
    # First action: 1.0*1.0 + 0.1*0.0 = 1.0
    # Second action: 1.0*0.0 + 0.1*1.0 = 0.1
    assert torch.all(output1 == torch.tensor([[1, 0]])), "Should select first action"


def test_neural_ts_forward_stochastic() -> None:
    """
    Test that NeuralTSBandit forward pass is stochastic (might choose different actions).
    """
    n_runs: int = 50

    n_features: int = 2
    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    network[0].weight.data = torch.tensor([[1.0, 1.0]])

    buffer = InMemoryDataBuffer[torch.Tensor](buffer_strategy=AllDataBufferStrategy())
    bandit: NeuralTSBandit = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        weight_decay=1.0,
        exploration_rate=1.0,
    )

    # Two actions with equal expected rewards
    contextualized_actions: torch.Tensor = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)

    outputs: list[torch.Tensor] = []
    for _ in range(n_runs):
        output, _ = bandit(contextualized_actions)
        outputs.append(output)

    outputs_tensor: torch.Tensor = torch.cat(outputs, dim=0)  # shape (n_runs, 2)

    # Count how many times each arm was chosen
    arm0_count = (outputs_tensor[:, 0] == 1).sum().item()
    arm1_count = (outputs_tensor[:, 1] == 1).sum().item()

    # With enough runs, both arms should be chosen at least once
    assert arm0_count > 0, "Arm 0 was never chosen in TS"
    assert arm1_count > 0, "Arm 1 was never chosen in TS"

    # For Thompson sampling with equal rewards, the distribution should be roughly balanced
    assert 0.2 <= arm0_count / n_runs <= 0.8, f"Expected balanced choices, got {arm0_count}/{n_runs} for arm 0"
