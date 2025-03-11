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
from calvera.utils.data_storage import AllDataRetrievalStrategy, SlidingWindowRetrievalStrategy, TensorDataBuffer
from calvera.utils.selectors import EpsilonGreedySelector, TopKSelector


@pytest.fixture(autouse=True)
def seed_tests() -> None:
    pl.seed_everything(42)


@pytest.fixture
def network_and_buffer() -> tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]]:
    """
    Create a simple network and buffer for bandit testing
    """
    n_features: int = 4
    network: nn.Module = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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

    assert buffer.contextualized_actions is None
    assert buffer.rewards.numel() == 0
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    trainer = pl.Trainer(fast_dev_run=True, max_steps=10)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    trainer.fit(bandit)

    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == actions.shape[0]
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

    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == 2 * actions.shape[0]
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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

    assert buffer.contextualized_actions is None
    assert buffer.rewards.numel() == 0

    trainer = pl.Trainer(fast_dev_run=True, max_steps=10)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    trainer.fit(bandit)

    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == actions.shape[0]
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

    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == 2 * actions.shape[0]
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    bandit_type: str,
) -> None:
    """
    Test that a training step runs with SlidingWindowRetrievalStrategy runs without error and
    updates parameters correctly for both UCB and TS bandits.

    It's hard to test that we actually trained on the correct data though.
    """
    actions, rewards, dataset = small_context_reward_batch
    n_features, network, _ = network_and_buffer
    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=SlidingWindowRetrievalStrategy(window_size=4))

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

    assert buffer.contextualized_actions is None
    assert buffer.rewards.numel() == 0
    assert bandit.is_initial_training_stage(), "Should be in initial training stage"

    trainer = pl.Trainer(fast_dev_run=True, max_steps=10)
    bandit.record_feedback(actions, rewards)
    assert bandit.should_train_network, "Should train network after new samples"
    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == actions.shape[0]
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
    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == 2 * actions.shape[0]
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

    assert cast(torch.Tensor, buffer.contextualized_actions).shape[0] == 3 * actions.shape[0]
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    new_buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    new_buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    new_buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
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

    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())
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

    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())
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


# ------------------------------------------------------------------------------
# 3) Tests for Combinatorial Setting
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_combinatorial_neural_bandit_feedback(
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
    bandit_type: str,
) -> None:
    """
    Test that Neural bandits in combinatorial setting correctly process reward feedback
    when multiple actions are selected.
    """
    n_features, network, buffer = network_and_buffer

    k = 2

    if bandit_type == "ucb":
        bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=TopKSelector(k=k),
            min_samples_required_for_training=4,
            train_batch_size=2,
            initial_train_steps=4,
        )
    else:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=TopKSelector(k=k),
            min_samples_required_for_training=4,
            train_batch_size=2,
            initial_train_steps=4,
        )

    batch_size, n_arms = 2, 4
    contextualized_actions = torch.randn(batch_size, n_arms, n_features)

    chosen_actions, _ = bandit(contextualized_actions)

    rewards = torch.zeros(batch_size, n_arms)

    for b in range(batch_size):
        chosen_indices = chosen_actions[b].nonzero().squeeze(1)
        rewards[b, chosen_indices[0]] = 1.0
        rewards[b, chosen_indices[1]] = 0.5

    bandit.record_feedback(contextualized_actions, rewards)

    # Verify that the buffer contains the correct data
    assert bandit.buffer.contextualized_actions.shape[0] == batch_size * n_arms  # type: ignore
    assert bandit.buffer.rewards.shape[0] == batch_size * n_arms  # type: ignore

    params_before = {name: param.clone() for name, param in bandit.theta_t.named_parameters()}

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(bandit)

    # Verify parameters changed
    for name, param in bandit.theta_t.named_parameters():
        assert not torch.allclose(param, params_before[name]), f"Parameter {name} did not change after training"


@pytest.mark.parametrize("bandit_type", ["ucb", "ts"])
def test_combinatorial_neural_bandit_save_load(
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    tmp_path: Path,
    bandit_type: str,
) -> None:
    """
    Test saving and loading a Neural bandit model with TopKSelector.
    Verifies that the selector type and k value are preserved.
    """
    n_features, network, buffer = network_and_buffer
    ـ, ـ, dataset = small_context_reward_batch

    k = 3

    if bandit_type == "ucb":
        original_bandit: NeuralBandit = NeuralUCBBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=TopKSelector(k=k),
            train_batch_size=1,
            initial_train_steps=1,
        )
    else:
        original_bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            selector=TopKSelector(k=k),
            train_batch_size=1,
            initial_train_steps=1,
        )

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        enable_checkpointing=True,
    )
    trainer.fit(original_bandit, torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=_collate_fn))

    checkpoint_path = tmp_path / f"neural_{bandit_type}_topk.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    new_network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    new_buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

    bandit_class = NeuralUCBBandit if bandit_type == "ucb" else NeuralTSBandit
    loaded_bandit = bandit_class.load_from_checkpoint(
        checkpoint_path,
        n_features=n_features,
        network=new_network,
        buffer=new_buffer,
    )

    # Verify selector was properly restored
    assert isinstance(loaded_bandit.selector, TopKSelector)
    assert loaded_bandit.selector.k == k

    # Verify selector produces the same outputs
    test_scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])

    original_selection = original_bandit.selector(test_scores)
    loaded_selection = loaded_bandit.selector(test_scores)

    assert torch.equal(original_selection, loaded_selection), "Selector states don't match after loading"
    assert original_selection.sum().item() == k, f"Expected {k} actions selected, got {original_selection.sum().item()}"


@pytest.mark.parametrize("bandit_class", [NeuralUCBBandit, NeuralTSBandit])
def test_neural_bandit_warm_start(
    network_and_buffer: tuple[int, nn.Module, InMemoryDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]],
    ],
    tmp_path: Path,
    bandit_class: type[NeuralBandit],
) -> None:
    """Test warm_start functionality and checkpoint behavior for NeuralBandits."""
    actions, rewards, _ = small_context_reward_batch
    n_features, _, _ = network_and_buffer

    cold_bandit = bandit_class(
        n_features=n_features,
        network=nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1)),
        buffer=InMemoryDataBuffer(retrieval_strategy=AllDataRetrievalStrategy()),
        train_batch_size=2,
        initial_train_steps=2,
        warm_start=False,
    )

    warm_bandit = bandit_class(
        n_features=n_features,
        network=nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1)),
        buffer=InMemoryDataBuffer(retrieval_strategy=AllDataRetrievalStrategy()),
        train_batch_size=2,
        initial_train_steps=2,
        warm_start=True,
    )

    # Test initialization behavior
    assert cold_bandit.theta_t_init is not None, "Cold bandit should store initial weights"
    assert warm_bandit.theta_t_init is None, "Warm bandit should not store initial weights"

    # Test checkpoint behavior
    trainer = pl.Trainer(default_root_dir=str(tmp_path), fast_dev_run=True)
    cold_bandit.record_feedback(actions, rewards)
    trainer.fit(cold_bandit)

    ckpt_path = tmp_path / f"{bandit_class.__name__}.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_bandit = bandit_class.load_from_checkpoint(
        ckpt_path,
        n_features=n_features,
        network=nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1)),
        buffer=InMemoryDataBuffer(retrieval_strategy=AllDataRetrievalStrategy()),
    )

    assert not loaded_bandit.hparams["warm_start"], "warm_start=False should be preserved in checkpoint"
    assert loaded_bandit.theta_t_init is not None, "Initial weights should be preserved in checkpoint"


# ------------------------------------------------------------------------------
# 2) Tests for num_samples_per_arm in NeuralTS
# ------------------------------------------------------------------------------
def test_neural_ts_num_samples_per_arm_parameter() -> None:
    """
    Test that NeuralTSBandit correctly handles different values of num_samples_per_arm.
    """
    n_features = 4
    network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

    for num_samples in [1, 5, 10, 20]:
        bandit = NeuralTSBandit(
            n_features=n_features,
            network=network,
            buffer=buffer,
            num_samples_per_arm=num_samples,
        )

        # Verify the parameter is saved
        assert bandit.hparams["num_samples_per_arm"] == num_samples

        batch_size, n_arms = 2, 3
        contextualized_actions = torch.randn(batch_size, n_arms, n_features)

        # Run forward pass (should not error)
        chosen_actions, p = bandit(contextualized_actions)

        # Basic shape checks
        assert chosen_actions.shape == (batch_size, n_arms)
        assert p.shape == (batch_size,)


def test_neural_ts_num_samples_per_arm_affects_exploration() -> None:
    """
    Test that num_samples_per_arm affects the exploration behavior of NeuralTS.
    When arms have different values, more samples should lead to more
    consistent selection of the best arm.
    """
    n_features = 2

    network = nn.Sequential(nn.Linear(n_features, 1, bias=False))
    network[0].weight.data = torch.tensor([[1.0, 0.1]])  # First feature matters most

    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

    bandit1 = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        num_samples_per_arm=1,
        exploration_rate=2.0,
    )

    bandit10 = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        num_samples_per_arm=10,
        exploration_rate=2.0,
    )

    # Create context with different values for the arms
    # Arm 0 is best, arms 1 and 2 are worse
    batch_size, n_arms = 1, 3
    context = torch.zeros(batch_size, n_arms, n_features)
    context[0, 0, 0] = 1.2  # Best arm (arm 0)
    context[0, 1, 0] = 1.0  # Middle arm (arm 1)
    context[0, 2, 0] = 0.8  # Worst arm (arm 2)

    context[0, :, 1] = 1.0

    n_trials = 200
    bandit1_choices = []
    bandit10_choices = []

    for i in range(n_trials):
        torch.manual_seed(42 + i)
        chosen1, _ = bandit1(context)
        torch.manual_seed(42 + i)
        chosen10, _ = bandit10(context)

        bandit1_choices.append(chosen1.argmax().item())
        bandit10_choices.append(chosen10.argmax().item())

    b1_best_arm_count = bandit1_choices.count(0)
    b10_best_arm_count = bandit10_choices.count(0)

    b1_best_arm_rate = b1_best_arm_count / n_trials
    b10_best_arm_rate = b10_best_arm_count / n_trials

    # The bandit with more samples should select the best arm more frequently
    assert b10_best_arm_rate > b1_best_arm_rate, (
        f"Expected bandit with 10 samples to select the best arm more often, "
        f"got {b10_best_arm_rate:.2%} vs {b1_best_arm_rate:.2%}"
    )

    b1_worst_arm_count = bandit1_choices.count(2)
    b10_worst_arm_count = bandit10_choices.count(2)

    b1_worst_arm_rate = b1_worst_arm_count / n_trials
    b10_worst_arm_rate = b10_worst_arm_count / n_trials

    # The bandit with more samples should select the worst arm less frequently
    assert b10_worst_arm_rate < b1_worst_arm_rate, (
        f"Expected bandit with 10 samples to select the worst arm less often, "
        f"got {b10_worst_arm_rate:.2%} vs {b1_worst_arm_rate:.2%}"
    )


def test_neural_ts_num_samples_per_arm_save_load(
    network_and_buffer: tuple[int, nn.Module, TensorDataBuffer[torch.Tensor]],
    small_context_reward_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    ],
    tmp_path: Path,
) -> None:
    """
    Test that num_samples_per_arm is preserved when saving/loading a NeuralTS checkpoint.
    """
    n_features, network, buffer = network_and_buffer
    ـ, ـ, dataset = small_context_reward_batch

    num_samples = 7
    original_bandit = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        num_samples_per_arm=num_samples,
        train_batch_size=2,
        initial_train_steps=1,
    )

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        enable_checkpointing=True,
    )
    trainer.fit(original_bandit, torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=_collate_fn))

    checkpoint_path = tmp_path / "neural_ts_num_samples.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    new_network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    new_buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

    loaded_bandit = NeuralTSBandit.load_from_checkpoint(
        checkpoint_path,
        n_features=n_features,
        network=new_network,
        buffer=new_buffer,
    )

    assert loaded_bandit.hparams["num_samples_per_arm"] == num_samples


def test_neural_ts_score_method_with_num_samples_per_arm() -> None:
    """
    Test that the _score method in NeuralTS behaves differently with different num_samples_per_arm values.
    """
    n_features = 4
    network = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1))
    buffer = TensorDataBuffer[torch.Tensor](retrieval_strategy=AllDataRetrievalStrategy())

    bandit1 = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        num_samples_per_arm=1,
    )

    bandit10 = NeuralTSBandit(
        n_features=n_features,
        network=network,
        buffer=buffer,
        num_samples_per_arm=10,
    )

    f_t_a = torch.ones(2, 3)
    exploration_terms = torch.ones(2, 3) * 0.1

    torch.manual_seed(42)
    score1 = bandit1._score(f_t_a, exploration_terms)

    torch.manual_seed(42)
    score10 = bandit10._score(f_t_a, exploration_terms)

    # Scores should be different due to different number of samples
    assert not torch.allclose(score1, score10), "Scores should differ when using different num_samples_per_arm values"

    # For bandit1, scores will be just one sample per arm
    # For bandit10, scores are the max of 10 samples, which should be more optimistic
    # So the mean score from bandit10 should be higher
    assert score10.mean() > score1.mean(), "Expected higher mean score with more samples due to optimistic sampling"
