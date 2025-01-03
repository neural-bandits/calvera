from typing import Any

import pytest

from neural_bandits.algorithms.linear_bandits import LinearBandit, LinearUCBBandit
from neural_bandits.datasets.wheel import WheelBanditDataset
from neural_bandits.environments.base_environment import BaseEnvironment
from neural_bandits.trainers.linear_trainer import LinearTrainer
from neural_bandits.utils.multiclass import MultiClassContextualiser


class TestBaseEnvironment:

    @pytest.fixture
    def environment(self, request: Any) -> BaseEnvironment[LinearBandit]:
        return BaseEnvironment(
            LinearUCBBandit(5, 2 * 5),
            LinearTrainer(),
            WheelBanditDataset(1000, 0.9),
            MultiClassContextualiser(n_arms=5),
            seed=42,
            max_steps=100,
            log_method=lambda x: print(x),
        )

    def test_run_linucb(self, environment: BaseEnvironment[LinearBandit]) -> None:
        results = environment.run()
        assert results["cumulative_rewards"][-1] // (0.9 * 1.2 + 0.1 * 50) > 0.7
