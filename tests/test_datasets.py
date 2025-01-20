import pytest
import torch

from neural_bandits.datasets.covertype import CovertypeDataset
from neural_bandits.datasets.mnist import MNISTDataset
from neural_bandits.datasets.statlog import StatlogDataset
from neural_bandits.datasets.wheel import WheelBanditDataset, get_optimal_actions


class TestCoverTypeDataset:
    @pytest.fixture
    def dataset(self) -> CovertypeDataset:
        return CovertypeDataset()

    def test_len(self, dataset: CovertypeDataset) -> None:
        assert len(dataset) == 581012

    def test_getitem(self, dataset: CovertypeDataset) -> None:
        for _ in range(10):
            X = dataset[0]
            assert X.shape == (54,)

    def test_reward(self, dataset: CovertypeDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, torch.tensor(1)).item()
            assert reward == (dataset.y[i] - 1 == 1)


class TestMNISTDataset:
    @pytest.fixture
    def dataset(self) -> MNISTDataset:
        return MNISTDataset()

    def test_len(self, dataset: MNISTDataset) -> None:
        assert len(dataset) == 70000

    def test_getitem(self, dataset: MNISTDataset) -> None:
        for _ in range(10):
            X = dataset[0]
            assert X.shape == (784,)

    def test_reward(self, dataset: MNISTDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, torch.tensor(1)).item()
            assert reward == (dataset.y[i] == 1)


class TestStatlogDataset:
    @pytest.fixture
    def dataset(self) -> StatlogDataset:
        return StatlogDataset()

    def test_len(self, dataset: StatlogDataset) -> None:
        assert len(dataset) == 6435

    def test_getitem(self, dataset: StatlogDataset) -> None:
        for _ in range(10):
            X = dataset[0]
            assert X.shape == (36,)

    def test_reward(self, dataset: StatlogDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, torch.tensor(1)).item()
            assert reward == (dataset.y[i] - 1 == 1)


class TestWheelBanditDataset:
    @pytest.fixture
    def dataset(self) -> WheelBanditDataset:
        return WheelBanditDataset(num_samples=1000, delta=0.8)

    def test_len(self, dataset: WheelBanditDataset) -> None:
        assert len(dataset) == 1000

    def test_getitem(self, dataset: WheelBanditDataset) -> None:
        for _ in range(50):
            X = dataset[0]
            assert X.shape == (2,)

    def test_reward(self, dataset: WheelBanditDataset) -> None:
        # reward for action 4 should around 1 - 1.2
        for i in range(100):
            reward = dataset.reward(i, torch.tensor(4))
            assert 0.7 <= reward <= 1.5

        for i in range(100):
            reward = dataset.reward(i, torch.tensor(0))
            assert 0.7 <= reward <= 1.5 or 49.5 <= reward <= 50.5

    def test_opt_action(self, dataset: WheelBanditDataset) -> None:
        for i in range(10):
            X = dataset[i]
            opt_action = get_optimal_actions(X.unsqueeze(0), dataset.delta)
            if torch.norm(X) <= dataset.delta:
                assert opt_action == 4
            else:
                a = (X[0] > 0).float()
                b = (X[1] > 0).float()

                assert opt_action == (3 - 2 * a - b)
