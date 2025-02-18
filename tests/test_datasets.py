import pytest

from neural_bandits.datasets.covertype import CovertypeDataset
from neural_bandits.datasets.imdb_reviews import ImdbMovieReviews
from neural_bandits.datasets.mnist import MNISTDataset
from neural_bandits.datasets.statlog import StatlogDataset
from neural_bandits.datasets.wheel import WheelBanditDataset


class TestCoverTypeDataset:
    @pytest.fixture
    def dataset(self) -> CovertypeDataset:
        return CovertypeDataset()

    def test_len(self, dataset: CovertypeDataset) -> None:
        assert len(dataset) == 581012

    def test_getitem(self, dataset: CovertypeDataset) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert X.shape == (7, 7 * 54)
            assert rewards.shape == (7,)

    def test_reward(self, dataset: CovertypeDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, 1)
            assert reward == (dataset.y[i] - 1 == 1)


class TestMNISTDataset:
    @pytest.fixture
    def dataset(self) -> MNISTDataset:
        return MNISTDataset()

    def test_len(self, dataset: MNISTDataset) -> None:
        assert len(dataset) == 70000

    def test_getitem(self, dataset: MNISTDataset) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert X.shape == (10, 10 * 784)
            assert rewards.shape == (10,)

    def test_reward(self, dataset: MNISTDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, 1)
            assert reward == (dataset.y[i] == 1)


class TestStatlogDataset:
    @pytest.fixture
    def dataset(self) -> StatlogDataset:
        return StatlogDataset()

    def test_len(self, dataset: StatlogDataset) -> None:
        assert len(dataset) == 58000

    def test_getitem(self, dataset: StatlogDataset) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert X.shape == (9, 7 * 9)
            assert rewards.shape == (9,)

    def test_reward(self, dataset: StatlogDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, 1)
            assert reward == (dataset.y[i] - 1 == 1)


class TestWheelBanditDataset:
    @pytest.fixture
    def dataset(self) -> WheelBanditDataset:
        return WheelBanditDataset(num_samples=1000, delta=0.8)

    def test_len(self, dataset: WheelBanditDataset) -> None:
        assert len(dataset) == 1000

    def test_getitem(self, dataset: WheelBanditDataset) -> None:
        for _ in range(50):
            X, rewards = dataset[0]
            assert X.shape == (5, 5 * 2)
            assert rewards.shape == (5,)

    def test_reward(self, dataset: WheelBanditDataset) -> None:
        # reward for action 4 should around 1 - 1.2
        for i in range(100):
            reward = dataset.reward(i, 4)
            assert 0.7 <= reward <= 1.5

        for i in range(100):
            reward = dataset.reward(i, 0)
            assert 0.7 <= reward <= 1.5 or 49.5 <= reward <= 50.5


class TestImdbReviewsDataset:
    @pytest.fixture
    def dataset(self) -> ImdbMovieReviews:
        return ImdbMovieReviews(dest_path="./data")

    def test_len(self, dataset: ImdbMovieReviews) -> None:
        assert len(dataset) == 24904

    def test_getitem(self, dataset: ImdbMovieReviews) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert len(X) == 3
            assert all(X_i.shape == (255) for X_i in X)
            assert rewards.shape == (2,)

    def test_reward(self, dataset: ImdbMovieReviews) -> None:
        for i in range(10):
            reward = dataset.reward(i, 1)
            assert reward == (dataset.data["label"][i] == 1)
