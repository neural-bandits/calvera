import pytest
import torch
from ucimlrepo.fetch import DatasetNotFoundError

from calvera.benchmark.datasets.covertype import CovertypeDataset
from calvera.benchmark.datasets.imdb_reviews import ImdbMovieReviews
from calvera.benchmark.datasets.mnist import MNISTDataset
from calvera.benchmark.datasets.movie_lens import MovieLensDataset
from calvera.benchmark.datasets.statlog import StatlogDataset
from calvera.benchmark.datasets.tiny_imagenet import TinyImageNetDataset
from calvera.benchmark.datasets.wheel import WheelBanditDataset


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
        try:
            ds = StatlogDataset()
        except DatasetNotFoundError as e:
            pytest.skip(f"Skipping StatlogDataset tests: {e}")
        return ds

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
        return WheelBanditDataset(num_samples=100, delta=0.8, seed=42)

    def test_len(self, dataset: WheelBanditDataset) -> None:
        assert len(dataset) == 100

    def test_reproducible(self, dataset: WheelBanditDataset) -> None:
        dataset2 = WheelBanditDataset(num_samples=100, delta=0.8, seed=42)

        assert dataset2[0][0].equal(dataset[0][0])
        assert dataset2[0][1].equal(dataset[0][1])

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
        return ImdbMovieReviews()

    def test_len(self, dataset: ImdbMovieReviews) -> None:
        assert len(dataset) == 24904

    def test_getitem(self, dataset: ImdbMovieReviews) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert len(X) == 3
            assert rewards.shape == (2,)
            assert [X_i.shape == (255,) for X_i in X]

    def test_reward(self, dataset: ImdbMovieReviews) -> None:
        for i in range(10):
            reward = dataset.reward(i, 1)
            assert reward == (dataset.data["sentiment"][i] == 1)


class TestMovieLensDataset:
    @pytest.fixture
    def dataset(self) -> MovieLensDataset:
        return MovieLensDataset()

    @pytest.fixture
    def dataset_concat(self) -> MovieLensDataset:
        return MovieLensDataset(outer_product=False)

    def test_len(self, dataset: MovieLensDataset) -> None:
        assert len(dataset) == 525

    def test_getitem(self, dataset: MovieLensDataset) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert X.shape == (200, 20 * 20)
            assert rewards.shape == (200,)

    def test_getitem_concat(self, dataset_concat: MovieLensDataset) -> None:
        for _ in range(10):
            X, rewards = dataset_concat[0]
            assert X.shape == (200, 40)
            assert rewards.shape == (200,)

    def test_reward(self, dataset: MovieLensDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, 0)
            assert reward == dataset.F[i, 0]


class TestTinyImageNetDataset:
    @pytest.fixture
    def dataset(self) -> TinyImageNetDataset:
        return TinyImageNetDataset(max_classes=100)

    def test_len(self, dataset: TinyImageNetDataset) -> None:
        assert len(dataset) == 100 * 500

    def test_getitem(self, dataset: TinyImageNetDataset) -> None:
        for _ in range(10):
            X, rewards = dataset[0]
            assert X.shape == torch.Size([1, 3 * 64 * 64])
            assert rewards.shape == (100,)

    def test_reward(self, dataset: TinyImageNetDataset) -> None:
        for i in range(10):
            reward = dataset.reward(i, 1)
            assert reward == (dataset.y[i] == 1)
