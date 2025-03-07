"""A dataset implements the `AbstractDataset` class.

There are currently 6 datasets for the benchmark:
- `CovertypeDataset` - classification of forest cover types

- `ImdbMovieReviews` - sentiment classification of movie reviews

- `MNIST` - classification of 28x28 images of digits

- `MovieLens` - recommendation of movies

- `Statlog (Shuttle)` - classification of different modes of the space shuttle

- `Synthetic` / `Synthetic + Combinatorial`

- `Wheel` - synthetic dataset described [here](https://arxiv.org/abs/1802.09127)

The `AbstractDataset` class is an abstract subclass of `torch.utils.data.Dataset` that provides a common interface for
all datasets. It requires the implementation of the following methods:

- `__len__(self) -> int`: Returns the number of samples in the dataset.

- `__getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]`: Returns the sample at the given index.

- `reward(self, idx: int, action: int) -> float`: Returns the reward for the given index and action.
"""

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset
from calvera.benchmark.datasets.covertype import CovertypeDataset
from calvera.benchmark.datasets.imdb_reviews import ImdbMovieReviews, TextActionInputType
from calvera.benchmark.datasets.mnist import MNISTDataset
from calvera.benchmark.datasets.movie_lens import MovieLensDataset
from calvera.benchmark.datasets.statlog import StatlogDataset
from calvera.benchmark.datasets.synthetic import (
    CubicSyntheticDataset,
    LinearCombinationSyntheticDataset,
    LinearSyntheticDataset,
    QuadraticSyntheticDataset,
    SinSyntheticDataset,
    SyntheticDataset,
)
from calvera.benchmark.datasets.synthetic_combinatorial import SyntheticCombinatorialDataset
from calvera.benchmark.datasets.tiny_imagenet import TinyImageNetDataset
from calvera.benchmark.datasets.wheel import WheelBanditDataset

__all__ = [
    "AbstractDataset",
    "CovertypeDataset",
    "ImdbMovieReviews",
    "MNISTDataset",
    "MovieLensDataset",
    "StatlogDataset",
    "SyntheticDataset",
    "SyntheticCombinatorialDataset",
    "WheelBanditDataset",
    "TextActionInputType",
    "LinearSyntheticDataset",
    "QuadraticSyntheticDataset",
    "CubicSyntheticDataset",
    "SinSyntheticDataset",
    "LinearCombinationSyntheticDataset",
    "TinyImageNetDataset",
]
