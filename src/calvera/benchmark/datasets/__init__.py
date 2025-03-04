"""A dataset implements the `AbstractDataset` class.

There are currently 6 datasets for the benchmark:
- `CovertypeDataset` - classification of forest cover types

- `ImdbMovieReviews` - sentiment classification of movie reviews

- `MNIST` - classification of 28x28 images of digits

- `MovieLens` - recommendation of movies

- `Statlog (Shuttle)` - classification of different modes of the space shuttle

- `Wheel` - synthetic dataset described [here](https://arxiv.org/abs/1802.09127)

The `AbstractDataset` class is an abstract subclass of `torch.utils.data.Dataset` that provides a common interface for
all datasets. It requires the implementation of the following methods:

- `__len__(self) -> int`: Returns the number of samples in the dataset.

- `__getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]`: Returns the sample at the given index.

- `reward(self, idx: int, action: int) -> float`: Returns the reward for the given index and action.
"""
