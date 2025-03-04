The benchmark is a collection of scripts that can be used to evaluate the performance of a bandit algorithm.

## **Datasets**
A dataset implements the `AbstractDataset` class. There are currently 6 datasets for the benchmark:

- `CovertypeDataset` - classification of forest cover types

- `ImdbMovieReviews` - sentiment classification of movie reviews

- `MNIST` - classification of 28x28 images of digits

- `MovieLens` - recommendation of movies

- `Statlog (Shuttle)` - classification of different modes of the space shuttle

- `Wheel` - synthetic dataset described [here](https://arxiv.org/abs/1802.09127) 

<br>

::: neural_bandits.benchmark.datasets.abstract_dataset.AbstractDataset
    handler: python
    options:
      heading: AbstractDataset
      show_root_heading: true
      show_root_full_path: false
      members: 
        - __getitem__
        - __len__
        - reward

::: neural_bandits.benchmark.datasets.covertype.CovertypeDataset
    handler: python
    options:
      heading: CovertypeDataset
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: neural_bandits.benchmark.datasets.imdb_reviews.ImdbMovieReviews
    handler: python
    options:
      heading: ImdbMovieReviews
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: neural_bandits.benchmark.datasets.mnist.MNISTDataset
    handler: python
    options:
      heading: MNIST
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: neural_bandits.benchmark.datasets.movie_lens.MovieLensDataset
    handler: python
    options:
      heading: MovieLens
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: neural_bandits.benchmark.datasets.statlog.StatlogDataset
    handler: python
    options:
      heading: StatlogShuttle
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: neural_bandits.benchmark.datasets.wheel.WheelBanditDataset
    handler: python
    options:
      heading: WheelBanditDataset
      show_root_heading: true
      show_root_full_path: false
      members: 
        - __getitem__
        - reward

## **Environment**

The `BanditBenchmarkEnvironment` class is used to simulate a bandit environment. It is used to evaluate the performance of a bandit algorithm.
TODO