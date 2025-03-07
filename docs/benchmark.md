The benchmark is a collection of scripts that can be used to evaluate the performance of a bandit algorithm.

## **Datasets**
A dataset implements the `AbstractDataset` class. There are currently 6 datasets for the benchmark:

- `CovertypeDataset` - classification of forest cover types

- `ImdbMovieReviews` - sentiment classification of movie reviews

- `MNIST` - classification of 28x28 images of digits

- `MovieLens` - recommendation of movies

- `Statlog (Shuttle)` - classification of different modes of the space shuttle

- `Tiny ImageNet` - more difficult image classification task for large image networks.

- `Wheel` - synthetic dataset described [here](https://arxiv.org/abs/1802.09127) 

<br>

::: calvera.benchmark.datasets.abstract_dataset.AbstractDataset
    handler: python
    options:
      heading: AbstractDataset
      show_root_heading: true
      show_root_full_path: false
      members: 
        - __getitem__
        - __len__
        - reward

::: calvera.benchmark.datasets.covertype.CovertypeDataset
    handler: python
    options:
      heading: CovertypeDataset
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: calvera.benchmark.datasets.imdb_reviews.ImdbMovieReviews
    handler: python
    options:
      heading: ImdbMovieReviews
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: calvera.benchmark.datasets.mnist.MNISTDataset
    handler: python
    options:
      heading: MNIST
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: calvera.benchmark.datasets.movie_lens.MovieLensDataset
    handler: python
    options:
      heading: MovieLens
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: calvera.benchmark.datasets.statlog.StatlogDataset
    handler: python
    options:
      heading: StatlogShuttle
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: calvera.benchmark.datasets.tiny_imagenet.TinyImageNetDataset
    handler: python
    options:
      heading: TinyImageNetDataset
      show_root_heading: true
      show_root_full_path: false
      members:
        - __getitem__
        - reward

::: calvera.benchmark.datasets.wheel.WheelBanditDataset
    handler: python
    options:
      heading: WheelBanditDataset
      show_root_heading: true
      show_root_full_path: false
      members: 
        - __getitem__
        - reward

## **Environment**

::: calvera.benchmark.environment.BanditBenchmarkEnvironment
    handler: python
    options:
      heading: BanditBenchmarkEnvironment
      show_root_heading: true
      show_root_full_path: false
      members: 
        - __iter__
        - __next__
        - get_feedback
        - compute_regret