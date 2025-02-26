# Welcome to the documentation of the Calvera library

Calvera is a Python library for offering a small collection of multi-armed bandit algorithms.
Currently the following algorithms are implemented:

- Linear Thompson Sampling

- Linear UCB

- Neural Linear

- Neural Thompson Sampling

- Neural UCB

By using different `selectors` these algorithms can be adapted.
Selectors are classes that determine which arm is pulled based on the scores of the arms.
You can provide a selector to the different algorithm classes to modify the selection strategy.
The following selectors are available:

- `ArgMaxSelector`: Selects the arm with the highest score.

- `EpsilonGreedySelector`: Selects the arm with the highest score with probability `1-epsilon` or a random arm with probability `epsilon`.

- `TopKSelector`: Selects the top `k` arms with the highest scores.


## Installation

Calvera is (will be) available on [PyPI](https://pypi.org/).
```bash
pip install calvera
```

## Usage
The general usage assumes t




## Contributing
Contributions are always welcome! Please refer to the [contribution guidelines](CONTRIBUTING.md) for more information.

As of 26th February 2025, the library is under active development. Current contributors are:

- [Philipp Kolbe](mailto:philipp.kolbe@student.hpi.uni-potsdam.de)

- [Robert Weeke](mailto:robert.weeke@student.hpi.uni-potsdam.de)

- [Parisa Shahabinejad](mailto:parisa.shahabinejad@student.hpi.uni-potsdam.de)

### License

Calvera is licensed under the MIT license. See the [LICENSE](https://github.com/neural-bandits/calvera/blob/main/LICENSE) file for more details.

### Contact

If you have further questions or feedback, you are welcome to contact one of the authors directly.

- [Philipp Kolbe](mailto:philipp.kolbe@student.hpi.uni-potsdam.de)

- [Robert Weeke](mailto:robert.weeke@student.hpi.uni-potsdam.de)

- [Parisa Shahabinejad](mailto:parisa.shahabinejad@student.hpi.uni-potsdam.de)
