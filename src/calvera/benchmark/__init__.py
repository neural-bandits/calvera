"""The subpackage containing the benchmarking functionality.

This subpackage contains the utilities for benchmarking the bandit algorithms. The main class is the `BanditBenchmark`.
We also provide an `Environment`, a special logger `OnlineBanditLoggerDecorator`, a `MultiClassContextualizer` for
disjoint model contextualization and a set of datasets in the subpackage `datasets`.
"""

from calvera.benchmark.analyzer import BenchmarkAnalyzer
from calvera.benchmark.benchmark import BanditBenchmark, run, run_comparison, run_from_yaml
from calvera.benchmark.environment import BanditBenchmarkEnvironment
from calvera.benchmark.logger_decorator import OnlineBanditLoggerDecorator
from calvera.benchmark.network_wrappers import BertWrapper, ResNetWrapper

__all__ = [
    "BanditBenchmark",
    "BanditBenchmarkEnvironment",
    "BenchmarkAnalyzer",
    "OnlineBanditLoggerDecorator",
    "ResNetWrapper",
    "BertWrapper",
    "run",
    "run_comparison",
    "run_from_yaml",
]
