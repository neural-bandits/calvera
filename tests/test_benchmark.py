from neural_bandits.benchmark.benchmark import filter_kwargs, run, run_comparison


def test_lin_ucb_benchmark() -> None:
    # we just test that the benchmark runs without errors
    run(
        {
            "bandit": "lin_ucb",
            "dataset": "covertype",
            "max_samples": 100,
            "feedback_delay": 1,
            "train_batch_size": 1,
            "forward_batch_size": 1,
            "bandit_hparams": {
                "exploration_rate": 1.0,
            },
        },
        suppress_plots=True,
    )


def test_neural_ts_benchmark() -> None:
    # we just test that the benchmark runs without errors
    run(
        {
            "bandit": "neural_ts",
            "dataset": "covertype",
            "network": "tiny_mlp",
            "max_samples": 100,
            "feedback_delay": 1,
            "train_batch_size": 1,
            "forward_batch_size": 1,
            "data_strategy": "sliding_window",
            "sliding_window_size": 1,
            "bandit_hparams": {},
        },
        suppress_plots=True,
    )


def test_neural_linear_benchmark() -> None:
    # we just test that the benchmark runs without errors
    run(
        {
            "bandit": "neural_linear",
            "dataset": "covertype",
            "network": "tiny_mlp",
            "max_samples": 100,
            "feedback_delay": 1,
            "train_batch_size": 1,
            "forward_batch_size": 1,
            "data_strategy": "sliding_window",
            "sliding_window_size": 1,
            "bandit_hparams": {
                "n_embedding_size": 128,
            },
        },
        suppress_plots=True,
    )


def test_run_comparison() -> None:
    run_comparison(
        {
            "bandit": ["neural_ucb", "neural_ts"],
            "dataset": "covertype",
            "network": "tiny_mlp",
            "max_samples": 100,
            "feedback_delay": 1,
            "train_batch_size": 1,
            "forward_batch_size": 1,
            "data_strategy": "all",
            "bandit_hparams": {
                "n_embedding_size": 128,
            },
        },
        suppress_plots=True,
    )


def test_filter_kwargs() -> None:
    class A:
        def __init__(self, a: int, b: int) -> None:
            self.a = a
            self.b = b

    kwargs = {"a": 1, "b": 2, "c": 3}
    filtered_kwargs = filter_kwargs(A, kwargs)
    assert filtered_kwargs == {"a": 1, "b": 2}
    a = A(**filtered_kwargs)
    assert a.a == 1
    assert a.b == 2
    assert hasattr(a, "c") is False
