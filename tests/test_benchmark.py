from neural_bandits.benchmark.benchmark import run


def test_benchmark() -> None:
    # we just test that the benchmark runs without errors
    run(
        "lin_ucb",
        "covertype",
        {
            "max_samples": 16,
            "batch_size": 1,
            "forward_batch_size": 1,
            "feedback_delay": 1,
        },  # training parameters
        {
            "alpha": 1.0,
        },  # bandit hyperparameters
        supress_plots=True,
    )
