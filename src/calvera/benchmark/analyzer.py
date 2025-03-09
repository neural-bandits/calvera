import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """Separates out the analysis of CSV logs produced during benchmark training.

    This class reads the CSV logs output by the logger (for example, a CSVLogger)
    and produces metrics, plots, or statistics exactly as you need.

    Keeping analysis separate from training improves modularity.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        results_dir: str = "results",
        bandit_logs_file: str = "metrics.csv",
        metrics_file: str = "env_metrics.csv",
        save_plots: bool = False,
        suppress_plots: bool = False,
    ) -> None:
        """Initializes the BenchmarkAnalyzer.

        Args:
            log_dir: Directory where the logs are stored/outputted to. Default is "logs".
            results_dir: Subdirectory of log_dir where the results are outputted to. Default is "results".
            bandit_logs_file: Name of the metrics file of the CSV Logger. Default is "metrics.csv".
            metrics_file: Name of the metrics file. Default is "env_metrics.csv".
            save_plots: If True, plots will be saved to the results directory. Default is False.
            suppress_plots: If True, plots will not be automatically shown. Default is False.
        """
        if not save_plots and suppress_plots:
            logging.warning("Suppressing plots and not saving them. Results will not be visible.")

        self.log_dir = log_dir
        self.results_dir = os.path.join(log_dir, results_dir)
        self.bandit_logs_file = bandit_logs_file
        self.env_metrics_file = metrics_file
        self.suppress_plots = suppress_plots
        self.save_plots = save_plots

        self.env_metrics_df = pd.DataFrame()
        self.bandit_logs_df = pd.DataFrame()

    def load_metrics(self, log_path: str, bandit: str = "bandit") -> None:
        """Loads the logs from the log path.

        Args:
            log_path: Path to the log data.
            bandit: A name of the bandit. Default is "bandit".
        """
        new_metrics_df = self._load_df(log_path, self.env_metrics_file)

        if new_metrics_df is not None:
            new_metrics_df["bandit"] = bandit

            self.env_metrics_df = pd.concat([self.env_metrics_df, new_metrics_df], ignore_index=True)

        bandit_metrics_df = self._load_df(log_path, self.bandit_logs_file)
        if bandit_metrics_df is not None:
            bandit_metrics_df["bandit"] = bandit

            self.bandit_logs_df = pd.concat([self.bandit_logs_df, bandit_metrics_df], ignore_index=True)

    def _load_df(self, log_path: str, file_name: str) -> pd.DataFrame | None:
        """Loads the logs from the log path.

        Args:
            log_path: Path to the log data.
            file_name: Name of the file to load.

        Returns:
            A pandas DataFrame containing the logs.
        """
        try:
            return pd.read_csv(os.path.join(log_path, file_name))
        except FileNotFoundError:
            logger.warning(f"Could not find metrics file {file_name} in {log_path}.")
            return None

    def plot_accumulated_metric(self, metric_name: str | list[str], comparison_key: str | None = None) -> None:
        """Plots the accumulated metric over training steps.

        If several metrics are passed they are all plotted in the same plot.
        If the analyzer has seen data from several bandits they are plotted in the same plot.

        Args:
            metric_name: The name(s) of the metric(s) to plot.
            comparison_key: The key to compare the metrics by. Default is None.
        """
        if isinstance(metric_name, str):
            metric_name = [metric_name]

        if any(name not in self.env_metrics_df.columns for name in metric_name):
            logger.warning(f"\One of {','.join(metric_name)} data not found in logs.")
            return

        if self.env_metrics_df["bandit"].nunique() > 1 and len(metric_name) > 1:
            raise ValueError("Cannot plot multiple metrics for multiple bandits.")

        plt.figure(figsize=(10, 5))
        if self.env_metrics_df["bandit"].nunique() > 1:
            for bandit_name, bandit_df in self.env_metrics_df.groupby("bandit"):
                accumulated_metric = bandit_df[metric_name[0]].fillna(0).cumsum()
                plt.plot(bandit_df["step"], accumulated_metric, label=bandit_name)
                plt.ylabel(f"Accumulated {metric_name[0]}")
        else:
            for metric in metric_name:
                accumulated_metric = self.env_metrics_df[metric].fillna(0).cumsum()
                plt.plot(self.env_metrics_df["step"], accumulated_metric, label=metric)

        plt.xlabel("Step")
        if comparison_key is not None:
            plt.legend(title=comparison_key)
        else:
            plt.legend()
        plt.title(f"Accumulated {', '.join(metric_name)} over training steps")

        if self.save_plots:
            path = os.path.join(self.results_dir, f"acc_{'_'.join(metric_name)}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        if not self.suppress_plots:
            plt.show()

    def plot_average_metric(self, metric_name: str, comparison_key: str | None = None) -> None:
        """Plots the average metric over training steps.

        Args:
            metric_name: The name of the metric to plot.
            comparison_key: The key to compare the metrics by. Default is None.
        """
        if metric_name not in self.env_metrics_df.columns:
            logger.warning(f"\nNo {metric_name} data found in logs.")
            return

        # Plot how average changes over time
        plt.figure(figsize=(10, 5))

        for bandit_name, bandit_df in self.env_metrics_df.groupby("bandit"):
            valid_idx = bandit_df[metric_name].dropna().index
            accumulated_metric = bandit_df.loc[valid_idx, metric_name].cumsum()
            steps = bandit_df.loc[valid_idx, "step"]

            # Plot how average changes over time
            plt.plot(steps, accumulated_metric / (steps + 1), label=bandit_name)

        plt.ylabel(f"Average {metric_name}")
        plt.xlabel("Step")
        if comparison_key is not None:
            plt.legend(title=comparison_key)
        else:
            plt.legend()
        plt.title(f"Average {metric_name} over training steps")

        if self.save_plots:
            path = os.path.join(self.results_dir, f"avg_{metric_name}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        if not self.suppress_plots:
            plt.show()

    def plot_loss(self) -> None:
        """Plots the loss over training steps."""
        # Generate a plot for the loss
        if "loss" not in self.bandit_logs_df.columns:
            logger.warning("\nNo loss data found in logs.")
            return

        plt.figure(figsize=(10, 5))
        for bandit_name, bandit_df in self.bandit_logs_df.groupby("bandit"):
            loss = bandit_df["loss"].dropna()
            if loss.empty:
                logger.warning(f"No loss data found in logs for {bandit_name}")
                continue
            plt.plot(bandit_df["step"], loss, label=bandit_name)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over training steps")

        if self.save_plots:
            path = os.path.join(self.results_dir, "loss.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        if not self.suppress_plots:
            plt.show()

    def log_metrics(self, bandit: str = "bandit") -> None:
        """Logs the metrics of the bandits run to the console.

        Args:
            bandit: The name of the bandit. Default is "bandit".
        """
        if self.env_metrics_df.empty:
            raise ValueError("No metrics found in logs. Please call load_metrics() first.")

        bandit_df = self.env_metrics_df[self.env_metrics_df["bandit"] == bandit]

        if bandit_df.empty:
            raise ValueError(f"No metrics found for {bandit}.")

        str = f"Metrics of {bandit}:\n"
        str += f"Avg Regret: {bandit_df['regret'].mean()}\n"
        str += f"Avg Reward: {bandit_df['reward'].mean()}\n"
        str += f"Accumulated Regret: {bandit_df['regret'].sum()}\n"
        str += f"Accumulated Reward: {bandit_df['reward'].sum()}\n"

        # log avg_regret from first 10, 100, 1000, 10000, ... steps
        i = 1
        while True:
            steps = 10**i
            if steps >= bandit_df["step"].max():
                break

            avg_regret = bandit_df[bandit_df["step"] < steps]["regret"].mean()
            str += f"Avg Regret (first {steps} steps): {avg_regret}\n"

            i += 1

        # log from last steps
        i = 1
        while True:
            steps = 10**i
            if steps >= bandit_df["step"].max():
                break

            avg_regret = bandit_df[bandit_df["step"] > bandit_df["step"].max() - steps]["regret"].mean()
            str += f"Avg Regret (last {steps} steps): {avg_regret}\n"

            i += 1

        print(str)

        # Write to file
        if self.save_plots:
            bandit_dir = os.path.join(self.results_dir, bandit) if bandit is not None else self.results_dir
            os.makedirs(bandit_dir, exist_ok=True)
            path = os.path.join(bandit_dir, "metrics.txt")
            with open(path, "w+") as f:
                f.write(str)
