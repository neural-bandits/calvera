<!-- ## Working with Custom Datasets

Calvera allows you to use your own datasets by implementing the `AbstractDataset` interface. This is useful when you want to apply bandit algorithms to your specific domain or problem.

### Creating a Custom Dataset

To create a custom dataset, you need to subclass `AbstractDataset` and implement the required methods:

```python
import torch
from calvera.benchmark.datasets import AbstractDataset

class MyCustomDataset(AbstractDataset[torch.Tensor]):
    """Custom dataset implementation for Calvera."""

    # Define these properties for your dataset
    num_actions: int = 5  # Number of available actions
    context_size: int = 20  # Size of the context vector

    def __init__(self, data_path: str = "./my_data.csv", needs_disjoint_contextualization: bool = True):
        """Initialize your custom dataset.

        Args:
            data_path: Path to your dataset file
            needs_disjoint_contextualization: Whether to use disjoint model contextualization
        """
        super().__init__(needs_disjoint_contextualization=needs_disjoint_contextualization)

        # Load and preprocess your data
        # This is just an example, adapt to your data source
        import pandas as pd
        self.data = pd.read_csv(data_path)

        # Convert features to tensor
        self.X = torch.tensor(self.data['features'].values.tolist(), dtype=torch.float32)

        # Convert labels/rewards to tensor (if available)
        if 'rewards' in self.data:
            self.rewards = torch.tensor(self.data['rewards'].values.tolist(), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the contextualized actions and rewards for a given index.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            contextualized_actions: The contextualized actions for the given index.
            rewards: The rewards for each action.
        """
        context = self.X[idx].reshape(1, -1)
        contextualized_actions = self.contextualizer(context).squeeze(0)

        # Either use precomputed rewards or generate them on-the-fly
        if hasattr(self, 'rewards'):
            rewards = self.rewards[idx]
        else:
            # Generate rewards based on some reward model
            rewards = torch.tensor(
                [self.reward(idx, action) for action in range(self.num_actions)],
                dtype=torch.float32,
            )

        return contextualized_actions, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a given index and action.

        In a real application, this could be a function of the features and action,
        or it could be pre-computed from historical data.

        Args:
            idx: The index of the context in this dataset.
            action: The action for which to compute the reward.

        Returns:
            The reward value as a float.
        """
        # This is just an example, implement your own reward logic
        # For instance, you might have a reward model that predicts rewards
        # based on the context and action
        if hasattr(self, 'rewards') and self.rewards.shape[1] > action:
            return float(self.rewards[idx, action])
        else:
            # Fallback reward logic (simple example)
            feature_sum = self.X[idx].sum().item()
            return float(0.5 + 0.1 * action * feature_sum)
```

### Using a Custom Dataset with Bandits

Once you've implemented your custom dataset, you can use it with Calvera bandits just like the built-in datasets:

```python
from torch.utils.data import DataLoader
from calvera.benchmark import BanditBenchmarkEnvironment
from calvera.bandits import LinearUCBBandit

# Create your custom dataset
my_dataset = MyCustomDataset(data_path="./my_domain_data.csv")

# Create DataLoader
batch_size = 32
train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# Create environment
env = BanditBenchmarkEnvironment(train_loader)

# Initialize a bandit
bandit = LinearUCBBandit(
    n_features=my_dataset.context_size,
    exploration_rate=1.0,
)

# Now you can run the training loop with your custom dataset
# (similar to the examples earlier in this guide)
```

### Logging Data from Custom Datasets

When using custom datasets, you might want to log additional metrics specific to your domain:

```python
import pandas as pd
from lightning.pytorch.loggers import CSVLogger
from calvera.benchmark import OnlineBanditLoggerDecorator

# Create a logger
logger = OnlineBanditLoggerDecorator(
    CSVLogger("logs", name="my_custom_dataset_experiment"),
    enable_console_logging=True,
)

# During training, log domain-specific metrics
for contextualized_actions in env:
    # ... perform bandit selection and get rewards ...

    # Log custom metrics
    logger.log_metrics(
        {
            "domain_specific_metric": calculated_value,
            "conversion_rate": conversions / total_attempts,
            # ... other custom metrics ...
        },
        step=current_step
    )

# After training, analyze your custom metrics
results = pd.read_csv(f"{logger._logger_wrappee.log_dir}/metrics.csv")
domain_metric_over_time = results["domain_specific_metric"]
``` -->