# Calvera Usage Examples

This guide demonstrates how to use the Calvera library for multi-armed bandit algorithms. We'll cover how to use various bandit algorithms, from simple linear bandits to more complex neural bandits, and show how to integrate them into your workflows.

## Quick Start

Let's start with a simple example using Linear Thompson Sampling:

```python
import torch
from calvera.bandits import LinearTSBandit

# 1. Create a bandit for a linear model with 128 features
N_FEATURES = 128
bandit = LinearTSBandit(n_features=N_FEATURES)

# 2. Generate sample data (batch_size, n_actions, n_features) and perform inference
data = torch.randn(100, 1, N_FEATURES)
chosen_arms_one_hot, probabilities = bandit(data)
chosen_arms = chosen_arms_one_hot.argmax(dim=1)

# 3. Retrieve rewards for the chosen arms
rewards = torch.randn(100, 1)

# 4. Add the data to the bandit
chosen_contextualized_actions = data[:, :, chosen_arms]
bandit.record_feedback(chosen_contextualized_actions, rewards)

# 5. Train the bandit
trainer = pl.Trainer(
    max_epochs=1,
    enable_progress_bar=False,
    enable_model_summary=False,
    accelerator=accelerator,
)
trainer.fit(bandit)

# (6. Repeat the process as needed)
```

This simple example demonstrates the core workflow with Calvera bandits:

1. Create a bandit model
2. Perform inference on your data to get action selections
3. Collect rewards for chosen actions
4. Record the data in the bandit's buffer
5. Train the bandit using PyTorch Lightning

Now, let's explore the different types of bandits in more detail.

## Linear Bandits

Linear bandits model the expected reward as a linear function of the context. Calvera provides two main variants: Linear Thompson Sampling and Linear UCB.

### Linear Thompson Sampling

Linear Thompson Sampling (LinTS) maintains a Bayesian posterior over the model parameters and, at each time step, plays an arm according to its posterior probability of being the best arm.

Let's look at a practical example using the [StatLog](https://archive.ics.uci.edu/dataset/148/statlog+shuttle) dataset:

```python
import torch
import lightning as pl
from torch.utils.data import DataLoader, Subset
from calvera.bandits import LinearTSBandit
from calvera.benchmark.datasets import StatlogDataset
from calvera.benchmark import BanditBenchmarkEnvironment
from calvera.utils import ArgMaxSelector

# Load the StatLog dataset
dataset = StatlogDataset()
print(f"Dataset context size: {dataset.context_size}")
print(f"Dataset sample count: {len(dataset)}")

# Create data loader for a subset of the data
train_loader = DataLoader(Subset(dataset, range(10000)), batch_size=32, shuffle=True)

# Set up the environment
accelerator = "cpu"
env = BanditBenchmarkEnvironment(train_loader, device=accelerator)

# Initialize the Linear TS bandit
bandit_module = LinearTSBandit(
    n_features=dataset.context_size,
    selector=ArgMaxSelector(),
    lazy_uncertainty_update=True,
).to(accelerator)
```

Now we can run the training loop:

```python
import numpy as np
import pandas as pd
from tqdm import tqdm

rewards = np.array([])
regrets = np.array([])
progress = tqdm(iter(env), total=len(env))

for contextualized_actions in progress:
    # 1. Select actions
    chosen_actions, _ = bandit_module.forward(contextualized_actions)

    # 2. Create a trainer for this step
    trainer = pl.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=accelerator,
    )

    # 3. Get feedback from environment
    chosen_contextualized_actions, realized_rewards = env.get_feedback(chosen_actions)
    batch_regret = env.compute_regret(chosen_actions)

    # 4. Track metrics
    rewards = np.append(rewards, realized_rewards.cpu().numpy())
    regrets = np.append(regrets, batch_regret.cpu().numpy())
    progress.set_postfix({
        "reward": realized_rewards.mean().item(),
        "regret": batch_regret.mean().item(),
        "avg_regret": regrets.mean()
    })

    # 5. Update the bandit
    bandit_module.record_feedback(chosen_contextualized_actions, realized_rewards)
    trainer.fit(bandit_module)
    bandit_module = bandit_module.to(accelerator)

# Store metrics
metrics = pd.DataFrame({
    "reward": rewards,
    "regret": regrets,
})
```

We can then analyze and visualize our results:

```python
import matplotlib.pyplot as plt

# Calculate cumulative metrics
cumulative_reward = np.cumsum(metrics["reward"])
cumulative_regret = np.cumsum(metrics["regret"])

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(cumulative_reward, label="reward")
plt.plot(cumulative_regret, label="regret")
plt.xlabel("steps")
plt.ylabel("cumulative reward/regret")
plt.legend()
plt.show()

# Print average metrics at different time horizons
print(f"Average reward (first 10 rounds): {np.mean(metrics['reward'][:10]):.4f}")
print(f"Average reward (first 100 rounds): {np.mean(metrics['reward'][:100]):.4f}")
print(f"Average reward (all rounds): {np.mean(metrics['reward']):.4f}")
print("")
print(f"Average regret (first 10 rounds): {np.mean(metrics['regret'][:10]):.4f}")
print(f"Average regret (first 100 rounds): {np.mean(metrics['regret'][:100]):.4f}")
print(f"Average regret (all rounds): {np.mean(metrics['regret']):.4f}")
```

### Linear UCB

The Linear UCB (LinUCB) algorithm uses an upper confidence bound strategy to balance exploration and exploitation.

The interface is similar to Linear TS, with the main difference being the algorithm used:

```python
from calvera.bandits import LinearUCBBandit

# Initialize the Linear UCB bandit
bandit_module = LinearUCBBandit(
    n_features=dataset.context_size,
    selector=ArgMaxSelector(),
    exploration_rate=1.0,  # Controls the amount of exploration
    lazy_uncertainty_update=True,
).to(accelerator)

# The rest of the code remains the same as in the Linear TS example
```

[View complete notebook on GitHub](https://github.com/neural-bandits/calvera/blob/main/examples/linear.ipynb)

## Neural Linear

Neural Linear performs a Bayesian linear regression on top of the representation of the last layer of a neural network.

Let's see an example:

```python
import torch.nn as nn
from calvera.bandits import NeuralLinearBandit
from calvera.utils import InMemoryDataBuffer, AllDataBufferStrategy

# Define a neural network architecture
class Network(nn.Module):
    def __init__(self, dim, hidden_size=100, n_embedding_size=10):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_embedding_size)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

# Load dataset
dataset = StatlogDataset()
network = Network(dataset.context_size, hidden_size=100, n_embedding_size=10)

# Set up buffer for storing interaction data
buffer = InMemoryDataBuffer(
    buffer_strategy=AllDataBufferStrategy(),
    max_size=10000,
)

# Create data loader
train_loader = DataLoader(Subset(dataset, range(10000)), batch_size=256, shuffle=True)
env = BanditBenchmarkEnvironment(train_loader)

# Initialize the Neural Linear bandit
bandit_module = NeuralLinearBandit(
    n_embedding_size=10,
    network=network,
    buffer=buffer,
    train_batch_size=32,
    early_stop_threshold=1e-3,
    weight_decay=1e-3,
    learning_rate=1e-3,
    min_samples_required_for_training=1024,
    initial_train_steps=2048,
)
```

Now we can run the training loop, which is similar to the Linear TS example, but with a few differences:

```python
from lightning.pytorch.loggers.csv_logs import CSVLogger
from calvera.benchmark import OnlineBanditLoggerDecorator

# Set up logger
logger = OnlineBanditLoggerDecorator(
    CSVLogger("logs", name="neural_linear_bandit", flush_logs_every_n_steps=100),
    enable_console_logging=False,
)

rewards = np.array([])
regrets = np.array([])
progress_bar = tqdm(enumerate(env), total=len(env))

for contextualized_actions in progress_bar:
    # 1. Select actions
    chosen_actions, _ = bandit_module.forward(contextualized_actions)

    # 2. Create a trainer for this step
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=1000,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    # 3. Get feedback from environment
    chosen_contextualized_actions, realized_rewards = env.get_feedback(chosen_actions)
    batch_regret = env.compute_regret(chosen_actions)

    # 4. Track metrics
    rewards = np.append(rewards, realized_rewards.cpu().numpy())
    regrets = np.append(regrets, batch_regret.cpu().numpy())
    progress_bar.set_postfix(
        reward=realized_rewards.mean().item(),
        regret=batch_regret.mean().item(),
        average_regret=regrets.mean(),
    )

    # 5. Update the bandit
    bandit_module.record_feedback(chosen_contextualized_actions, realized_rewards)
    trainer.fit(bandit_module)
```

The key differences in Neural Linear compared to Linear bandits:

1. We need to define a neural network architecture
2. We use a buffer to store interaction data
3. The training process involves both updating the neural network and the linear head

In Neural Linear, the neural network and the Bayesian linear regression components are updated at different time scales

[View complete notebook on GitHub](https://github.com/neural-bandits/calvera/blob/main/examples/neural_linear.ipynb).

[Extension of the above notebook to Sentiment Analysis using a simple BertModel](https://github.com/neural-bandits/calvera/blob/main/examples/neural_linear_bert.ipynb).

## Neural Bandits

Neural bandits use neural networks to model the expected reward function, which allows them to capture non-linear relationships between contexts and rewards.

### Neural Thompson Sampling

Neural Thompson Sampling (NeuralTS) uses deep neural networks for both exploration and exploitation by sampling rewards from a posterior distribution based on the network's uncertainty.

Here's how to use it:

```python
from calvera.bandits import NeuralTSBandit

# Define a simple network architecture
class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

# Load dataset and create network
dataset = StatlogDataset()
network = Network(dataset.context_size, hidden_size=100)

# Set up buffer
buffer = InMemoryDataBuffer(
    buffer_strategy=AllDataBufferStrategy(),
    max_size=10000,
)

# Create data loader
train_loader = DataLoader(Subset(dataset, range(10000)), batch_size=256, shuffle=True)
env = BanditBenchmarkEnvironment(train_loader)

# Initialize the Neural TS bandit
bandit_module = NeuralTSBandit(
    n_features=dataset.context_size,
    network=network,
    buffer=buffer,
    train_batch_size=32,
    early_stop_threshold=1e-3,
    weight_decay=1e-3,
    exploration_rate=1e-5,
    learning_rate=1e-3,
    min_samples_required_for_training=1024,
    initial_train_steps=2048,
)
```

The training loop is very similar to the Neural Linear case:

```python
# Set up logger
logger = OnlineBanditLoggerDecorator(
    CSVLogger("logs", name="neural_ts_bandit", flush_logs_every_n_steps=100),
    enable_console_logging=False
)

# Then run the same training loop as with Neural Linear
```

[View complete notebook on GitHub](https://github.com/neural-bandits/calvera/blob/main/examples/neuralts.ipynb)

### Neural UCB

Neural UCB (NeuralUCB) uses neural networks to estimate rewards and guides exploration through upper confidence bounds based on the network's uncertainty.

Here's how to use it:

```python
from calvera.bandits import NeuralUCBBandit

# Network architecture is the same as for Neural TS

# Initialize the Neural UCB bandit
bandit_module = NeuralUCBBandit(
    n_features=dataset.context_size,
    network=network,
    buffer=buffer,
    train_batch_size=32,
    early_stop_threshold=1e-3,
    weight_decay=1e-3,
    exploration_rate=1e-5,  # Controls the amount of exploration
    learning_rate=1e-3,
    min_samples_required_for_training=128,
    initial_train_steps=2048,
)

# The training loop is the same as for Neural TS
```

The main difference between NeuralUCB and NeuralTS is how they calculate action selection scores:

- NeuralUCB adds an uncertainty bonus to the predicted reward
- NeuralTS samples from an approximated posterior over the neural network outputs

[View complete notebook on GitHub](https://github.com/neural-bandits/calvera/blob/main/examples/neuralucb.ipynb)

## Combinatorial Bandits

Combinatorial bandits extend the standard multi-armed bandit framework by allowing the selection of multiple arms simultaneously. This is particularly useful in settings where actions are combinations (e.g., recommending a slate of items) and the reward depends on the joint selection.

### Combinatorial Neural Thompson Sampling

Combinatorial Neural Thompson Sampling adapts the neural network-based Thompson Sampling algorithm to the combinatorial setting.

Let's explore an example using a synthetic dataset and Neural Thompson Sampling:

```python
from calvera.benchmark.datasets import SyntheticCombinatorialDataset
from calvera.bandits import NeuralTSBandit
from calvera.utils import TopKSelector


# Define a Neural Network for the Combinatorial Bandit
class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

# Create a Synthetic Combinatorial Dataset
dataset = SyntheticCombinatorialDataset(
    n_samples=2000,
    num_actions=20,
    context_size=40,
    function_type="quadratic",
    noise_std=0,
    seed=42
)
```

Now, let's set up the Combinatorial Neural Thompson Sampling bandit:

```python
# Set up the environment
train_loader = DataLoader(Subset(dataset, range(2000)), batch_size=100, shuffle=True)
env = BanditBenchmarkEnvironment(train_loader)

# Create a data buffer
buffer = InMemoryDataBuffer(
    buffer_strategy=AllDataBufferStrategy(),
    max_size=None,
)

# Initialize the Neural TS bandit with TopK selector
bandit_module = NeuralTSBandit(
    n_features=dataset.context_size,
    network=Network(dataset.context_size, hidden_size=100),
    selector=TopKSelector(k=K),
    buffer=buffer,
    train_batch_size=32,
    early_stop_threshold=1e-4,
    weight_decay=1e-4,
    exploration_rate=1,
    learning_rate=1e-2,
    min_samples_required_for_training=16,
    initial_train_steps=1024,
    num_samples_per_arm=10,
)

# Training Loop
rewards = np.array([])
regrets = np.array([])
progress_bar = tqdm(enumerate(env), total=len(env))

for i, contextualized_actions in progress_bar:
    # Select actions
    chosen_actions, _ = bandit_module.forward(contextualized_actions)

    # Create a trainer
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=1000,
        gradient_clip_val=20.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    # Get feedback from environment
    chosen_contextualized_actions, realized_scores = env.get_feedback(chosen_actions)
    realized_rewards = realized_scores.sum(dim=1)
    batch_regret = env.compute_regret(chosen_actions)

    # Track metrics
    rewards = np.append(rewards, realized_rewards.cpu().numpy())
    regrets = np.append(regrets, batch_regret.cpu().numpy())
    progress_bar.set_postfix(
        reward=realized_rewards.mean().item(),
        regret=batch_regret.mean().item(),
        avg_regret=np.mean(regrets),
        acc_regret=np.sum(regrets),
    )

    # Update the bandit
    bandit_module.record_feedback(chosen_contextualized_actions, realized_scores)
    trainer.fit(bandit_module)
```

Note that we are defining a NeuralTSBandit module here, and the key difference is the use of a TopKSelector, which enables the selection of multiple actions simultaneously. This selector can be customized based on your specific requirements to match your particular use case. (See [Customization](#customization) for more details on adapting selectors.)

[View complete notebook on GitHub](https://github.com/neural-bandits/calvera/blob/main/examples/combinatorial_neuralts.ipynb)

### Combinatorial Neural UCB

Similar to Combinatorial Neural Thompson Sampling, the Combinatorial Neural UCB (NeuralUCB) approach can be applied to combinatorial action spaces:

```python
from calvera.bandits import NeuralUCBBandit

# Initialize the Neural UCB bandit
bandit_module = NeuralUCBBandit(
    n_features=dataset.context_size,
    network=Network(dataset.context_size, hidden_size=100),
    selector=TopKSelector(k=K),
    buffer=buffer,
    train_batch_size=32,
    early_stop_threshold=1e-4,
    weight_decay=1e-4,
    exploration_rate=1,
    learning_rate=1e-2,
    min_samples_required_for_training=16,
    initial_train_steps=1024,
)

# The training loop remains the same as the Neural Thompson Sampling example
```

[View complete notebook on GitHub](https://github.com/neural-bandits/calvera/blob/main/examples/combinatorial_neuralucb.ipynb)

## Customization

Calvera is designed to be highly customizable, allowing you to adapt it to your specific needs.

### Custom Selectors

Selectors determine how actions are chosen based on the scores produced by the bandit. Calvera provides three built-in selectors:

1. **ArgMaxSelector**: Chooses the arm with the highest score
2. **EpsilonGreedySelector**: Uses epsilon-greedy exploration
3. **TopKSelector**: Selects the top k arms (for combinatorial bandits)

Here's how to use them:

```python
from calvera.utils import ArgMaxSelector, EpsilonGreedySelector, TopKSelector

# Use ArgMaxSelector (default if none specified)
bandit = LinearTSBandit(
    n_features=n_features,
    selector=ArgMaxSelector()
)

# Use EpsilonGreedySelector for exploration
bandit = LinearTSBandit(
    n_features=n_features,
    selector=EpsilonGreedySelector(epsilon=0.1)
)

# Use TopKSelector for combinatorial bandits
bandit = LinearTSBandit(
    n_features=n_features,
    selector=TopKSelector(k=3)  # Select top 3 arms
)
```

You can also implement custom selectors by subclassing `AbstractSelector`:

```python
from calvera.utils import AbstractSelector

class MyCustomSelector(AbstractSelector):
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        # Your custom selection logic here
        # scores shape: (batch_size, n_arms)
        # should return one-hot encoding of shape (batch_size, n_arms)
        pass
```

### Data Storage Strategies

Calvera provides different strategies for managing interaction data:

1. **AllDataBufferStrategy**: Stores all data
2. **SlidingWindowBufferStrategy**: Stores only the most recent data

Here's how to use them:

```python
from calvera.utils import InMemoryDataBuffer, AllDataBufferStrategy, SlidingWindowBufferStrategy

# Store all data
buffer = InMemoryDataBuffer(
    buffer_strategy=AllDataBufferStrategy(),
    max_size=10000,  # Optional: limit to 10000 samples
)

# Store only the most recent 1000 samples
buffer = InMemoryDataBuffer(
    buffer_strategy=SlidingWindowBufferStrategy(window_size=1000),
    max_size=None,  # No overall limit
)
```

### Custom Networks

For neural bandits, you can provide any PyTorch neural network:

```python
import torch.nn as nn

# Simple MLP
network = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, output_size)
)

# More complex architecture
class MyCustomNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Use the custom network
network = MyCustomNetwork(input_dim=128, embedding_dim=32)
bandit = NeuralUCBBandit(
    n_features=10,
    network=network,
    # ... other parameters
)
```

The benchmark module also provides pre-defined network architectures:

- `none`: Identity mapping
- `linear`: Simple linear layer
- `tiny_mlp`: Small MLP with one hidden layer (64 units)
- `small_mlp`: MLP with two hidden layers (128 units each)
- `large_mlp`: MLP with three hidden layers (256 units each)
- `deep_mlp`: Deep MLP with seven hidden layers (64 units each)
- `bert`: BERT model for text data

These can be specified in the benchmark configuration:

```python
run(
    {
        "bandit": "neural_linear",
        "dataset": "statlog",
        "network": "tiny_mlp",  # Specify the network architecture
        # ... other parameters
    }
)
```

### Data Samplers

Calvera provides custom data samplers that give you control over how data is sampled from your datasets during training. These can be particularly useful when you want to prioritize certain samples or implement specialized sampling strategies.

#### Random Data Sampler

The `RandomDataSampler` samples elements randomly without replacement:

```python
from torch.utils.data import DataLoader
from calvera.utils import RandomDataSampler
from calvera.benchmark.datasets import StatlogDataset

# Create a dataset
dataset = StatlogDataset()

# Create a random sampler with optional reproducibility
generator = torch.Generator().manual_seed(42)  # For reproducible sampling
sampler = RandomDataSampler(dataset, generator=generator)

# Use the sampler with a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,  # Use our custom sampler instead of shuffle=True
    num_workers=4
)

# This dataloader will now sample data randomly using our sampler
for batch in dataloader:
    # Process batch...
    pass
```

#### Sorted Data Sampler

The `SortedDataSampler` allows you to sample elements in a specific order based on a key function:

```python
from calvera.utils import SortedDataSampler

# Define a key function that determines the sorting order
def key_fn(idx):
    # Sort by the sum of features for each sample
    return dataset.X[idx].sum().item()

# Create a sorted sampler
sampler = SortedDataSampler(dataset, key_fn=key_fn, reverse=True)  # reverse=True for descending order

# Use the sampler with a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4
)

# This dataloader will provide batches in descending order of feature sums
for batch in dataloader:
    # Process batch...
    pass
```

This approach is useful for curriculum learning or when you want to train first on samples with certain characteristics.

## Checkpointing

Calvera supports checkpointing via PyTorch Lightning's checkpoint system. This allows you to save and resume training, which is especially valuable for long-running experiments or when deploying models to production.

### Saving Checkpoints

Here's how to enable checkpointing:

```python
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from calvera.bandits import NeuralTSBandit

# Initialize your bandit
bandit = NeuralTSBandit(
    n_features=dataset.context_size,
    network=network,
    buffer=buffer,
    # ... other parameters ...
)

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",  # Directory to save checkpoints
    filename="neural_ts-{step}",  # Filename pattern including the step number
    save_top_k=3,  # Save the top 3 models (based on monitor)
    monitor="loss",  # Monitor this metric for determining the "best"
    mode="min",  # Lower loss is better
    every_n_train_steps=100,  # Save every 100 training steps
)

# Create trainer with checkpoint callback
trainer = pl.Trainer(
    max_epochs=1,
    callbacks=[checkpoint_callback],
    enable_checkpointing=True,
    # ... other trainer parameters ...
)

# Run training
trainer.fit(bandit)

# Get the best checkpoint path
best_model_path = checkpoint_callback.best_model_path
print(f"Best checkpoint saved at: {best_model_path}")
```

### Loading Checkpoints

You can load a model from a checkpoint to resume training or for inference:

```python
# Load model from checkpoint
loaded_bandit = NeuralTSBandit.load_from_checkpoint(
    best_model_path,
    n_features=dataset.context_size,
    network=network,  # Provide the network architecture
    buffer=buffer,  # Provide a buffer (can be empty)
    # ... any other required parameters not stored in the checkpoint ...
)

# Use the loaded model for inference
context = torch.randn(10, 5, dataset.context_size)
chosen_actions, probabilities = loaded_bandit(context)

# Or resume training
trainer = pl.Trainer(
    max_epochs=1,
    # ... other trainer parameters ...
)
trainer.fit(loaded_bandit)
```

### What Gets Checkpointed

When a Calvera bandit is checkpointed, the following components are saved:

1. **Model Parameters**: All learnable parameters of the bandit, including neural network weights
2. **Buffer State**: The data buffer state, including stored contexts and rewards
3. **Linear Head Parameters**: For linear and neural linear bandits, the precision matrix, b vector, and theta
4. **Selector State**: The state of the action selector (e.g., epsilon value and RNG state for EpsilonGreedySelector)
5. **Training State**: Counters and flags related to training progress
6. **Hyperparameters**: The hyperparameters used to initialize the model

### Checkpoint Management

For managing checkpoints over long experiments:

```python
from lightning.pytorch.callbacks import ModelCheckpoint

# Save only the best model
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best-{step}",
    save_top_k=1,
    monitor="loss",
    mode="min",
)

# Also save the latest model periodically
latest_checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="latest-{step}",
    save_top_k=1,
    save_last=True,
    every_n_train_steps=500,
)

# Add both callbacks to the trainer
trainer = pl.Trainer(
    callbacks=[checkpoint_callback, latest_checkpoint_callback],
    enable_checkpointing=True,
    # ... other trainer parameters ...
)
```

This setup ensures you always have the best-performing model saved, as well as regular snapshots of recent training progress.

## Benchmarking

Calvera provides a benchmarking module to easily compare different bandit algorithms on various datasets. Here's how to use it:

```python
from calvera.benchmark import run

# Benchmark a Linear UCB bandit on the Covertype dataset
run(
    {
        "bandit": "lin_ucb",
        "dataset": "covertype",
        "max_samples": 5000,
        "feedback_delay": 1,
        "train_batch_size": 1,
        "forward_batch_size": 1,
        "bandit_hparams": {
            "alpha": 1.0,
        },
    }
)

# Benchmark a Neural Linear bandit on the Covertype dataset
run(
    {
        "bandit": "neural_linear",
        "dataset": "covertype",
        "network": "tiny_mlp",
        "max_samples": 5000,
        "feedback_delay": 1,
        "train_batch_size": 1,
        "forward_batch_size": 1,
        "data_strategy": "sliding_window",
        "sliding_window_size": 1,
        "bandit_hparams": {
            "n_embedding_size": 128,
        },
    }
)
```

The benchmark module supports the following bandits:

- `lin_ucb`: Linear UCB
- `approx_lin_ucb`: Diagonal Precision Approximation Linear UCB
- `lin_ts`: Linear Thompson Sampling
- `approx_lin_ts`: Diagonal Precision Approximation Linear Thompson Sampling
- `neural_linear`: Neural Linear
- `neural_ucb`: (Combinatorial + Standard) Neural UCB
- `neural_ts`: (Combinatorial + Standard) Neural Thompson Sampling

And the following datasets:

- `covertype`: Covertype dataset from UCI
- `mnist`: MNIST dataset
- `statlog`: Statlog dataset from UCI
- `wheel`: Synthetic Wheel Bandit dataset
- `imdb`: IMDB movie reviews dataset
- `movielens`: MovieLens dataset
- `tiny_imagenet`: Tiny ImaneNet dataset
- `synthetic`: A binary classification synthetic dataset
- `synthetic_combinatorial`: Synthetic dataset for Combinatorial Bandit experiments

## Further Resources

- For more examples, check the [examples directory](https://github.com/neural-bandits/calvera/tree/main/examples) on GitHub.
- For API documentation, see the [API reference](https://neural-bandits.github.io/calvera/bandits/).
- For implementation details, see the [source code](https://github.com/neural-bandits/calvera).
