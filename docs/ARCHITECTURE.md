# Architecture Specification of the library

## Overview
![Architecture UML](./architecture_12_12_2024.drawio.svg)

## Tasks

### Datasets
- MNIST
- STATLOG
- CoverType
- TODO Some kind of RAG dataset
- Synthetic dataset (wheel bandit, generation functions)

and an AbstractDataset, inheriting from torch Dataset, which provides the option to contextualize the actions using a disjoint module by using the MultiClassContextualizer.

### Feedback -- offline vs. online
- NO immediate online feedback because it is unrealistic in practice. Though it is possible if a batch_size of 1 is used.
- Instead: Batched Semi-Online Feedback => retrain once a batch of realized rewards per chosen action is available.
- Offline feedback: user can add as much data into the training process as is available.
- it would be the users responsibility to store probabilities for logged feedback-based training. We return a probability distribution over actions.

### Algorithms
*NOTE*: everything is contextual
#### Exploration Strategies
- Linear Bandits (LinUCB, LinTS)
- ($\epsilon$)-greedy (through an EpsilonGreedySelector)
- NeuralUCB (UCB with gradients)
- NeuralTS (optional?)
- NeuralLinear
- Combinatorial Bandits (maybe we need to figure the integration of this out)

#### Architectures
- Bootstrap (optional?)
- Neural Networks

## Usage
Usage can be found in a seperate example notebook.