# Architecture Specification of the library

## Overview
![Architecture UML](./architecture_12_12_2024.drawio.svg)

## Tasks

### Datasets
- MNIST
- STATLOG
- Some kind of RAG dataset
- Synthetic dataset (wheel bandit, generation functions)

### Feedback -- offline vs. online
- Online Feedback (rewards per action)
- Offline feedback (Optional)
- store probabilities for logged feedback-based training  (Optional)


### Algorithms
*NOTE*: everything is contextual
#### Exploration Strategies
- Linear Bandits (LinUCB, LinTS)
- ($\epsilon$)-greedy (?)
- NeuralUCB (UCB with gradients)
- NeuralTS
- Combinatorial Bandits (maybe we need to figure the integration of this out)

#### Architectures
- Bootstrap
- Neural Networks
- Low Rank Adaption of Neural Networks (Optional)

---

# Architecture specifics
 

## Model (forward pass)
This is just a sketch: 

## General Case: Multiple Feature Vectors
(multiple feature vectors / contextualized actions)-> NN  (multiple embeddings)-> Bandit Head (actions)

*Note 1: The usage of a NN is optional*
*Note 2: The NN or model we refer to here should not be confused with a general embedding model. This is not part of the bandit*

 ```python
import torch.nn as nn

class NeuralUCB(nn.Module): 
    def __init__(self, model: nn.Module, bandit_head: Type[nn.Module]):
        self.model = model
        self.bandit_head = bandit_head

    def forward(self, x): # <- predicts a single action based on a single feature vector
        x = self.model(x)
        x = self.bandit_head(x)
        return x
        
```



## Special Case: Single Feature Vector
(single feature vector)-> NN  (embedding)-> DisjointModel (contextualized actions)-> Bandit Head (actions)

 ```python
import torch.nn as nn

class NeuralLinear(nn.Module):
    def __init__(self, model: nn.Module, bandit_head: Type[nn.Module]):
        self.model = model
        self.loss = loss
        self.disjoint_model = DisjointModelContext
        self.bandit_head = bandit_head

    def forward(self, x): # <- predicts a single action based on a single feature vector
        x = self.model(x)
        x = self.disjoint_model(x)
        x = self.bandit_head(x)
        return x


aModel = NeuralLinear(model, bandit_head)
aModel(x)
```

## Bandit Updater

```python
class NeuralUCBTrainer:
    def __init__(self, neural_ucb: NeuralUCB, optimizer: torch.optim.Optimizer):
        self.neural_ucb = neural_ucb
        self.optimizer = optimizer

    def update(self, x, y):
        self.optimizer.zero_grad()
        action_pred = self.model(x)
        
        
        return loss
```

