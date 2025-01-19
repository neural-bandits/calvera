import torch
from torch import optim
import numpy as np

from ..algorithms.neural_ucb_bandit import NeuralUCB
from .abstract_trainer import AbstractTrainer

class NeuralUCBTrainer(AbstractTrainer):
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        
    def update(
        self,
        bandit: NeuralUCB,
        rewards: torch.Tensor,
        chosen_actions: torch.Tensor,
    ) -> NeuralUCB:
        """Update using simplified training approach"""
        rewards = rewards.to(bandit.device)
        chosen_actions = chosen_actions.to(bandit.device)
        
        bandit.context_list.append(chosen_actions)
        bandit.reward_list.append(rewards)
        
        self._train_network(bandit)
        
        return bandit
    
    def _train_network(self, bandit: NeuralUCB):
        """Simplified training procedure"""
        optimizer = optim.SGD(bandit.theta.parameters(), lr=self.eta, weight_decay=bandit.lambda_)
        
        indices = np.arange(len(bandit.reward_list))
        np.random.shuffle(indices)
        tot_loss = 0
        cnt = 0
        while True:
            batch_loss = 0
            for idx in indices:
                context = bandit.context_list[idx]
                reward = bandit.reward_list[idx]
                
                optimizer.zero_grad()
                pred = bandit.theta(context)
                loss = (pred - reward) ** 2
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                
                if cnt >= 1000:
                    return
            
            if batch_loss / len(bandit.reward_list) <= 1e-3:
                break