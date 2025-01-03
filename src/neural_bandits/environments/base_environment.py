import random
from typing import Any, Callable, Dict, Generic

import numpy as np
import torch

from ..datasets.abstract_dataset import AbstractDataset
from ..trainers.abstract_trainer import AbstractTrainer, BanditType
from ..utils.abtract_contextualiser import AbstractContextualiser


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class BaseEnvironment(Generic[BanditType]):
    """
    A simple environment for training and evaluating bandit algorithms
    on a given dataset/scenario. The environment takes in a bandit instance,
    a dataset, and some optional parameters (like random seeds for reproducibility).
    """

    def __init__(
        self,
        bandit: BanditType,
        trainer: AbstractTrainer[BanditType],
        dataset: AbstractDataset,
        contextualiser: AbstractContextualiser | None = None,
        seed: int = 42,
        max_steps: int | None = None,
        log_method: Callable[[Dict[str, Any]], None] = lambda x: None,
    ) -> None:
        self.bandit = bandit
        self.dataset = dataset
        self.trainer = trainer
        self.contextualiser = contextualiser
        self.log = log_method
        self.max_steps = max_steps or len(dataset)
        self.seed = seed

    def run(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Run the bandit algorithm on the given dataset scenario and record metrics.
        """
        seed_all(self.seed)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)

        chosen_arms = []
        result_rewards = []
        cumulative_rewards = []
        cum_reward = 0.0

        for t in range(0, self.max_steps, batch_size):
            context_list = []

            for idx in indices[t : t + batch_size]:
                context_list.append(self.dataset[idx])

            contexts = torch.stack(context_list)
            contextualised_actions = (
                self.contextualiser.contextualise(contexts)
                if self.contextualiser
                else contexts
            )

            pred = self.bandit(contextualised_actions)
            arms = torch.argmax(pred, dim=-1).float()
            chosen_arms.append(arms)

            rewards = torch.stack(
                [
                    self.dataset.reward(int(idx), action)
                    for idx, action in zip(indices[t : t + batch_size], arms)
                ]
            ).squeeze(1)
            result_rewards.append(rewards)
            self.bandit = self.trainer.update(
                self.bandit,
                rewards=rewards,
                chosen_actions=contextualised_actions[
                    torch.arange(len(contextualised_actions)), arms.long()
                ],
            )

            for i in range(len(rewards)):
                cum_reward += rewards[i].item()
                self.log(
                    {
                        "step": t + i,
                        "arm": arms[i].item(),
                        "reward": rewards[i].item(),
                        "cumulative_reward": cum_reward,
                    }
                )
                cumulative_rewards.append(cum_reward)

        results = {
            "chosen_arms": torch.cat(chosen_arms).tolist(),
            "rewards": torch.cat(result_rewards).tolist(),
            "cumulative_rewards": cumulative_rewards,
            "final_cumulative_reward": cum_reward,
        }
        return results
