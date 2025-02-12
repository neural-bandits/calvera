from abc import ABC, abstractmethod

import torch

class AbstractSelector(ABC):
    """Defines the interface for all bandit action selectors.
    Given a tensor of scores per action, the selector chooses the best action (i.e. an arm)
    or the best set of actions (i.e. a super arm in combinatorial bandits). The selector
    returns a one hot encoded tensor of the chosen actions.
    """

    @abstractmethod
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Selects a single best action, or a set of actions in the case of combinatorial bandits.

        Args:
            scores (torch.Tensor): Tensor of shape (batch_size, n_actions).
            This may contain a probability distribution per sample (when used for thompson sampling) or simply a score per arm (e.g. for UCB).
            In case of combinatorial bandits, these are the scores per arm from which the oracle selects a super arm (e.g. simply top-k).

        Returns:
            chosen_actions (torch.Tensor): One hot encoded actions that were chosen.
                Shape: (batch_size, n_actions).
        """
        pass


# TODO: Documentation
class ArgMaxSelector(AbstractSelector):
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.argmax(scores, dim=1), num_classes=scores.shape[1]
        )
