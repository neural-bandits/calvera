from typing import Generic, cast

import torch
from torch.utils.data import Dataset

from neural_bandits.bandits.abstract_bandit import ActionInputType


class BanditFeedbackDataset(
    Generic[ActionInputType], Dataset[tuple[ActionInputType, torch.Tensor]]
):
    """
    Dataset that contains only those actions & rewards chosen by the bandit. It is used to do a single update on the bandit.
    Supports multiple chosen actions (combinatorial bandits) but must be the same for all rows.
    This is form of feedback is called semi-bandit feedback because we receive one reward per chosen action.

    ActionInputType is the type of the chosen contextualized actions that are input to the bandit.
    """

    input_tuple: bool
    chosen_contextualized_actions: torch.Tensor  # shape: [n, m, i, k]
    realized_rewards: torch.Tensor  # shape: [n, i]

    def __init__(
        self,
        chosen_contextualized_actions: ActionInputType,
        realized_rewards: torch.Tensor,
    ) -> None:
        """
        n = # of rows in the dataset
        i = # of actions chosen per row
        k = # size of the contextualized action vector

        Args:
            chosen_contextualized_actions: shape (n, i, k) or list of m tensors of shape (n, i, k)
            realized_rewards: shape (n, i)
        """
        super().__init__()

        if isinstance(chosen_contextualized_actions, torch.Tensor):
            tuple_of_chosen_contextualized_actions = (chosen_contextualized_actions,)
            self.input_tuple = False
        elif isinstance(chosen_contextualized_actions, tuple):
            tuple_of_chosen_contextualized_actions = chosen_contextualized_actions
            self.input_tuple = True
        else:
            raise ValueError(
                f"chosen_contextualized_actions must be a torch.Tensor or a tuple. Received {type(chosen_contextualized_actions)}."
            )

        assert (
            len(tuple_of_chosen_contextualized_actions) > 0
        ), "chosen_contextualized_actions must be a non-empty tuple."
        assert (
            tuple_of_chosen_contextualized_actions[0].ndim == 3
        ), f"chosen_contextualized_actions must have shape (n, num_actions, num_features). Received {tuple_of_chosen_contextualized_actions[0].shape}."

        n, i, k = tuple_of_chosen_contextualized_actions[0].shape
        assert all(
            action_item.shape == (n, i, k)
            for action_item in tuple_of_chosen_contextualized_actions
        ), "All elements of tuple of chosen_contextualized_actions must have the same shape (n, i, k)."

        assert (
            realized_rewards.size(0) == n and realized_rewards.size(1) == i
        ), f"Mismatched size of realized_rewards tensors. Expected ({n}, {i}). Received {realized_rewards.shape}."

        # chosen_contextualized_actions: [n, m, i, k]
        self.chosen_contextualized_actions = torch.cat(
            [
                action_item.unsqueeze(1)
                for action_item in tuple_of_chosen_contextualized_actions
            ]
        )

        # realized_rewards: [n, i]
        self.realized_rewards = realized_rewards

    def __len__(self) -> int:
        # We store realized_rewards in an [n, i] shape, so # of rows = n
        return self.realized_rewards.size(0)

    def __getitem__(self, idx: int) -> tuple[ActionInputType, torch.Tensor]:
        chosen_contextualized_action_tensor = self.chosen_contextualized_actions[
            idx
        ]  # shape [m, i, k]

        if self.input_tuple:  # input was a tuple
            chosen_contextualized_action = cast(
                ActionInputType,
                tuple(torch.unbind(chosen_contextualized_action_tensor, dim=0)),
            )
        else:  # input was a single tensor, so we squeeze the m dimension
            chosen_contextualized_action = cast(
                ActionInputType, chosen_contextualized_action_tensor.squeeze(0)
            )

        return chosen_contextualized_action, self.realized_rewards[idx]
