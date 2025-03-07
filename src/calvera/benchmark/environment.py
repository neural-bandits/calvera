from typing import Generic, cast

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from calvera.bandits.action_input_type import ActionInputType


class BanditBenchmarkEnvironment(Generic[ActionInputType]):
    """Environment that iterates over a DataLoader, yielding only `contextualized_actions`.

    Internally stores `rewards`, which can be retrieved by a helper method.
    This is used to simulate a bandit environment with delayed feedback where the bandit can only see the actions and
    not the rewards.

    Internally stores `rewards`, which can be retrieved by a helper method.
    This is used to simulate a bandit environment with delayed feedback where the bandit can only see the actions
    and not the rewards. The bandit should first sample `contextualized_actions` by iterating over the environment.
    The bandit can then choose the best actions.
    Finally, the bandit can receive rewards by calling `get_rewards_dataset(chosen_actions)`.
    Since this is a simulation, the bandit can also compute the regret by calling `compute_regret(chosen_actions)`.

    Usage:
    ```python
    environment = BanditBenchmarkEnvironment(dataloader)
    for contextualized_actions in environment:
        chosen_actions = bandit.forward(contextualized_actions)
        feedback_dataset = environment.get_feedback(chosen_actions)
        bandit.update(feedback_dataset)
        regret = environment.compute_regret(chosen_actions)
    ```
    """

    _dataloader: DataLoader[tuple[ActionInputType, torch.Tensor]]
    _iterator: _BaseDataLoaderIter | None
    _last_contextualized_actions: ActionInputType | None
    _last_all_rewards: torch.Tensor | None
    device: torch.device | None

    def __init__(
        self,
        dataloader: DataLoader[tuple[ActionInputType, torch.Tensor]],
        device: torch.device | None = None,
    ) -> None:
        """Initializes a BanditBenchmarkEnvironment.

        Args:
            dataloader: DataLoader that yields batches of (contextualized_actions, all_rewards) tuples.
            device: The device the tensors should be moved to. If None, the default device is used.
        """
        self._dataloader: DataLoader[tuple[ActionInputType, torch.Tensor]] = dataloader
        self._iterator: _BaseDataLoaderIter | None = None
        self._last_contextualized_actions: ActionInputType | None = None
        self._last_all_rewards: torch.Tensor | None = None
        self.device = device

    def __iter__(self) -> "BanditBenchmarkEnvironment[ActionInputType]":
        """Returns an iterator object for the BanditBenchmarkEnvironment.

        This method initializes an iterator for the dataloader and returns the
        BanditBenchmarkEnvironment instance itself, allowing it to be used as an
        iterator in a loop. Needs to be called before the first iteration.

        Returns:
            BanditBenchmarkEnvironment: The instance of the environment itself.
        """
        self._iterator = iter(self._dataloader)
        return self

    def __next__(self) -> ActionInputType:
        """Returns the next batch of contextualized actions from the DataLoader.

        Returns:
            The contextualized actions for the bandit to pick from.

        Raises:
            AssertionError: If the iterator is not initialized with `__iter__`.
        """
        assert self._iterator is not None, "No iterator was created."

        # Retrieve one batch from the DataLoader
        batch = next(self._iterator)
        contextualized_actions: ActionInputType = batch[0]
        all_rewards: torch.Tensor = batch[1].to(device=self.device)

        if isinstance(contextualized_actions, torch.Tensor):
            batch_size, num_actions = contextualized_actions.shape[:2]
            contextualized_actions = cast(ActionInputType, contextualized_actions.to(device=self.device))
        elif isinstance(contextualized_actions, tuple | list):
            contextualized_actions = cast(
                ActionInputType,
                tuple(action_tensor.to(device=self.device) for action_tensor in contextualized_actions),
            )
            batch_size, num_actions = contextualized_actions[0].shape[:2]
        else:
            raise ValueError(
                f"contextualized_actions must be a torch.Tensor or a tuple. Received {type(contextualized_actions)}."
            )

        assert batch_size == all_rewards.size(0), (
            f"Mismatched batch size of contextualized_actions and all_rewards tensors."
            f"Received {batch_size} and {all_rewards.size(0)}."
        )
        assert num_actions == all_rewards.size(1) or num_actions == 1, (
            f"Mismatched number of actions in contextualized_actions and all_rewards tensors."
            f"Received {num_actions} and {all_rewards.size(1)}."
        )

        # Store them so we can fetch them later when building the update dataset
        self._last_contextualized_actions = contextualized_actions
        self._last_all_rewards = all_rewards
        # Return only the contextualized actions for the bandit to pick from
        return contextualized_actions

    def get_feedback(self, chosen_actions: torch.Tensor) -> tuple[ActionInputType, torch.Tensor]:
        """Returns the chosen actions & realized rewards of the last batch.

        For combinatorial bandits, this feedback is semi-bandit feedback.

        Args:
            chosen_actions: shape (n, m) (one-hot, possibly multiple "1"s). The actions chosen by the bandit. Must
                contain at least one and the same number of chosen actions ("1s") for all rows.

        Returns:
            BanditFeedbackDataset with the chosen actions (shape: (n, m, k)) and realized rewards (shape: (n, m)).
        """
        self._validate_chosen_actions(chosen_actions)

        chosen_contextualized_actions = self._get_chosen_contextualized_actions(chosen_actions)
        realized_rewards = self._get_realized_rewards(chosen_actions)

        return (
            chosen_contextualized_actions,
            realized_rewards,
        )

    def compute_regret(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        """Computes the regret for the most recent batch.

        Definition:
          best_reward = max over top i actions (where i is the number of chosen actions)
          chosen_reward = sum over chosen actions (handles multiple 1s per row)
          regret = best_reward - chosen_reward
        Important: For combinatorial bandits assumes that the reward of a super-action is the sum of each chosen arm.

        Args:
            chosen_actions: shape (n, k), one-hot, possibly multiple "1"s. The actions chosen by the bandit. Must
                contain at least one and the same number of chosen actions ("1s") for all rows.

        Returns:
            Tensor of regrets shape (n, ).
        """
        self._validate_chosen_actions(chosen_actions)

        best_action_rewards = self._get_best_action_rewards(chosen_actions).sum(dim=1)
        chosen_reward = self._get_realized_rewards(chosen_actions).sum(dim=1)

        eps = 1e-4
        assert torch.all(best_action_rewards >= chosen_reward - eps), (
            "Best action rewards should be greater than chosen rewards. "
            f"Best: {best_action_rewards}, Chosen: {chosen_reward}"
        )

        regret = best_action_rewards - chosen_reward
        # clamp to 0 to avoid negative regrets
        regret = torch.clamp(regret, min=0.0)
        return regret

    def _validate_chosen_actions(self, chosen_actions: torch.Tensor) -> None:
        if self._last_contextualized_actions is None:
            return

        batch_size, num_actions = (
            self._last_contextualized_actions.shape[:2]
            if isinstance(self._last_contextualized_actions, torch.Tensor)
            else self._last_contextualized_actions[0].shape[:2]
        )

        assert chosen_actions.size(0) == batch_size, (
            "Mismatched batch size of chosen_actions and contextualized_actions tensors."
            "Received {chosen_actions.size(0)} and {batch_size}."
        )

        assert (chosen_actions.sum(dim=1) > 0).all(), "No actions were chosen in some rows."
        assert (
            chosen_actions.sum(dim=1) == chosen_actions.sum(dim=1)[0]
        ).all(), "Number of chosen actions is not the same for all rows."

    def _get_best_action_rewards(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        assert self._last_all_rewards is not None, "No rewards were stored."

        # For each row, find how many actions were chosen (i) and take the top i rewards in that row.
        batch_size = chosen_actions.size(0)  # n
        best_rewards = []  # will have shape (n, i)
        for batch_idx in range(batch_size):
            row = self._last_all_rewards[batch_idx]  # shape (m, )
            i = int(chosen_actions[batch_idx].sum().item())  # number of chosen actions in this row

            assert i > 0, "No actions were chosen in this row!"

            # topk returns a namedtuple (values, indices)
            top_values = torch.topk(row, i).values
            best_rewards.append(top_values)

        # assert that all rows have the same number of chosen actions
        assert all(
            len(best_rewards[0]) == len(row) for row in best_rewards
        ), "Mismatched number of chosen actions in rows."

        # Stack all sums into a single tensor of shape (n, i)
        best_action_rewards = torch.stack(best_rewards, dim=0)
        return best_action_rewards  # shape (n, i)

    def _get_chosen_contextualized_actions(self, chosen_actions: torch.Tensor) -> ActionInputType:
        assert self._last_contextualized_actions is not None, "No actions were stored."

        mask = chosen_actions.bool()

        if isinstance(self._last_contextualized_actions, torch.Tensor):
            # Make shape match contextualized_actions for masked_select

            if self._last_contextualized_actions.shape[1] == 1:
                return cast(ActionInputType, self._last_contextualized_actions)
            else:
                expanded_mask = mask.unsqueeze(-1).expand_as(self._last_contextualized_actions)

            return cast(
                ActionInputType,
                torch.masked_select(self._last_contextualized_actions, expanded_mask).view(
                    self._last_contextualized_actions.size(0),
                    -1,
                    *self._last_contextualized_actions.shape[2:],
                ),
            )  # shape (n, m, k)
        elif isinstance(self._last_contextualized_actions, tuple | list):
            if self._last_contextualized_actions[0].shape[1] == 1:
                return cast(ActionInputType, tuple(action for action in self._last_contextualized_actions))

            first_tensor = self._last_contextualized_actions[0]
            expanded_mask = mask.unsqueeze(-1).expand_as(first_tensor)
            return cast(
                ActionInputType,
                tuple(
                    torch.masked_select(action_item, expanded_mask).view(
                        first_tensor.size(0),
                        -1,
                        *action_item.shape[2:],
                    )
                    for action_item in self._last_contextualized_actions
                ),
            )  # tuple of shape (n, m, k)
        else:
            raise ValueError(
                f"chosen_contextualized_actions must be a torch.Tensor or a tuple."
                f"Received {type(self._last_contextualized_actions)}."
            )

    def _get_realized_rewards(self, chosen_actions: torch.Tensor) -> torch.Tensor:
        assert self._last_all_rewards is not None, "No rewards were stored."

        mask = chosen_actions.bool()
        return (
            (self._last_all_rewards * chosen_actions.float())
            .masked_select(mask)
            .view(self._last_all_rewards.size(0), -1)
        )  # shape (n, m)

    def __len__(self) -> int:
        """Returns the number of batches in the DataLoader."""
        assert self._iterator is not None, "No iterator was created."
        return len(self._iterator)
