import torch
from torch.utils.data import Dataset


class BanditFeedbackDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that contains only those actions & rewards chosen by the bandit.
    
    It is used to do a single update on the bandit. Supports multiple chosen actions (combinatorial bandits) but must be
    the same for all rows. This is form of feedback is called semi-bandit feedback because we receive one reward per chosen
    action.
    """

    def __init__(
        self,
        chosen_contextualized_actions: torch.Tensor,
        realized_rewards: torch.Tensor,
    ) -> None:
        """Initialize the BanditFeedbackDataset.
        
        n = # of rows in the dataset
        i = # of actions chosen per row
        k = # size of the contextualized action vector

        Args:
            chosen_contextualized_actions: shape (n, i, k)
            realized_rewards: shape (n, i)
        """
        super().__init__()

        assert chosen_contextualized_actions.size(0) == realized_rewards.size(
            0
        ), f"Mismatched size of chosen_contextualized_actions and realized_rewards tensors. Received {chosen_contextualized_actions.size(0)} and {realized_rewards.size(0)}."
        assert chosen_contextualized_actions.size(1) == realized_rewards.size(
            1
        ), f"Mismatched size of chosen_contextualized_actions and realized_rewards tensors. Received {chosen_contextualized_actions.size(1)} and {realized_rewards.size(1)}."

        # chosen_contextualized_actions: [n, i, k]
        self.chosen_contextualized_actions = chosen_contextualized_actions

        # realized_rewards: [n, i]
        self.realized_rewards = realized_rewards

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        # We store everything in an [n, i, *] shape, so # of rows = n
        return self.realized_rewards.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the chosen contextualized actions and realized rewards for the given index.
        
        Args:
            idx: The index of the sample to retrieve.
        
        Returns:
            A tuple containing the chosen contextualized actions and realized rewards.
        """
        return self.chosen_contextualized_actions[idx], self.realized_rewards[idx]
