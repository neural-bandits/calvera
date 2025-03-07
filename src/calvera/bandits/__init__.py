"""A subpackage containing the (neural) bandit implementations.

In this subpackage are all current bandit implementations. Each bandit is a PyTorch Lightning module that implements the
`forward()` method for the input and in the `training_step()` method for the
update using the provided rewards and chosen contextualized actions. The outwards facing methods are `forward()` and
`training_step()`. `forward()` is used for inference and `training_step()` is used for training.
So, when implementing a new bandit, the following methods need to be implemented:

- `_predict_action(self, x: torch.Tensor) -> torch.Tensor`: Predicts the action for the given context.
- `_update(self, x: torch.Tensor, y: torch.Tensor) -> None`: Updates the bandit with the given context and reward.
"""

from calvera.bandits.abstract_bandit import AbstractBandit, DummyBandit
from calvera.bandits.action_input_type import ActionInputType
from calvera.bandits.linear_ts_bandit import DiagonalPrecApproxLinearTSBandit, LinearTSBandit
from calvera.bandits.linear_ucb_bandit import DiagonalPrecApproxLinearUCBBandit, LinearUCBBandit
from calvera.bandits.neural_linear_bandit import HelperNetwork, NeuralLinearBandit
from calvera.bandits.neural_ts_bandit import NeuralTSBandit
from calvera.bandits.neural_ucb_bandit import NeuralUCBBandit

__all__ = [
    "AbstractBandit",
    "DummyBandit",
    "ActionInputType",
    "LinearTSBandit",
    "DiagonalPrecApproxLinearTSBandit",
    "LinearUCBBandit",
    "DiagonalPrecApproxLinearUCBBandit",
    "NeuralLinearBandit",
    "HelperNetwork",
    "NeuralTSBandit",
    "NeuralUCBBandit",
]
