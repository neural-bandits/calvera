Below is the interface that all bandit algorithms share, defined in the `AbstractBandit` class. The idea is that assertions happen in the `forward()` method for the input and in the `training_step()` method for the update using the provided rewards and chosen contextualized actions.
The outwards facing methods are `forward()` and `training_step()`. `forward()` is used for inference and `training_step()` is used for training.
So, when implementing a new bandit, the following methods need to be implemented:

- `_predict_action(self, x: torch.Tensor) -> torch.Tensor`: Predicts the action for the given context.
- `_update(self, x: torch.Tensor, y: torch.Tensor) -> None`: Updates the bandit with the given context and reward.



::: neural_bandits.bandits.abstract_bandit.AbstractBandit
    handler: python
    options:
      heading_level: 2
      heading: AbstractBandit
      show_root_heading: true
      show_root_full_path: false
      members:
        - forward
        - training_step
        - _predict_action
        - _update