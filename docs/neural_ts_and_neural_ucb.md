The `NeuralUCBBandit` and the `NeuralTSBandit` share the same interface.
Both use a neural network to learn the reward function for a given contextualized actions.
To estimate the uncertainty the gradients of the estimated reward of the chosen action with respect to the network parameters are used.
These gradients are used to build a precision matrix which is used to compute the UCB or perform Thompson sampling.


::: neural_bandits.bandits.neural_ucb_bandit.NeuralUCBBandit
    handler: python
    options:
      heading: LinearTSBandit
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: false
      inherited_members:
        - __init__

::: neural_bandits.bandits.neural_ts_bandit.NeuralTSBandit
    handler: python
    options:
      heading: LinearTSBandit
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: false
      inherited_members:
        - __init__