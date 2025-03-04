# LinearTS and LinearUCB
Linear Thompson Sampling and Linear UCB are two of the basic contextual bandit algorithms.
The main idea is to use a linear model to regress the reward on the context and combine
this, with an uncertainty measure, to also account for unexplored parts of the context space.
The uncertainty measure is usually the inverse of the covariance matrix of the chosen contexts.
This measure is then used to sample the parameters in the case of Thompson Sampling or to determine
the upper confidence bound in the case of UCB.

The key part is that both need to compute the inverse of the covariance matrix of the chosen contexts.
This calculation can become expensive for high-dimensional contexts. Therefore, we also provide a `DiagonalPrecApprox-`
variant. This variant uses a diagonal approximation of the covariance matrix and is much faster to compute.

::: calvera.bandits.linear_ucb_bandit.LinearUCBBandit
    handler: python
    options:
      heading: LinearUCBBandit
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members:
        - __init__
      show_source: True

::: calvera.bandits.linear_ucb_bandit.DiagonalPrecApproxLinearUCBBandit
    handler: python
    options:
      heading: DiagonalPrecApproxLinearUCBBandit
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: false

::: calvera.bandits.linear_ts_bandit.LinearTSBandit
    handler: python
    options:
      heading: LinearTSBandit
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members:
        - __init__

::: calvera.bandits.linear_ts_bandit.DiagonalPrecApproxLinearTSBandit
    handler: python
    options:
      heading: DiagonalPrecApproxLinearTSBandit
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: false