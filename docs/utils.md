## **Selectors**

Selectors are used to control the action selection behavior of a bandit.
You can provide a selector to each bandit. If you don't provide one, a default selector will be used.

Currently, we provide the following selectors:

- `ArgMaxSelector`: Select the action with the highest estimated reward.

- `EpsilonGreedySelector`: Select the action with the highest estimated reward with probability `1 - epsilon` and a random action with probability `epsilon`.

- `TopKSelector`: Select the top `k` actions with the highest estimated reward.

If you want to implement your own selector, you can subclass the `Selector` class and implement the `__call__` method making your class callable.

<br>

::: calvera.utils.AbstractSelector
    handler: python
    options:
      heading: AbstractSelector
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members:
        - __call__

::: calvera.utils.ArgMaxSelector
    handler: python
    options:
      heading: ArgMaxSelector
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: False

::: calvera.utils.EpsilonGreedySelector
    handler: python
    options:
      heading: EpsilonGreedySelector
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: 
        - __init__

::: calvera.utils.RandomSelector
    handler: python
    options:
      heading: EpsilonGreedySelector
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: False

::: calvera.utils.TopKSelector
    handler: python
    options:
      heading: TopKSelector
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: 
        - __init__


<br>
<br>

## **Data Samplers**
To simulate a bandit in a scenario with non-i.i.d. contexts, we need to modify the data sampler of our benchmark datasets.
To be consistent we provide a `DataSampler` class that can be used to sample data from a dataset.
<br>

::: calvera.utils.data_sampler.AbstractDataSampler
    handler: python
    options:
      heading: AbstractDataSampler
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members:
        - __init__
        - __len__
        - __iter__


::: calvera.utils.data_sampler.RandomDataSampler
    handler: python
    options:
      heading: RandomDataSampler
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: False

::: calvera.utils.data_sampler.SortedDataSampler
    handler: python
    options:
      heading: SortedDataSampler
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: False

<br>
<br>


::: calvera.utils.MultiClassContextualizer
    handler: python
    options:
      heading: MultiClassContextualizer
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
      members: 
        - __init__
        - __call__