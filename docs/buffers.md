DataBuffers store data for the bandits. This way a bandit is not limited to the current batch of data but can
look at previous data e.g. during a complete retraining.
Currently we only provide a simple in-memory buffer but more sophisticated buffer can be implemented by
subclassing the `AbstractBanditDataBuffer` class. 

To achieve different buffer strategies, one can provide a `DataRetrievalStrategy` to the buffer.
The buffer will then use the `get_training_indices` method to get the indices of the data to use for training.

Currently, we provide two strategies:

- `AllDataRetrievalStrategy`: Use all data for training.

- `SlidingWindowRetrievalStrategy`: Use a sliding window of the last `window_size` data points for training.

Custom strategies can be implemented by subclassing the `DataRetrievalStrategy` class and implementing the `get_training_indices` method.

<br>
<br>

## **Data Buffers**

::: calvera.utils.AbstractBanditDataBuffer
    handler: python
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: 
        - add_batch
        - get_batch
        - update_embeddings
        - __len__
        - state_dict
        - load_state_dict


::: calvera.utils.InMemoryDataBuffer
    handler: python
    options:
      heading: InMemoryDataBuffer
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members:
        - __init__


::: calvera.utils.ListDataBuffer
    handler: python
    options:
      heading: ListDataBuffer
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members:
        - __init__

<br>
<br>

## **Strategies**

::: calvera.utils.data_storage.DataRetrievalStrategy
    handler: python
    options:
      heading: DataRetrievalStrategy
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: 
        - get_training_indices

::: calvera.utils.data_storage.SlidingWindowRetrievalStrategy
    handler: python
    options:
      heading: SlidingWindowRetrievalStrategy
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: False

::: calvera.utils.data_storage.AllDataRetrievalStrategy
    handler: python
    options:
      heading: AllDataRetrievalStrategy
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: False
