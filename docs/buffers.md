DataBuffers store data for the bandits. This way a bandit is not limited to the current batch of data but can
look at previous data e.g. during a complete retraining.
Currently we only provide a simple in-memory buffer but more sophisticated buffer can be implemented by
subclassing the `AbstractBanditDataBuffer` class. 

To achieve different buffer strategies, one can provide a `DataBufferStrategy` to the buffer.
The buffer will then use the `get_training_indices` method to get the indices of the data to use for training.

Currently, we provide two strategies:

- `AllDataBufferStrategy`: Use all data for training.

- `SlidingWindowBufferStrategy`: Use a sliding window of the last `window_size` data points for training.

Custom strategies can be implemented by subclassing the `DataBufferStrategy` class and implementing the `get_training_indices` method.

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


::: calvera.utils.TensorDataBuffer
    handler: python
    options:
      heading: TensorDataBuffer
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

::: calvera.utils.data_storage.DataBufferStrategy
    handler: python
    options:
      heading: DataBufferStrategy
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: 
        - get_training_indices

::: calvera.utils.data_storage.SlidingWindowBufferStrategy
    handler: python
    options:
      heading: SlidingWindowBufferStrategy
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: False

::: calvera.utils.data_storage.AllDataBufferStrategy
    handler: python
    options:
      heading: AllDataBufferStrategy
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
      members: False
