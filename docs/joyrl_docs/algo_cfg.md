默认的算法参数配置在`joyrl/algos/[algo_name]/config.py`中，具体请分别参考各算法说明。

### Q-learning

```python
class AlgoConfig:
    def __init__(self) -> None:
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 300 # epsilon decay rate
        self.gamma = 0.90 # discount factor
        self.lr = 0.1 # learning rate
```

注意：

* 设置`epsilon_start=epsilon_end`可以得到固定的`epsilon=epsilon_end`。
参数说明：

* 适当调整`epsilon_decay`以保证`epsilon`在训练过程中不会过早衰减。
* 由于传统强化学习算法面对的环境都比较简单，因此`gamma`一般设置为`0.9`，且`lr`可以设置得比较大如`0.1`，不用太担心过拟合的情况。

### DQN

```python
class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
        self.gamma = 0.95  # discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_size = 100000  # size of replay buffer
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}]
```