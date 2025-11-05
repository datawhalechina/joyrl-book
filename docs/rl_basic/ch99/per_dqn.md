# PER DQN 算法实战

## 伪代码

$\text{PER DQN}$ 算法的核心看起来简单，就是把普通的经验回放改进成了优先级经验回放，但是实现起来却比较复杂，因为我们需要实现一个 $\text{SumTree}$ 结构，并且在模型更新的时候也需要一些额外的操作，因此我们先从伪代码开始，如图 $\text{8-7}$ 所示。


<div align=center>
<img width="500" src="figs/per_dqn_pseu.png"/>
</div>
<div align=center>图 $\text{8-7}$ $\text{PER DQN}$ 伪代码</div>

### SumTree 结构

如代码清单 $\text{8-6}$ 所示，我们可以先实现 $\text{SumTree}$ 结构。

<div style="text-align: center;">
    <figcaption> 代码清单 $\text{8-6}$ $\text{SumTree}$ 结构 </figcaption>
</div>

```python
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # 树的大小，叶节点数等于capacity
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        '''向树中添加数据
        '''
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        '''更新树中节点的优先级
        '''
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        '''根据给定的值v，找到对应的叶节点
        '''
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def max_priority(self):
        '''获取当前树中最大的优先级
        '''
        return self.tree[-self.capacity:].max()
        
    @property
    def total_priority(self):
        '''获取当前树中所有优先级的和
        '''
        return self.tree[0]

```

其中，除了需要存放各个节点的值`tree`之外，我们需要定义要给`data`来存放叶子节点的样本。此外，`add`函数用于添加一个样本到叶子节点，并更新其父节点的优先级；`update`函数用于更新叶子节点的优先级，并更新其父节点的优先级；`get_leaf`函数用于根据优先级的值采样对应区间的叶子节点样本；`get_data`函数用于根据索引获取对应的样本。

## 优先级经验回放

基于 $\text{SumTree}$ 结构，并结合优先级采样和重要性采样的技巧，如代码清单 $\text{8-7}$ 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 $\text{8-7}$ 优先级经验回放结构 </figcaption>
</div>

```python
class ReplayBuffer:
    def __init__(self, cfg):
        self.capacity = cfg.buffer_size
        self.alpha = cfg.per_alpha
        self.beta = cfg.per_beta
        self.beta_increment_per_sampling = cfg.per_beta_increment_per_sampling
        self.epsilon = cfg.per_epsilon
        self.tree = SumTree(self.capacity)
        
    def push(self, transition):
        # max_prio = self.tree.tree[-self.tree.capacity:].max()
        max_prio = self.tree.max_priority
        if max_prio == 0:
            max_prio = 1.0
        self.tree.add(max_prio, transition)

    def sample(self, batch_size):
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        minibatch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            minibatch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        batch = list(zip(*minibatch))
        return tuple(map(lambda x: np.array(x), batch)), idxs, is_weight

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, (np.abs(priority) + self.epsilon) ** self.alpha)

    def __len__(self):
        return len(self.tree.data)
```

我们可以看到，优先级经验回放的核心是 SumTree，它可以在 $O(\log N)$ 的时间复杂度内完成添加、更新和采样操作。在实践中，我们可以将经验回放的容量设置为 $10^6$，并将 $\alpha$ 设置为 $0.6$，$\epsilon$ 设置为 $0.01$，$\beta$ 设置为 $0.4$，$\beta_{\text{step}}$ 设置为 $0.0001$。 当然我们也可以利用 Python 队列的方式实现优先级经验回放，形式上会更加简洁，并且在采样的时候减少了 for 循环的操作，会更加高效，如代码清单 $\text{8-8}$ 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 $\text{8-8}$ 基于队列实现优先级经验回放 </figcaption>
</div>

```python
class PrioritizedReplayBufferQue:
    def __init__(self, cfg):
        self.capacity = cfg.buffer_size
        self.alpha = cfg.per_alpha # 优先级的指数参数，越大越重要，越小越不重要
        self.epsilon = cfg.per_epsilon # 优先级的最小值，防止优先级为0
        self.beta = cfg.per_beta # importance sampling的参数
        self.beta_annealing = cfg.per_beta_annealing # beta的增长率
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.count = 0 # 当前存储的样本数量
        self.max_priority = 1.0
    def push(self,exps):
        self.buffer.append(exps)
        self.priorities.append(max(self.priorities, default=self.max_priority))
        self.count += 1
    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities/sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (self.count*probs[indices])**(-self.beta)
        weights /= weights.max()
        exps = [self.buffer[i] for i in indices]
        return zip(*exps), indices, weights
    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities)
        priorities = (priorities + self.epsilon) ** self.alpha
        priorities = np.minimum(priorities, self.max_priority).flatten()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    def __len__(self):
        return self.count
```
最后，我们可以将优先级经验回放和 $\text{DQN}$ 结合起来，实现一个带有优先级的 $\text{DQN}$ 算法，并展示它在 $\text{CartPole}$ 环境下的训练结果，如图 $\text{8-8}$ 所示。

<div align=center>
<img width="400" src="figs/PERDQN_CartPole-v1_training_curve.png"/>
</div>
<div align=center>图 $\text{8-8}$ $\text{CartPole}$ 环境 $\text{PER DQN}$ 算法训练曲线</div>