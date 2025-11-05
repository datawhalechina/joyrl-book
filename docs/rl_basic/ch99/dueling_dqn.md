# Dueling DQN 算法实战

$\text{Dueling DQN}$ 算法主要是改了网络结构，其他地方跟 $\text{DQN}$ 是一模一样的，如代码清单 1 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 1 $\:$ $\text{Dueling DQN}$ 网络结构 </figcaption>
</div>

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        # 隐藏层
        self.hidden_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        #  优势层
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # 价值层
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        x = self.hidden_layer(state)
        advantage = self.advantage_layer(x)
        value     = self.value_layer(x)
        return value + advantage - advantage.mean() # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
```

最后我们展示一下它在 $\text{CartPole}$ 环境下的训练结果，如图 1 所示，完整的代码同样可以参考本书的代码仓库。

<div align=center>
<img width="400" src="figs/DuelingDQN_CartPole-v1_training_curve.png"/>
</div>
<div align=center>图 1 $\:$ $\text{CartPole}$ 环境 $\text{Dueling DQN}$ 算法训练曲线</div>

由于环境比较简单，暂时还看不出来 $\text{Dueling DQN}$ 算法的优势，但是在复杂的环境下，比如 $\text{Atari}$ 游戏中，$\text{Dueling DQN}$ 算法的效果就会比 $\text{DQN}$ 算法好很多，读者可以在 $\text{JoyRL}$ 仓库中找到更复杂环境下的训练结果便于更好地进行对比。