# DQN 算法进阶

本章开始介绍一些基于 DQN 改进的一些算法，由于本章的算法都是网络模型、经验回放等层面的改进，整体的训练模式跟 DQN 算法是一模一样的，因此出于简洁考虑，本章算法不再贴出对应的伪代码。另一方面来讲，建议读者在阅读本章之前，牢牢掌握 DQN 算法的原理。
## Double DQN 算法

Double DQN 算法<sup>①</sup>是谷歌 DeepMind 于 2015 年 12 月提出的一篇解决 $Q$ 值过估计（overestimate）的论文。

> ① 论文链接：http://papers.neurips.cc/paper/3964-double-q-learning.pdf

回忆一下 DQN 算法的更新公式，如下：
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_t+\gamma\max _{a}Q^{\prime}(s_{t+1},a)-Q(s_t,a_t)]
$$

其中 $y_t = r_t+\gamma\max _{a}Q^{\prime}(s_{t+1},a)$ 是估计值，注意这里的 $Q^{\prime}$ 是目标网络（DQN 算法中有两个网络，一个是目标网络，一个是当前网络或者说策略网络）。

在 Double DQN 算法中则不再是直接在目标网络中寻找各动作最大的 $Q$ 值，而是现在策略网络中找出最大 $Q$ 值对应的动作如下：

$$
a^{max}_{\theta}(s_{t+1}) = \arg \max _{a}Q_{\theta}(s_{t+1},a)
$$

然后将这个找出来的动作代入到目标网络里面去计算目标的 $Q$ 值，进而计算估计值，如下：

$$
y_t = r_t+\gamma\max _{a}Q^{\prime}_{\theta^{\prime}}(s_{t+1},a^{max}_{\theta}(s_{t+1}))
$$

到这里我们就讲完了 Double DQN 算法的核心，在原论文中还通过拟合曲线实验证明了过估计是真实存在且对实验结果有重大影响等相关细节，感兴趣的读者深入研究。为了方便读者理解，我们接着用皇帝与大臣的例子来举例说明为什么 Double DQN 算法能够估计得更准确。我们知道在 DQN 算法中策略网络直接与环境交互相当于大臣们搜集情报，然后定期更新到目标网络的过程相当于大臣向皇帝汇报然后皇帝做出最优决策。在 DQN 算法中，大臣是不管好的还是坏的情报都会汇报给皇帝的，而在  Double DQN 算法中大臣会根据自己的判断将自己认为最优的情报汇报给皇帝，即先在策略网络中找出最大 $Q$ 值对应的动作。这样一来皇帝这边得到的情报就更加精简并且质量更高了，以便于皇帝做出更好的判断和决策，也就是估计得更准确了。

Double DQN 算法核心代码如下，完整代码参考 JoyRL ：

```python
# 计算当前Q(s_t|a=a_t)，即Q网络的输出，这里的gather函数的作用是根据action_batch中的值，从Q网络的输出中选出对应的Q值
q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1)
# 计算 Q(s_t+1|a)
next_q_value_batch = self.policy_net(next_state_batch)
# 计算 Q'(s_t+1|a)，也是与DQN不同的地方
next_target_value_batch = self.target_net(next_state_batch)
# 计算 Q'(s_t+1|a=argmax Q(s_t+1|a))
next_target_q_value_batch = next_target_value_batch.gather(1, torch.max(next_q_value_batch, 1)[1].unsqueeze(1)) # shape(batchsize,1)
```

整体代码跟 DQN 算法基本一致，只是在计算目标 $Q$ 值的时候多了一步，即先在策略网络中找出最大 $Q$ 值对应的动作，然后再在目标网络中计算目标 $Q$ 值。

## Dueling DQN 算法

在 Double DQN 算法中我们是通过改进目标 $Q$ 值的计算来优化算法的，而在 Dueling DQN 算法<sup>②</sup>中则是通过优化神经网络的结构来优化算法的。

> ② Dueling DQN 算法论文：Dueling Network Architectures for Deep Reinforcement Learning

回忆 DQN 算法的网络结构，如图 8.1 所示，输入层的维度就是状态数，输出层的维度就是动作数。

<div align=center>
<img width="300" src="../figs/ch8/dqn_network.png"/>
</div>
<div align=center>图 8.1 DQN 网络结构</div>

而 Dueling DQN 算法中则是在输出层之前分流（Dueling）出了两个层，一个是优势层（Advantage Layer），用于估计每个动作带来的优势，输出维度为动作数一个是价值层（Value Layer），用于估计每个状态的价值，输出维度为 $1$ 。
<div align=center>
<img width="400" src="../figs/ch8/dueling_network.png"/>
</div>
<div align=center>图 8.2 Dueling DQN 网络结构</div>

在 DQN 算法中我们用 $Q_{\theta}(\boldsymbol{s},\boldsymbol{a})$ 表示 一个 $Q$ 网络，而在这里优势层可以表示为 $A_{\theta,\alpha}(\boldsymbol{s},\boldsymbol{a})$，这里 $\theta$ 表示共享隐藏层的参数，$\alpha$ 表示优势层自己这部分的参数，相应地价值层可以表示为 $V_{\theta,\beta}(\boldsymbol{s})$。这样 Dueling DQN 算法中网络结构可表示为：

$$
Q_{\theta,\alpha,\beta}(\boldsymbol{s},\boldsymbol{a}) = A_{\theta,\alpha}(\boldsymbol{s},\boldsymbol{a}) + V_{\theta,\beta}(\boldsymbol{s})
$$

去掉这里的价值层即优势层就是普通的 $Q$ 网络了，另外我们会对优势层做一个中心化处理，即减掉均值，如下：

$$
Q_{\theta,\alpha,\beta}(\boldsymbol{s},\boldsymbol{a}) = (A_{\theta,\alpha}(\boldsymbol{s},\boldsymbol{a})-\frac{1}{\mathcal{A}} \sum_{a \in \mathcal{A}} A_{\theta,\alpha}\left(\boldsymbol{s}, a\right)) - + V_{\theta,\beta}(\boldsymbol{s})
$$

其实 Dueling DQN 的网络结构跟我们后面要讲的 Actor-Critic 算法是类似的，这里优势层相当于 Actor，价值层相当于 Critic，不同的是在 Actor-Critic 算法中 Actor 和 Critic 是独立的两个网络，而在这里是合在一起的，在计算量以及拓展性方面都完全不同，具体我们会在后面的 Actor-Critic 算法对应章节中展开。

总的来讲，Dueling DQN 算法在某些情况下相对于 DQN 是有好处的，因为它分开评估每个状态的价值以及某个状态下采取某个动作的 $Q$ 值。当某个状态下采取一些动作对最终的回报都没有多大影响时，这个时候 Dueling DQN 这种结构的优越性就体现出来了。或者说，它使得目标值更容易计算，因为通过使用两个单独的网络，我们可以隔离每个网络输出上的影响，并且只更新适当的子网络，这有助于降低方差并提高学习**鲁棒性（Robustness）**。

根据上面的分析，我们就可以写出 Dueling DQN 网络的代码，如下：

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, n_states, n_actions,hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        # 隐藏层
        self.hidden_layer = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU()
        )
        #  优势层
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
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

## Noisy DQN 算法

Noisy DQN 算法<sup>③</sup> 也是通过优化网络结构的方法来提升 DQN 算法的性能，但与 Dueling 算法不同的是，它的目的并不是为了提高 $Q$ 值的估计，而是增强网络的探索能力。

> ③ 论文链接：https://arxiv.org/abs/1706.10295

从 Q-learning 算法开始，我们就讲到了探索策略以及探索-利用窘境的问题，常见的 $\varepsilon-greedy$ 策略是从智能体与环境的交互过程改善探索能力，以避免陷入局部最优解。而在深度强化学习中，由于引入了深度学习，深度学习本身也会因为网络模型限制或者梯度下降方法陷入局部最优解问题。也就是说，深度强化学习既要考虑与环境交互过程中的探索能力，也要考虑深度模型本身的探索能力，从而尽量避免陷入局部最优解的困境之中，这也是为什么经常有人会说强化学习比深度学习更难“炼丹”的原因之一。

回归正题，Noisy DQN 算法其实是在 DQN 算法基础上在神经网络中引入了噪声层来提高网络性能的，即将随机性应用到神经网络中的参数或者说权重，增加了 $Q$ 网络对于状态和动作空间的探索能力，从而提高收敛速度和稳定性。在实践上也比较简单，就是通过添加随机性参数到神经网络的线性层，对应的 $Q$ 值则可以表示为 $Q_{\theta+\epsilon}$，注意不要把这里的 $\epsilon$ 跟 $\varepsilon-greedy$ 策略中的 $\varepsilon$ 混淆了。虽然都叫做 epsilon，但这里 $\epsilon$ 是由高斯分布生成的总体分类噪声参数。

在实战中，我们首先可以定义引入了噪声层的线性层，如下：

```python
class NoisyLinear(nn.Module):
    '''在Noisy DQN中用NoisyLinear层替换普通的nn.Linear层
    '''
    def __init__(self, in_dim, out_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.std_init  = std_init
        
        self.weight_mu    = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        # 将一个 tensor 注册成 buffer，使得这个 tensor 不被当做模型参数进行优化。
        self.register_buffer('weight_epsilon', torch.empty(out_dim, in_dim)) 
        
        self.bias_mu    = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer('bias_epsilon', torch.empty(out_dim))
        
        self.reset_parameters() # 初始化参数
        self.reset_noise()  # 重置噪声
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / self.in_dim ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_dim ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_dim ** 0.5)
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_dim)
        epsilon_out = self._scale_noise(self.out_dim)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_dim))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
```

根据写好的 NoisyLinear 层，我们可以在 DQN 算法中将普通的线性层替换为 NoisyLinear 层，如下：

```python
class NoisyQNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 =  nn.Linear(n_states, hidden_dim)
        self.noisy_fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_fc3 = NoisyLinear(hidden_dim, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy_fc2(x))
        x = self.noisy_fc3(x)
        return x

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()
```

注意在训练过程中，我们需要在每个 mini-batch 中重置噪声，更多细节请参考 JoyRL 源码。另外，我们可以直接利用 torchrl 模块中中封装好的 NoisyLinear 层来构建 Noisy Q 网络，跟我们自己定义的功能是一样的，如下：

```python
import torchrl
class NoisyQNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 =  nn.Linear(n_states, hidden_dim)
        self.noisy_fc2 = torchrl.NoisyLinear(hidden_dim, hidden_dim,std_init=0.1)
        self.noisy_fc3 = torchrl.NoisyLinear(hidden_dim, n_actions,std_init=0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy_fc2(x))
        x = self.noisy_fc3(x)
        return x

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()
```

## PER DQN 算法

在 DQN 算法章节中我们讲到经验回放，从另一个角度来说也是为了优化深度网络中梯度下降的方式，或者说网络参数更新的方式。在本节要讲的 PER DQN 算法<sup>④</sup>中，进一步优化了经验回放的设计从而提高模型的收敛能力和鲁棒性。PER 英文全称为 Prioritized Experience Replay，即优先经验回放，跟数据结构中优先队列与普通队列一样，会在采样过程中赋予经验回放中样本的优先级。

> ④ 论文链接：https://arxiv.org/abs/1511.05952

具体以什么为依据来为经验回放中的样本赋予不同优先级呢？答案是TD误差。TD误差最开始我们是在时序差分方法的提到的，广义的定义是值函数（包括状态价值函数和动作价值函数）的估计值与实际值之差，在 DQN 算法中就是目标网络计算的 $Q$ 值和策略网络（当前网络）计算的 $Q$ 值之差，也就是 DQN 网络中损失函数的主要构成部分。我们每次从经验回访中取出一个批量的样本，进而计算的TD误差一般是不同的，对于 DQN 网络反向传播的作用也是不同的。**TD误差越大，损失函数的值也越大，对于反向传播的作用也就越大。** 这样一来如果TD误差较大的样本更容易被采到的话，那么我们的算法也会更加容易收敛。因此我们只需要设计一个经验回放，根据经验回放中的每个样本计算出的TD误差赋予对应的优先级，然后在采样的时候取出优先级较大的样本。

原理听上去比较简单，但具体如何实现呢？在实践中，我们通常用 SumTree 这样的二叉树结构来实现。这里建议没有了解过数据结构或者二叉树的读者先花个十几分钟的时间快速了解一下二叉树的基本概念，比如根节点、叶节点、父节点与子节点等等。

<div align=center>
<img width="300" src="../figs/ch8/sumtree.png"/>
</div>
<div align=center>图 8.3 SumTree 结构</div>

如图 8.3 所示，每个父节点的值等于左右两个子节点值之和。在强化学习中，所有的样本只保存在最下面的叶子节点中，并且除了保存样本数据之外，还会保存对应的优先级，即对应叶子节点中的值（例如图中的31、13、14以及8等，也对应样本的 TD 误差）。并且根据叶子节点的值，我们从 $0$ 开始依次划分采样区间。然后在采样中，例如这里根节点值为 66，那么我们就可以在$[0,66)$这个区间均匀采样，采样到的值落在哪个区间中，就说明对应的样本就是我们要采样的样本。例如我们采样到了 $25$ 这个值，即对应区间 $[0,31)$,那么我们就采样到了第一个叶子节点对应的样本。注意到，第一个样本对应的区间也是最长的，这意味着第一个样本的优先级最高，也就是 TD 误差最大，反之第四个样本的区间最短，优先级也最低。这样一来，我们就可以通过采样来实现优先经验回放的功能。

每个叶节点的值就是对应样本的 TD 误差（例如途中的）。我们可以通过根节点的值来计算出每个样本的 TD 误差占所有样本 TD 误差的比例，这样就可以根据比例来采样样本。在实际的实现中，我们可以将每个叶节点的值设置为一个元组，其中包含样本的 TD 误差和样本的索引，这样就可以通过索引来找到对应的样本。

## HER DQN 算法

## QR DQN 算法

## Rainbow DQN 算法