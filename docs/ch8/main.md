# DQN 算法进阶

本章开始介绍一些基于 DQN 改进的一些算法
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

总的来讲，Dueling DQN 算法在某些情况下相对于 DQN 是有好处的，因为它分开评估每个状态的价值以及某个状态下采取某个动作的 $Q$ 值。当某个状态下采取一些动作对最终的回报都没有多大影响时，这个时候 Dueling DQN 这种结构的优越性就体现出来了。或者说，它使得目标值更容易计算，因为通过使用两个单独的网络，我们可以隔离每个网络输出上的影响，并且只更新适当的子网络，这有助于降低方差并提高学习稳定性。
## PER DQN 算法
## HER DQN 算法

## Noisy DQN 算法

## QR DQN 算法

## Rainbow DQN 算法