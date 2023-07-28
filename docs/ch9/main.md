# 策略梯度


本章开始介绍基于策略梯度（policy based）的算法，与前面介绍的基于价值（value based）的算法（包括 DQN 等算法）不同，这类算法直接对策略本身进行近似优化。在这种情况下，我们可以将策略描述成一个带有参数$\theta$的连续函数，该函数将某个状态作为输入，输出的不再是某个确定性（deterministic）的离散动作，而是对应的动作概率分布，通常用$\pi_{\theta}(a|s)$表示，称作随机性（stochastic）策略。

## 基于价值算法的缺点

尽管以 DQN 算法为代表的基于价值的算法在很多任务上都取得了不错的效果，并且具备较好的收敛性，但是这类算法也存在一些缺点。

* **无法表示连续动作**。由于 DQN 等算法是通过学习状态和动作的价值函数来间接指导策略的，因此它们只能处理离散动作空间的问题，无法表示连续动作空间的问题。而在一些问题中，比如机器人的运动控制问题，连续动作空间是非常常见的，比如要控制机器人的运动速度、角度等等，这些都是连续的量。

* **高方差**。基于价值的方法通常都是通过采样的方式来估计价值函数，这样会导致估计的方差很高，从而影响算法的收敛性。尽管一些 DQN 改进算法，通过改善经验回放、目标网络等方式，可以在一定程度上减小方差，但是这些方法并不能完全解决这个问题。

* **探索与利用的平衡问题**。DQN 等算法在实现时通常选择贪心的确定性策略，而很多问题的最优策略是随机策略，即需要以不同的概率选择不同的动作。虽然可以通过 $\epsilon$-greedy 等方式来实现一定程度的随机策略，但是实际上这种方式并不是很理想，因为它并不能很好地平衡探索与利用的关系。

## 策略梯度算法

策略梯度算法是一类直接对策略进行优化的算法，但它的优化目标与基于价值的算法是一样的，都是累积的价值期望$V^{*}(s)$。我们通常用 $\pi_{\theta}(a|s)$ 来表示策略，即 在状态 $s$ 下采取动作 $a$ 的概率分布 $ p(a|s)$，其中 $\theta$ 是我们要去求出来的模型参数。

我们知道智能体在与环境交互的过程时，首先环境会产生一个初始状态 $s_0$，然后智能体相应地执行动作 $a_0$, 然后环境会转移到下一个状态 $s_1$ 并反馈一个奖励 $r_1$，智能体再根据当前状态 $s_1$ 选择动作 $a_1$，以此类推，直到环境转移到终止状态。我们将这个过程称为一个**回合**（episode），然后把所有的状态和动作按顺序组合起来，记作 $\tau$，称为**轨迹**（trajectory），即

$$
\tau=\left\{s_{0}, a_{0}, s_{1}, a_{1}, \cdots, s_{T}, a_{T}\right\}
$$

其中 $T$ 表示回合的终止时刻。由于环境初始化是随机的，我们假设产生初始状态 $s_0$ 的概率为 $p(s_0)$，那么给定策略函数$\pi_{\theta}(a|s)$的情况下，其实是很容易计算出轨迹 $\tau$ 产生的概率的，用 $P_{\theta}(\tau)$ 表示。为了方便读者理解，我们假设有一个很短的轨迹 $\tau_0 = \{s_0,a_0,s_1\}$，即智能体执行一个动作之后就终止本回合了。要计算该轨迹产生的概率，我们可以拆分一下在这条轨迹产生的过程中出现了那些概率事件，首先是环境初始化产生状态 $s_0$，接着是智能体采取动作 $a_0$，然后环境转移到状态 $s_1$，即整个过程有三个概率事件，那么根据条件概率的乘法公式，该轨迹出现的概率应该为 环境初始化产生状态 $s_0$的概率 $p(s_0)$ 乘以智能体采取动作 $a_0$ 的概率 $\pi_{\theta}(a_0|s_0)$ 乘以环境转移到状态 $s_1$ 的概率 $p(s_1|s_0,a_0)$，即 $P_{\theta}(\tau_0) = \pi_{\theta}(a_0|s_0)p(s_1|s_0,a_0)$。依此类推，对于任意轨迹 $\tau$ ，其产生的概率为

$$
\begin{aligned}
P_{\theta}(\tau)
&=p(s_{0}) \pi_{\theta}(a_{0} | s_{0}) p(s_{1} | s_{0}, a_{0}) \pi_{\theta}(a_{1} | s_{1}) p(s_{2} | s_{1}, a_{1}) \cdots \\
&=p(s_{0}) \prod_{t=0}^{T} \pi_{\theta}\left(a_{t} | s_{t}\right) p\left(s_{t+1} | s_{t}, a_{t}\right)
\end{aligned}
$$

注意公式中所有的概率都是大于 0 的，否则也不会产生这条轨迹了。前面提到，同基于价值的算法一样，策略梯度算法的优化目标也基本是每回合的累积奖励期望，即我们通常讲的回报 $G$（return）。 我们将环境在每一步状态和动作下产生的奖励记作一个函数 $r_{t+1}=r(s_t,a_t),t=0,1,\cdots$，那么对于一条轨迹 $\tau$ 来说，对应的累积奖励就可以计算为 $R(\tau)=\sum_{t=0}^T r\left(s_t, a_t\right)$，注意这里出于简化考虑我们忽略了折扣因子 $\gamma$。那么在给定的策略下，即参数 $\theta$ 固定，对于不同的初始状态，会形成不同的轨迹 $\tau_{1},\tau_{2},\cdots$，对应轨迹的出现概率前面已经推导出来为 $P_{\theta}(\tau_{1}),P_{\theta}(\tau_{2}),\cdots$，累积奖励则为 $R(\tau_{1}),R(\tau_{2}),\cdots$。结合概率论中的全期望公式，我们可以得到策略的价值期望公式，如下：

$$
\begin{aligned}
J(\pi_{\theta}) = \underset{\tau \sim \pi_\theta}{E}[R(\tau)] 
& = P_{\theta}(\tau_{1})R(\tau_{1})+P_{\theta}(\tau_{2})R(\tau_{2})+\cdots \\
&=\int_\tau P_{\theta}(\tau) R(\tau) \\ 
&=E_{\tau \sim P_\theta(\tau)}[\sum_t r(s_t, a_t)] 
\end{aligned}
$$

换句话说，我们的目标就是最大化策略的价值期望 $J(\pi_{\theta})$，因此 $J(\pi_{\theta})$ 又称作目标函数。有了目标函数之后，只要能求出梯度，就可以使用万能的梯度上升或下降的方法来求解对应的最优参数 $\theta^*$了，这里由于我们的目标是最大化目标函数，因此我们使用梯度上升的方法。那么问题来了，我们发现策略梯度的目标函数过于复杂，这种情况下要怎么求梯度呢？这就是策略梯度算法的核心问题。乍一看，这个策略梯度公式是很复杂，但是仔细观察之后，首先会发现我们的目标是求关于参数 $\theta$ 的梯度，而公式中的$R(\tau)$跟$\theta$其实是没有关联的，因此在求解梯度的时候可以将这一项看作常数，这样一来问题就稍稍简化成了如何求解 $P_{\theta}(\tau)$ 的梯度了。这个时候我们就需要回忆起中学就用过的一个对数微分技巧，即 $\log x$ 的导数是 $1/x$。注意，有同学可能会奇怪不是 $ \ln x$ 的导数才是 $ 1/x$ 吗，这其实涉及到一个国际的沿用标准问题，国际上通常使用 $\log x$ 表示以 $e$ 为底的对数，国内数学教材基本沿用了早期的 ISO 标准，即使用 $\ln x$ 表示以 $e$ 为底的对数，东直门只需要记住在算法领域默认使用 $\log x$ 表示以 $e$ 为底的对数即可。回到我们的问题，使用这个对数微分技巧，我们就可以将目标函数的梯度做一个转化，即：

$$
\nabla_\theta P_{\theta}(\tau)= P_{\theta}(\tau) \frac{\nabla_\theta P_{\theta}(\tau)}{P_{\theta}(\tau) }= P_{\theta}(\tau) \nabla_\theta \log P_{\theta}(\tau)
$$


现在的问题就从求$P_{\theta}(\tau)$的梯度变成了求$\log P_{\theta}(\tau)$的梯度了，即求$\nabla_\theta \log P_{\theta}(\tau)$。我们先求出$\log P_{\theta}(\tau)$，根据 $P_{\theta}(\tau)=p(s_{0}) \prod_{t=0}^{T} \pi_{\theta}\left(a_{t} | s_{t}\right) p\left(s_{t+1}  s_{t}, a_{t}\right)$，再根据对数公式$log (ab) = log a + log b$，即可求出：

$$
\label{eq:station_dist_log}
\log P_{\theta}(\tau)= \log p(s_{0})  +  \sum_{t=0}^T(\log \pi_{\theta}(a_t \mid s_t)+\log p(s_{t+1} \mid s_t,a_t))
$$

我们会惊奇地发现$\log P_{\theta}(\tau)$展开之后只有中间的项$\log \pi_{\theta}(a_t \mid s_t)$跟参数$\theta$有关，也就是说其他项关于$\theta$的梯度为0，即：

$$
\label{eq:station_dist_log_grad}
\begin{aligned}
\nabla_\theta \log P_{\theta}(\tau) &=\nabla_\theta \log \rho_0\left(s_0\right)+\sum_{t=0}^T\left(\nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)+\nabla_\theta \log p\left(s_{t+1} \mid s_t, a_t\right)\right) \\
&=0+\sum_{t=0}^T\left(\nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)+0\right) \\
&=\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)
\end{aligned}
$$


现在我们就可以很方便地求出目标函数的梯度了，如下

$$
\begin{aligned}
\nabla_\theta J\left(\pi_\theta\right) &=\nabla_\theta \underset{\tau \sim \pi_\theta}{\mathrm{E}}[R(\tau)] \\
&=\nabla_\theta \int_\tau P_{\theta}(\tau) R(\tau) \\
&=\int_\tau \nabla_\theta P_{\theta}(\tau) R(\tau) \\
&=\int_\tau P_{\theta}(\tau) \nabla_\theta \log P_{\theta}(\tau) R(\tau) \\
&=\underset{\tau \sim \pi_\theta}{\mathrm{E}}\left[\nabla_\theta \log P_{\theta}(\tau) R(\tau)\right]\\
&= \underset{\tau \sim \pi_\theta}{\mathrm{E}}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) R(\tau)\right]
\end{aligned}
$$

这里简单解释一下上述公式中的步骤，首先第一行就是目标函数的表达形式，到第二行就是全期望展开式，到第三行就是利用了积分的梯度性质，即梯度可以放到积分号的里面也就是被积函数中，第四行到最后就是对数微分技巧了。回过头来看下，我们为什么要用到对数微分技巧呢？这其实是一个常见的数学技巧：当我们看到公式中出现累乘的项时，我们通常都会取对数简化，因为根据对数公式的性质可以将累乘的项转换成累加的项，这样一来问题会更加便于处理。