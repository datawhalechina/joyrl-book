# 蒙特卡洛方法

蒙特卡洛方法（ $\text{Monte Carlo，MC}$ ）的核心思想是通过大量的随机采样来近似估计期望或积分。在强化学习中，一方面可以用来解决预测问题，即估计状态价值函数 $V(s)$ 或动作价值函数 $Q(s,a)$。另一方面，可以用来优化策略，即通过采样来评估和改进策略。

## 状态价值计算示例

为帮助理解蒙特卡洛方法，我们先举一个简单的例子来根据定义计算状态价值，然后再介绍蒙特卡洛预测算法。

如图 1 所示，考虑智能体在 $2 \times 2$ 的网格中使用随机策略进行移动，以左上角为起点，右下角为终点，规定每次只能向右或向下移动，动作分别用 $a_1$ 和 $a_2$ 表示。用智能体的位置不同的状态，即$s_1,s_2,s_3,s_4$，初始状态为$S_0=s_1$。设置每走一步接收到的奖励为 $-1$， 折扣因子 $\gamma=0.9$，目标是计算各个状态的价值函数 $V(s)$。

<div align=center>
<img width="200" src="figs/simple_maze.png"/>
</div>
<div align=center>图 1 迷你网格示例</div>

回顾状态价值函数的定义，如式 $\eqref{eq:state_value}$ 所示。

$$
\begin{equation}\label{eq:state_value}
\begin{aligned}
V_\pi(s) &=\mathbb{E}_{\pi}[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3} + \cdots |S_t=s ] \\
&=\mathbb{E}_{\pi}[G_t|S_t=s ] 
\end{aligned}
\end{equation}
$$

现在根据定义来分别计算各状态的价值，首先由于 $s_4$ 是终止状态，因此 $V(s_4)=0$。

接下来计算 $s_2$ 的价值 $V(s_2)$，从 $s_2$ 出发只能向下走到达终点 $s_4$，对应的轨迹为 $\tau_1 = \{s_2,a_2,r(s_2,a_2),s_4\}$，回报为 $G_{\tau_1} = r(s_2,a_2)=-1$，因此 $V(s_2) = G_{\tau_1} = -1$。

然后计算 $s_3$ 的价值 $V(s_3)$，从 $s_3$ 出发只能向右走到达终点 $s_4$，对应的轨迹为 $\tau_2 = \{s_3,a_1,r(s_3,a_1),s_4\}$，回报为 $G_{\tau_2} = r(s_3,a_1)=-1$，因此 $V(s_3) = G_{\tau_2} = -1$。

最后计算起始状态 $s_1$ 的价值 $V(s_1)$，从 $s_1$ 出发有两条可能的轨迹，其一是 $s_1 \to s_2 \to s_4$，其二是 $s_1 \to s_3 \to s_4$，对应的轨迹分别如式 $\eqref{eq:tau3}$ 和 $\eqref{eq:tau4}$ 所示。

$$
\begin{equation}\label{eq:tau3}
\tau_3 = \{s_1,a_1,r(s_1,a_1),s_2,a_2,r(s_2,a_2),s_4\}
\end{equation}
$$

$$
\begin{equation}\label{eq:tau4}
\tau_4 = \{s_1,a_2,r(s_1,a_2),s_3,a_1,r(s_3,a_1),s_4\}
\end{equation}
$$

相应地，对应的回报计算分别如式 $\eqref{eq:G_tau3}$ 和 $\eqref{eq:G_tau4}$ 所示。

$$
\begin{equation}\label{eq:G_tau3}
G_{\tau_3} = r(s_1,a_1) + \gamma r(s_2,a_2)= (-1) + 0.9 * (-1) = -1.9
\end{equation}
$$

$$
\begin{equation}\label{eq:G_tau4}
G_{\tau_4} = r(s_1,a_2) + \gamma r(s_3,a_1)= (-1) + 0.9 * (-1) = -1.9
\end{equation}
$$

由于智能体采用随机策略，因此两条轨迹的概率相等，均为 $0.5$。因此，$V(s_1)$ 可以表示为式 $\eqref{eq:V_s1}$ 。

$$
\begin{equation}\label{eq:V_s1}
V(s_1) = 0.5 * G_{\tau_3} + 0.5 * G_{\tau_4} = 0.5 * (-1.9) + 0.5 * (-1.9) = -1.9
\end{equation}
$$

下面将介绍蒙特卡洛方法是如何通过采样来估计状态价值的。

## 蒙特卡洛预测

蒙特卡洛估计是一种用随机采样近似求期望、积分或概率分布特征的通用方法。换句话说，如果想求一个复杂的数学期望（或积分），而无法直接解析求解时，就可以用大量随机样本的平均值去逼近它。

假设我们想要估计某个函数 $f(x)$ 的期望，如式 $\eqref{eq:expectation}$ 所示。

$$
\begin{equation}\label{eq:expectation}
\mathbb{E}[f(X)] = \int f(x) p(x) dx
\end{equation}
$$

其中 $p(x)$ 是 随机变量 $X$ 的概率密度函数。直接计算这个积分可能很复杂，但我们可以通过蒙特卡洛采样来近似估计它。具体步骤如下：

1. 从概率分布 $p(x)$ 中采样 $N$ 个独立同分布的样本 $\{x_1, x_2, \ldots, x_N\}$。
2. 计算函数值的平均，如式 $\eqref{eq:monte_carlo_estimate}$ 所示。

$$
\begin{equation}\label{eq:monte_carlo_estimate}
\hat{\mathbb{E}}[f(X)] = \frac{1}{N} \sum_{i=1}^{N} f(x_i)
\end{equation}
$$

根据大数定律，当样本数量 $N$ 足够大时，估计值 $\hat{\mathbb{E}}[f(X)]$ 会收敛到真实的期望值 $\mathbb{E}[f(X)]$。

为帮助理解，我们来演示如何通过蒙特卡洛方法来估计圆周率 $\pi$。考虑一个单位正方形内切一个单位圆，圆的面积为 $\pi r^2 = \pi$，正方形的面积为 $4$。如果我们在正方形内随机撒点，落在圆内（即满足 $x^2 + y^2 \leq 1$）的点数与总点数的比例应该接近于圆的面积与正方形面积的比例，即 $\pi / 4$。

用 `Python` 代码实现这个过程，如代码 1 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 1 使用蒙特卡洛方法估计 $\pi$ 的值</b> </figcaption>
</div>

```python
import random

def monte_carlo_pi(num_samples=1000000):
    count_in_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            count_in_circle += 1
    pi_estimate = 4 * count_in_circle / num_samples
    return pi_estimate

print("Estimated π:", monte_carlo_pi())
```

运行代码后，可以得到一个接近 $\pi$ 的估计值。随着采样数量的增加，估计值会越来越精确。   

**蒙特卡洛预测**（$\text{Monte Carlo Prediction}$）则指的是，在强化学习中，利用蒙特卡洛估计来预测给定策略 $\pi$ 下的状态价值 $V_\pi(s)$ 或动作价值 $Q_\pi(s,a)$。

我们先看如何预测或者说估计状态价值，思路是多次完整地执行策略 $\pi$，每次执行都会产生一条完整的轨迹（从初始状态到终止状态），然后根据这些轨迹来计算各个状态的回报，最后取平均作为该状态的价值估计，如式 $\eqref{eq:mc_value_estimate}$ 所示。

$$
\begin{equation}\label{eq:mc_value_estimate}
V_\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}
\end{equation}
$$

蒙特卡洛方法主要分成两种算法，一种是首次访问蒙特卡洛（$\text{first-visit Monte Carlo，FVMC}$）方法，另外一种是每次访问蒙特卡洛（$\text{every-visit Monte Carlo，EVMC}$）方法。$\text{FVMC}$ 方法主要包含两个步骤，首先是产生一个回合的完整轨迹，然后遍历轨迹计算每个状态的回报。注意，只在第一次遍历到某个状态时会记录并计算对应的回报，对应伪代码如图 $\text{4-3}$ 所示。

而在 $\text{EVMC}$ 方法中不会忽略同一状态的多个回报，在前面的示例中，我们计算价值函数的方式就是 $\text{every-visit}$ ，比如对于状态 $s_4$ ，我们考虑了所有轨迹即 $G_{\tau_3}$ 和 $G_{\tau_4}$ 的回报，而在 $\text{FVMC}$ 我们只会记录首次遍历的回报，即 $G_{\tau_3}$ 和 $G_{\tau_4}$ 其中的一个，具体取决于遍历到 $s_4$ 时对应的轨迹是哪一条。 

<div align=center>
<img width="400" src="figs/fvmc_pseu.png"/>
</div>
<div align=center>图 $\text{4-3}$ 首次访问蒙特卡洛算法伪代码</div>

实际上无论是 $\text{FVMC}$ 还是 $\text{EVMC}$ 在实际更新价值函数的时候是不会像伪代码中体现的那样 $V\left(S_t\right) \leftarrow \operatorname{average}\left(\operatorname{Returns}\left(S_t\right)\right)$，每次计算到新的回报 $ G_t = \operatorname{average}\left(\operatorname{Returns}\left(S_t\right)\right)$ 直接就赋值到已有的价值函数中，而是以一种递进更新的方式进行的，如式 $\text{(4.3)}$ 所示。

$$
\tag{4.3}
新的估计值 \leftarrow 旧的估计值 + 步长 *（目标值-旧的估计值）
$$

这样的好处就是不会因为个别不好的样本而导致更新的急剧变化，从而导致学习得不稳定，这种模式在今天的深度学习中普遍可见，这里的步长就是深度学习中的学习率。

对应到蒙特卡洛方法中，更新公式可表示为式 $\text{(4.4)}$ 。

$$
\tag{4.4}
V(s_t) \leftarrow V(s_t) + \alpha[G_t- V(s_{t})]
$$

其中 $\alpha$ 表示学习率，$G_t- V(S_{t+1})$为目标值与估计值之间的误差（ $\text{error}$ ）。此外，$\text{FVMC}$ 是一种基于回合的增量式方法，具有无偏性和收敛快的优点，但是在状态空间较大的情况下，依然需要训练很多个回合才能达到稳定的结果。而 $\text{EVMC}$ 则是更为精确的预测方法，但是计算的成本相对也更高。


## 小结

本章主要介绍了蒙特卡洛方法的基本思想及其在强化学习中的应用，重点讲解了蒙特卡洛预测算法。通过随机采样和计算回报，蒙特卡洛方法能够有效地估计状态价值函数和动作价值函数。我们还讨论了首次访问蒙特卡洛（$\text{FVMC}$）和每次访问蒙特卡洛（$\text{EVMC}$）两种不同的实现方式，以及它们各自的优缺点。蒙特卡洛方法作为一种无模型的强化学习方法，为后续更复杂的算法奠定了基础。

