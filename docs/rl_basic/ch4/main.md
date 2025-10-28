# 蒙特卡洛方法

蒙特卡洛方法的核心思想是通过大量的随机采样来近似估计期望或积分。在强化学习中，一方面可以用来解决预测问题，即估计状态价值函数 $V(s)$ 或动作价值函数 $Q(s,a)$。另一方面，可以用来优化策略，即通过采样来评估和改进策略来解决控制问题。

蒙特卡洛预测包括首次访问法和每次访问法两种基本方法，前者只在每个状态的首次访问时更新价值估计，后者则在每次访问时都进行更新。

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

综上所述，各状态的价值函数结果如表 1 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>表 1 各状态的价值函数</b> </figcaption>
</div>

| 状态 | $s_1$ | $s_2$ | $s_3$ | $s_4$ |
| :--: | :---: | :---: | :---: | :---: |
| 价值 | $-1.9$  | $-1.0$  | $-1.0$  |  $0.0$  |

下面将介绍蒙特卡洛方法是如何通过采样来估计状态价值的。

## 蒙特卡洛预测

### 蒙特卡洛估计

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

**蒙特卡洛预测**（$\text{Monte Carlo Prediction}$）则指的是，在强化学习中，利用蒙特卡洛估计来预测给定策略 $\pi$ 下的状态价值 $V_\pi(s)$。具体思路是多次完整地执行策略 $\pi$，每次执行都会产生一条完整的轨迹（从初始状态到终止状态），然后根据这些轨迹来计算各个状态的回报，最后取平均作为该状态的价值估计，如式 $\eqref{eq:mc_value_estimate}$ 所示。

$$
\begin{equation}\label{eq:mc_value_estimate}
V_\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}
\end{equation}
$$

### 增量式更新

在实际强化学习应用中，由于状态空间可能非常大，估计状态价值所需的轨迹数量可能上万甚至更多。一方面，轨迹是通过智能体与环境交互产生的，这一交互过程可能也会非常耗时；另一方面，存储和处理大量轨迹数据也会带来计算和内存的压力。

为了解决这些问题，蒙特卡洛预测通常采用增量式更新的方式来估计状态价值，即**边采样边更新**，而不是等采样完所有轨迹后再进行批量更新。

如式 $\eqref{eq:incremental_update}$ 所示，增量式更新的基本思想是每次采样到一个新的回报 $G$ 后，立即用它来更新对应状态 $s$ 的价值估计 $V(s)$。

$$
\begin{equation}\label{eq:incremental_update}
V(s) \leftarrow V(s) + \frac{1}{N(s)} [G - V(s)]
\end{equation}
$$

其中 $N(s)$ 是状态 $s$ 被访问的次数，$G$ 是当前采样到的回报。或者使用常数步长，如式 $\eqref{eq:constant_step_size}$ 所示。

$$
\begin{equation}\label{eq:constant_step_size}
V(s) \leftarrow V(s) + \alpha [G - V(s)]
\end{equation}
$$

其中 $\alpha \in (0,1]$ 是学习率，$\alpha$ 越大，收敛速度很快但波动也较大；$\alpha$ 越小，收敛速度较慢但更稳定。

可以发现，增量式更新的核心思想如式 $\eqref{eq:incremental_update_core}$ 所示。

$$
\begin{equation}\label{eq:incremental_update_core}
新的估计值 \leftarrow 旧的估计值 + 步长 \times（目标值-旧的估计值）
\end{equation}
$$

### 首次访问蒙特卡洛

在增量式更新的基础上，蒙特卡洛方法主要分成两种算法，一种是首次访问蒙特卡洛（$\text{first-visit Monte Carlo，FVMC}$）方法，另外一种是每次访问蒙特卡洛（$\text{every-visit Monte Carlo，EVMC}$）方法。$\text{FVMC}$ 方法主要包含两个步骤，首先是产生一个回合的完整轨迹，然后遍历轨迹计算每个状态的回报。

我们先来看首次访问蒙特卡洛（$\text{FVMC}$）方法的具体实现，算法流程如图 2 所示。

<div align=center>
<img width="600" src="figs/fvmc_pseu.png"/>
</div>
<div align=center>图 2 首次访问蒙特卡洛算法伪代码</div>

假设我们已经采样得到了一条完整的轨迹 $\tau = \{S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T, S_T\}$，其中 $S_T$ 是终止状态。对于轨迹中的每个状态 $S_t$，我们检查它是否是该状态在当前轨迹中的首次出现。如果是首次出现，我们计算从该时间步 $t$ 开始的回报 $G_t$，并将其添加到该状态的回报列表中，最后更新该状态的价值函数 $V(S_t)$ 为回报列表的平均值。

回头看我们前面的示例，可以用 $\text{FVMC}$ 方法来实现状态价值函数的估计，如代码 2 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 2 首次访问蒙特卡洛方法估计状态价值函数</b> </figcaption>
</div>

```python
import numpy as np
from collections import defaultdict

# ----------- 环境定义 -----------
states = ['s1', 's2', 's3', 's4']
gamma = 0.9
R = -1
terminal = 's4'

# 状态转移（确定性）
transitions = {
    's1': {'right': 's2', 'down': 's3'},
    's2': {'down': 's4'},
    's3': {'right': 's4'},
}

# 策略 π：在合法动作间随机选择
def policy(state):
    actions = list(transitions[state].keys())
    return np.random.choice(actions)

# 生成一条完整轨迹（从 s1 到 s4）
def generate_episode():
    episode = []
    state = 's1'
    while state != terminal:
        action = policy(state)
        next_state = transitions[state][action]
        episode.append((state, action, R))
        state = next_state
    episode.append((terminal, None, 0))  # 终止
    return episode

def first_visit_mc(num_episodes=1000):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode()
        G = 0
        visited = set()  # 用于记录首访

        # 反向遍历轨迹
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited:
                visited.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    return V

if __name__ == "__main__":
    V_first = first_visit_mc()
    print("First-Visit MC:")
    for s in states:
        print(f"  {s}: {V_first[s]:.2f}")
```

运行结果如代码 3 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 3 首次访问蒙特卡洛方法估计状态价值函数结果</b> </figcaption>
</div>

```
First-Visit MC:
  s1: -1.90
  s2: -1.00
  s3: -1.00
  s4: 0.00
```

可以发现，估计的状态价值函数与我们前面根据定义计算的结果是一致的。

注意，只在第一次遍历到某个状态时会记录并计算对应的回报，对应伪代码如图 2 所示。

### 每次访问蒙特卡洛

在 $\text{EVMC}$ 方法中则不会忽略同一状态的多个回报，具体代码实现如代码 4 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 4 每次访问蒙特卡洛方法估计状态价值函数</b> </figcaption>
</div>

```python
import numpy as np
from collections import defaultdict

# ----------- 环境定义 -----------
states = ['s1', 's2', 's3', 's4']
gamma = 0.9
R = -1
terminal = 's4'

# 状态转移（确定性）
transitions = {
    's1': {'right': 's2', 'down': 's3'},
    's2': {'down': 's4'},
    's3': {'right': 's4'},
}

# 策略 π：在合法动作间随机选择
def policy(state):
    actions = list(transitions[state].keys())
    return np.random.choice(actions)

# 生成一条完整轨迹（从 s1 到 s4）
def generate_episode():
    episode = []
    state = 's1'
    while state != terminal:
        action = policy(state)
        next_state = transitions[state][action]
        episode.append((state, action, R))
        state = next_state
    episode.append((terminal, None, 0))  # 终止
    return episode

def every_visit_mc(num_episodes=1000):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode()
        G = 0

        # 反向遍历轨迹（每次出现都更新）
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            returns[state].append(G)
            V[state] = np.mean(returns[state])
    return V

    return V

if __name__ == "__main__":
    V_every = every_visit_mc()

    print("\nEvery-Visit MC:")
    for s in states:
        print(f"  {s}: {V_every[s]:.2f}")
```

同样运行结果如代码 5 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 5 每次访问蒙特卡洛方法估计状态价值函数结果</b> </figcaption>
</div>

```
Every-Visit MC:
  s1: -1.90
  s2: -1.00
  s3: -1.00
  s4: 0.00
```

总的来说，$\text{FVMC}$ 是一种基于回合的增量式方法，具有无偏性和收敛快的优点，但是在状态空间较大的情况下，依然需要训练很多个回合才能达到稳定的结果。而 $\text{EVMC}$ 则是更为精确的预测方法，但是计算的成本相对也更高。

### 蒙特卡洛动作价值

蒙特卡洛预测或者估计动作价值函数 $Q(s,a)$ 的方法与状态价值函数类似，只不过需要同时考虑状态和动作的组合。具体来说，蒙特卡洛动作价值估计的步骤如下：

1. **生成完整轨迹**：与状态价值函数相同，首先需要通过与环境的交互生成一条完整的轨迹，包括状态、动作和奖励。

2. **计算回报**：对于轨迹中的每个状态-动作对 $(s,a)$，计算从该对开始的回报 $G_t$。

3. **更新价值函数**：根据计算得到的回报更新动作价值函数 $Q(s,a)$，可以使用首次访问或每次访问的方式。
具体的增量式更新公式与状态价值函数类似，如式 $\eqref{eq:mc_action_value_update}$ 所示。

$$
\begin{equation}\label{eq:mc_action_value_update}
Q(s,a) \leftarrow Q(s,a) + \alpha [G - Q(s
,a)]
\end{equation}
$$

其中 $\alpha$ 是学习率，$G$ 是从状态 $s$ 执行动作 $a$ 后得到的回报。

使用`Python` 代码实现蒙特卡洛动作价值估计来解决前面示例的问题，如代码 6 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 6 蒙特卡洛方法估计动作价值函数</b> </figcaption>
</div>

```python
import numpy as np
from collections import defaultdict

states = ['s1', 's2', 's3', 's4']
actions = ['right', 'down']
gamma = 0.9
R = -1
terminal = 's4'

# 转移定义
transitions = {
    ('s1', 'right'): 's2',
    ('s1', 'down'): 's3',
    ('s2', 'down'): 's4',
    ('s3', 'right'): 's4',
}

def policy(state):
    # 随机策略 π(a|s)
    legal = [a for (s,a) in transitions if s == state]
    return np.random.choice(legal)

def generate_episode():
    episode = []
    state = 's1'
    while state != terminal:
        action = policy(state)
        next_state = transitions[(state, action)]
        episode.append((state, action, R))
        state = next_state
    episode.append((terminal, None, 0))
    return episode

def mc_action_value(num_episodes=1000):
    Q = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode()
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if action is not None and (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
    return Q

Q = mc_action_value()
for (s,a), v in Q.items():
    print(f"Q({s},{a}) = {v:.2f}")
```

运行结果如代码 7 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 7 蒙特卡洛方法估计动作价值函数结果</b> </figcaption>
</div>

```python
Q(s1,down) = -1.90
Q(s1,right) = -1.90
Q(s2,down) = -1.00
Q(s3,right) = -1.00
```

联系状态价值和动作价值的关系，如式 $\eqref{eq:state_action_value_relation}$ 所示。

$$
\begin{equation}\label{eq:state_action_value_relation}
V_\pi(s) = \sum_{a} \pi(a|s) Q_\pi(s,a)
\end{equation}
$$

以状态 $s_1$ 为例，由于智能体采用随机策略，即在动作 $a_1$ 和 $a_2$ 之间按同等概率选择，因此可以计算出 $V(s_1)$ 如式 $\eqref{eq:V_s1_from_Q}$ 所示。

$$
\begin{equation}\label{eq:V_s1_from_Q}
\begin{aligned}
V(s_1) &= 0.5 * Q(s_1,a_1) + 0.5 * Q(s_1,a_2)  \\
&= 0.5 *(-1.9) + 0.5 * (-1.9) \\
&= -1.9
\end{aligned}
\end{equation}
$$

可以发现，计算结果与前面直接估计的状态价值是一致的。

## 蒙特卡洛控制

