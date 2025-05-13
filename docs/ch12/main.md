# 第 12 章 PPO 算法

$\qquad$ 本章我们开始讲解强化学习中比较重要的 $\text{PPO}$ 算法，它在相关应用中有着非常重要的地位，是一个里程碑式的算法。不同于 $\text{DDPG}$ 算法，$\text{PPO}$ 算法是一类典型的 $\text{Actor-Critic}$ 算法，既适用于连续动作空间，也适用于离散动作空间。

$\qquad$ $\text{PPO}$ 算法是一种基于策略梯度的强化学习算法，由 $\text{OpenAI}$ 的研究人员 $\text{Schulman}$ 等人在 $\text{2017}$ 年提出。$\text{PPO}$ 算法的主要思想是通过在策略梯度的优化过程中引入一个重要性权重来限制策略更新的幅度，从而提高算法的稳定性和收敛性。$\text{PPO}$ 算法的优点在于简单、易于实现、易于调参，应用十分广泛，正可谓 “遇事不决 $\text{PPO}$ ”。

$\qquad$ $\text{PPO}$ 的前身是 $\text{TRPO}$ 算法，旨在克服 $\text{TRPO}$ 算法中的一些计算上的困难和训练上的不稳定性。$\text{TRPO}$ 是一种基于策略梯度的算法，它通过定义策略更新的信赖域来保证每次更新的策略不会太远离当前的策略，以避免过大的更新引起性能下降。然而，$\text{TRPO}$ 算法需要解决一个复杂的约束优化问题，计算上较为繁琐。本书主要出于实践考虑，这种太复杂且几乎已经被淘汰的 $\text{TRPO}$ 算法就不再赘述了，需要深入研究或者工作面试的读者可以自行查阅相关资料。 接下来将详细讲解 $\text{PPO}$ 算法的原理和实现，希望能够帮助读者更好地理解和掌握这个算法。

## 12.1 重要性采样

$\qquad$ 在展开 $\text{PPO}$ 算法之前，我们先铺垫一个概念，即重要性采样（ $\text{importance sampling}$ ）。重要性采样是一种估计随机变量的期望或者概率分布的统计方法。它的原理也很简单，假设有一个函数 $f(x)$ ，需要从分布 $p(x)$ 中采样来计算其期望值，但是在某些情况下我们可能很难从 $p(x)$ 中采样，这个时候我们可以从另一个比较容易采样的分布 $q(x)$ 中采样，来间接地达到从 $p(x)$ 中采样的效果。这个过程的数学表达式如式 $\text{(12.1)}$ 所示。

$$
\tag{12.1}
E_{p(x)}[f(x)]=\int_{a}^{b} f(x) \frac{p(x)}{q(x)} q(x) d x=E_{q(x)}\left[f(x) \frac{p(x)}{q(x)}\right]
$$

$\qquad$ 对于离散分布的情况，可以表达为式 $\text{(12.2)}$ 。

$$
\tag{12.2}
\begin{aligned}
E_{p(x)}[f(x)]=\frac{1}{N} \sum f\left(x_{i}\right) \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}
\end{aligned}
$$

$\qquad$ 这样一来原问题就变成了只需要从 $q(x)$ 中采样，然后计算两个分布之间的比例 $\frac{p(x)}{q(x)}$ 即可，这个比例称之为**重要性权重**。换句话说，每次从 $q(x)$ 中采样的时候，都需要乘上对应的重要性权重来修正采样的偏差，即两个分布之间的差异。当然这里可能会有一个问题，就是当 $p(x)$ 不为 $\text{0}$ 的时候，$q(x)$ 也不能为 $\text{0}$，但是他们可以同时为 $\text{0}$ ，这样 $\frac{p(x)}{q(x)}$ 依然有定义，具体的原理由于并不是很重要，因此就不展开讲解了。

$\qquad$ 通常来讲，我们把这个 $p(x)$ 叫做目标分布， $q(x)$ 叫做提议分布（ $\text{Proposal Distribution}$ ）, 那么重要性采样对于提议分布有什么要求呢? 其实理论上 $q(x)$ 可以是任何比较好采样的分布，比如高斯分布等等，但在实际训练的过程中，聪明的读者也不难想到我们还是希望 $q(x)$ 尽可能 $p(x)$，即重要性权重尽可能接近于 $\text{1}$ 。我们可以从方差的角度来具体展开讲讲为什么需要重要性权重尽可能等于 $1$ ，回忆一下方差公式，如式 $\text{(12.3)}$ 所示。

$$
\tag{12.3}
Var_{x \sim p}[f(x)]=E_{x \sim p}\left[f(x)^{2}\right]-\left(E_{x \sim p}[f(x)]\right)^{2}
$$

$\qquad$ 结合重要性采样公式，我们可以得到式 $\text{(12.4)}$ 。

$$
\tag{12.4}
\begin{aligned}
Var_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]=E_{x \sim q}\left[\left(f(x) \frac{p(x)}{q(x)}\right)^{2}\right]-\left(E_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]\right)^{2} \\
= E_{x \sim p}\left[f(x)^{2} \frac{p(x)}{q(x)}\right]-\left(E_{x \sim p}[f(x)]\right)^{2}
\end{aligned}
$$

$\qquad$ 不难看出，当 $q(x)$ 越接近 $p(x)$ 的时候，方差就越小，也就是说重要性权重越接近于 $1$ 的时候，反之越大。

$\qquad$ 其实重要性采样也是蒙特卡洛估计的一部分，只不过它是一种比较特殊的蒙特卡洛估计，允许我们在复杂问题中利用已知的简单分布进行采样，从而避免了直接采样困难分布的问题，同时通过适当的权重调整，可以使得蒙特卡洛估计更接近真实结果。

## 12.2 PPO 算法

$\qquad$ 既然重要性采样本质上是一种在某些情况下更优的蒙特卡洛估计，再结合前面 $\text{Actor-Critic}$ 章节中我们讲到策略梯度算法的高方差主要来源于 $\text{Actor}$ 的策略梯度采样估计，读者应该不难猜出 $\text{PPO}$ 算法具体是优化在什么地方了。没错，$\text{PPO}$ 算法的核心思想就是通过重要性采样来优化原来的策略梯度估计，其目标函数表示如式 $\text{(12.5)}$ 所示。

$$
\tag{12.5}
\begin{gathered}
J^{\mathrm{TRPO}}(\theta)=\mathbb{E}\left[r(\theta) \hat{A}_{\theta_{\text {old }}}(s, a)\right] \\
r(\theta)=\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)}
\end{gathered}
$$

$\qquad$ 这个损失就是置信区间的部分，一般称作 $\text{TRPO}$ 损失。这里旧策略分布 $\pi_{\theta_{\text {old }}}(a \mid s)$ 就是重要性权重部分的目标分布 $p(x)$ ，目标分布是很难采样的，所以在计算重要性权重的时候这部分通常用上一次与环境交互采样中的概率分布来近似。相应地， $\pi_\theta(a \mid s)$ 则是提议分布，即通过当前网络输出的 `probs` 形成的类别分布 $\text{Catagorical}$ 分布（离散动作）或者 $\text{Gaussian}$ 分布（连续动作）。

$\qquad$ 读者们可能对这个写法感到陌生，似乎少了 $\text{Actor-Critic}$ 算法中的 `logit_p`，但其实这个公式等价于式 $\text{(12.6)}$ 。

$$
\tag{12.6}
J^{\mathrm{TRPO}}(\theta)=E_{\left(s_t, a_t\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_\theta\left(a_t \mid s_t\right)}{p_{\theta^{\prime}}\left(a_t \mid s_t\right)} A^{\theta^{\prime}}\left(s_t, a_t\right) \nabla \log p_\theta\left(a_t^n \mid s_t^n\right)\right]
$$

$\qquad$ 换句话说，本质上 $\text{PPO}$ 算法就是在 $\text{Actor-Critic}$ 算法的基础上增加了重要性采样的约束而已，从而确保每次的策略梯度估计都不会过分偏离当前的策略，也就是减少了策略梯度估计的方差，从而提高算法的稳定性和收敛性。

$\qquad$ 前面我们提到过，重要性权重最好尽可能地等于 $\text{1}$ ，而在训练过程中这个权重它是不会自动地约束到 $1$ 附近的，因此我们需要在损失函数中加入一个约束项或者说正则项，保证重要性权重不会偏离 $\text{1}$ 太远。具体的约束方法有很多种，比如 $\text{KL}$ 散度、$\text{JS}$ 散度等等，但通常我们会使用两种约束方法，一种是 $\text{clip}$ 约束 ，另一种是 $\text{KL}$ 散度。$\text{clip}$ 约束定义如式 $\text{(12.7)}$ 所示。

$$
\tag{12.7}
J_{\text {clip }}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$

$\qquad$ 其中 $\epsilon$ 是一个较小的超参，一般取 $\text{0.1}$ 左右。这个 $\text{clip}$ 约束的意思就是始终将重要性权重 $r(\theta)$ 裁剪在 $1$ 的邻域范围内，实现起来非常简单。

$\qquad$ 另一种 $\text{KL}$ 约束定义如式 $\text{(12.8)}$ 所示。

$$
\tag{12.8}
J^{KL}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right]
$$

$\qquad$ $\text{KL}$ 约束一般也叫 $\text{KL-penalty}$，它的意思是在 $\text{TRPO}$ 损失的基础上，加上一个 $\text{KL}$ 散度的惩罚项，这个惩罚项的系数 $\beta$ 一般取 $0.01$ 左右。这个惩罚项的作用也是保证每次更新的策略分布都不会偏离上一次的策略分布太远，从而保证重要性权重不会偏离 $1$ 太远。在实践中，我们一般用 $\text{clip}$ 约束，因为它更简单，计算成本较低，而且效果也更好。

$\qquad$ 到这里，我们就基本讲完了 $\text{PPO}$ 算法的核心内容，其实在熟练掌握 $\text{Actor-Critic}$ 算法的基础上，去学习这一类的其他算法是不难的，读者只需要注意每个算法在 $\text{Actor-Critic}$ 框架上做了哪些改进，取得了什么效果即可。
## 12.3 一个常见的误区

$\qquad$ 在之前的章节中，我们讲过 $\text{on-policy}$ 和 $\text{off-policy}$ 算法，前者使用当前策略生成样本，并基于这些样本来更新该策略，后者则可以使用过去的策略采集样本来更新当前的策略。$\text{on-policy}$ 算法的数据利用效率较低，因为每次策略更新后，旧的样本或经验可能就不再适用，通常需要重新采样。而 $\text{off-policy}$ 算法由于可以利用历史经验，一般使用经验回放来存储和重复利用之前的经验，数据利用效率则较高，因为同一批数据可以被用于多次更新。但由于经验的再利用，可能会引入一定的偏见，但这也有助于稳定学习。但在需要即时学习和适应的环境中，$\text{on-policy}$ 算法可能更为适合，因为它们直接在当前策略下操作。

$\qquad$ 那么 $\text{PPO}$ 算法究竟是 $\text{on-policy}$ 还是 $\text{off-policy}$ 的呢？有读者可能会因为 $\text{PPO}$ 算法在更新时重要性采样的部分中利用了旧的 $\text{Actor}$ 采样的样本，就觉得 $\text{PPO}$ 算法会是 $\text{off-policy}$ 的。实际上虽然这批样本是从旧的策略中采样得到的，但我们并没有直接使用这些样本去更新我们的策略，而是使用重要性采样先将数据分布不同导致的误差进行了修正，即是两者样本分布之间的差异尽可能地缩小。换句话说，就可以理解为重要性采样之后的样本虽然是由旧策略采样得到的，但可以近似为从更新后的策略中得到的，即我们要优化的 $\text{Actor}$ 和采样的 $\text{Actor}$ 是同一个，因此 **$\text{PPO}$ 算法是 $\text{on-policy}$ 的**。

## 12.4 实战：PPO 算法
### 12.4.1 PPO 伪代码

$\qquad$ 如图 $\text{12-1}$ 所示，与 $\text{off-policy}$ 算法不同，$\text{PPO}$ 算法每次会采样若干个时步的样本，然后利用这些样本更新策略，而不是存入经验回放中进行采样更新。

<div align=center>
<img width="500" src="../figs/ch12/ppo_pseu.png"/>
</div>
<div align=center>图 $\text{12-1}$ $\text{PPO}$ 算法伪代码</div>

### 12.4.2 PPO 算法更新

$\qquad$ 无论是连续动作空间还是离散动作空间，$\text{PPO}$ 算法的动作采样方式跟前面章节讲的 $\text{Actor-Critic}$ 算法是一样的，在本次实战中就不做展开，读者可在 $\text{JoyRL}$ 代码仓库上查看完整代码。我们主要看看更新策略的方式，如代码清单 $\text{12-1}$ 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 $\text{12-1}$ $\text{PPO}$ 算法更新 </figcaption>
</div>

```Python
def update(self):
    # 采样样本
    old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
    # 转换成tensor
    old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
    old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
    old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
    # 计算回报
    returns = []
    discounted_sum = 0
    for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
        if done:
            discounted_sum = 0
        discounted_sum = reward + (self.gamma * discounted_sum)
        returns.insert(0, discounted_sum)
    # 归一化
    returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
    for _ in range(self.k_epochs): # 小批量随机下降
        #  计算优势
        values = self.critic(old_states) 
        advantage = returns - values.detach()
        probs = self.actor(old_states)
        dist = Categorical(probs)
        new_probs = dist.log_prob(old_actions)
        # 计算重要性权重
        ratio = torch.exp(new_probs - old_log_probs) #
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        # 注意dist.entropy().mean()的目的是最大化策略熵
        actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
        critic_loss = (returns - values).pow(2).mean()
        # 反向传播
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
```

$\qquad$ 注意在更新时由于每次采样的轨迹往往包含的样本数较多，我们通过利用小批量随机下降将样本随机切分成若干个部分，然后一个批量一个批量地更新网络参数。最后我们展示算法在 $\text{CartPole}$ 上的训练效果，如图 $\text{12-2}$ 所示。此外，在更新 $\text{Actor}$ 参数时，我们增加了一个最大化策略熵的正则项，这部分原理我们会在接下来的一章讲到。

<div align=center>
<img width="500" src="../figs/ch12/PPO_Cartpole_training_curve.png"/>
</div>
<div align=center>图 $\text{12-2}$ $\text{CartPole}$ 环境 $\text{PPO}$ 算法训练曲线</div>

$\qquad$ 可以看到，与 $\text{A2C}$ 算法相比，$\text{PPO}$ 算法的收敛是要更加快速且稳定的。

## 12.5 本章小结

$\qquad$ 本章主要介绍了强化学习中最为泛用的 $\text{PPO}$ 算法，它既适用于离散动作空间，也适用于连续动作空间，并且快速稳定，调参相对简单。与其他算法相比， $\text{PPO}$ 算法更像是一种实践上的创新，主要利用了重要性采样来提高 $\text{Actor-Critic}$ 架构的收敛性，也是各类强化学习研究中比较常见的一类基线算法。

## 12.6 练习题

1. 为什么 $\text{DQN}$ 和 $\text{DDPG}$ 算法不使用重要性采样技巧呢？
2. $\text{PPO}$ 算法原理上是 $\text{on-policy}$ 的，但它可以是 $\text{off-policy}$ 的吗，或者说可以用经验回放来提高训练速度吗?为什么？（提示：是可以的，但条件比较严格）
3. $\text{PPO}$ 算法更新过程中在将轨迹样本切分个多个小批量的时候，可以将这些样本顺序打乱吗？为什么？
4. 为什么说重要性采样是一种特殊的蒙特卡洛采样？