# PPO 算法

$\qquad$ 本章我们开始讲解强化学习中最最最泛用的 $\text{PPO}$ 算法（$\text{proximal policy optimization}$），这个算法在强化学习领域的研究和应用中有着非常重要的地位，可以说是强化学习领域的一个里程碑式的算法。$\text{PPO}$ 算法是一种基于策略梯度的强化学习算法，由 $\text{OpenAI}$ 的研究人员 $\text{Schulman}$ 等人在 $\text{2017}$ 年提出。$\text{PPO}$ 算法的主要思想是通过在策略梯度的优化过程中引入一个重要性比率来限制策略更新的幅度，从而提高算法的稳定性和收敛性。$\text{PPO}$ 算法的优点在于简单、易于实现、易于调参，而且在实际应用中的效果也非常好，因此在强化学习领域得到了广泛的应用。

$\qquad$ $\text{PPO}$ 的前身是 $\text{TRPO}$ 算法，旨在克服 $\text{TRPO}$ 算法中的一些计算上的困难和训练上的不稳定性。$\text{TRPO}$ 是一种基于策略梯度的算法，它通过定义策略更新的信赖域来保证每次更新的策略不会太远离当前的策略，以避免过大的更新引起性能下降。然而，$\text{TRPO}$ 算法需要解决一个复杂的约束优化问题，计算上较为繁琐。本书主要出于实践考虑，这种太复杂且几乎已经被淘汰的 $\text{TRPO}$ 算法就不再赘述了，需要深入研究或者工作面试的读者可以自行查阅相关资料。 接下来将详细讲解 $\text{PPO}$ 算法的原理和实现，希望能够帮助读者更好地理解和掌握这个算法。

## 重要性采样

$\qquad$ 在将 $\text{PPO}$ 算法之前，我们需要铺垫一个概念，那就是重要性采样（ $\text{importance sampling}$ ）。重要性采样是一种估计随机变量的期望或者概率分布的统计方法。它的原理也很简单，假设有一个函数 $f(x)$ ，需要从分布 $p(x)$ 中采样来计算其期望值，但是在某些情况下我们可能很难从 $p(x)$ 中采样，这个时候我们可以从另一个比较容易采样的分布 $q(x)$ 中采样，来间接地达到从 $p(x)$ 中采样的效果。这个过程的数学表达式如式 $\text{(12.1)}$ 所示。

$$
\tag{12.1}
E_{p(x)}[f(x)]=\int_{a}^{b} f(x) \frac{p(x)}{q(x)} q(x) d x=E_{q(x)}\left[f(x) \frac{p(x)}{q(x)}\right]
$$

$\qquad$ 对于离散分布的情况，可以表达为式 $\text{(12.2)}$ 。

$$
\tag{12.2}
E_{p(x)}[f(x)]=\frac{1}{N} \sum f\left(x_{i}\right) \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}
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
Var_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]=E_{x \sim q}\left[\left(f(x) \frac{p(x)}{q(x)}\right)^{2}\right]-\left(E_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]\right)^{2} \\
=E_{x \sim p}\left[f(x)^{2} \frac{p(x)}{q(x)}\right]-\left(E_{x \sim p}[f(x)]\right)^{2}
$$

$\qquad$ 不难看出，当 $q(x)$ 越接近 $p(x)$ 的时候，方差就越小，也就是说重要性权重越接近于 $1$ 的时候，反之越大。

$\qquad$ 其实重要性采样也是蒙特卡洛估计的一部分，只不过它是一种比较特殊的蒙特卡洛估计，允许我们在复杂问题中利用已知的简单分布进行采样，从而避免了直接采样困难分布的问题，同时通过适当的权重调整，可以使得蒙特卡洛估计更接近真实结果。

## PPO 算法

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

$\text{KL}$ 约束一般也叫 $\text{KL-penalty}$，它的意思是在 $\text{TRPO}$ 损失的基础上，加上一个 $\text{KL}$ 散度的惩罚项，这个惩罚项的系数 $\beta$ 一般取 $0.01$ 左右。这个惩罚项的作用也是保证每次更新的策略分布都不会偏离上一次的策略分布太远，从而保证重要性权重不会偏离 $1$ 太远。在实践中，我们一般用 $\text{clip}$ 约束，因为它更简单，计算成本较低，而且效果也更好。

## 一个常见的误区

在很早的章节之前，我们讲过 `on-policy` 和

## 实战：PPO 算法