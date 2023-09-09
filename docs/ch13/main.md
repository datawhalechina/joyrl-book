# 第 13 章 SAC 算法

$\qquad$ 本章开始介绍最后一种经典的策略梯度算法，即 $\text{Soft Actor-Critic}$ 算法，简写为 $\text{SAC}$ 。相比于前两个算法，$\text{SAC}$ 算法要更为复杂，因此本章涉及的公式推导也要多很多，但是最后的结果还是相对简洁的。因此读者可以根据自己的需求选择性阅读，只需要关注伪代码中变量的涵义以及结果公式即可。$\text{SAC}$ 算法是一种基于最大熵强化学习的策略梯度算法，它的目标是最大化策略的熵，从而使得策略更加鲁棒。$\text{SAC}$ 算法的核心思想是，通过最大化策略的熵，使得策略更加鲁棒，经过超参改良后的 $\text{SAC}$ 算法在稳定性方面是可以与 $\text{PPO}$ 算法华山论剑的。注意，由于 $\text{SAC}$ 算法理论相对之前的算法要复杂一些，因此推导过程

## 13.1 最大熵强化学习

$\qquad$ 由于 $\text{SAC}$ 算法相比于之前的策略梯度算法独具一路，它走的是最大熵强化学习的路子。为了让读者更好地搞懂什么是 $\text{SAC}$ ，我们先介绍一下最大熵强化学习，然后从基于价值的 $\text{Soft Q-Learning}$ 算法开始讲起。我们先回忆一下确定性策略和随机性策略，确定性策略是指在给定相同状态下，总是选择相同的动作，随机性策略则是在给定状态下可以选择多种可能的动作，不知道读者们有没有想过这两种策略在实践中有什么优劣呢？或者说哪种更好呢？这里我们先架空实际的应用场景，只总结这两种策略本身的优劣，首先看确定性策略：

* 优势：**稳定性且可重复性**。由于策略是确定的，因此可控性也比较好，在一些简单的环境下，会更容易达到最优解，因为不会产生随机性带来的不确定性，实验也比较容易复现。

* 劣势：**缺乏探索性**。由于策略是确定的，因此在一些复杂的环境下，可能会陷入局部最优解，无法探索到全局最优解，所以读者会发现目前所有的确定性策略算法例如 $\text{DQN}$ 、$\text{DDPG}$ 等等，都会增加一些随机性来提高探索。此外，面对不确定性和噪音的环境时，确定性策略可能显得过于刻板，无法灵活地适应环境变化。

$\qquad$ 再看看随机性策略：

* 优势：**更加灵活**。由于策略是随机的，这样能够在一定程度上探索未知的状态和动作，有助于避免陷入局部最优解，提高全局搜索的能力。在具有不确定性的环境中，随机性策略可以更好地应对噪音和不可预测的情况。

* 劣势：**不稳定**。正是因为随机，所以会导致策略的可重复性太差。另外，如果随机性太高，可能会导致策略的收敛速度较慢，影响效率和性能。

$\qquad$ 不知道读者有没有发现，这里字里行间都透露着随机性策略相对于确定性策略来说存在碾压性的优势。为什么这么说呢？首先我们看看确定性策略的优点，其实这个优点也不算很大的优点，因为所有可行的算法虽然可能不能保证每次的结果都是一模一样的，但是也不会偏差得太过离谱，而且我们一般也不会对可复现性要求那么高，一定要精确到每个小数点都正确，因此容易复现本身就是个伪命题。其次，这里也说了在一些简单的环境中更容易达到最优解，简单的环境是怎么简单呢？可能就是在九宫格地图里面寻找最短路径或者石头剪刀布的那种程度，而实际的应用环境是不可能有这么简单的场景的。

$\qquad$ 再看看随机性策略的缺点，其实也不算是什么缺点，因为在随机性策略中随机性是我们人为赋予的，换句话说就是可控的，反而相对来说是可控的稳定性。结合我们实际的生活经验，比如在和别人玩游戏对战的时候，是不是通常会觉得招式和套路比较多的人更难对付呢？因为即使是相同的情况，高手可能会有各种各样的方式来应对，反之如果对方只会一种打法，这样会很快让我们抓住破绽并击败对方。在强化学习中也是如此，我们会发现实际应用中，如果有条件的话，我们会尽量使用随机性策略，诸如 $\text{A2C}$ 、$\text{PPO}$ 等等，因为它更加灵活，更加鲁棒，更加稳定。

$\qquad$ 然而，最大熵强化学习认为，即使我们目前有了成熟的随机性策略，即 $\text{Actor-Critic}$ 一类的算法，但是还是没有达到最优的随机。因此，它引入了一个信息熵的概念，在最大化累积奖励的同时最大化策略的熵，使得策略更加鲁棒，从而达到最优的随机性策略。我们先回顾一下标准的强化学习框架，其目标是得到最大化累积奖励的策略，如式 $\text{(13.1)}$ 所示。

$$
\tag{13.1}
\pi^*=\arg \max _\pi \sum_t \mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \rho_\pi}\left[\gamma^t r\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]
$$

$\qquad$ 而最大熵强化学习则是在这个基础上加上了一个信息熵的约束，如式 $\text{(13.2)}$ 所示。

$$
\tag{13.2}
\pi_{\mathrm{MaxEnt}}^*=\arg \max _\pi \sum_t \mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \rho_\pi}\left[\gamma^t\left(r\left(\mathbf{s}_t, \mathbf{a}_t\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_t\right)\right)\right)\right]
$$

$\qquad$ 其中 $\alpha$ 是一个超参，称作温度因子（ $\text{temperature}$ ），用于平衡累积奖励和策略熵的比重。这里的 $\mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_t\right)\right)$ 就是策略的信息熵，定义如式 $\text{(13.3)}$ 所示。

$$
\tag{13.3}
\mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_t\right)\right)=-\sum_{\mathbf{a}_t} \pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \log \pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)
$$

$\qquad$ 它表示了随机策略 $\pi\left(\cdot \mid \mathbf{s}_t\right)$ 对应概率分布的随机程度，策略越随机，熵越大。后面我们可以发现，虽然理论推导起来比较复杂，但实际实践起来是比较简单的。

## 13.2 Soft Q-Learning

$\qquad$ 前面小节中我们引入了带有熵的累积奖励期望，接下来我们需要基于这个重新定义的奖励来重新推导一下相关的量。后面我们会发现虽然推导起来比较复杂，但用代码实现起来是比较简单的，因为几乎跟传统的 $\text{Q-Learning}$ 算法没有多大区别。因此着重于实际应用的同学可以直接跳过本小节的推导部分，直接看后面的算法实战部分。

$\qquad$ 现在我们开始进行枯燥地推导过程了，首先是 $Q$ 值函数和 $V$ 值函数，我们先回顾一下传统的最优 $Q$ 值函数的定义，如式 $\text{(13.4)}$ 所示。

$$
\tag{13.4}
Q^*\left(\mathbf{s}_t, \mathbf{a}_t\right)=r_t+\mathbb{E}_{\left(\mathbf{s}_{t+1}, \ldots\right) \sim \rho_{\pi^*}}\left[\sum_{l=1}^{\infty} \gamma^l r_{t+l}\right]
$$

$\qquad$ 相应地，最优的 $V$ 值函数只需要根据最优 $Q$ 值函数取 $\text{argmax}$ 即可，如式 $\text{(13.5)}$ 所示。

$$
\tag{13.5}
V^*\left(\mathbf{s}_t\right)=\max _{\mathbf{a}_t} Q^*\left(\mathbf{s}_t, \mathbf{a}_t\right)
$$

$\qquad$ 在最大熵强化学习中，我们需要重新定义最优 $Q$ 值函数需要改成式 $\text{(13.6)}$ 。

$$
\tag{13.6}
Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right)=r_t+\mathbb{E}_{\left(\mathbf{s}_{t+1}, \ldots\right) \sim \rho_{\pi_{\text {MaxEnt }}^*}}\left[\sum_{l=1}^{\infty} \gamma^l\left(r_{t+l}+\alpha \mathcal{H}\left(\pi_{\mathrm{MaxEnt}}^*\left(\cdot \mid \mathbf{s}_{t+l}\right)\right)\right)\right]
$$

$\qquad$ 即增加了一个熵的项，但是对应地最优 $V$ 值函数就不能简单地取 $\text{argmax}$ 了。但是根据贝尔曼方程，我们可以得到如式 $\text{(13.7)}$ 所示的关系。

$$
\tag{13.7}
Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right)=r_t+\gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p_{\mathrm{s}}, \mathbf{a}_{t+1} \sim p_{\mathrm{a}}}\left[Q_{\mathrm{soft}}^*\left(\mathbf{s}_{t+1}, \mathbf{a}_{t+1}\right)-\alpha \log \pi_{\mathrm{MaxEnt}}^*\left(\mathbf{a}_{t+1} \mid \mathbf{s}_{t+1}\right)\right]
$$

$\qquad$ 注意 $Q$ 和 $V$ 之间存在着线性关系，如式 $\text{(13.8)}$ 所示。

$$
\tag{13.8}
Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right)=r_t+\gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p_{\mathrm{s}}}\left[V_{\mathrm{soft}}^*\left(\mathbf{s}_{t+1}\right)\right]
$$

$\qquad$ 根据这个关系就可以得到式 $\text{(13.9)}$ 。

$$
\tag{13.9}
V_{\mathrm{soft}}^*\left(\mathbf{s}_t\right)=\mathbb{E}_{\mathbf{a}_t \sim \pi_{\mathrm{MaxEnt}}^*}\left[Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right)-\alpha \log \pi_{\mathrm{MaxEnt}}^*\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right]
$$

$\qquad$ 进一步地，可以得到式 $\text{(13.10)}$ 。

$$
\tag{13.10}
V_{\mathrm{soft}}^*\left(\mathbf{s}_t\right) \propto \alpha \log \int_{\mathcal{A}} \exp \left(\frac{1}{\alpha} Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}^{\prime}\right)\right) d \mathbf{a}^{\prime}
$$

$\qquad$ 可能这一步的求解读者们理解起来有点困难，其实是这样的，在 $V$ 值到达最优即最大的时候，也就是 $V_{\mathrm{soft}}^*\left(\mathbf{s}_t\right)$ ，相当于 $Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right)-\alpha \log \pi_{\mathrm{MaxEnt}}^*\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$ 也是最大的，我们设这个最大值为 $c$，那么就可以得到式 $\text{(13.11)}$ 。

$$
\tag{13.11}
\pi_{\mathrm{MaxEnt}}^*\left(\mathbf{a}_t \mid \mathbf{s}_t\right) = \exp \frac{1}{\alpha} Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right) - \exp \frac{c}{\alpha}
$$

$\qquad$ 两边都对动作 $\mathbf{a}_t$ 积分之后，就可以得到式 $\text{(13.12)}$ 。

$$
\tag{13.12}
\int_{\mathcal{A}} \pi_{\mathrm{MaxEnt}}^*\left(\mathbf{a}_t \mid \mathbf{s}_t\right) d \mathbf{a}_t = V_{\mathrm{soft}}^*\left(\mathbf{s}_t\right) = \int_{\mathcal{A}} \exp \frac{1}{\alpha} Q_{\mathrm{soft}}^*\left(\mathbf{s}_t, \mathbf{a}_t\right) - \exp \frac{c}{\alpha} d \mathbf{a}_t
$$

$\qquad$ 这里策略分布对动作的积分其实就是等于值函数，去掉一些无关的常数项，就是上面我们得到的最终式子了。注意到，这里求解出来的 $V$ 函数其实是一种 $\text{softmax}$ 的形式，如式 $\text{(13.13)}$ 所示。

$$
\tag{13.13}
\operatorname{softmax}_{\mathbf{a}} f(\mathbf{a}):=\log \int \exp f(\mathbf{a}) d \mathbf{a}
$$

$\qquad$ 意思就是需要平滑地取最大值，而当 $\alpha \rightarrow 0$ 的时候，就退化成了传统的 $Q$ 函数，对应的 $\text{softmax}$ 也就变成了 $\text{hardmax}$，这也是为什么叫 $\text{soft}$ 的原因。这其实就是玻尔兹曼分布的一种形式，即给所有动作赋予一个非零的概率，让智能体倾向于学习到能够处理相应任务的所有行为，注意是所有。针对这个玻尔兹曼分布，原论文提出了一个 $\text{energy-based policy}$ 的模型，感兴趣的读者可以深入了解，但由于对我们理解 $\text{SAC}$ 算法并不是很重要，因此这里就不展开讲解了。

$\qquad$ 另外注意，其实这里的 $V_{\mathrm{soft}}^*\left(\mathbf{s}_t\right)$ 是不太好算的，因为需要对整个动作空间积分然后进行 $\text{softmax}$，这个计算量是非常大的，因此也可以借助于 $\text{PPO}$ 中用到的重要性采样，将其转化为一个包含 $Q$ 函数的期望的形式，如式 $\text{(13.14)}$ 所示。

$$
\tag{13.14}
V_{\text {soft }}^\theta\left(\mathbf{s}_t\right)=\alpha \log \mathbb{E}_{q_{\mathrm{a}^{\prime}}}\left[\frac{\exp \left(\frac{1}{\alpha} Q_{\text {soft }}^\theta\left(\mathbf{s}_t, \mathbf{a}^{\prime}\right)\right)}{q_{\mathrm{a}^{\prime}}\left(\mathbf{a}^{\prime}\right)}\right]
$$

$\qquad$ 其中 $q_{\mathrm{a}^{\prime}}$ 一般用 $\text{Categorica}$ 分布来表示，最后我们就可以得到 $\text{Soft Q-Learning}$ 的损失函数了，如式 $\text{(13.15)}$ 所示。

$$
\tag{13.15}
J_Q(\theta)=\mathbb{E}_{\mathrm{s}_t \sim q_{s_t}, \mathrm{a}_t \sim q_{\mathrm{a}_t}}\left[\frac{1}{2}\left(\hat{Q}_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_t, \mathrm{a}_t\right)-Q_{\mathrm{soft}}^\theta\left(\mathrm{s}_t, \mathrm{a}_t\right)\right)^2\right]
$$

$\qquad$ 其中 $\hat{Q}_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_t, \mathrm{a}_t\right)=r_t+\gamma \mathbb{E}_{\mathrm{s}_{t+1} \sim p_{\mathrm{s}}}\left[V_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_{t+1}\right)\right]$ 表示目标网络的 $Q$ 值。

$\qquad$ 这里还有一个问题，就是理论上直接从当前策略分布 $\pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \propto \exp \left(\frac{1}{\alpha} Q_{\mathrm{soft}}^\theta\left(\mathbf{s}_t, \mathbf{a}_t\right)\right)$ 中采样也是比较困难的，此时我们可以借助一个额外的采样网络。首先构建一个参数为 $\phi$ 的 $\mathbf{a}_t=f^\phi\left(\xi ; \mathbf{s}_t\right)$，这个网络可以在状态 $s_t$ 下，将一个服从任意分布的噪声样本 $\xi$ 映射到 $\pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \propto \exp \left(\frac{1}{\alpha} Q_{\mathrm{soft}}^\theta\left(\mathbf{s}_t, \mathbf{a}_t\right)\right)$，将 $\mathbf{a}_t=f^\phi\left(\xi ; \mathbf{s}_t\right)$ 的动作分布记为 $\pi^\phi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$，那么参数 $\phi$ 的更新目标就是让 $\pi^\phi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$ 尽可能接近 $\pi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$，这种方式很容易想到用 KL 散度来做，如式 $\text{(13.16)}$ 所示。

$$
\tag{13.16}
J_\pi\left(\phi ; \mathbf{s}_t\right)=\mathrm{D}_{\mathrm{KL}}\left(\pi^\phi\left(\cdot \mid \mathbf{s}_t\right) \| \exp \left(\frac{1}{\alpha}\left(Q_{\mathrm{soft}}^\theta\left(\mathbf{s}_t, \cdot\right)-V_{\mathrm{soft}}^\theta\right)\right)\right)
$$

$\qquad$ 其梯度如式 $\text{(13.17)}$ 所示。

$$
\tag{13.17}
\begin{aligned}
\Delta f^\phi\left(\cdot ; \mathrm{s}_t\right)= & \mathbb{E}_{\mathbf{a}_t \sim \pi^\phi}\left[\left.\kappa\left(\mathbf{a}_t, f^\phi\left(\cdot ; \mathrm{s}_t\right)\right) \nabla_{\mathbf{a}^{\prime}} Q_{\mathrm{soft}}^\theta\left(\mathrm{s}_t, \mathrm{a}^{\prime}\right)\right|_{\mathbf{a}^{\prime}=\mathbf{a}_t}\right. \\
& \left.+\left.\alpha \nabla_{\mathbf{a}^{\prime}} \kappa\left(\mathbf{a}^{\prime}, f^\phi\left(\cdot ; \mathbf{s}_t\right)\right)\right|_{\mathbf{a}^{\prime}=\mathbf{a}_t}\right]
\end{aligned}
$$

## 13.3 SAC

$\qquad$ 实际上 $\text{SAC}$ 算法有两个版本，第一个版本是由 $\text{Tuomas Haarnoja}$ 于 $\text{2018}$ 年提出来的<sup>①</sup>，，第二个版本也是由 $\text{Tuomas Haarnoja}$ 于 $\text{2019}$ 年提出来的<sup>②</sup>，一般称作 $\text{SAC v2}$。第二个版本主要在前一版本的基础上做了简化，并且实现了温度因子的自动调节，从而使得算法更加简单稳定。

> ① Haarnoja T , Zhou A , Abbeel P ,et al.Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.2018[2023-08-02].DOI:10.48550/arXiv.1801.01290.

> ② Haarnoja T , Zhou A , Hartikainen K ,et al.Soft Actor-Critic Algorithms and Applications.2018[2023-08-02].DOI:10.48550/arXiv.1812.05905.

$\qquad$ 我们先讲讲第一版，第一版的 $\text{SAC}$ 算法思想基本上是和 $\text{Soft Q-Learning} 是特别相似的，只是额外增加了两个 $V$ 值网络（即包含目标网络和当前网络）来估计价值。$V$ 网络的目标函数定义如式 $\text{(13.18)}$ 所示。

$$
\tag{13.18}
J_V(\psi)=\mathbb{E}_{\mathbf{s}_t \sim \mathcal{D}}\left[\frac{1}{2}\left(V_\psi\left(\mathbf{s}_t\right)-\mathbb{E}_{\mathbf{a}_t \sim \pi_\phi}\left[Q_\theta\left(\mathbf{s}_t, \mathbf{a}_t\right)-\log \pi_\phi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right]\right)^2\right]
$$

$\qquad$ 其中 $\mathcal{D}$ 主要来自经验回放中的样本分布，其梯度如式 $\text{(13.19)}$ 所示。

$$
\tag{13.19}
\hat{\nabla}_\psi J_V(\psi)=\nabla_\psi V_\psi\left(\mathbf{s}_t\right)\left(V_\psi\left(\mathbf{s}_t\right)-Q_\theta\left(\mathbf{s}_t, \mathbf{a}_t\right)+\log \pi_\phi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right)
$$

$\qquad$ $\text{Soft Q}$ 函数的目标函数定义如式 $\text{(13.20)}$ 所示。

$$
\tag{13.20}
J_Q(\theta)=\mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \mathcal{D}}\left[\frac{1}{2}\left(Q_\theta\left(\mathbf{s}_t, \mathbf{a}_t\right)-\hat{Q}\left(\mathbf{s}_t, \mathbf{a}_t\right)\right)^2\right]
$$

$\qquad$ 对应梯度如式 $\text{(13.21)}$ 所示。

$$
\tag{13.21}
\hat{\nabla}_\theta J_Q(\theta)=\nabla_\theta Q_\theta\left(\mathbf{a}_t, \mathbf{s}_t\right)\left(Q_\theta\left(\mathbf{s}_t, \mathbf{a}_t\right)-r\left(\mathbf{s}_t, \mathbf{a}_t\right)-\gamma V_{\bar{\psi}}\left(\mathbf{s}_{t+1}\right)\right)
$$

$\qquad$ 策略的目标函数相比于 $\text{Soft Q-Learning}$ 要更简洁一些，如式 $\text{(13.22)}$ 所示。

$$
\tag{13.22}
J_\pi(\phi)=\mathbb{E}_{\mathbf{s}_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}}\left[\log \pi_\phi\left(f_\phi\left(\epsilon_t ; \mathbf{s}_t\right) \mid \mathbf{s}_t\right)-Q_\theta\left(\mathbf{s}_t, f_\phi\left(\epsilon_t ; \mathbf{s}_t\right)\right)\right]
$$

$\qquad$ 其梯度如式 $\text{(13.23)}$ 所示。

$$
\tag{13.23}
\begin{aligned}
& \hat{\nabla}_\phi J_\pi(\phi)=\nabla_\phi \log \pi_\phi\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \\
& \quad+\left(\nabla_{\mathbf{a}_t} \log \pi_\phi\left(\mathbf{a}_t \mid \mathbf{s}_t\right)-\nabla_{\mathbf{a}_t} Q\left(\mathbf{s}_t, \mathbf{a}_t\right)\right) \nabla_\phi f_\phi\left(\epsilon_t ; \mathbf{s}_t\right)
\end{aligned}
$$

## 13.4 自动调节温度因子

$\qquad$ 本小节主要讲解如何推导出自动调节因子的版本，整体推导的思路其实很简单，就是转换成规划问题，然后用动态规划、拉格朗日乘子法等方法简化求解，只关注结果的读者可以直接跳到本小节最后一个关于温度调节因子 $\alpha$ 的梯度下降公式即可。

$\qquad$ 注意第一个版本和 $\text{Soft Q-Learning}$ 都存在一个温度因子 $\alpha$ 的超参，并且这个超参是比较敏感的，第二个版本则设计了一个可以自动调节温度因子的方法。首先回顾一下累积奖励期望公式，如式 $\text{(13.24)}$ 所示。

$$
\tag{13.24}
\pi_{\mathrm{MaxEnt}}^*=\arg \max _\pi \sum_t \mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \rho_\pi}\left[\gamma^t\left(r\left(\mathbf{s}_t, \mathbf{a}_t\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_t\right)\right)\right)\right]
$$

$\qquad$ 其中左边项是奖励部分，右边项是熵部分，我们的目标是同时最大化这两个部分，但是由于这两个部分的量纲不同，因此需要引入一个温度因子 $\alpha$ 来平衡这两个部分。第二版 $\text{SAC}$ 的思路就是，我们可以把熵的部分转换成一个约束项，即只需要满足式 $\text{(13.25)}$ 即可。

$$
\tag{13.25}
\mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_t\right)\right) = \mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \rho_\pi}\left[-\log \left(\pi_t\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right)\right] \geq \mathcal{H}_0 
$$

$\qquad$ 换句话说，我们只需要让策略的熵大于一个固定的值 $\mathcal{H}_0$ 即可，这样就可以不需要温度因子了。这样一来我们的目标就变成了在最小期望熵的约束条件下最大化累积奖励期望，如式 $\text{(13.26)}$ 所示。

$$
\tag{13.26}
\max _{\pi_{0: T}} \mathbb{E}_{\rho_\pi}\left[\sum_{t=0}^T r\left(\mathbf{s}_t, \mathbf{a}_t\right)\right] \text { s.t. } \mathbb{E}_{\left(\mathbf{s}_t, \mathbf{a}_t\right) \sim \rho_\pi}\left[-\log \left(\pi_t\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right)\right] \geq \mathcal{H}_0 \quad \forall t
$$

$\qquad$ 可以看出，当我们忽略掉这个约束条件时，就变成了一个标准的强化学习问题。这里将策略 $\pi$ 拆分成了每一时刻的策略 $\pi_{o: T}=\left\{\pi_1, \ldots, \pi_T\right\}$，以便于求解每一时刻 $t$ 的最优温度因子 $\alpha^*_t$。由于每一时刻 $t$ 的策略只会对未来奖励造成影响（马尔可夫性质），因此可以利用动态规划的思想，自顶向下（从后往前）对策略进行求解，即分解为式 $\text{(13.27)}$ 。

$$
\tag{13.27}
\underbrace{\max _{\pi_0}(\mathbb{E}\left[r\left(s_0, a_0\right)\right]+\underbrace{\max _{\pi_1}(\mathbb{E}[\ldots]+\underbrace{\max _{\pi_T} \mathbb{E}\left[r\left(s_T, a_T\right)\right]}_{ \text { 第一次最大（子问题一） }})}_{\text { 倒数第二次最大 }})}_{\text { 倒数第一次最大 }}
$$

$\qquad$ 这样一来我们只需要求出第一个子问题，即式 $\text{(13.28)}$ 。

$$
\tag{13.28}
\max _{\pi_T} \mathbb{E}_{\left(\mathbf{s}_T, \mathbf{a}_T\right) \sim \rho_{\pi_T}}\left[r\left(\mathbf{s}_T, \mathbf{a}_T\right)\right] \text { s.t. } \mathbb{E}_{\left(\mathbf{s}_T, \mathbf{a}_T\right) \sim \rho_{\pi_T}}\left[-\log \left(\pi_t\left(\mathbf{a}_T \mid \mathbf{s}_T\right)\right)\right] \geq \mathcal{H}_0 \quad \forall t
$$

$\qquad$ 这个问题可以通过拉格朗日乘子法求解，首先我们做一个简化，如式 $\text{(13.29)}$ 所示。

$$
\tag{13.29}
\begin{aligned}
& h\left(\pi_T\right)=\mathcal{H}\left(\pi_T\right)-\mathcal{H}_0=\mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[-\log \pi_T\left(a_T \mid s_T\right)\right]-\mathcal{H}_0 \\
& f\left(\pi_T\right)= \begin{cases}\mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[r\left(s_T, a_T\right)\right], & \text { if } h\left(\pi_T\right) \geq 0 \\
-\infty, & \text { 否则 }\end{cases}
\end{aligned}
$$

$\qquad$ 进而原问题简化为式 $\text{(13.30)}$ 。

$$
\tag{13.30}
\operatorname{maximize} f\left(\pi_T\right) \text { s.t. } h\left(\pi_T\right) \geq 0
$$

$\qquad$ 通过乘上一个拉格朗日乘子 $\alpha_T$ （也叫做对偶变量，也相当于温度因子），我们可以得到式 $\text{(13.31)}$ 。

$$
\tag{13.31}
L\left(\pi_T, \alpha_T\right)=f\left(\pi_T\right)+\alpha_T h\left(\pi_T\right)
$$

$\qquad$ 当 $\alpha_T=0$ 时，可得到 $L(\pi_T, 0) = f\left(\pi_T\right)$，当 $\alpha_T \rightarrow \infty$ 时，可得到 $L\left(\pi_T, \alpha_T\right) = \alpha_T h\left(\pi_T\right)$，一般来说 $h\left(\pi_T\right) < 0$， 因此 $L\left(\pi_T, \alpha_T\right) = \alpha_T h\left(\pi_T\right) = f\left(\pi_T\right)$。

$\qquad$ 我们的目标是最大化 $f\left(\pi_T\right)$，如式 $\text{(13.32)}$ 所示。

$$
\tag{13.32}
\max _{\pi_T} f\left(\pi_T\right)=\min _{\alpha_T \geq 0} \max _{\pi_T} L\left(\pi_T, \alpha_T\right)
$$

$\qquad$ 因此原问题可以转化为相对易解的对偶问题，如式 $\text{(13.33)}$ 所示。

$$
\tag{13.33}
\begin{aligned}
\max _{\pi_T} \mathbb{E}\left[r\left(s_T, a_T\right)\right] & =\max _{\pi_T} f\left(\pi_T\right) \\
& =\min _{\alpha_T \geq 0} \max _{\pi_T} L\left(\pi_T, \alpha_T\right) \\
& =\min _{\alpha_T \geq 0} \max _{\pi_T} f\left(\pi_T\right)+\alpha_T h\left(\pi_T\right) \\
& =\min _{\alpha_T \geq 0} \max _{\pi_T} \mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[r\left(s_T, a_T\right)\right]+\alpha_T\left(\mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[-\log \pi_T\left(a_T \mid s_T\right)\right]-\mathcal{H}_0\right) \\
& =\min _{\alpha_T \geq 0} \max _{\pi_T} \mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[r\left(s_T, a_T\right)-\alpha_T \log \pi_T\left(a_T \mid s_T\right)\right]-\alpha_T \mathcal{H}_0 \\
& =\min _{\alpha_T \geq 0} \max _{\pi_T} \mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[r\left(s_T, a_T\right)+\alpha_T \mathcal{H}\left(\pi_T\right)-\alpha_T \mathcal{H}_0\right]
\end{aligned}
$$

$\qquad$ 首先固定住温度因子 $\alpha_T$，就能够得到最佳策略 $\pi_T^*$ 使得 $f\left(\pi_T^*\right)$ 最大化，如式 $\text{(13.34)}$ 所示。

$$
\tag{13.34}
\pi_T^*=\arg \max _{\pi_T} \mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[r\left(s_T, a_T\right)+\alpha_T \mathcal{H}\left(\pi_T\right)-\alpha_T \mathcal{H}_0\right]
$$

$\qquad$ 求出最佳策略之后，就可以求出最佳的温度因子 $\alpha_T^*$，如式 $\text{(13.35)}$ 所示。

$$
\tag{13.35}
\alpha_T^*=\arg \min _{\alpha_T \geq 0} \mathbb{E}_{\left(s_T, a_T\right) \sim \rho_\pi}\left[r\left(s_T, a_T\right)+\alpha_T \mathcal{H}\left(\pi_T^*\right)-\alpha_T \mathcal{H}_0\right]
$$

$\qquad$ 这样回到第一个子问题，即在时刻 $T$ 下的奖励期望，如式 $\text{(13.36)}$ 所示。

$$
\tag{13.36}
\max _{\pi_T} \mathbb{E}\left[r\left(s_T, a_T\right)\right]=\mathbb{E}_{\left(s_T, a_T\right) \sim \rho_{\pi^*}}\left[r\left(s_T, a_T\right)+\alpha_T^* \mathcal{H}\left(\pi_T^*\right)-\alpha_T^* \mathcal{H}_0\right]
$$

$\qquad$ 接下来我们就可以求解第二个子问题，即在时刻 $T-1$ 下的奖励期望。回顾一下 $\text{Soft Q}$ 函数公式，如式 $\text{(13.37)}$ 所示。

$$
\tag{13.37}
\begin{aligned}
Q_{T-1}\left(s_{T-1}, a_{T-1}\right) & =r\left(s_{T-1}, a_{T-1}\right)+\mathbb{E}\left[Q\left(s_T, a_T\right)-\alpha_T \log \pi\left(a_T \mid s_T\right)\right] \\
& =r\left(s_{T-1}, a_{T-1}\right)+\mathbb{E}\left[r\left(s_T, a_T\right)\right]+\alpha_T \mathcal{H}\left(\pi_T\right)
\end{aligned}
$$

$\qquad$ 代入第一个子问题的最优策略 $\pi_T^*$ 之后，就可以得到式 $\text{(13.38)}$ 。

$$
\tag{13.38}
Q_{T-1}^*\left(s_{T-1}, a_{T-1}\right)=r\left(s_{T-1}, a_{T-1}\right)+\max _{\pi_T} \mathbb{E}\left[r\left(s_T, a_T\right)\right]+\alpha_T \mathcal{H}\left(\pi_T^*\right)
$$

$\qquad$ 同样地，利用拉格朗日乘子法，就能得到第二个子问题即 $T-1$ 下的奖励期望最大化为式 $\text{(13.39)}$ 。

$$
\tag{13.39}
\begin{aligned}
& \max _{\pi_{T-1}}\left(\mathbb{E}\left[r\left(s_{T-1}, a_{T-1}\right)\right]+\max _{\pi_T} \mathbb{E}\left[r\left(s_T, a_T\right]\right)\right. \\
& =\max _{\pi_{T-1}}\left(Q_{T-1}^*\left(s_{T-1}, a_{T-1}\right)-\alpha_T^* \mathcal{H}\left(\pi_T^*\right)\right) \\
& =\min _{\alpha_{T-1} \geq 0} \max _{\pi_{T-1}}\left(Q_{T-1}^*\left(s_{T-1}, a_{T-1}\right)-\alpha_T^* \mathcal{H}\left(\pi_T^*\right)+\alpha_{T-1}\left(\mathcal{H}\left(\pi_{T-1}\right)-\mathcal{H}_0\right)\right) \\
& =\min _{\alpha_{T-1} \geq 0} \max _{\pi_{T-1}}\left(Q_{T-1}^*\left(s_{T-1}, a_{T-1}\right)+\alpha_{T-1} \mathcal{H}\left(\pi_{T-1}\right)-\alpha_{T-1} \mathcal{H}_0\right)-\alpha_T^* \mathcal{H}\left(\pi_T^*\right)
\end{aligned}
$$

$\qquad$ 进而得到式 $\text{(13.40)}$ 。

$$
\tag{13.40}
\begin{aligned}
& \pi_{T-1}^*=\arg \max _{\pi_{T-1}} \mathbb{E}_{\left(s_{T-1}, a_{T-1}\right) \sim \rho_\pi}\left[Q_{T-1}^*\left(s_{T-1}, a_{T-1}\right)+\alpha_{T-1} \mathcal{H}\left(\pi_{T-1}\right)-\alpha_{T-1} \mathcal{H}_0\right] \\
& \alpha_{T-1}^*=\arg \min _{\alpha_{T-1} \geq 0} \mathbb{E}_{\left(s_{T-1}, a_{T-1}\right) \sim \rho_{\pi^*}}\left[\alpha_{T-1} \mathcal{H}\left(\pi_{T-1}^*\right)-\alpha_{T-1} \mathcal{H}_0\right]
\end{aligned}
$$

$\qquad$ 我们会发现第二个子问题的答案形式其实和第一个子问题是一样的，依此类推，我们就可以得到温度因子的损失函数，如式 $\text{(13.41)}$ 所示。

$$
\tag{13.41}
J(\alpha)=\mathbb{E}_{a_t \sim \pi_t}\left[-\alpha \log \pi_t\left(a_t \mid s_t\right)-\alpha \mathcal{H}_0\right]
$$

$\qquad$ 这样一来就能实现温度因子的自动调节了。这一版本由于引入了温度因子的自动调节，因此不需要额外的 $V$ 值网络，直接使用两个 $Q$ 网络（包含目标网络和当前网络）来作为 $\text{Critic}$ 估计价值即可。

## 13.5 实战：SAC 算法

$\qquad$ 在实战中，我们主要讲解 $SAC$ 算法的第二个版本，即自动调节温度因子的版本。该版本的如图 $\text{13-1}$ 所示，整个训练过程相对来说还是比较简洁的，只是需要额外定义一些网络，比如用来调节温度因子等。

<div align=center>
<img width="500" src="../figs/ch13/sac_pseu.png"/>
</div>
<div align=center>图 $\text{13-1}$ $\text{SAC}$ 算法伪代码</div>

### 15.5.1 定义模型

$\qquad$ 首先我们定义 $\text{Actor}$ 和 $\text{Critic}$，即值网络和策略网络，跟 $\text{A2C}$ 算法其实是一样的，如代码清单 $\text{13-1}$ 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 $\text{13-1}$ $\text{Actor}$ 和 $\text{Critic}$ 网络 </figcaption>
</div>

```Python
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNet, self).__init__()
        '''定义值网络
        '''
        self.linear1 = nn.Linear(state_dim, hidden_dim) # 输入层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏层
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w) # 初始化权重
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化权重
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # 计算动作
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        # 计算动作概率
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()
        return action[0]
```

$\qquad$ 然后再额外定义一个 $\text{Soft Q}$ 网络，如代码清单 $\text{13-2}$ 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 $\text{13-2}$ $\text{Soft Q}$ 网络 </figcaption>
</div>

```Python
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(SoftQNet, self).__init__()
        '''定义Q网络，state_dim, action_dim, hidden_dim, init_w分别为状态维度、动作维度隐藏层维度和初始化权重
        '''
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
```

### 15.5.2 算法更新

$\qquad$ 我们再看看 