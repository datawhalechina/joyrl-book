# CQL（Conservative Q-Learning）

我们已经知道离线 RL 算法，都是在”无法在线验证策略性能“的前提下，用不同手段保证”策略评估+改进“仍可靠。CQL也不例外，在本节，我们将详细介绍CQL是如何保证”策略评估+改进“的可靠性的。


## OOD（Out-of Distribution） Q 值高估问题


离线场景里，策略一旦部署便无法与环境交互，也就永远没机会“打脸”那些过度乐观的 Q 值；误差遂在每次 Bellman 备份中层层叠加，最终让结果一落千丈。

Kumar 等人（2019）用 SAC 在 HalfCheetah-v2 上做了经典演示：

<div align=center>
<img width="450" src="figs/CQL_overestimateQ_pic.png"/>
<figcaption style="font-size: 14px;">图 1 增加样本数并未普遍抑制“反学习”现象图示。</figcaption>
</div>
 
左图中横轴是 Q 网络梯度步数，纵轴是贪婪策略的实际回报。回报先上升，随后随训练持续而急剧下降，形似过拟合，却随样本倍增而依旧出现——说明问题并非传统过拟合，而是 OOD 误差在目标 Q 中持续发酵，最终把整个价值函数拖垮（见右图）。

Conservative Q-Learning（CQL）对症下药：迫使学到的 Q 函数对任意策略都保持“保守”，其期望价值天然低于真实回报。由此，策略即便贪婪，也不会被虚假的 OOD 峰值引入歧途。实现仅需在标准 Bellman 损失外增添一项 **Q 值正则**，几行代码即可嵌入现有 DQN 或 actor-critic 框架。离散与连续任务实验表明，CQL 最终回报普遍高出先前最佳离线算法 2–5 倍，在复杂多模态数据集上优势尤为显著

## CQL 的正则化机制

CQL 在标准的 Bellman 误差损失基础上，添加了两个正则化项：

$$
\begin{equation}\label{eq:CQLBase}\ \  
J = \argmin_Q \alpha \cdot \left(\mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(⋅∣s)}[Q(s,a)] - \mathbb{E}_{s∼D, a\sim \pi_\beta(a|s)}[Q(s,a)]\right) + \frac{1}{2}\mathbb{E}_{s, a, s^\prime \sim \mathcal{D}}\left[\left(Q(s, a) - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k(s, a)\right)^2\right] 
\end{equation}
$$

1. 最小化策略动作（包括 OOD）的 Q 值（如下），其中 $\mu$ 是策略或**某种探索分布**(uniform distribution)，**用于生成 OOD 动作**。
$$\min_Q \mathbb{E}_{s \sim D, a \sim \mu(⋅∣s)}[Q(s,a)]$$
2. 最大化数据集中动作的 Q 值：
$$\mathbb{E}_{s∼D, a\sim \pi_\beta(a|s)}[Q(s,a)]$$

这两个项的组合，**使得 Q 函数在数据分布内的动作上保持高值，而在 OOD 动作上被压低**，从而防止策略被 OOD 动作吸引。从而确保学习到的 Q 函数是真实 Q 值的下界估计。这意味着：1) OOD 动作的 Q 值被压低，不会被策略误选；2) 数据分布内的动作 Q 值保持较高，策略更稳定。


接下来我们需要考虑策略优化的问题，最直接的想法是：在每一次策略迭代 $\hat{\pi}_k$ 上，先完整地做一遍离策略评估，再执行一步策略改进。另一种思路是：由于策略 $\hat{\pi}_k$ 通常由当前 Q 函数导出，我们可直接令 $\mu(a|s)$ 近似“最大化当前 Q 函数”的策略，从而得到一个**在线算法**。为形式化描述这类在线算法，我们在 $\eqref{eq:CQLBase}$ 的基础上定义一族关于$\mu(a|s)$ 的优化问题，下方给出通用模板:

$$
\begin{equation}\label{eq:CQL(\mathcal{R})}\ \  
J = \min_Q \max_\mu \alpha \cdot \left(\mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(⋅∣s)}[Q(s,a)] - \mathbb{E}_{s∼D, a\sim \pi_\beta(a|s)}[Q(s,a)]\right) + \frac{1}{2}\mathbb{E}_{s, a, s^\prime \sim \mathcal{D}}\left[\left(Q(s, a) - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k(s, a)\right)^2\right] + \mathcal{R}(\mu)
\end{equation}
$$

当我们将 $\mathcal{R}(\mu)$ 选为与先验分布 $\rho(a|s)$ 的 KL 散度，即$\mathcal{R}(\mu)=−D_{KL}(\mu, \rho)$，则可得到 $\mu(a|s) \propto \rho(a|s) \cdot e^{Q(s, a)}$（推导见附录:推导A）。特别地，当 $\rho = \text{Unif}(a)$ 时，$\eqref{eq:CQL(\mathcal{R})}$ 的首项对应于状态 s 处 Q 值的软最大化，由此得到$\eqref{eq:CQL(\mathcal{R})}$ 的一个变体，记为 $CQL(\mathcal{H})$:

$$
\begin{equation}\label{eq:CQL(\mathcal{H})}\ \  
J = \min_Q \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}}\left[log\sum_a e^{Q(s, a)}- \mathbb{E}_{a\sim \pi_\beta(a|s)}[Q(s,a)]\right] + \frac{1}{2}\mathbb{E}_{s, a, s^\prime \sim \mathcal{D}}\left[\left(Q(s, a) - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k(s, a)\right)^2\right]
\end{equation}
$$

从上式可以看出，我们在实现上是非常便捷的

```python
# 离散动作环境
q_sa = self.critic(obs)
loss = loss + self.cql_alpha * (
   torch.logsumexp(q_sa/self.cql_temperature, dim=1) 
   - q_sa.gather(1, action.long().view(-1, 1))
).mean()
```

## CQL算法实现


## 训练及效果展示

```python

```

## 附录

### 推导A

