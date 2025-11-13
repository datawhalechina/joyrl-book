## PPO 算法实战
## 算法流程

如图 $\text{1}$ 所示，与 $\text{off-policy}$ 算法不同，$\text{PPO}$ 算法每次会采样若干个时步的样本，然后利用这些样本更新策略，而不是存入经验回放中进行采样更新。

<div align=center>
<img width="500" src="figs/ppo_pseu.png"/>
</div>
<div align=center>图 $\text{1}$ $\:$ $\text{PPO}$ 算法伪代码</div>

### PPO 算法更新

无论是连续动作空间还是离散动作空间，$\text{PPO}$ 算法的动作采样方式跟前面章节讲的 $\text{Actor-Critic}$ 算法是一样的，在本次实战中就不做展开，读者可在 $\text{JoyRL}$ 代码仓库上查看完整代码。我们主要看看更新策略的方式，如代码 $\text{1}$ 所示。

<div style="text-align: center;">
    <figcaption> 代码 $\text{1}$ $\text{PPO}$ 算法更新 </figcaption>
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

注意在更新时由于每次采样的轨迹往往包含的样本数较多，我们通过利用小批量随机下降将样本随机切分成若干个部分，然后一个批量一个批量地更新网络参数。最后我们展示算法在 $\text{CartPole}$ 上的训练效果，如图 $\text{2}$ 所示。此外，在更新 $\text{Actor}$ 参数时，我们增加了一个最大化策略熵的正则项，这部分原理我们会在接下来的一章讲到。

<div align=center>
<img width="500" src="figs/PPO_Cartpole_training_curve.png"/>
</div>
<div align=center>图 $\text{2}$ $\:$ $\text{CartPole}$ 环境 $\text{PPO}$ 算法训练曲线</div>

可以看到，与 $\text{A2C}$ 算法相比，$\text{PPO}$ 算法的收敛是要更加快速且稳定的。