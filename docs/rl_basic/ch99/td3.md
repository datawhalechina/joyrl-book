## TD3 算法实战

如代码 $\text{1}$ 所示，$\text{TD3}$ 算法只是在策略更新上与 $\text{DDPG}$ 算法有所差异，其它地方基本相同。

<div style="text-align: center;">
    <figcaption><b> 代码 $\text{1}$ $\:$ $\text{TD3}$ 算法的策略更新 </b></figcaption>
</div>

```Python
def update(self):
    if len(self.memory) < self.explore_steps: # 当经验回放中不满足一个批量时，不更新策略
        return
    state, action, reward, next_state, done = self.memory.sample(self.batch_size) # 从经验回放中随机采样一个批量的转移(transition)
           # 将数据转换为tensor
    state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
    action = torch.tensor(np.array(action), device=self.device, dtype=torch.float32)
    next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
    reward = torch.tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
    done = torch.tensor(done, device=self.device, dtype=torch.float32).unsqueeze(1)
    noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) # 构造加入目标动作的噪声
          # 计算加入了噪声的目标动作
    next_action = (self.actor_target(next_state) + noise).clamp(-self.action_scale+self.action_bias, self.action_scale+self.action_bias)
          # 计算两个critic网络对t+1时刻的状态动作对的评分，并选取更小值来计算目标q值
    target_q1, target_q2 = self.critic_1_target(next_state, next_action).detach(), self.critic_2_target(next_state, next_action).detach()
    target_q = torch.min(target_q1, target_q2)
    target_q = reward + self.gamma * target_q * (1 - done)
          # 计算两个critic网络对t时刻的状态动作对的评分
    current_q1, current_q2 = self.critic_1(state, action), self.critic_2(state, action)
          # 计算均方根损失
    critic_1_loss = F.mse_loss(current_q1, target_q)
    critic_2_loss = F.mse_loss(current_q2, target_q)
    self.critic_1_optimizer.zero_grad()
    critic_1_loss.backward()
    self.critic_1_optimizer.step()
    self.critic_2_optimizer.zero_grad()
    critic_2_loss.backward()
    self.critic_2_optimizer.step()
    if self.sample_count % self.policy_freq == 0:
              # 延迟策略更新，actor的更新频率低于critic
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
              #目标网络软更新
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

同样我们展示一下训练效果，在合适的参数设置下它会比 $\text{DDPG}$ 算法收敛的更快，如图 $\text{1}$ 所示。

<div align=center>
<img width="500" src="figs/TD3_Pendulum_training_curve.png"/>
</div>
<div align=center>图 $\text{1}$ $\:$ $\text{Pendulum}$ 环境 $\text{TD3}$ 算法训练曲线</div>
