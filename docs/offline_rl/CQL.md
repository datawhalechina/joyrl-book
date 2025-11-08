



Issues with generalization are not corrected
- $Q(s,a) \leftarrow y(s,a) = r(s, a) + E_{a^\prime \sim \pi_{new}}[Q(s^\prime, a^\prime)]$
- 目标函数：$\min_Q E_{(s, a)\sim\pi_\beta(s, a)}[(Q(s, a) - y(s,a))^2]$
  - 期望 $\pi_\beta(a|s) = \pi_{new}(a|s) = \argmax_\pi E_{a\sim \pi(a|s)}[Q(s, a)]$
  - 但是实际上，从下图中可以看出 $\pi_{new}(a|s)$ 可能更差
    - 在offline-RL中只能用四个点去预估R， 所以在OOD上可能会出现高估
    - 在online-RL中开始只有四个点去预估R, 交互后会产生中间的第五个点，来调整预估
![17_overEstimateQ](../pic/17_overEstimateQ.png)

avoid all OOD actions in the Q update -> IQL & CQL


# 三、 CQL 解决 OOD 问题的原理

CQL（Conservative Q-Learning）之所以能有效解决离线强化学习中的 OOD（Out-of-Distribution）问题，核心在于它通过<font color=darkred>正则化机制显式地惩罚分布外动作的 Q 值估计</font>，从而防止策略被错误的、过高的 OOD Q 值误导。


## 1. OOD 问题的本质
在离线 RL 中，策略只能基于固定的数据集学习，无法与环境交互。这导致在数据集中未出现的状态-动作对（即OOD动作）上，Q 函数的估计可能严重偏高。这种高估会误导策略选择这些 OOD 动作，造成策略性能下降甚至崩溃。


## 2. CQL 的正则化机制
CQL 在标准的 Bellman 误差损失基础上，添加了两个正则化项：
- 最小化策略动作（包括 OOD）的 Q 值：
$$\min_Q E_{s \sim D, a \sim \mu(⋅∣s)}[Q(s,a)]$$
其中 μ 是策略或**某种探索分布**(uniform distribution)，**用于生成 OOD 动作**。

- 最大化数据集中动作的 Q 值：
$$−E_{(s,a)∼D}[Q(s,a)]$$

regularization:
- maximun entropy regularization
$$R(\mu)=R(H)=E_{s\sim D}[H(\mu(\cdot | s))]=-E_{s\sim D, a\sim \mu(\cdot | s)}[log(\mu(a | s))]$$

before两个项的组合，**使得 Q 函数在数据分布内的动作上保持高值，而在 OOD 动作上被压低**，从而防止策略被 OOD 动作吸引。从而确保学习到的 Q 函数是真实 Q 值的下界估计。这意味着：
- OOD 动作的 Q 值被压低，不会被策略误选；
- 数据分布内的动作 Q 值保持较高，策略更稳定。


# 4、 CQL 的正则化机制 - code 

- normal loss:
```python
critic_1_loss = 0.5 * torch.mean((q1 - td_target.float().detach())**2)
```

- 最小化策略动作（包括 OOD）的 Q 值：
$$\min_Q E_{s \sim D, a \sim \mu(⋅∣s)}[Q(s,a)]$$

```python
# uniform distribution
random_act_tensor = torch.FloatTensor(q2.shape[0] * self.num_random, action.shape[-1]).uniform_(
   -self.action_bound, self.action_bound).to(self.device)
q1_rand =  self._get_tensor_values(state, random_act_tensor, self.critic_1)
# -------------------------------
# \mu sampling a - state
state_temp = state.unsqueeze(1).repeat(1, self.num_random, 1).view(state.shape[0] * self.num_random, state.shape[1])
cur_act, cur_log_proba = self.actor(state_temp)
q1_curr_actions = self._get_tensor_values(state, cur_act, network=self.critic_1)

# \mu sampling a - next state
next_state_temp = next_state.unsqueeze(1).repeat(1, self.num_random, 1).view(
   next_state.shape[0] * self.num_random, next_state.shape[1])
next_act, next_log_proba = self.actor(next_state_temp)
q1_next_actions = self._get_tensor_values(state, next_act, network=self.critic_1)
```
- maximun entropy regularization
$$R(\mu)=R(H)=E_{s\sim D}[H(\mu(\cdot | s))]=-E_{s\sim D, a\sim \mu(\cdot | s)}[log(\mu(a | s))]$$

$$Q_{MaxEnt}(s, a) = \min_Q E_{s \sim D, a \sim \mu(⋅∣s)}[Q(s,a)] + R(\mu) \propto log \sum_a exp(Q_{MaxEnt}(s, a))$$


- 均匀分布 $[−1,1)$ 其熵为 $H(X) = -\frac{1}{2}log\frac{1}{2} \int^1_{-1}1dx= -log\frac{1}{2} = log2$，动作维度为`action_dim=cur_act.shape[-1]`
   - `H = action_dim * np.log(2)`
```python
# entropy
cur_log_proba = cur_log_proba.view(state.shape[0], self.num_random, 1)
next_log_proba = next_log_proba.view(next_state.shape[0], self.num_random, 1) 
random_h = cur_act.shape[-1] * np.log(2)
# Q_{MaxEnt}(s, a)  \propto log \sum_a exp(Q_{MaxEnt}(s, a))
cat_q1 = torch.cat([
   q1_rand + random_h, 
   q1_next_actions - next_log_proba.detach(),
   q1_curr_actions - cur_log_proba.detach()
], 1
)
min_qf1_loss = torch.logsumexp(cat_q1, dim=1).mean()
```

final loss
- `q1.mean()`: 最大化数据集中动作的 Q 值($−E_{(s,a)∼D}[Q(s,a)]$)
```python
critic_final_loss = critic_1_loss + (min_qf1_loss - q1.mean()) * self.min_q_weight
```

# 5. 实证效果

在 D4RL 等基准测试中，CQL 在多个任务（如 MuJoCo、AntMaze）中显著优于 BEAR、BCQ 等基线算法，尤其是在复杂、多模态数据分布和稀疏奖励环境中。
![17_cql_pp1](../pic/17_cql_pp1.png)

| 机制                 | 作用                |
| ------------------ | ----------------- |
| 正则化项惩罚 OOD 动作的 Q 值 | 防止策略被高估的 OOD 动作误导 |
| 最大化数据集中动作的 Q 值     | 保持策略在数据分布内的稳定性    |
| 理论保证 Q 值为下界估计      | 提供安全策略改进的保障       |
| 实证上在复杂任务中表现优异      | 验证其在实际场景中的有效性     |


因此，CQL 通过保守地估计 Q 值，有效缓解了离线 RL 中由于 OOD 动作引起的策略坍塌问题，是当前离线强化学习中最具代表性的稳健算法之一。

## Train Test

[detial python code: test_cql.py](../../src/test/test_cql.py)

```python

def cql_Walker2d_v4_test():
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'CQL-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=2.5e-4,
        critic_lr=4.5e-4,
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        num_epoches=1200,
        batch_size=256,
        CQL_kwargs=dict(
            temp=1.2,
            min_q_weight=1.0,
            num_random=10,
            tau=0.05,
            target_entropy=-torch.prod(torch.Tensor(env.action_space.shape)).item(),
            action_bound=1.0,
            reward_scale=2.5
        )
    )
    agent = CQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-3,
        gamma=cfg.gamma,
        CQL_kwargs=cfg.CQL_kwargs,
        device=cfg.device
    )
    
    batch_rl_training(
        agent, 
        cfg,
        env_name,
        data_level='simple',# 'medium', #
        test_episode_freq=10,
        episode_count=5,
        play_without_seed=True, 
        render=False
    )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    agent.eval()
    cfg.max_episode_steps = 600
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=True)

```
![CQL_Walk2d-v4.gif](../pic/CQL_Walk2d-v4.gif)


--------------------------------
which offline RL algrithm do I use

1. only train offline...
   1. CQL: +just one hyperparameter          + well understood and widely tested
   2. IQL: +more fiexible(offline + online)  -more hyperparameter
2. train offline and finetune online 
   1. Adavantage-weighted actor-critic (AWAC) +widely used and well tested 
   2. IQL: +seems to perform much better !
3. have a good way to train models in your domain
   1. COMBO 
      1. + similar properties as CQL, but benifits from models
      2. -not always easy to train a good model in your domain!
   2. TT: Trajectory Transformer
      1. +very powerful and effective models
      2. -extremely computationally expensive to train and evaluate 


standard real-world RL process 

1. instrument the task so taht we can run RL
   1. safety mechanisms
   2. autonomous collection
   3. rewards, reset, etc
2. wait a long time for online RL to run
3. change the algorithm in some small way (and repeat 2)
4. thorw it all in the garbage and start over for next task 


offline RL process
1. collect inital dataset
   1. human-provided
   2. scripted controller
   3. baseline policy
   4. all of the above
2. Train a policy offline
3. change the algoritnm in some small way  (and repeat 2)
4. collect more data, add to growing dataset (back to 2 and go downstream)
5. **keep the dataset and use it again for the next project!**


1. An offline RL workflow
   - Supervised learning workflow: train/test split
   - Offline RL workflow: ??? OPE ?
2. Statisticak guarantees
   - Biggest challenge: distributional shift/counterfactuals
3. scalable methods, large-scale applications 






# Reference

1. [berkeley CS285 lec-15](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-15.pdf)