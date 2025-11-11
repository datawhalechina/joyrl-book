# Sarsa 算法实战

## 算法流程

如图 $\text{1}$ 所示，$\text{Sarsa}$ 算法流程跟 $\text{Q-learning}$ 算法基本相同，主要区别在于 $\text{Sarsa}$ 算法使用的是智能体实际执行的动作 $a'$ 来更新动作价值函数，而不是选择的最大动作值。

<div align=center>
<img width="500" src="figs/sarsa_pseu.png"/>
</div>
<div align=center>图 $\text{1}$ $\:$ $\text{Sarsa}$ 算法流程 </div>

## 定义超参数

为了便于调整和实验，我们把所有的超参数都定义在一个`Python`类中，如代码 $\text{1}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{1}$ $\:$ 定义超参数</b></figcaption>
</div>

```python
class Config:
    def __init__(self) -> None:
        ## 通用参数
        self.env_id = "CliffWalking-v0" # 环境id
        self.n_states = 48 # 状态数
        self.n_actions = 4 # 动作数
        self.render_mode = None # 渲染模式
        self.algo_name = "Qlearning" # 算法名称
        self.seed = 1 # 随机种子
        self.device = "cuda" # 训练设备，"cpu" or "cuda"
        self.max_episode = 300 # 最大回合数
        self.max_step = 200 # 每个回合的最大步数

        ## 算法参数
        self.epsilon_start = 0.95 # epsilon 初始值
        self.epsilon_end = 0.01 # epsilon 终止值
        self.epsilon_decay = 300 # epsilon 衰减率
        self.gamma = 0.90 # 奖励折扣因子
        self.lr = 0.1 # 学习率
```

## 定义策略

在 $\text{Sarsa}$ 算法中，我们同样使用 $\epsilon$-贪婪策略来选择动作。具体实现与 $\text{Q-learning}$ 算法中的实现相同，如代码 $\text{2}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{2}$ $\:$ 定义策略</b></figcaption>
</div>

```python
import numpy as np
import math
from collections import defaultdict

class Policy(object):
    def __init__(self, cfg: Config):
        ''' 初始化
        '''
        self.n_actions: int = cfg.n_actions # 动作数
        self.lr: float = cfg.lr 
        self.gamma: float = cfg.gamma    
        self.epsilon: float = cfg.epsilon_start
        self.sample_count = 0  # 采样计数，用于 epsilon 衰减
        self.epsilon_start: float = cfg.epsilon_start 
        self.epsilon_end: float = cfg.epsilon_end
        self.epsilon_decay: float = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions)) # 使用默认字典来表示 Q(s,a)，初始值为 0

    def sample_action(self, state):
        ''' 采样动作
        ''' 
        self.sample_count += 1
        # epsilon 值需要衰减，衰减方式可以是线性、指数等，以平衡探索和开发
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # 选择具有最大 Q 值的动作
        else:
            action = np.random.choice(self.n_actions) # 随机选择一个动作
        return action
    
    def predict_action(self, state):
        ''' 预测动作
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        ''' 更新策略
        '''
        Q_estimate = self.Q_table[str(state)][action]
        if done:
            Q_target = reward  # 终止状态 
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action] # 与 Q-learning 的唯一区别
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_estimate)
```



## 定义工具函数

为了保证实验的可复现性，通常需要固定随机种子。因此，我们定义一个工具函数 `set_seed` 来设置所有相关模块的随机种子。另外，为了更好地观察训练过程中的变化情况，我们定义了一些绘图函数来可视化训练结果，如代码 $\text{3}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{3}$ $\:$ 定义工具函数</b></figcaption>
</div>

```python
import random
import os
import torch
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

def set_seed(seed = 0):
    ''' 固定随机种子
    '''
    if seed == 0: # 不设置随机种子
        return 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def smooth(data, weight=0.9):  
    '''用于平滑曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards, ylabel = "rewards", title="learning curve"):
    ''' 绘制奖励曲线
    '''
    sns.set_theme()
    plt.figure()  
    plt.title(f"{title}") # 设置标题
    plt.xlim(0, len(rewards)) # x轴范围
    plt.xlabel('episodes') # x轴标签
    plt.ylabel(ylabel) # y轴标签
    plt.plot(rewards, label='original') # 绘制原始奖励曲线
    plt.plot(smooth(rewards), label='smoothed') # 绘制平滑后的奖励曲线
    plt.legend() # 显示图例
    plt.show() 
```

## 定义环境

同 $\text{Q-learning}$ 算法一样，我们使用 `OpenAI Gym` 提供的 `CliffWalking-v0` 环境来测试 $\text{Sarsa}$ 算法的性能表现，如代码 $\text{4}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{4}$ $\:$ 定义环境</b></figcaption>
</div>

```python
import gymnasium as gym

ACTION_MAP = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}

def create_env(cfg: Config):
    ''' 创建环境并设置随机种子
    '''
    env = gym.make(cfg.env_id, render_mode = cfg.render_mode) # 创建环境
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)
    print(f"状态数：{n_states}，动作数：{n_actions}")
    return env
```

## 定义训练与测试

我们定义了 `train` 和 `test` 函数来分别进行训练和测试。在训练过程中，智能体根据当前策略选择动作，并根据环境反馈更新动作价值函数。在测试过程中，智能体使用贪婪策略选择动作，以评估学习到的策略的性能，如代码 $\text{5}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{5}$ $\:$ 定义训练与测试</b></figcaption>
</div>

```python
import time
def train(cfg: Config, env, policy: Policy):
    ''' 训练
    '''
    print("开始训练！")
    s_t = time.time()
    rewards = []  # 记录所有回合的奖励
    steps = []  # 记录所有回合的步数
    for i_ep in range(cfg.max_episode):
        ep_reward = 0  # 单回合总奖励
        ep_step = 0
        state, info = env.reset(seed = cfg.seed)  # 重置环境并获取初始状态
        action = policy.sample_action(state)  # 采样动作 
        for _ in range(cfg.max_step):
            ep_step += 1
            next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            next_action =  policy.sample_action(next_state)
            done = terminated or truncated
            policy.update(state, action, reward, next_state, next_action, done)  # 更新 policy
            state = next_state  # 更新状态 
            action = next_action # 更新动作
            ep_reward += reward 
            ep_step += 1
            if done:
                break
        rewards.append(ep_reward)
        steps.append(ep_step)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep+1}/{cfg.max_episode}，奖励：{ep_reward:.2f}, 步数：{ep_step}")
    env.close()
    print(f"完成训练！用时：{time.time()-s_t:.2f} 秒")
    return {'rewards':rewards, 'steps':steps}

def test(cfg: Config, env, policy: Policy):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    s_t = time.time()
    for i_ep in range(cfg.max_episode):
        ep_reward = 0  # 一轮的累计奖励 
        ep_step = 0
        state, info = env.reset(seed = cfg.seed)  # 重置环境并获取初始状态
        action_sequence = []
        for _ in range(cfg.max_step):
            action = policy.predict_action(state)  # 预测动作 
            next_state, reward, terminated, truncated , info = env.step(action)
            done = terminated or truncated
            state = next_state  # 更新状态 
            action_sequence.append(ACTION_MAP[action])
            ep_reward += reward  # 增加奖励
            ep_step += 1
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.max_episode}，奖励：{ep_reward:.2f}, 步数：{ep_step}, 动作序列：{action_sequence}")
    print(f"完成测试！用时：{time.time()-s_t:.2f} 秒")
    env.close()
    return {'rewards':rewards, 'steps':steps}
```

## 开始训练

定义好以上各个部分后，我们就可以开始训练智能体了。训练过程中，我们会记录每个回合的总奖励和步数，以便后续分析和可视化，如代码 $\text{6}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{6}$ $\:$ 开始训练</b></figcaption>
</div>

```python

cfg = Config()
set_seed(cfg.seed)
env = create_env(cfg)
policy = Policy(cfg)
train_res = train(cfg, env, policy)
plot_rewards(train_res['rewards'], title=f"{cfg.algo_name} on {cfg.env_id} - Training")
```

得到的训练曲线如图 $\text{2}$ 所示。

<div align=center>
<img width="600" src="figs/sarsa_CliffWalking-v0_train_curve.png"/>
</div>
<div align=center>图 $\text{2}$ $\:$ $\text{Sarsa}$ 算法在 $\text{CliffWalking-v0}$ 环境中的训练曲线</div>


## 开始测试

训练完成后，我们可以使用测试函数来评估智能体的性能表现，如代码 $\text{7}$ 所示。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{7}$ $\:$ 开始测试</b></figcaption>
</div>

```python
cfg.max_episode = 10 # 测试时只跑10个回合
# cfg.render_mode = 'human' # 测试时渲染环境, 不要在Notebook中开启渲染，会卡死
env_test = create_env(cfg)
test_res = test(cfg, env_test, policy)
plot_rewards(test_res['rewards'], title=f"{cfg.algo_name} on {cfg.env_id} - Testing")
```

得到的测试曲线如图 $\text{3}$ 所示。

<div align=center>
<img width="600" src="figs/sarsa_CliffWalking-v0_test_curve.png"/>
</div>
<div align=center>图 $\text{3}$ $\:$ $\text{Sarsa}$ 算法在 $\text{CliffWalking-v0}$ 环境中的测试曲线</div>

对比 $\text{Q-learning}$ 算法的实战结果可以发现，$\text{Sarsa}$ 算法在该环境中表现得更为保守，避免了掉入悬崖的风险，但收敛速度稍慢一些，最终的总奖励也略低一些。打印的动作序列如代码 $\text{8}$ 所示，显示智能体选择更为保守的路径。

<div style="text-align: center;">
    <figcaption style="font-size: 14px;"> <b>代码 $\text{8}$ $\:$ 测试时打印的动作序列</b></figcaption>
</div>

```python
回合：1/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：2/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：3/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：4/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：5/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：6/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：7/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：8/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：9/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
回合：10/10，奖励：-15.00, 步数：15, 动作序列：['Up', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Down']
```
