# Noisy DQN 算法实战

$\text{Noisy DQN}$ 算法的核心思想是将 $\text{DQN}$ 算法中的线性层替换成带有噪声的线性层，如代码 1 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 1 $\:$ 带有噪声的线性层网络 </figcaption>
</div>

```python
class NoisyLinear(nn.Module):
    '''在Noisy DQN中用NoisyLinear层替换普通的nn.Linear层
    '''
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.std_init  = std_init
        self.weight_mu    = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.empty(output_dim, input_dim))
        # 将一个 tensor 注册成 buffer，使得这个 tensor 不被当做模型参数进行优化。
        self.register_buffer('weight_epsilon', torch.empty(output_dim, input_dim)) 
        
        self.bias_mu    = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))
        self.register_buffer('bias_epsilon', torch.empty(output_dim))
        
        self.reset_parameters() # 初始化参数
        self.reset_noise()  # 重置噪声
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / self.input_dim ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.input_dim ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.output_dim ** 0.5)
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
```

根据写好的 $\text{NoisyLinear}$ 层，我们可以在 $\text{DQN}$ 算法中将普通的线性层替换为 $\text{NoisyLinear}$ 层，如代码 2  所示。

<div style="text-align: center;">
    <figcaption> 代码清单 2 $\:$ 带噪声层的全连接网络 </figcaption>
</div>

```python
class NoisyQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 =  nn.Linear(state_dim, hidden_dim)
        self.noisy_fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_fc3 = NoisyLinear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy_fc2(x))
        x = self.noisy_fc3(x)
        return x

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()
```

注意在训练过程中，我们需要在每次更新后重置噪声，这样有助于提高训练的稳定性，更多细节请参考 $\text{JoyRL}$ 源码。另外，我们也可以直接利用 $\text{torchrl}$ 模块中中封装好的 $\text{NoisyLinear}$ 层来构建 $\text{Noisy Q}$ 网络，跟我们自己定义的功能是一样的，如代码 3 所示。

<div style="text-align: center;">
    <figcaption> 代码清单 3 $\:$ 使用 $\text{torchrl}$ 模块构造的 $\text{Noisy Q}$ 网络 </figcaption>
</div>

```python
import torchrl
class NoisyQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 =  nn.Linear(state_dim, hidden_dim)
        self.noisy_fc2 = torchrl.NoisyLinear(hidden_dim, hidden_dim,std_init=0.1)
        self.noisy_fc3 = torchrl.NoisyLinear(hidden_dim, action_dim,std_init=0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy_fc2(x))
        x = self.noisy_fc3(x)
        return x

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()
```

同样我们展示一下它在 $\text{CartPole}$ 环境下的训练结果，如图 1 所示。

<div align=center>
<img width="400" src="figs/NoisyDQN_CartPole-v1_training_curve.png"/>
</div>
<div align=center>图 1 $\:$ $\text{CartPole}$ 环境 $\text{Noisy DQN}$ 算法训练曲线</div>