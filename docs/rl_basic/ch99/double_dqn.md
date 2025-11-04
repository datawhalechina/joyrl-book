# Double DQN 算法实战

$\text{Double DQN}$ 算法的整体训练方式跟 $\text{DQN}$ 是一样的，只是区别在于目标值的计算方式，如代码 1 所示。

<div style="text-align: center;">
    <figcaption> 代码 1 $\:$ $\text{Double DQN}$目标值的计算 </figcaption>
</div>

```python
# 计算当前网络的Q值，即Q(s_t+1|a)
next_q_value_batch = self.policy_net(next_state_batch)
# 计算目标网络的Q值，即Q'(s_t+1|a)
next_target_value_batch = self.target_net(next_state_batch)
# 计算 Q'(s_t+1|a=argmax Q(s_t+1|a))
next_target_q_value_batch = next_target_value_batch.gather(1, torch.max(next_q_value_batch, 1)[1].unsqueeze(1)) 
```

最后与 $\text{DQN}$ 算法相同，可以得到 $\text{Double DQN}$ 算法在 $\text{CartPole}$ 环境下的训练结果，如图 $\text{8-5}$ 所示，完整的代码可以参考本书的代码仓库。

<div align=center>
<img width="400" src="figs/DoubleDQN_CartPole-v1_training_curve.png"/>
</div>
<div align=center>图 1 $\:$ $\text{CartPole}$ 环境 $\text{Double DQN}$ 算法训练曲线</div>

与 $\text{DQN}$ 算法的训练曲线对比可以看出，在实践上 $\text{Double DQN}$ 算法的效果并不一定比 $\text{DQN}$ 算法好，比如在这个环境下其收敛速度反而更慢了，因此读者需要多多实践才能摸索并体会到这些算法适合的场景。