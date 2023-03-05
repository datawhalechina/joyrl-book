# DQN 算法

DQN 算法，英文全称 Deep Q-learning，顾名思义就是基于深度网络模型的 Q-learning 算法，主要由 DeepMind 公司于2013年和2015年分别提出的两篇论文来实现，即《Playing Atari with Deep Reinforcement Learning》和《Human-level Control through Deep Reinforcement Learning》。DQN 算法相对于 Q-learning 算法来说更新方法本质上是一样的，而 DQN 算法最重要的贡献之一就是本章节开头讲的，用神经网络替换表格的形式来近似动作价值函数$Q(\boldsymbol{s},\boldsymbol{a})$。