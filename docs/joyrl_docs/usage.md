# 使用说明

该部分主要讲述`JoyRL`的基本使用方法。

## 快速开始

### 参数配置简介

`JoyRL`旨在让用户只需要通过调参就能进行相关的强化学习实践，主要的参数包括：

* 通用参数(`GeneralConfig`)：跟运行模式相关的参数，如算法名称`algo_name`、环境名称`env_name`、随机种子`seed`等等；
* 算法参数(`AlgoConfig`)：算法本身相关参数，也是用户需要调参的主要参数；
* 环境参数(`EnvConfig`)：环境相关参数，比如`gym`环境中的`render_mode`等；

`JoyRL`提供多种超参数的配置方式，包括`yaml`文件、`python`文件等等，其中`yaml`文件是推荐新手使用的配置方式。以`DQN`为例，用户可以新建一个`yaml`文件，例如`DQN.yaml`，然后在其中配置相关参数，并执行：

```python
import joyrl
if __name__ == "__main__":
    print(joyrl.__version__) # 打印版本号
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path)
```

即可开始训练。对于`yaml`文件的配置，`JoyRL`提供了内置的终端命令执行，即：

```bash
joyrl --yaml ./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
```

用户也可以新建`python`文件自定义相关的参数类来运行，如下：

```python
import joyrl

class GeneralConfig:
    ''' General parameters for running
    '''
    def __init__(self) -> None:
        # basic settings
        self.env_name = "gym" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train, test
        self.device = "cpu" # device to use
        self.seed = 0 # random seed
        self.max_episode = -1 # number of episodes for training, set -1 to keep running
        self.max_step = 200 # number of episodes for testing, set -1 means unlimited steps
        self.collect_traj = False # if collect trajectory or not
        # multiprocessing settings
        self.n_interactors = 1 # number of workers
        # online evaluation settings
        self.online_eval = True # online evaluation or not
        self.online_eval_episode = 10 # online eval episodes
        self.model_save_fre = 500 # model save frequency per update step
        # load model settings
        self.load_checkpoint = False # if load checkpoint
        self.load_path = "Train_single_CartPole-v1_DQN_20230515-211721" # path to load model
        self.load_model_step = 'best' # load model at which step

class EnvConfig(object):
    def __init__(self) -> None:
        self.id = "CartPole-v1" # environment id
        
if __name__ == "__main__":
    general_cfg = GeneralConfig()
    env_cfg = EnvConfig()
    joyrl.run(general_cfg = general_cfg, env_cfg = env_cfg)
```

注意必须以准确的关键字(`kwarg`)形式传入到`joyrl.run`函数中！！！

同时，`JoyRL`自带默认的参数配置，在用户传入自定义参数时，会优先考虑`yaml`文件中的参数，其次是传入的参数类，默认的参数配置优先级最低。用户在配置参数时不需要同时配置所有参数，对于一些不关心的参数使用默认的配置即可。下面部分我们将介绍几种常用的参数配置方式。

### 训练与测试

想要训练一个算法，我们首先需要把`mode`改成`train`，并且配置好算法名称`algo_name`和环境名称`env_name`，以及环境的`id`，然后设置`max_episode`和`max_step`，如下：

```yaml
general_cfg:
  algo_name: DQN 
  env_name: gym 
  device: cpu 
  mode: train 
  max_episode: -1 
  max_step: 200 
  load_checkpoint: false
  load_path: Train_single_CartPole-v1_DQN_20230515-211721
  load_model_step: best 
  seed: 1 
  online_eval: true 
  online_eval_episode: 10 
  model_save_fre: 500
env_cfg:
  id: CartPole-v1
  render_mode: null
```

其中`max_episode`表示最大训练回合数，设置为-1时将持续训练直到手动停止，`max_step`表示每回合最大步数，设置为-1时将持续训练直到环境返回`done=True`或者`truncate=True`，**请根据实际环境情况设置，通常来讲每回合的步数过长不利于强化学习训练**。

配置好之后，用前面提到的任一种方式运行即可开始训练，训练过程中会在当前目录下生成一个`tasks`文件夹，里面包含了训练过程中的模型文件、日志文件等等，如下：

<div align=center>
<img width="500" src=".figs/tasks_dir.png"/>
<div align=center>图 1 tasks文件夹构成</div>
</div>

其中`logs`文件夹下会保存终端输出的日志，`models`文件夹下会保存训练过程中的模型文件，`tb_logs`文件夹下会保存训练过程中的`tensorboard`文件，例如奖励曲线、损失曲线等等，`results`文件夹下会以`csv`的形式保存奖励、损失等，便于后续单独绘图分析。`videos`文件夹下会保存运行过程中的视频文件，主要在测试过程中使用。`config.yaml`则保存本次运行过程中的参数配置，便于复现训练结果。

如果想要测试一个算法，我们只需要把`mode`改成`test`，然后将`load_checkpoint`(是否加载模型文件)改成`True`，并配好模型文件路径`load_path`和模型文件步数`load_model_step`，如下：

```yaml
mode: test
load_checkpoint: true
load_path: Train_single_CartPole-v1_DQN_20230515-211721
load_model_step: 1000
```

### 在线测试模式

在训练过程中，我们往往需要对策略进行定期的测试，以便于及时发现问题和保存效果最好的模型。因此，`JoyRL`提供了在线测试模式，只需要将`online_eval`设置为`True`，并设置好`online_eval_episode`（测试的回合数），即可开启在线测试模式，如下：

```yaml
online_eval: true 
online_eval_episode: 10 
model_save_fre: 500
```

其中`model_save_fre`表示模型保存频率，开启在线测试模式时，每保存一次模型，就会进行一次在线测试，并且会额外输出一个名为`best`的模型，用于保存训练过程中测试效果最好的模型，但不一定是最新的模型。

### 多进程模式

`JoyRL`支持多进程模式，但与向量化环境不同，`JoyRL`的多进程模式能够同时异步运行多个交互器和学习器，这样的好处是某一个交互器和学习器出现异常了，不会影响其他交互器和学习器的运行，从而提高训练的稳定性。在`JoyRL`中开启多进程的方式非常简单，只需要将`n_interactors`和`n_learners`设置为大于1的整数即可，如下：

```yaml
n_interactors: 2
n_learners: 2
```

注意，目前还不支持多个学习器的模式，即`n_learners`必须设置为1，未来会支持多个学习器的模式。

### 网络配置

`JoyRL`支持通过配置文件来建立网络，如下：

```yaml
merge_layers:
  - layer_type: linear
    layer_size: [256]
    activation: relu
  - layer_type: linear
    layer_size: [256]
    activation: relu
```

该配置等价为：

```python
class MLP(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) 
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, action_dim)  
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

其中输入的`state_dim`和`action_dim`会自动从环境中的状态动作空间中获取，用户只需要配置网络的结构即可。

### 多头网络

在上一小节配置网络中，会发现网络配置输入是`merge_layers`，这是因为`JoyRL`支持多头网络，即可以同时输入多个网络，然后将多个网络的输出进行合并。例如当状态输入同时包含图像和线性输入时，此时可以分别配置两个网络，然后将两个网络的输出进行合并，这就是多头网络的用法，如下图：

<div align=center>
<img width="500" src=".figs/branch_merge.png"/>
<div align=center>图 2 branch和merge网络</div>
</div>

其中`branch_layers`表示分支网络，`merge_layers`表示合并网络，`branch_layers`和`merge_layers`的配置方式与`merge_layers`相同，只是需要在每个网络的配置中加入`name`，如下：

```yaml
branch_layers:
    - name: feature_1
      layers:
      - layer_type: conv2d
        in_channel: 4
        out_channel: 16 
        kernel_size: 4
        stride: 2
        activation: relu
      - layer_type: pooling
        pooling_type: max2d
        kernel_size: 2
        stride: 2
        padding: 0
      - layer_type: flatten
      - layer_type: norm
        norm_type: LayerNorm
        normalized_shape: 512
      - layer_type: linear
        layer_size: [128]
        activation: relu
    - name: feature_2
        layers:
        - layer_type: linear
          layer_size: [128]
          activation: relu
        - layer_type: linear
          layer_size: [128]
          activation: relu
merge_layers:
  - layer_type: linear
    layer_size: [256]
    activation: relu
  - layer_type: linear
    layer_size: [256]
    activation: relu
```

如果只是配置简单的线性网络，则可以只配置`merge_layers`或者`branch_layers`，而如果是非线性网络例如`CNN`，则只能配置`branch_layers`，因为在逻辑上`merge_layers`只能接收线性输入。

## 自定义策略

用户可通过继承`algos`中的任意一个策略类来自定义策略，如下：

```python
import joyrl
from joyrl.algos.base import BasePolicy
from joyrl.algos.DQN.policy import Policy as DQNPolicy

class CustomPolicy1(BasePolicy):
    ''' 继承BasePolicy
    '''
    def __init__(self, cfg) -> None:
        super(BasePolicy, self).__init__(cfg)

class CustomPolicy2(DQNPolicy):
    ''' 继承DQNPolicy
    '''
    def __init__(self, cfg) -> None:
        super(DQNPolicy, self).__init__(cfg)

if __name__ == "__main__":
    my_policy = CustomPolicy1()
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path,policy = my_policy)
```

注意自定义策略必须传入`cfg`参数，便于`JoyRL`框架导入相应的参数配置。

## 自定义环境

`JoyRL`同时也支持自定义环境，如下：

```python
import env
class CustomEnv:
    def __init__(self,*args,**kwargs):
        pass
    def reset(self, seed = 0):
        return state, info
    def step(self, action):
        return state, reward, terminated, truncated, info
if __name__ == "__main__":
    my_env = CustomEnv()
    yaml_path = "xxx.yaml"
    joyrl.run(yaml_path = yaml_path, env = my_env)
```

注意，目前仅支持`gymnasium`接口的环境，即必须包含`reset`和`step`等函数。
