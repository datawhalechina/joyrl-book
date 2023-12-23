# 使用说明

该部分主要讲述`JoyRL`的基本使用方法。

## 快速开始

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
        self.interactor_mode = "dummy" # dummy, only works when learner_mode is serial
        self.learner_mode = "serial" # serial, parallel, whether workers and learners are in parallel
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

同时，`JoyRL`自带默认的参数配置，在用户传入自定义参数时，会优先考虑`yaml`文件中的参数，其次是传入的参数类，默认的参数配置优先级最低。用户在配置参数时不需要同时配置所有参数，对于一些不关心的参数使用默认的配置即可。

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
