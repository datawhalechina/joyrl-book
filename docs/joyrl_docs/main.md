# JoyRL

[![PyPI](https://img.shields.io/pypi/v/joyrl)](https://pypi.org/project/joyrl/)  [![GitHub issues](https://img.shields.io/github/issues/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/issues) [![GitHub stars](https://img.shields.io/github/stars/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/stargazers) [![GitHub forks](https://img.shields.io/github/forks/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/network) [![GitHub license](https://img.shields.io/github/license/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/blob/master/LICENSE)


`JoyRL` 是一个基于 `PyTorch` 和 `Ray` 开发的强化学习(`RL`)框架，支持串行和并行等方式。相比于其他`RL`库，`JoyRL` 旨在帮助用户摆脱算法实现繁琐、`API`不友好等问题。`JoyRL`设计的宗旨是，用户只需要通过**超参数配置**就可以训练和测试强化学习算法，这对于初学者来说更加容易上手，并且`JoyRL`支持大量的强化学习算法。`JoyRL` 为用户提供了一个**模块化**的接口，用户可以自定义自己的算法和环境并使用该框架训练。

## 安装

>⚠️注意：不要使用任何镜像源安装 `JoyRL`！！！

安装 `JoyRL` 推荐先安装 `Anaconda`，然后使用 `pip` 安装 `JoyRL`。

```bash
# 创建虚拟环境
conda create -n joyrl python=3.8
conda activate joyrl
pip install -U joyrl
```

`Torch` 安装：

推荐使用 `pip` 安装，但是如果遇到网络问题，可以尝试使用 `conda` 安装或者使用镜像源安装。

```bash
# pip CPU only
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
# pip GPU with mirror image
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
# CPU only
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## 使用说明

### 快速开始

以下是一个使用 `JoyRL` 的示例。如下所示，首先创建一个 `yaml` 文件来**设置超参数**，然后在终端中运行以下命令。这就是你需要做的所有事情，就可以在 `CartPole-v1` 环境上训练一个 `DQN` 算法。

```bash
joyrl --yaml ./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
```

或者你可以在`python` 文件中运行以下代码。

```python
import joyrl
if __name__ == "__main__":
    print(joyrl.__version__)
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path)
```

### 串行与并行

`JoyRL` 支持串行和并行两种方式。串行方式指的是先让智能体与环境交互产生样本，模型再使用这些样本进行训练。并行方式指的是智能体与环境交互产生样本的同时，模型也能同时进行采样训练。并行方式可以在复杂环境中加速训练，但是由于多进程的基础通信开销，可能会导致训练速度变慢。因此，我们建议在简单环境中使用串行方式，在复杂环境中使用并行方式。具体使用方式如下所示。

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
        self.seed = 0 # random seed, set -1 means using random seed
        self.max_episode = -1 # number of episodes for training, set -1 to keep running
        self.max_step = 200 # number of episodes for testing, set -1 means unlimited steps
        # multiprocessing settings
        self.n_interactors = 1 # number of interactors
        self.interactor_mode = "dummy" # dummy, only works when learner_mode is serial
        self.learner_mode = "serial" # serial, parallel, whether workers and learners are in parallel
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

当`learner_mode`为`serial`时，即开始串行训练。`n_interactors`表示环境交互器的数量，当`n_interactors`大于1时，可通过`interactor_mode`调整采样模式，`dummy`表示所有交互器依次采样，`ray`表示使用`Ray`库实现并行采样。当`learner_mode`为`parallel`时，即开始并行训练，此时所有交互器默认使用`Ray`库实现并行采样。

更多的使用方式请参考说明文档。

## 文档

[点击](https://datawhalechina.github.io/joyrl/)查看更详细的教程和`API`文档。


## 算法列表

算法讲解请参考[“蘑菇书”](https://github.com/datawhalechina/easy-rl)和[JoyRL Book](https://github.com/datawhalechina/joyrl-book)

|       名称       |                          参考文献                          |                    作者                     | 备注 |
| :--------------: | :----------------------------------------------------------: | :-------------------------------------------: | :---: |
| DQN | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) |       |
