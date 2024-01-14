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

## 文档

[点击](https://datawhalechina.github.io/joyrl/)查看更详细的教程和`API`文档。
