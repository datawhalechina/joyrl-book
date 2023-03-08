# JoyRL附书代码

## 环境安装

```bash
conda create -n joyrl python=3.7
conda activate joyrl
pip install -r requirements.txt
```

安装Gym：

```bash
pip install gym=0.25.2
```

安装Torch：

```bash
# CPU版本
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU版本
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# GPU版本镜像安装
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113

```