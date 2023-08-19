
创建Conda环境（需先安装Anaconda）

```bash
conda create -n joyrl-offline python=3.8
conda activate joyrl-offline
```

安装Gym：

```bash
pip install gymnasium==0.28.1
```

安装Torch：

```bash
# CPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# GPU镜像安装
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

安装其他依赖：

```bash
pip install -r requirements.txt
```