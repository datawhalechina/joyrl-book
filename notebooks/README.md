
创建Conda环境（需先安装Anaconda）

```bash
conda create -n joyrl-book python=3.10
conda activate joyrl-book
```

安装Torch：

```bash
# CPU
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
# CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

安装其他依赖：

```bash
pip install -r requirements.txt
```