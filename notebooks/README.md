

## 环境准备

创建`Python`环境(需先安装[Anaconda3](https://www.anaconda.com/download)或[Miniforge3](https://github.com/conda-forge/miniforge/releases/tag/24.11.3-0))：

```bash
conda create -n joyrl-book python=3.10
conda activate joyrl-book
```

安装Torch：

```bash
# CPU
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
# CUDA 12.1
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

安装其他依赖：

```bash
pip install -r requirements.txt
```

## 运行

选择感兴趣的`*.ipynb`文件，然后使用`Jupyter Notebook`打开并运行即可，推荐使用`VS Code`的`Jupyter`插件