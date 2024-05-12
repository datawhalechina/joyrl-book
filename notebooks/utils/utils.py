#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-26 19:33:59
LastEditor: JiangJi
LastEditTime: 2024-05-13 00:22:58
Discription: 
'''
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output

def all_seed(seed: int = 0):
    ''' 设置随机种子
    '''
    if seed == 0:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def smooth(data: list, weight: float = 0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(frames, rewards, device = 'cpu', algo_name = 'PPO', env_id= 'Pendulum-v1',  tag='train'):
    ''' 画图
    '''
    sns.set_theme(style="darkgrid")
    clear_output(True)
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {device} of {algo_name} for {env_id}")
    plt.xlabel('frames')
    plt.plot(frames, rewards, label='rewards')
    plt.plot(frames, smooth(rewards), label='smoothed rewards')
    plt.legend()
    plt.show()

def plot_losses(frames, losses, device = 'cpu', algo_name = 'PPO', env_id= 'Pendulum-v1',  tag='train'):
    ''' 画图
    '''
    sns.set_theme(style="darkgrid")
    clear_output(True)
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {device} of {algo_name} for {env_id}")
    plt.xlabel('frames')
    plt.plot(frames, losses, label='losses')
    plt.plot(frames, smooth(losses), label='smoothed losses')
    plt.legend()
    plt.show()