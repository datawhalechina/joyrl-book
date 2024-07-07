#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-07-06 23:59:45
LastEditor: JiangJi
LastEditTime: 2024-07-08 00:41:37
Discription: 
'''
import numpy as np

class Exp:
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

def compute_returns_for_exps(exps, gamma=0.95, gae_lambda=0.95):
    def _get_exp_len(exps, max_step: int = 1):
        ''' get exp len
        '''
        exp_len = len(exps)
        if exp_len <= max_step or exps[-1].done:
            exp_len = max(exp_len, 0)
        else:
            exp_len = exp_len - max_step
        return exp_len
    exp_len = _get_exp_len(exps)
    next_value = exps[-1].value
    return_mc = 0
    return_td = next_value   
    adv_gae = 0
    returns_mc = []
    returns_td = []
    returns_gae = []
    for t in reversed(range(exp_len)):
        delta = exps[t].reward + gamma * next_value * (1 - exps[t].done) - exps[t].value
        adv_gae = delta + gamma * gae_lambda * (1 - exps[t].done) * adv_gae
        return_mc = exps[t].reward + gamma * return_mc * (1 - exps[t].done)
        return_td = exps[t].reward + gamma * return_td * (1 - exps[t].done)
        returns_mc.insert(0, return_mc)
        returns_td.insert(0, return_td)
        returns_gae.insert(0, adv_gae + exps[t].value)
        exps[t].return_mc = return_mc
        exps[t].return_td = return_td
        exps[t].adv_gae = adv_gae
        exps[t].return_gae = adv_gae + exps[t].value
        next_value = exps[t].value
    return_mc_normed = (returns_mc - np.mean(returns_mc)) / (np.std(returns_mc) + 1e-8)
    return_td_normed = (returns_td - np.mean(returns_td)) / (np.std(returns_td) + 1e-8)
    return_gae_normed = (returns_gae - np.mean(returns_gae)) / (np.std(returns_gae) + 1e-8)
    for t in range(exp_len):
        exps[t].return_mc_normed = return_mc_normed[t]
        exps[t].return_td_normed = return_td_normed[t]
        exps[t].return_gae_normed = return_gae_normed[t]
    exps = exps[:exp_len]
    return exps