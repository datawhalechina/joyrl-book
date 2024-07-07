#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-07-06 14:29:40
LastEditor: JiangJi
LastEditTime: 2024-07-08 00:39:48
Discription: 
'''
import ray
import numpy as np
from utils.experience import Exp
        
@ray.remote
class EnvWorker:
    def __init__(self, cfg, env, idx, **kwargs):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.cfg = cfg
        self.idx = idx
        self.env = env
        self.seed = cfg.seed + idx
        self.state, _  = self.env.reset(seed=self.seed)

    def run(self, policy):
        exps = []
        for _ in range(self.cfg.n_steps):
            action = policy.sample_action(self.state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            policy_transition = policy.get_policy_transition()
            interact_transition = {'state':self.state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':terminated or truncated}
            exps.append(Exp(**interact_transition, **policy_transition))
            self.state = next_state
            if truncated or terminated:
                self.state, _ = self.env.reset(seed=self.seed)
        return exps
    
def evaluate_policy(env, policy, vis=False, n_episodes=10):
    state,_ = env.reset()
    if vis: env.render()
    terminated = False
    rewards = []
    for _ in range(n_episodes):
        ep_reward = 0
        ep_step = 0
        while True:
            action = policy.predict_action(np.array(state).reshape(1, -1))
            next_state, reward, terminated, truncated , _ = env.step(action)
            state = next_state
            # if vis: env.render()
            ep_reward += reward
            ep_step += 1
            if truncated or terminated or ep_step >=200:
                state,_ = env.reset()
                rewards.append(ep_reward)
                break
    return np.round(np.mean(rewards),3)