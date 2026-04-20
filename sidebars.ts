import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  rlBasicSidebar: [
    {
      type: 'category',
      label: '基础强化学习',
      link: {
        type: 'doc',
        id: 'rl_basic/README',
      },
      collapsible: false,
      items: [
        {type: 'doc', id: 'rl_basic/ch0/README', label: '前言'},
        {type: 'doc', id: 'rl_basic/ch0_1/README', label: '术语与符号说明'},
        {type: 'doc', id: 'rl_basic/ch1/README', label: '绪论'},
        {type: 'doc', id: 'rl_basic/ch2/README', label: '马尔可夫决策过程'},
        {type: 'doc', id: 'rl_basic/ch3/README', label: '动态规划'},
        {type: 'doc', id: 'rl_basic/ch4/README', label: '蒙特卡洛方法'},
        {type: 'doc', id: 'rl_basic/ch4_1/README', label: '时序差分方法'},
        {type: 'doc', id: 'rl_basic/ch5/README', label: 'Dyna-Q 算法'},
        {type: 'doc', id: 'rl_basic/ch6/README', label: '深度学习基础'},
        {type: 'doc', id: 'rl_basic/ch7/README', label: 'DQN 算法'},
        {type: 'doc', id: 'rl_basic/ch8/README', label: 'DQN 算法进阶'},
        {type: 'doc', id: 'rl_basic/ch9/README', label: '策略梯度方法'},
        {type: 'doc', id: 'rl_basic/ch10/README', label: 'Actor-Critic 算法'},
        {type: 'doc', id: 'rl_basic/ch11/README', label: 'DDPG 算法'},
        {type: 'doc', id: 'rl_basic/ch11_1/README', label: 'TRPO 算法'},
        {type: 'doc', id: 'rl_basic/ch12/README', label: 'PPO 算法'},
        {type: 'doc', id: 'rl_basic/ch13/README', label: 'SAC 算法'},
        {type: 'doc', id: 'rl_basic/ch14/README', label: '模仿学习'},
        {
          type: 'category',
          label: '实战篇',
          link: {
            type: 'doc',
            id: 'rl_basic/ch99/README',
          },
          items: [
            {type: 'doc', id: 'rl_basic/ch99/gym_intro_1', label: 'Gymnasium 环境介绍'},
            {type: 'doc', id: 'rl_basic/ch99/q-learning', label: 'Q-learning 算法'},
            {type: 'doc', id: 'rl_basic/ch99/sarsa', label: 'Sarsa 算法'},
            {type: 'doc', id: 'rl_basic/ch99/torch', label: 'PyTorch 入门'},
            {type: 'doc', id: 'rl_basic/ch99/dqn', label: 'DQN 算法'},
            {type: 'doc', id: 'rl_basic/ch99/double_dqn', label: 'Double DQN 算法'},
            {type: 'doc', id: 'rl_basic/ch99/dueling_dqn', label: 'Dueling DQN 算法'},
            {type: 'doc', id: 'rl_basic/ch99/noisy_dqn', label: 'Noisy DQN 算法'},
            {type: 'doc', id: 'rl_basic/ch99/per_dqn', label: 'PER DQN 算法'},
            {type: 'doc', id: 'rl_basic/ch99/a2c', label: 'A2C 算法'},
            {type: 'doc', id: 'rl_basic/ch99/ddpg', label: 'DDPG 算法'},
            {type: 'doc', id: 'rl_basic/ch99/td3', label: 'TD3 算法'},
            {type: 'doc', id: 'rl_basic/ch99/trpo', label: 'TRPO 算法'},
            {type: 'doc', id: 'rl_basic/ch99/ppo', label: 'PPO 算法'},
            {type: 'doc', id: 'rl_basic/ch99/sac', label: 'SAC 算法'},
          ],
        },
      ],
    },
  ],
  offlineRlSidebar: [
    {
      type: 'category',
      label: '离线强化学习',
      link: {
        type: 'doc',
        id: 'offline_rl/README',
      },
      collapsible: false,
      items: [
        {type: 'doc', id: 'offline_rl/offlineRL', label: '离线强化学习综述'},
        {type: 'doc', id: 'offline_rl/CQL', label: 'CQL'},
      ],
    },
  ],
  llmRlSidebar: [
    {type: 'doc', id: 'llm_rl/README', label: '大模型与强化学习'},
  ],
  joyrlDocsSidebar: [
    {
      type: 'category',
      label: 'JoyRL 中文文档',
      link: {
        type: 'doc',
        id: 'joyrl_docs/main',
      },
      collapsible: false,
      items: [
        {type: 'doc', id: 'joyrl_docs/basic_concept', label: '基本概念'},
        {type: 'doc', id: 'joyrl_docs/usage', label: '使用说明'},
        {type: 'doc', id: 'joyrl_docs/hyper_cfg', label: '参数说明'},
        {type: 'doc', id: 'joyrl_docs/general_cfg', label: '通用参数说明'},
        {type: 'doc', id: 'joyrl_docs/algo_cfg', label: '算法参数说明'},
      ],
    },
  ],
};

export default sidebars;
