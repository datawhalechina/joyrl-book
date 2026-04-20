import React from 'react';
import Layout from '@theme/Layout';
import DqnPlayground from '@site/src/components/interactive/dqn/DqnPlayground';

export default function DqnPlaygroundPage() {
  return (
    <Layout
      title="DQN 交互模式"
      description="通过可视化沙盘理解 DQN 中的 epsilon-greedy、经验回放、目标网络和 TD 更新。"
    >
      <DqnPlayground />
    </Layout>
  );
}
