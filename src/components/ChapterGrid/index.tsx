import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type Chapter = {
  title: string;
  icon: string;
  desc: string;
  to: string;
  accent: string;
};

const CHAPTERS: Chapter[] = [
  {
    title: '强化学习基础',
    icon: '🎯',
    desc: 'MDP · DQN · Policy Gradient · Actor-Critic · PPO · SAC',
    to: '/docs/rl_basic',
    accent: '#0284c7',
  },
  {
    title: '离线强化学习',
    icon: '📦',
    desc: 'BCQ · CQL · IQL · 数据集与评估方法',
    to: '/docs/offline_rl',
    accent: '#0ea5e9',
  },
  {
    title: '大模型 + 强化学习',
    icon: '🧠',
    desc: 'RLHF · DPO · GRPO · 对齐训练',
    to: '/docs/llm_rl',
    accent: '#38bdf8',
  },
  {
    title: 'JoyRL 框架',
    icon: '⚙️',
    desc: 'API · 配置 · 自定义环境 · 训练脚本',
    to: '/docs/joyrl_docs/',
    accent: '#fbbf24',
  },
];

export default function ChapterGrid(): React.ReactElement {
  return (
    <section className={styles.section}>
      <div className={styles.header}>
        <div className={styles.eyebrow}>核心章节</div>
        <h2 className={styles.title}>四大板块，按学习路径组织</h2>
      </div>
      <div className={styles.grid}>
        {CHAPTERS.map((c) => (
          <Link
            key={c.to}
            to={c.to}
            className={styles.card}
            style={{borderLeftColor: c.accent}}
          >
            <div className={styles.icon} aria-hidden>{c.icon}</div>
            <h3 className={styles.cardTitle}>{c.title}</h3>
            <p className={styles.cardDesc}>{c.desc}</p>
          </Link>
        ))}
      </div>
    </section>
  );
}
