import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import GridWorldHero from '@site/src/components/GridWorldHero';
import ChapterGrid from '@site/src/components/ChapterGrid';
import Features from '@site/src/components/Features';
import LearningPath from '@site/src/components/LearningPath';
import CommunityStats from '@site/src/components/CommunityStats';

import styles from './index.module.css';

export default function Home(): React.ReactElement {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="JoyRL Book — 从零到实战的中文强化学习宝典"
    >
      <header className={styles.hero}>
        <div className={styles.heroGrid}>
          <div className={styles.heroLeft}>
            <div className={styles.badge}>✨ 由 Datawhale 开源维护</div>
            <h1 className={styles.heroTitle}>
              从零到实战的<br />中文强化学习宝典
            </h1>
            <p className={styles.heroSub}>
              覆盖 DQN / PPO / SAC / 离线 RL / RLHF，配套 Notebook 与 JoyRL 框架
            </p>
            <div className={styles.ctas}>
              <Link to="/docs/" className={styles.btnPrimary}>开始学习 →</Link>
              <Link
                to="https://github.com/datawhalechina/joyrl-book"
                className={styles.btnSecondary}
              >
                GitHub ★
              </Link>
            </div>
          </div>
          <div className={styles.heroRight}>
            <GridWorldHero />
          </div>
        </div>
        <div className={styles.heroBgGrid} aria-hidden />
      </header>

      <main>
        <ChapterGrid />
        <Features />
        <LearningPath />
        <CommunityStats />
        <section className={styles.ctaSection}>
          <h2 className={styles.ctaTitle}>准备好训练你的第一个 Agent 了吗？</h2>
          <p className={styles.ctaSub}>从 MDP 开始，一步步走到 RLHF</p>
          <Link to="/docs/" className={styles.btnPrimary}>开始第一章 →</Link>
        </section>
      </main>
    </Layout>
  );
}
