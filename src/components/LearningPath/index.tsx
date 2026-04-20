import React from 'react';
import styles from './styles.module.css';

type Step = {label: string; n: string; isGoal?: boolean};

const STEPS: Step[] = [
  {label: '基础', n: '第 1 步'},
  {label: '经典算法', n: '第 2 步'},
  {label: '离线 / LLM', n: '第 3 步'},
  {label: 'JoyRL 实战', n: '目标', isGoal: true},
];

export default function LearningPath(): React.ReactElement {
  return (
    <section className={styles.section}>
      <h2 className={styles.title}>推荐学习路径</h2>
      <div className={styles.row}>
        {STEPS.map((s, i) => (
          <React.Fragment key={s.label}>
            <div className={`${styles.step} ${s.isGoal ? styles.goal : ''}`}>
              <div className={styles.stepN}>{s.n}</div>
              <strong className={styles.stepLabel}>{s.label}</strong>
            </div>
            {i < STEPS.length - 1 && <div className={styles.arrow} aria-hidden>→</div>}
          </React.Fragment>
        ))}
      </div>
    </section>
  );
}
