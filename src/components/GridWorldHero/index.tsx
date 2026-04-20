import React from 'react';
import styles from './styles.module.css';

export default function GridWorldHero(): React.ReactElement {
  return (
    <div className={styles.container} aria-label="强化学习 GridWorld 示意动画">
      <svg viewBox="0 0 200 160" className={styles.svg} role="img">
        <g className={styles.grid}>
          {/* 行 0 */}
          <rect x="10" y="10" width="30" height="30" />
          <rect x="40" y="10" width="30" height="30" />
          <rect x="70" y="10" width="30" height="30" />
          <rect x="100" y="10" width="30" height="30" className={styles.reward} />
          <rect x="130" y="10" width="30" height="30" />
          <rect x="160" y="10" width="30" height="30" className={styles.goal} />
          {/* 行 1 */}
          <rect x="10" y="40" width="30" height="30" />
          <rect x="40" y="40" width="30" height="30" className={styles.danger} />
          <rect x="70" y="40" width="30" height="30" />
          <rect x="100" y="40" width="30" height="30" />
          <rect x="130" y="40" width="30" height="30" />
          <rect x="160" y="40" width="30" height="30" />
          {/* 行 2 */}
          <rect x="10" y="70" width="30" height="30" />
          <rect x="40" y="70" width="30" height="30" />
          <rect x="70" y="70" width="30" height="30" />
          <rect x="100" y="70" width="30" height="30" />
          <rect x="130" y="70" width="30" height="30" />
          <rect x="160" y="70" width="30" height="30" />
          {/* 行 3 */}
          <rect x="10" y="100" width="30" height="30" />
          <rect x="40" y="100" width="30" height="30" />
          <rect x="70" y="100" width="30" height="30" />
          <rect x="100" y="100" width="30" height="30" />
          <rect x="130" y="100" width="30" height="30" />
          <rect x="160" y="100" width="30" height="30" />
        </g>
        <path
          className={styles.path}
          d="M25 115 L55 115 L55 85 L85 85 L85 55 L115 55 L115 25 L175 25"
        />
        <g className={styles.agent}>
          <circle r="7" />
          <text textAnchor="middle" dy="4">★</text>
        </g>
        <text x="100" y="152" textAnchor="middle" className={styles.formula}>
          policy π(a|s) · reward +1 · γ=0.99
        </text>
      </svg>
      <p className={styles.caption}>▶ Agent 每 4s 沿最优策略走一次</p>
    </div>
  );
}
