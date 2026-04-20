import React from 'react';
import styles from './styles.module.css';

const STATS = [
  {num: '4+', label: '大板块'},
  {num: '40+', label: '章节'},
  {num: '20+', label: 'Notebook'},
  {num: '∞', label: '贡献者'},
];

export default function CommunityStats(): React.ReactElement {
  return (
    <section className={styles.section}>
      <h2 className={styles.title}>由 Datawhale 社区共创</h2>
      <p className={styles.subtitle}>开源 · 免费 · 可贡献</p>
      <div className={styles.grid}>
        {STATS.map((s) => (
          <div key={s.label} className={styles.stat}>
            <strong className={styles.num}>{s.num}</strong>
            <div className={styles.label}>{s.label}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
