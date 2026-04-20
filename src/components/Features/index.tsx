import React from 'react';
import styles from './styles.module.css';

const ITEMS = [
  {icon: '🇨🇳', title: '中文原创', desc: '不是翻译，用中文思维讲清'},
  {icon: '💻', title: '代码即教程', desc: '每章配 Notebook，可运行'},
  {icon: '🔬', title: '前沿覆盖', desc: '持续更新 RLHF / GRPO 等'},
];

export default function Features(): React.ReactElement {
  return (
    <section className={styles.section}>
      <div className={styles.eyebrow}>为什么选这本书</div>
      <div className={styles.grid}>
        {ITEMS.map((it) => (
          <div key={it.title} className={styles.item}>
            <div className={styles.icon} aria-hidden>{it.icon}</div>
            <h3 className={styles.title}>{it.title}</h3>
            <p className={styles.desc}>{it.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
