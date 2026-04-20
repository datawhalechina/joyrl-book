import React, {useEffect, useState} from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

type CustomFields = {
  visitorStatsApiUrl?: string;
};

type VisitorStatsResponse = {
  todayVisits: number;
  totalVisits: number;
  updatedAt?: string;
};

type DisplayStat = {
  num: string;
  label: string;
  helper?: string;
  accent?: boolean;
};

const COMMUNITY_STATS = [
  {num: '4+', label: '大板块'},
  {num: '40+', label: '章节'},
  {num: '20+', label: 'Notebook'},
  {num: '∞', label: '贡献者'},
];

function normalizeApiBaseUrl(rawUrl: string): string {
  return rawUrl.endsWith('/') ? rawUrl.slice(0, -1) : rawUrl;
}

function formatVisitCount(value: number): string {
  return new Intl.NumberFormat('zh-CN').format(value);
}

export default function CommunityStats(): React.ReactElement {
  const {siteConfig} = useDocusaurusContext();
  const customFields = (siteConfig.customFields ?? {}) as CustomFields;
  const apiBaseUrl = normalizeApiBaseUrl((customFields.visitorStatsApiUrl ?? '').trim());
  const [visitorStats, setVisitorStats] = useState<VisitorStatsResponse | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error' | 'disabled'>(
    apiBaseUrl ? 'loading' : 'disabled',
  );

  useEffect(() => {
    if (!apiBaseUrl) {
      setStatus('disabled');
      setVisitorStats(null);
      return;
    }

    let cancelled = false;

    const loadVisitorStats = async (): Promise<void> => {
      setStatus('loading');

      try {
        const response = await fetch(`${apiBaseUrl}/stats`, {
          method: 'GET',
          mode: 'cors',
          cache: 'no-store',
          headers: {
            accept: 'application/json',
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to load visitor stats: ${response.status}`);
        }

        const data = (await response.json()) as Partial<VisitorStatsResponse>;
        if (
          cancelled ||
          typeof data.todayVisits !== 'number' ||
          typeof data.totalVisits !== 'number'
        ) {
          if (!cancelled) {
            throw new Error('Invalid visitor stats payload');
          }
          return;
        }

        setVisitorStats({
          todayVisits: data.todayVisits,
          totalVisits: data.totalVisits,
          updatedAt: data.updatedAt,
        });
        setStatus('ready');
      } catch {
        if (!cancelled) {
          setStatus('error');
        }
      }
    };

    void loadVisitorStats();

    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const visitorCards: DisplayStat[] =
    status === 'ready' && visitorStats
      ? [
          {
            num: formatVisitCount(visitorStats.todayVisits),
            label: '今日访问',
            helper: '全站 PV',
            accent: true,
          },
          {
            num: formatVisitCount(visitorStats.totalVisits),
            label: '累计访问',
            helper: '全站 PV',
            accent: true,
          },
        ]
      : [
          {
            num: status === 'loading' ? '...' : '--',
            label: '今日访问',
            helper:
              status === 'disabled'
                ? '待接入 Worker'
                : status === 'error'
                  ? '统计接口异常'
                  : '正在同步',
            accent: true,
          },
          {
            num: status === 'loading' ? '...' : '--',
            label: '累计访问',
            helper:
              status === 'disabled'
                ? '待接入 Worker'
                : status === 'error'
                  ? '统计接口异常'
                  : '正在同步',
            accent: true,
          },
        ];

  const stats: DisplayStat[] = [...visitorCards, ...COMMUNITY_STATS];

  return (
    <section className={styles.section}>
      <h2 className={styles.title}>由 Datawhale 社区共创</h2>
      <p className={styles.subtitle}>开源 · 免费 · 可贡献 · 动态访问统计</p>
      <div className={styles.grid}>
        {stats.map((s) => (
          <div
            key={s.label}
            className={`${styles.stat} ${s.accent ? styles.statAccent : ''}`.trim()}
          >
            <strong className={styles.num}>{s.num}</strong>
            <div className={styles.label}>{s.label}</div>
            {s.helper ? <div className={styles.helper}>{s.helper}</div> : null}
          </div>
        ))}
      </div>
    </section>
  );
}
