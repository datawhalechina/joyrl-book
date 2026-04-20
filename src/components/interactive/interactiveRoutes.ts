export type InteractiveDocEntry = {
  interactiveHref: string;
  readingHref: string;
  title: string;
};

export type ChapterEntry = {
  docId: string;
  label: string;
  navLabel: string;
  readingHref: string;
  preferredHref: string;
  hasInteractive: boolean;
};

export const READING_MODE_QUERY_KEY = 'view';
export const READING_MODE_QUERY_VALUE = 'read';

export const INTERACTIVE_DOC_MAP: Record<string, InteractiveDocEntry> = {
  'rl_basic/ch7/README': {
    interactiveHref: '/rl_basic/ch7/playground',
    readingHref: '/rl_basic/ch7/',
    title: 'DQN 训练机制沙盘',
  },
};

const RL_BASIC_CHAPTER_DEFINITIONS = [
  {docId: 'rl_basic/ch0/README', label: '前言', navLabel: '前言'},
  {
    docId: 'rl_basic/ch0_1/README',
    label: '术语与符号说明',
    navLabel: '术语与符号说明',
  },
  {docId: 'rl_basic/ch1/README', label: '绪论', navLabel: '第 1 章 · 绪论'},
  {
    docId: 'rl_basic/ch2/README',
    label: '马尔可夫决策过程',
    navLabel: '第 2 章 · 马尔可夫决策过程',
  },
  {docId: 'rl_basic/ch3/README', label: '动态规划', navLabel: '第 3 章 · 动态规划'},
  {
    docId: 'rl_basic/ch4/README',
    label: '蒙特卡洛方法',
    navLabel: '第 4 章 · 蒙特卡洛方法',
  },
  {
    docId: 'rl_basic/ch4_1/README',
    label: '时序差分方法',
    navLabel: '第 4.1 章 · 时序差分方法',
  },
  {docId: 'rl_basic/ch5/README', label: 'Dyna-Q 算法', navLabel: '第 5 章 · Dyna-Q 算法'},
  {
    docId: 'rl_basic/ch6/README',
    label: '深度学习基础',
    navLabel: '第 6 章 · 深度学习基础',
  },
  {docId: 'rl_basic/ch7/README', label: 'DQN 算法', navLabel: '第 7 章 · DQN 算法'},
  {
    docId: 'rl_basic/ch8/README',
    label: 'DQN 算法进阶',
    navLabel: '第 8 章 · DQN 算法进阶',
  },
  {
    docId: 'rl_basic/ch9/README',
    label: '策略梯度方法',
    navLabel: '第 9 章 · 策略梯度方法',
  },
  {
    docId: 'rl_basic/ch10/README',
    label: 'Actor-Critic 算法',
    navLabel: '第 10 章 · Actor-Critic 算法',
  },
  {docId: 'rl_basic/ch11/README', label: 'DDPG 算法', navLabel: '第 11 章 · DDPG 算法'},
  {docId: 'rl_basic/ch11_1/README', label: 'TRPO 算法', navLabel: '第 11.1 章 · TRPO 算法'},
  {docId: 'rl_basic/ch12/README', label: 'PPO 算法', navLabel: '第 12 章 · PPO 算法'},
  {docId: 'rl_basic/ch13/README', label: 'SAC 算法', navLabel: '第 13 章 · SAC 算法'},
  {docId: 'rl_basic/ch14/README', label: '模仿学习', navLabel: '第 14 章 · 模仿学习'},
] as const;

export function getInteractiveDocEntry(docId: string): InteractiveDocEntry | undefined {
  return INTERACTIVE_DOC_MAP[docId];
}

export function getDocReadingHref(docId: string): string {
  const normalizedPath = docId.endsWith('/README')
    ? docId.slice(0, -'README'.length)
    : docId;
  return `/${normalizedPath}`;
}

export function getPreferredDocHref(docId: string): string {
  return getInteractiveDocEntry(docId)?.interactiveHref ?? getDocReadingHref(docId);
}

export const RL_BASIC_CHAPTERS: ChapterEntry[] = RL_BASIC_CHAPTER_DEFINITIONS.map((chapter) => ({
  ...chapter,
  readingHref: getDocReadingHref(chapter.docId),
  preferredHref: getPreferredDocHref(chapter.docId),
  hasInteractive: Boolean(getInteractiveDocEntry(chapter.docId)),
}));

export function getRlBasicChapter(docId: string): ChapterEntry | undefined {
  return RL_BASIC_CHAPTERS.find((chapter) => chapter.docId === docId);
}

export function getAdjacentRlBasicChapters(docId: string): {
  previousChapter?: ChapterEntry;
  nextChapter?: ChapterEntry;
} {
  const chapterIndex = RL_BASIC_CHAPTERS.findIndex((chapter) => chapter.docId === docId);

  if (chapterIndex === -1) {
    return {};
  }

  return {
    previousChapter: RL_BASIC_CHAPTERS[chapterIndex - 1],
    nextChapter: RL_BASIC_CHAPTERS[chapterIndex + 1],
  };
}

export function isReadingModeSearch(search: string): boolean {
  return (
    new URLSearchParams(search).get(READING_MODE_QUERY_KEY) ===
    READING_MODE_QUERY_VALUE
  );
}

export function getReadingModeHref(href: string): string {
  const params = new URLSearchParams();
  params.set(READING_MODE_QUERY_KEY, READING_MODE_QUERY_VALUE);
  return `${href}?${params.toString()}`;
}
