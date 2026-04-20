import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeMathjax from './src/rehype-mathjax-twopass.mjs';

const repoUrl = 'https://github.com/datawhalechina/joyrl-book';
const mathjaxConfig = {
  tex: {
    tags: 'ams',
    tagSide: 'right',
  },
};

const config: Config = {
  title: 'JoyRL Book',
  tagline: '强化学习实践教程与 JoyRL 框架文档',
  favicon: 'img/favicon-brand.png',

  future: {
    v4: true,
  },

  url: 'https://datawhalechina.github.io',
  baseUrl: '/joyrl-book/',
  trailingSlash: true,

  organizationName: 'datawhalechina',
  projectName: 'joyrl-book',

  onBrokenLinks: 'warn',
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
    localeConfigs: {
      'zh-Hans': {
        label: '简体中文',
        htmlLang: 'zh-CN',
      },
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: '/docs',
          sidebarPath: './sidebars.ts',
          exclude: ['**/_sidebar.md', '**/docsify.md', '**/index.html'],
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          editUrl: `${repoUrl}/tree/main/`,
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeMathjax, mathjaxConfig]],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          filename: 'sitemap.xml',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['zh', 'en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        indexBlog: false,
        indexPages: false,
        searchResultLimits: 12,
        searchResultContextMaxLength: 100,
      },
    ],
  ],

  themeConfig: {
    metadata: [
      {
        name: 'description',
        content: 'JoyRL Book 是面向强化学习实践的中文教程站点，覆盖基础强化学习、离线强化学习与 JoyRL 框架使用说明。',
      },
      {
        name: 'keywords',
        content: 'JoyRL,强化学习,Reinforcement Learning,离线强化学习,DQN,PPO,SAC,教程,中文文档',
      },
    ],
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      hideOnScroll: true,
      logo: {
        alt: 'JoyRL Book Logo',
        src: 'img/logo-wordmark.png',
        srcDark: 'img/logo-wordmark.png',
        width: 96,
        height: 54,
      },
      items: [
        {to: '/', position: 'left', label: '首页', activeBaseRegex: '^/$'},
        {type: 'doc', docId: 'rl_basic/README', position: 'left', label: '强化学习基础'},
        {type: 'doc', docId: 'offline_rl/README', position: 'left', label: '离线强化学习'},
        {type: 'doc', docId: 'llm_rl/README', position: 'left', label: '大模型与强化学习'},
        {type: 'doc', docId: 'joyrl_docs/main', position: 'left', label: 'JoyRL 文档'},
        {href: repoUrl, label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '文档',
          items: [
            {label: '文档首页', to: '/docs/'},
            {label: '强化学习基础', to: '/docs/rl_basic'},
            {label: '离线强化学习', to: '/docs/offline_rl'},
            {label: 'JoyRL 文档', to: '/docs/joyrl_docs'},
          ],
        },
        {
          title: '资源',
          items: [
            {label: '项目仓库', href: repoUrl},
            {label: 'JoyRL 框架', href: 'https://github.com/datawhalechina/joyrl'},
            {label: 'Notebooks', href: `${repoUrl}/tree/main/notebooks`},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Datawhale China`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
