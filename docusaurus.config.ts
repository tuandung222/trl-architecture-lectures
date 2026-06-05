import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'TRL Internals',
  tagline: 'Chuỗi bài giảng phân tích chi tiết kiến trúc & hiện thực thư viện TRL (Transformer Reinforcement Learning) của Hugging Face',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://tuandung222.github.io',
  baseUrl: '/trl-architecture-lectures/',

  organizationName: 'tuandung222',
  projectName: 'trl-architecture-lectures',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'vi',
    locales: ['vi'],
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl:
            'https://github.com/tuandung222/trl-architecture-lectures/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    metadata: [
      {name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet'},
    ],
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'TRL Internals',
      logo: {
        alt: 'TRL Internals Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Bài học (Lectures)',
        },
        {
          href: 'https://github.com/tuandung222/trl-architecture-lectures',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Bài học',
          items: [
            {
              label: 'Bài 0: Nền tảng Alignment & RLHF',
              to: '/docs/lesson_0_alignment_fundamentals',
            },
            {
              label: 'Bài 4: GRPO Internals',
              to: '/docs/lesson_4_grpo_internals',
            },
            {
              label: 'Bài 8: Thực hành Toy Alignment Pipeline',
              to: '/docs/lesson_8_toy_alignment_pipeline',
            },
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/tuandung222/trl-architecture-lectures',
            },
            {
              label: 'TRL GitHub',
              href: 'https://github.com/huggingface/trl',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} TRL Internals. Biên soạn bởi tuandung222. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'markdown'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
