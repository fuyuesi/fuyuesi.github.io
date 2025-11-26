import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "付悦思的面经",
  description: "计算机基础与算法笔记",
  themeConfig: {
    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      { text: '大模型', link: '/notes/llm-transformer' },
      { text: '机器学习', link: '/notes/ml-metrics' }
    ],

    // 左侧侧边栏
    sidebar: [
      {
        text: '大模型与算法',
        items: [
          { text: 'Transformer 与 LoRA', link: '/notes/llm-transformer' },
        ]
      },
      {
        text: '机器学习基础',
        items: [
          { text: '准确率与召回率', link: '/notes/ml-metrics' },
        ]
      },
      {
        text: '前沿技术',
        items: [
          { text: 'VLM 文档理解', link: '/notes/vlm-ocr' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/fuyuesi/fuyuesi.github.io' }
    ]
  }
})