import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "付悦思的面经",
  description: "计算机基础与算法笔记",
  themeConfig: {
    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      { text: '大模型', link: '/notes/llm-transformer' }
    ],

    // 左侧侧边栏
    sidebar: [
      {
        text: '大模型与算法',
        items: [
          { text: 'Transformer 与 LoRA', link: '/notes/llm-transformer' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/fuyuesi/fuyuesi.github.io' }
    ]
  }
})