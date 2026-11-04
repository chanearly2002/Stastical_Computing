---
name: html-explainer
description: 为本仓库的 ITE/CFR 论文某一部分生成自包含的交互式 HTML 讲解页（SVG 示意图、滑块演示、分步动画），存到 docs/。当用户说"生成 html""做个可视化页面""交互式讲解""图解"时使用。
---

# 交互式 HTML 讲解页

## 产出

- 一个**单文件、完全离线可用**的 `docs/<主题>.html`（内联 CSS 与 JS，不依赖 CDN 与外部字体）。
- 页面内容来自 `notes/` 与论文 PDF，不添加论文里没有的主张。

## 页面规范

1. **标题** `<title>` 用 2-4 个词的名称（如 "CFR 算法图解"），不要写成"……：讲解"。
2. **配色用 token**：`:root` 定义浅色，`@media (prefers-color-scheme: dark)`（加 `:root:not([data-theme="light"])`）与 `:root[data-theme="dark"]` 重定义深色；`body` 显式背景。
3. **颜色语义固定**：处理组 t=1 暖色，对照组 t=0 冷色，表示 Φ 青绿，IPM 玫红，其余用中性色。
4. **响应式**：手机宽度（约 400px）不得横向滚动；左右留白 `max(16px, env(safe-area-inset-*))`；宽图放进 `overflow-x:auto` 容器。
5. **图**：内联 SVG，`<figure>` + `<figcaption>`，`role="img"` 与 `aria-label`；箭头要有文字标签。
6. **交互**：滑块、按钮要有 `id` 和可见焦点；`prefers-reduced-motion` 下不做动画。
7. **来源标注**：每个演示用小标签区分"论文内容"与"自编演示/示意"。自编演示不得暗示是论文结果。
8. **诚实**：补充材料里的内容（PDF 中没有）要写明"论文未给出"。
9. **公式与变量不省略**：页面涉及的论文公式、定义、定理必须完整写出，每个公式放进"公式卡片"（标题 + 论文位置 + 公式 + 变量表 + 折叠的英文原文），所有变量逐个解释；Theorem/Lemma 要有显式陈述。

## 验证（写完后做一次）

用 Playwright（已预装，`require('playwright')`）打开文件，在 1280 与 400 两种宽度、浅色与深色下检查：
- 无 `pageerror` 与 console error；
- `document.documentElement.scrollWidth <= window.innerWidth`；
- 截图查看一次。

数值演示的计算核心可先用 node 单独跑一遍，确认曲线形状合理。

## 交付

写入 `docs/`，提交并推送；需要即时查看时，用 SendUserFile（display 为 render）发给用户。
