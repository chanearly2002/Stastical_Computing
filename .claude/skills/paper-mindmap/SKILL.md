---
name: paper-mindmap
description: 为本仓库的 ITE/CFR 论文生成结构脑图和概念依赖图（Mermaid mindmap/flowchart），帮助把握论文全貌。当用户说"脑图""mindmap""论文结构""概念关系图"时使用。
---

# 论文结构可视化

## 产出

1. **mindmap**（Mermaid `mindmap`）：一级分支为 问题 / 假设 / 理论 / 算法 / 实验 / 局限，往下展开概念与公式号。
2. **论证链 flowchart**（Mermaid `flowchart TD`）：
   强可忽略性 → ITE 可识别 → PEHE 分解为 ε_F + ε_CF → 用 IPM 约束 ε_CF（Lemma 1）→ Theorem 1 → 目标函数 (3) → CFR / TARNet → 实验验证。
3. **符号表**：符号、含义、所在位置。

## 做法

- 内容来自 `notes/paper_fulltext.txt` 与 PDF，不添加论文里没有的主张。
- 写成 `notes/mindmap.md`（Mermaid 代码块，GitHub 可直接渲染）。
- 用户需要图片或可交互页面时，再用 matplotlib/graphviz 导出到 `figures/`，或发布为 HTML 页面。
- 节点文字保持简短，公式号标在节点里（如 "Thm 1 (Eq.2)"）。
