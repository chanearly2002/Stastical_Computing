# CLAUDE.md

## 项目目标

复现并彻底理解论文：**Shalit, Johansson, Sontag (ICML 2017), "Estimating individual treatment effect: generalization bounds and algorithms"**（CFR / TARNet）。

用户是学习者，目标是"完全理解"，不只是跑通代码。所有产出以**讲清楚、可验证**为先。

## 仓库结构

```
Estimating individual treatment effect ：generalization bounds and algorithms.pdf   # 原文（文件名含全角冒号 "："，引用时务必加引号）
notes/paper_fulltext.txt     # pdftotext -layout 提取的全文，供检索（公式为纯文本，不完全可靠，以 PDF 为准）
notes/                       # 讲解笔记，按章节编号：01_problem.md, 02_theory.md ...
docs/                        # 交互式 HTML 讲解页（自包含单文件）：index / theory / algorithm / experiments
src/                         # 复现代码（见下）
figures/                     # 生成的图（推导可视化、实验复现图）
.claude/skills/              # 项目级 skills（见下）
```

## 工作约定

- **语言**：与用户交流和写笔记一律用中文；公式、符号、代码标识符保持英文/LaTeX。
- **符号**：统一沿用论文记号（Φ, h, ε_F, ε_CF, IPM_G, B_Φ, u=p(t=1), PEHE 等），首次出现必须给出定义和直觉。
- **讲解节奏**：按 论文结构 → 问题设定 → 假设 → 引理/定理推导 → 算法 → 实验 → 局限 逐步推进，每一步结束时用 2-3 个自测问题确认理解，再进入下一步。
- **引用**：讲解中引用论文位置（章节、公式号、图表号）。论文正文引用"补充材料"的证明（Lemma 1、Theorem 1 的完整证明、Theorem 2/3 的 B_Φ 常数）**不在本 PDF 内**；讲到这些时要明确说明是"根据正文思路重建的推导"，不要假装是原文逐字证明。
- **不编造**：不确定的结论（如论文没给的超参数、补充材料里的细节）要标注"论文未给出"。数值结果必须来自实际运行或论文表格，并注明来源。
- **推导**：每个公式推导逐步写出，每一步注明用到的假设或不等式；能用数值例子或代码验证的就验证（例如用 numpy 数值检验 IPM 不等式、Wasserstein/MMD 估计）。
- **可视化**：matplotlib 出图存到 `figures/`，文件名带章节编号。图要标注坐标轴、符号含义。

## 复现计划（src/）

数据：
- **IHDP**：Hill (2011) 半合成数据，747 样本、25 协变量；结果用 NPCI setting "A"。需联网获取，若环境网络不通则退回到自行模拟的同构数据，并在笔记中说明差异。
- **Jobs**：LaLonde / Dehejia-Wahba，需单独获取。

模型（PyTorch，当前环境**未安装** torch / scipy，用前先 `pip install torch scipy matplotlib numpy`）：
- `TARNet`：共享表示网络 Φ（3 层 ELU）+ 两个输出头 h₀、h₁（各 3 层），每个样本只更新对应观测处理的头。
- `CFR`：在 TARNet 目标上加 `α · IPM(Φ(x)|t=0, Φ(x)|t=1)`，实现 Wasserstein 与线性 MMD² 两种；Φ 需归一化（投影或固定尺度 batch-norm）。
- 样本重加权 `w_i = t_i/(2u) + (1-t_i)/(2(1-u))`。
- 评价：√PEHE、|ε_ATE|、policy risk（Jobs）。

建议复现顺序：①合成小例子验证 IPM 与 bound 的直觉 → ②TARNet → ③加 MMD → ④加 Wasserstein → ⑤IHDP 全流程 → ⑥对照论文表 1。

## 项目 skills（`.claude/skills/`）

- `paper-walkthrough`：按固定节奏逐章讲解论文。
- `derive-formula`：对公式逐步推导，并用代码数值验证 / 画图。
- `paper-mindmap`：生成论文结构脑图（Mermaid）和概念依赖图。
- `html-explainer`：生成交互式 HTML 讲解页（SVG 图、滑块演示、分步动画），存到 `docs/`。

用户说"讲解第 X 节""推导 Theorem 1""画脑图""生成 html / 图解"时，调用对应 skill。

## 常用命令

```bash
# 重新提取全文
pdftotext -layout "Estimating individual treatment effect ：generalization bounds and algorithms.pdf" notes/paper_fulltext.txt
# 安装依赖
pip install torch scipy matplotlib numpy
```

## 进度

Sec 1-5 已讲解并有笔记（notes/00-06）；HTML 图解已有 docs/theory.html（Sec 1-3）、docs/algorithm.html（Sec 4）、docs/experiments.html（Sec 5）。Sec 6 与复现尚未开始。HTML 页面里论文的公式和变量必须写全并列变量表（见 html-explainer skill）。

## Git

- 开发分支：`claude/funny-euler-8bki0s`，只往这个分支提交。
- 不要主动创建 PR。
