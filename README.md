# Stastical_Computing

复现并读懂一篇因果推断与机器学习的论文：

> Uri Shalit, Fredrik D. Johansson, David Sontag.
> **Estimating individual treatment effect: generalization bounds and algorithms.** ICML 2017.
> 提出 CFR（Counterfactual Regression）与 TARNet，用"表示学习 + 两组分布距离惩罚"估计个体处理效应（ITE），并给出对应的泛化误差界。

论文 PDF 在仓库根目录：`Estimating individual treatment effect ：generalization bounds and algorithms.pdf`（文件名含全角冒号，命令行里请加引号）。

## 论文一句话

个体效应估计的误差，不超过两个处理组各自的预测误差，加上两组在表示空间里的分布距离。所以要学一个既预测得准、又让处理组与对照组"看起来像"的表示。

## 学习进度

| 部分 | 内容 | 笔记 |
|---|---|---|
| 总览 | 用通俗语言走一遍全文 | [notes/00_overview.md](notes/00_overview.md) |
| Sec 1 | 潜在结果、ITE、混杂、强可忽略性 | [notes/01_problem.md](notes/01_problem.md)，交互页面 [docs/theory.html](docs/theory.html) |
| Sec 2 | 相关工作 | [notes/02_related_work.md](notes/02_related_work.md) |
| Sec 3.1 | 符号与定义（Φ、ε_F、ε_CF、PEHE、IPM） | [notes/03_definitions.md](notes/03_definitions.md)，[docs/theory.html](docs/theory.html) |
| Sec 3.2 | Lemma 1、Theorem 1 的推导与数值验证 | [notes/04_bounds.md](notes/04_bounds.md)，[docs/theory.html](docs/theory.html) |
| Sec 4 | 算法：网络结构、目标函数、训练过程 | [notes/05_algorithm.md](notes/05_algorithm.md)，交互页面 [docs/algorithm.html](docs/algorithm.html) |
| Sec 5 | 实验：数据集、指标、表 1、图 2-3 | [notes/06_experiments.md](notes/06_experiments.md)，交互页面 [docs/experiments.html](docs/experiments.html) |
| Sec 6 | 结论与局限 | 待讲 |
| 复现 | TARNet / CFR 的 PyTorch 实现与对照论文表 1 | 未开始 |

论文结构脑图与符号表见 [notes/mindmap.md](notes/mindmap.md)。

## 目录结构

```
CLAUDE.md                    # 项目约定（给 Claude Code 读的）
Estimating individual ….pdf  # 论文原文
notes/                       # 按章节编号的讲解笔记
docs/index.html              # 三个图解页面的目录
docs/theory.html             # Sec 1-3 理论部分图解（公式、变量、Theorem 1 全文）
docs/algorithm.html          # 算法部分的交互式图解（单文件，离线可用）
docs/experiments.html        # 实验部分的交互式图解（含论文 Fig 2-3 的截图）
src/                         # 数值验证脚本；后续放复现代码
figures/                     # 生成的图
.claude/skills/              # 项目级 skills
```

## 怎么看交互页面

`docs/` 下的几个 HTML（从 `docs/index.html` 进入）是自包含的网页，GitHub 网页上只会显示源码。克隆仓库后直接用浏览器打开即可，也可以开启 GitHub Pages 在线查看。每页里论文的公式与变量都写全了，并区分"论文内容"和"自编演示"。

## 运行验证脚本

依赖：`numpy`、`matplotlib`（复现阶段还需要 `torch`、`scipy`）。

```bash
pip install numpy matplotlib
```

```bash
python3 src/toy_sec31.py       # Sec 3.1 的玩具例子：ε_F、ε_CF、PEHE、Wasserstein
```

```bash
python3 src/verify_thm1.py     # 随机问题上数值检验 Lemma 1 与 Theorem 1
```

```bash
python3 src/plot_thm1.py       # 生成 figures/03_thm1_chain.png
```

## 项目 skills

放在 `.claude/skills/`，在 Claude Code 里按名称或自然语言触发：

| skill | 用途 |
|---|---|
| `paper-walkthrough` | 按章节逐步讲解论文，每节末尾自测 |
| `derive-formula` | 公式逐步推导，配代码数值验证与出图 |
| `paper-mindmap` | 生成论文结构脑图与论证链（Mermaid） |
| `html-explainer` | 生成交互式 HTML 讲解页，存到 `docs/` |

## 内容来源的约定

- 论文正文引用的"补充材料"（Lemma 1、Theorem 1 的完整证明，B_Φ 的具体常数，IPM 梯度细节，超参数选择流程）**不在这份 PDF 里**。笔记中凡是推导，都标明是"根据正文思路重建"，并用代码核对过。
- 自编的玩具例子和演示数据都会明确标注，它们不是论文的实验结果。
- 论文没有给出的信息记为"论文未给出"。

## 参考

- 论文作者代码：<https://github.com/clinicalml/cfrnet>（论文脚注给出）
- 前作：Johansson, Shalit, Sontag. Learning representations for counterfactual inference. ICML 2016.
