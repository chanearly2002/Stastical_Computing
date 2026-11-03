# 论文结构脑图

## Mindmap

```mermaid
mindmap
  root((ITE 泛化界与 CFR))
    问题
      目标 tau x = E Y1 - Y0 given x
      反事实不可见
      混杂导致分布不平衡
    假设
      强可忽略性
        Y1, Y0 独立于 t, 给定 x
        0 < p t=1 given x < 1
      一致性
      Phi 可逆可微 Assumption 1
    理论 Sec 3
      定义
        事实损失 eps_F
        反事实损失 eps_CF
        PEHE Eq.1
        IPM
      Lemma 1
        eps_CF <= 加权 eps_F + B_Phi * IPM
      Theorem 1 Eq.2
        PEHE <= 2 eps_F0 + eps_F1 + B_Phi * IPM - 2 sigma^2
      IPM 特例
        Wasserstein 1-Lipschitz
        MMD RKHS 单位球
    算法 Sec 4
      TARNet alpha = 0
        共享 Phi + 双头 h0, h1
      CFR alpha > 0
        目标函数 Eq.3
        权重 w_i
        Phi 归一化
      Algorithm 1 mini-batch SGD
    实验 Sec 5
      IHDP 半合成
        sqrt PEHE , ATE 误差
      Jobs 真实数据
        policy risk, ATT 误差
      结论
        IHDP 上显著优于基线
        Jobs 上相当
    局限
      无隐藏混杂假设
      alpha 与 B_Phi 难确定
      无置信区间
```

## 论证链

```mermaid
flowchart TD
    A["强可忽略性"] --> B["tau(x) 可识别"]
    B --> C["PEHE <= 2(eps_F + eps_CF - 2 sigma^2)"]
    C --> D["eps_CF 无法直接估计"]
    D --> E["Lemma 1: 用 IPM 约束 eps_CF 与 eps_F 的差"]
    E --> F["Theorem 1: PEHE <= 2(eps_F0 + eps_F1 + B_Phi * IPM - 2 sigma^2)"]
    F --> G["目标函数 Eq.3: 预测损失 + alpha * IPM"]
    G --> H["CFR / TARNet"]
    H --> I["IHDP / Jobs 实验"]
```

## 符号表

| 符号 | 含义 | 位置 |
|---|---|---|
| Y₀, Y₁ | 潜在结果 | Sec 1 |
| τ(x) | ITE = m₁(x) − m₀(x) | Sec 1 |
| Φ, Ψ | 表示函数及其逆 | Assumption 1 |
| h | 表示空间上的假设 | Sec 3.1 |
| ε_F, ε_CF | 事实 / 反事实损失 | Def 2 |
| u | p(t=1) | Def 3 |
| PEHE | ITE 的均方误差 | Def 6 / Eq.1 |
| IPM_G | 积分概率度量 | Sec 3.1 |
| B_Φ | 使 ℓ/B_Φ ∈ G 的常数 | Lemma 1 |
| α, λ | IPM / 复杂度惩罚权重 | Eq.3 |
