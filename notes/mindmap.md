# 论文结构脑图

## Mindmap

```mermaid
mindmap
  root((ITE 泛化界与 CFR))
    问题
      目标 τ(x)=E[Y1-Y0|x]
      反事实不可见
      混杂导致分布不平衡
    假设
      强可忽略性
        (Y1,Y0)⊥t|x
        0<p(t=1|x)<1
      一致性 SUTVA
      Φ 可逆可微 (Assumption 1)
    理论 Sec 3
      定义
        ε_F 事实损失
        ε_CF 反事实损失
        PEHE (Eq.1)
        IPM
      Lemma 1
        ε_CF ≤ 加权ε_F + B_Φ·IPM
      Theorem 1 (Eq.2)
        PEHE ≤ 2(ε_F^0 + ε_F^1 + B_Φ·IPM − 2σ²)
      IPM 特例
        Wasserstein (1-Lipschitz)
        MMD (RKHS 单位球)
    算法 Sec 4
      TARNet α=0
        共享 Φ + 双头 h0 h1
      CFR α>0
        目标函数 Eq.3
        权重 w_i
        Φ 归一化
      Algorithm 1 mini-batch SGD
    实验 Sec 5
      IHDP 半合成
        √PEHE, ATE 误差
      Jobs 真实
        policy risk, ATT 误差
      结论
        IHDP 上显著优于基线
        Jobs 上相当
    局限
      无隐藏混杂假设
      α / B_Φ 难确定
      无置信区间
```

## 论证链

```mermaid
flowchart TD
    A[强可忽略性] --> B[τ(x) 可识别]
    B --> C[PEHE ≤ 2(ε_F + ε_CF − 2σ²)]
    C --> D[ε_CF 不可估计]
    D --> E[Lemma 1: 用 IPM 约束 ε_CF − ε_F]
    E --> F[Theorem 1: PEHE ≤ 2(ε_F^0 + ε_F^1 + B_Φ·IPM − 2σ²)]
    F --> G[目标函数 Eq.3: 预测损失 + α·IPM]
    G --> H[CFR / TARNet]
    H --> I[IHDP / Jobs 实验]
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
