# S5b/E4 预注册：GN-MLP 对照（dying-gate 决定性检验）

**写于 2026-10-25 14:55，先于任何 s5bgn 数据落盘。**

## 假设（dying-gate hypothesis 的关键预测）

S5/CNN 双判死给出的解释：own-bias 修复起效的前提是 kWTA 神经元死亡。
决定性检验：**在 MLP 上插入 GroupNorm**（与 S1 唯一差异：
`t = GN(Linear(x))`），其余协议逐项不变。GN 归一化特征应显著降低
死亡率；若修复增益同时消失，则"死亡率是修复的必要条件"得到
同架构内的因果支持（比跨架构对比更强——排除了 conv 归纳偏置等
一切其他架构差异）。

## 设计（固定）

- 模型：GNFFNet = exp4.ADNet + GroupNorm(16, nh)，nh=2000，
  dend/kWTA/protect_grads/out 逐项同 S1
- 臂：b ∈ {0,1,3,10} × {mnist,fashion} × T=10 × seeds 0–9 = 80 runs
- 参照臂（已有，不重跑）：c2_*（bare MLP）b=0/1/3/10
- 超参：λ=1000，Adam 1e-3，15 epochs，bs 256（同 S1）

## 预注册判据

**E4 确证（dying-gate 成立）**，需同时满足：
1. GN-MLP b=0 的平均 dead_frac ≤ 0.10（bare MLP 是 0.572）
2. GN-MLP 上 b=10 vs b=0 配对 Δℓ 不显著（|t|<2）于两数据集，
   或显著但 < bare-MLP 增益的 25%

**E4 否证（dying-gate 不成立）**：
1' dead_frac 仍 >0.3，或
2' GN-MLP 上 Δℓ ≥ bare-MLP 增益的 50% 且显著
→ 修复不依赖死亡通道，S5/CNN 的塌缩另有原因（如 conv 特征的
路由无关性），dying-gate 撤回，CNN 篇叙事重写

**混合结果**（死亡消失但增益保留部分）：报事实，机制叙事降级为
"死亡是伪影的必要条件但不是修复的唯一通道"。

## 风险预告

- GN 的可学习仿射（γ,β）可能不足以救活 kWTA——若 dead_frac 仍高，
  判据 1' 自动触发，属合法否证路径，不算失败实验。
- GN-MLP 的 b=0 基线本身可能比 bare MLP 好（归一化收益）——判据
  全部用配对差值，不受基线水平影响。
