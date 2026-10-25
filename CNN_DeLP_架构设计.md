# CNN-DeLP 架构设计（v1，2026-10-25）

## 目标

把 S1 的"段专门化审计 → 死神经元伪影 → 因果修复"链迁移到卷积骨干上，
检验三条主张（G1 修复泛化 / G2 伪影复现 / G3 机制保持）。

## 接口约束（决定复用率）

branch_delp.py 的审计/修复代码通过以下接口与模型解耦，CNN 版只要
逐项满足即可 90% 复用：

| 接口 | MLP-ADNet 现状 | CNN 版要求 |
|---|---|---|
| `model(x, ctx) -> (logits, seg_id)` | ✓ | ✓（x 为图像张量） |
| `model.protect_grads(task_id)` | 冻结非当前段的 dend 梯度 | 同左（dend 结构不变） |
| `model.dend(ctx)` -> (nh*nseg, ntask) 权重 | ✓ | ✓（dend 结构不变） |
| `flat_params(model)` -> {"ff.w","ff.b","out.w","out.b"} | nn.Linear 直取 | conv 特征器需展平命名 |
| `kwta(t * sigmoid(d))` | t 是 (B,nh) | t 是 (B,nh)（pool 后） |
| eval 协议 | permute + one-hot ctx | 同左 |

关键洞察：**dendritic 段层不关心 t 的来源**。MLP 里 `t = ff(x)`，
CNN 里 `t = pool(conv_stack(x))`——只要 conv 栈输出 (B, nh) 特征，
dend/kWTA/out/protect_grads 全部原样保留。

## 三方案对比（segment 挂哪里）

### 方案 A：尾部 dendritic 层（tail-mounted）★ 推荐

```
x (B,1,28,28) → conv 栈 [32,64] × GN → GAP → t (B,nh=2048)
ctx (B,ntask) → dend (ntask → nh*nseg) → u (B,nseg,nh)
win = argmax(u, dim=1); h = kWTA(t ⊙ sigmoid(u_win))
→ out (nh → 10) → logits
```

- 优点：与 MLP-ADNet **逐接口同构**；flat_params 只需把 ff 换成
  conv 栈（"ff.w" 变成 conv 栈参数的串联视图——锚构造对参数字典
  逐 key 记 theta/fisher，只需展平命名）；审计仪表（win-rate、
  dom_hist、dead_frac）全部不动。
- 缺点：伪影发生在"特征已经学完"之后——若 CNN 的 conv 特征本身
  不死（ReLU+GN 比 kWTA 温和），伪影可能弱于 MLP。**这本身就是
  G2 要测的东西**，不是缺陷。

### 方案 B：每段独立卷积分支（branch-parallel）

nseg 个并行 conv 栈，ctx 选路。忠实模仿生物（apical vs basal），
但参数 ×nseg、protect_grads 语义要改（冻结非当前栈的 conv 梯度）、
锚的参数命名爆炸。**审稿人会问"为什么不用 A"**，且与 S1 的
"shared substrate"叙事断裂（S1 的 granularity null 正是建立在
共享 ff 上）。弃。

### 方案 C：kWTA 作用于 conv feature map（spatial kWTA）

把 kWTA 放进 conv 层间。最激进，但 (a) 死神经元伪影在空间维度
的定义不明（一个 filter 死 ≠ 一个位置死），(b) 与 S1 的
neuron-level 锚无法对齐。留作 discussion，不进主实验。

## 选型：方案 A

理由：接口同构 → branch_delp.py 的 run_gran/BiasADNet/审计插桩
只需换 `ff` 定义与 flat_params 命名；S1 的全部预注册判据模板
（剂量格、配对种子、win-rate 口径）直接搬。

## 实现细节

1. **conv 栈**（MNIST/Fashion 28×28 → 适配 CIFAR 32×32 改 stride）：
   - Conv(1→32,3,s1) GN ReLU → Conv(32→64,3,s2) GN ReLU →
     Conv(64→128,3,s2) GN ReLU → Conv(128→256,3,s2) GN ReLU → GAP
   - nh=2048（对标 MLP 的 2000，kWTA k=64 同）
   - BN→GN：延续 anchor_budget_cifar 的选择（CL 下 BN 统计量漂移
     是已知混淆源）
2. **flat_params 命名**：conv 栈序列化为 "ff.c{i}.w"/"ff.c{i}.b"/
   "ff.gn{i}.w"/... —— anchor_penalty 对 key 集合是 agnostic 的
   （逐 key 做二次型），只需 theta/fisher 字典同构。
   Fisher 插桩 `fisher_raw` 用 `model(x, ctx)` 的 loss 对参数求导，
   同样 agnostic。
3. **usage_stats / dom_instr**：读 `model.dend(ctx)`，dend 结构
   未变 → 原样复用。
4. **bias 注入**（BiasConvNet）：与 BiasADNet 相同——
   forward 里对 u 的 own-segment 项加标量/表偏置。
5. **训练超参**：lr 1e-3 Adam、epochs 15、bs 256 与 MLP 版一致
   （先对齐协议再谈调参；CIFAR 上如 loss 不收敛再单独预注册调参，
   不与主实验混）。

## 与 S1 判据的对齐

- G1（修复泛化）：剂量格 b∈{0,1,3,10} × {mnist,fashion} × T∈{10,20}
  × gran=neuron × 10 seeds = 160 runs。判据：Δℓ 显著为正 ≥15/20 格、
  符号一致 ≥18/20。
- G2（伪影复现）：untrained probe（3 seeds）+ b=0 臂的 dom_hist
  插桩 → naive vs active 对比。
- G3（机制保持）：signed-drift（b∈{0,10}×3 seeds）+ task-relevance
  （s3_corr 协议，4 组 × b∈{0,1,3,10} × 10 seeds）。
- 判死分支：G1 失败 + G2 成立 → 转"伪影泛化、修复架构特异"论文。

## 风险

- conv 栈在 permuted-MNIST 上可能"太容易"（ℓ 天花板高），剂量
  响应天花板效应 → 判据里加"若 b=0 臂 ℓ>90 则报饱和"的护栏。
- 160 runs × ~3 min/run（T20 conv 比 MLP 慢 ~3×）≈ GPU 时 2 天
  （4 GPU 并行 ~12h）。
