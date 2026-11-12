# R2T 预注册：R2「保护对象」结论向 ResNet-18+GN 的迁移检验

**落盘时间**：2026-11-11 21:59:47 +0800（**先于任何 R2T 数据**）
**性质**：R2 触发条款 3（`R2_预注册.md` §3.3）要求的新预注册；R2 走了 H1 成立（阳性）分支，成文前必须补本检验。
**母预注册**：`bench/results_r2/R2_预注册.md`（MLP 版，其中文结论不得因本实验结果反推修改）。

---

## 0. 科学问题（数据前写死）

R2 在小饱和 MLP（Permuted-MNIST/Fashion，T≤20，λ=1000）上得到：**在 renorm 前向基座上，把锚点保护预算按 `exp(−g/τ)` 向近期任务重分配（α=1.0，τ=T/4，总权重守恒）可以同时保住 gain 并改善 retention**（`renorm_rec10`：T20 `Δℓ vs base +8.08`、`Δbwt vs renorm +1.52`，Holm p=6.42e−05；`renorm_anti10` 反向对照 20/20 全负）。

但 E9 已证 **renorm 前向本身在 ResNet-18+GN / Split-CIFAR-100 上是最差臂**（`Δℓ vs base = −8.83`，p=0.001，t=−4.5）。因此本实验回答一个 E9 与 R2 交织出的、两者单独都无法回答的问题：

> **当「前向干预」在真实骨干上有害时，只改「锚点预算方向」（不动前向）是否仍能改善？**

三个可能结局都预先写死判读（§5），无事后择优空间。

## 1. 继承的协议（与 E9 逐字一致，除非本节明写）

- 骨干：torchvision ResNet-18（3×3 stem、无 maxpool、BN→GN(32,C)）；门挂 avgpool 后 512 维特征：`d = max_seg(dend(ctx))`，`h = kwta(t·σ(d), k=128)`（25% 保持率）。
- 数据：Split-CIFAR-100，T=10 任务 × 10 类，task-IL 评估（own-slice argmax），seed 决定类序（`torch.Generator().manual_seed(100+seed)` 的 `randperm(100)`，与 E9 同源）。
- 训练：SGD(0.05, 0.9, wd=5e-4)，5 epochs/task，BS=128，masked softmax，clip 5.0。
- 锚：对角 Fisher（50 draws，max-log-softmax，one-hot ctx），backbone 全局归一化，覆盖 backbone+dend、排除 fc head（E9 口径）。
- **λ=0.1**（S2 round-6 校准值，E9 同款；**不**沿用 MLP 的 1000）。
- G=0.5749（E9 试点冻结值，pin/renorm 使用）。
- 评估一律 oracle ctx；`ell/avg(=final_avg)/bwt` 与 `acc_matrix`（下三角）、`gate_stats` 与 E9 同口径落盘。

## 2. 干预定义（逐字复用 R2）

锚点惩罚的总权重守恒重分配：

```
w_j = (1−α) + α·(v·exp(−g_j/τ)/Σ_k exp(−g_k/τ)),   j = 0..v−1
g_j = (v−1) − j   （锚 j 的年龄；j=v−1 为刚结束的任务）
τ   = T/4 = 2.5（T=10）
```

性质（数学恒等式，非随机条款）：`Σ_j w_j = v` 对任意 α 恒成立；α=0 ⇒ `w_j ≡ 1`（uniform，E9 行为）；所有 `w_j > 0`。反向（`reverse=True`）时指数取 `+g_j/τ`（保护早期）。实现逐字复用 `r2_recency.recency_weights`（含 `phase_weights` 落盘）。

**唯一新自由度是权重方向**：前向、λ、锚构造、Fisher、优化器、数据划分、种子全部与 E9 逐字一致。

## 3. 臂与网格

| arm | 前向 | α | reverse | 作用 |
|---|---|---|---|---|
| `base` | `kwta(t·σ(d))` | 0 | — | E9 共享对照（**直接复用 E9 网格数据，不重跑**） |
| `renorm` | `kwta(t/G)` | 0 | — | E9 共享对照（**直接复用 E9 网格数据，不重跑**） |
| `base_rec10` | base | 1.0 | 否 | **主候选**：不引入有害前向，只重分配锚预算 |
| `renorm_rec10` | renorm | 1.0 | 否 | R2 主候选的完整迁移（在 ResNet 上前向有害） |
| `base_anti10` | base | 1.0 | 是 | 方向对照（保护早期） |

- 新跑 runs = **3 臂 × 10 seeds = 30**（T=10）；`base`/`renorm` 用 `results_e9/e9_grid_s*.json` 的 60 runs（10 seeds 配对齐全）作共享对照，不重跑。
- 分片：`job_id % 2` → GPU 2 shard 0（15 runs）/ GPU 3 shard 1（15 runs）。
- **T=20 不跑**（E9 预注册已排除：T=10 无信号则 T=20 无增量；且单 run ~3 分钟 × T=20 ≈ 6 分钟，成本翻倍无助于 T=10 的判读）。

### 条件触发（写死，不默认执行）

1. **`base_rec10` 剂量轴**（α∈{0.25, 0.5}）：仅当 §5 判读为「rescue」或「fail」且 `base_rec10` 的 Δ 95%CI 跨越 0（信号方向不明）时追加，用于判断是剂量不足还是机制失效。3 档 × 10 seeds = 20 runs。
2. **n 扩到 20**：仅当任一主比较的 raw p ∈ (0.01, 0.1) 时追加（E8 教训：边缘 null 可被扩样本翻转）。
3. toyCNN / GN-MLP 的 R2 迁移：不属于本预注册，另议。

## 4. 指标（字段名与 E9/R2 一致）

`ell`、`avg`、`bwt`、`acc_matrix`（下三角 T=10）、`gate_stats`、`phase_weights`（**新**：每 phase 的 w 向量，round 6 位）、`weight_sums`（Σw，round 6 位）、`arm/alpha/tau/lam/g/nseg`。

配对单位：seed（n=10，单数据集 CIFAR-100，无 pooling）。

## 5. 预注册判读（数据前写死；三层结果都预先定义）

记 `ΔX(arm vs ref) = mean_seed [X_arm − X_ref]`，配对 t（双尾）。**主比较只有 4 个**（预注册家族）：

| # | 比较 | 问题 |
|---|---|---|
| C1 | `base_rec10` vs `base` | 只改锚预算方向（不引入有害前向）是否改善 |
| C2 | `renorm_rec10` vs `renorm` | R2 主候选在 ResNet 上是否仍有边际改善 |
| C3 | `base_anti10` vs `base_rec10` | 方向特异性（与 R2 的 anti 对照同构） |
| C4 | `base_rec10` vs `renorm` | 「改预算方向」vs「改前向」哪个更好 |

**Holm 家族 = {C1, C2} 的 Δell 与 Δbwt（4 个配对 t）**；C3/C4 为描述性（报 p 不进家族）。

### 三层判读（写死）

**T1「transfer」**：C1 的 Δell > 0 且 Δbwt > −1.0（不恶化遗忘）且 Holm 后 C1.ell 显著（p<0.05）→ 「R2 的预算方向结论**迁移**到 ResNet：不动前向、只重分配锚预算即可改善」。
**T2「rescue」**：C1 不达 T1 但 C2 达标（C2.ell Holm p<0.05 且 C2.bwt > −1.0）→ 「renorm 前向在 ResNet 上全局有害，但 recency 加权在 renorm 基座内仍有边际改善——R2 结论**部分迁移**（仅限 renorm 基座）」。
**T3「fail」**：C1、C2 均不达标 → 「R2 的预算方向结论与 renorm 前向一起**不迁移**：在 λ 已按架构标定（0.1）的真实骨干上，锚预算方向不是有效自由度」→ 新论文主结论收窄为 MLP-only，按素材账本 §4 相应条款处理。

**方向特异性（探索性）**：C3 若 `base_anti10` 的 bwt 显著差于 `base_rec10` → 方向特异性成立，与 R2 §8.1 一致。

**判读红线**：
- 任一层级成立都不改写 E9 结论（renorm 前向有害、门是承重墙仍是 E9 事实）。
- T1/T2 成立**不得**写成「recency 加权是通用持续学习改进」；适用范围仅「本协议 ResNet-18+GN / Split-CIFAR-100 / task-IL / λ=0.1 / T=10 / n=10」。
- 也不得反向抹杀 R2：MLP 上 H1 成立与 ResNet 上 fail 并不矛盾，须并列陈述（架构依赖性）。
- `ell` 在本协议上基线低（base=27.28，sd=3.97），Δell 的实际意义按比例解读，报告原始单位。

## 6. 统计口径

- 配对 t（双尾）+ Holm（4 比较家族）；|d_z| 一律报告。
- 逐 seed 方向一致性单独计数，与 p 并列；混合符号记「未达严格形式」，不降级。
- E9 复用数据（base/renorm）与新增数据在同设备（GPU 2/3）同 venv 产生，但 E9 网格运行于 GPU 0/2/3 三卡——**跨卡配对**（新臂在卡 2/3 vs E9 臂在卡 0/2/3）存在设备间非确定性（§7 D2 实测同卡同 seed 两次运行 Δell 最大 6.83），故本实验**主比较 C1/C2 的新臂之间及新臂与 E9 臂的配对都写明「含跨运行非确定性」**；显著性以 paired t 为准，不因非确定性追加门槛。

## 6b. 功效核算（数据前写死，防「样本不够」事后争议）

E9 实测：`renorm−base` 配对差 sd=6.17（前向不同、最坏情形）；同前向只改锚权重的配对差 sd 预计 ≤3（R2 MLP 上 rec10−renorm sd≈1.2，bwt≈1.9；E9 上 bwt 类配对差 sd=0.77–1.31）。按保守 sd=4：
- n=10 → 可检出 |Δ|≥3.1 pp（α=0.05 双尾，power≈0.8）；
- R2 的 MLP 效应（+8.08 / +1.52）若按比例缩放到 ResNet 的 ell≈27 基线（≈2.7 pp / 0.5 pp bwt），n=10 对 ell 检验力勉强、对 bwt 不足——**这正是本实验要测的问题本身**（迁移后效应是否还存在），若效应缩水到检验阈以下，按 T3 判 fail 并如实报告功效限制，不扩样本硬凑显著（扩样本条款见 §3.2）。

## 7. 实现正确性门槛（未达则数据作废、排查后重跑）

**总则（三次门槛教训的固化）**：本节只写**数学恒等式与运行时断言**，不写任何依赖跨运行数值复现的条款——D 探针已证 conv 骨干上 cuDNN 非确定性使逐位门槛不可能成立（同卡同 seed 两次 Δell 差 6.83）。

1. **G1 权重守恒（恒等式）**：每个 run 的每 phase，`|Σw_j − v| < 1e-6` 且所有 `w_j > 0`；`phase_weights`/`weight_sums` 落盘完整（T=10 → 10 个向量）。
2. **G2 α 分派正确（恒等式）**：`base_rec10`/`renorm_rec10`/`base_anti10` 的 α=1.0；落盘字段 `alpha` 与臂名一致；anti 的 `phase_weights` 须满足 `w[0] > w[-1]`（早期权重更大）且非 anti 满足 `w[0] < w[-1]`。
3. **G3 复用数据完整性**：E9 的 base/renorm 各 10 seeds 齐全（60 runs 文件里按 (arm,seed) 索引）；新跑 30 runs 无重复键 (arm, seed)，臂集合 = {base_rec10, renorm_rec10, base_anti10}。
4. **G4 运行时断言（脚本内置，非收数后检查）**：`loss = loss + lam * pen_weighted` 中的 `lam` 必须是 `args.lam`（默认 0.1）且写进每条记录（防 S4/E6 lam 分派 bug 复现）；`k=128`、`NH=512`、`EPOCHS=5`、`BS=128` 写死并落盘。
5. **G5 增量落盘**：每 run 完成即 append 写盘（共享卡若闪断不丢已完成 runs）。
6. **G6 门统计合理带**：新臂 `gate_stats` 的 `gate_mean` ∈ [0.50, 0.65] 且 sd < 0.05（与 E9 H4 同窗口）——超出则先排查实现再收数。

## 8. 诚实披露与边界

- 任何偏离（臂增减、阈值替换、样本量变更、条件触发追加）必须写独立 `R2T_修订说明.md`，**不修改本文件**。
- 本实验**不修改** `main.tex`；结论进新论文素材账本。
- 若 T3 fail：素材账本 §4 的「ResNet 迁移检验」项照常打勾（完成了检验、结论为不迁移），新论文叙事按 §5 T3 条款收窄，**不因结果不理想而隐藏**。
- 若 T1/T2：E9 的「renorm 前向有害」与 R2 的「renorm 基座上预算方向有效」须并列陈述，不得只报一侧。

---

**预期结局的三条分支（写死防事后择优）**：
- **T1** → 新论文主张「保护对象自由度跨架构成立（MLP+ResNet），when 否证 + what 成立 + 外部效度边界」。
- **T2** → 「预算方向仅在前向被重标定的基座内有效——与 MLP 上的特异性对照（base_rec10 无 gain）呼应，形成『基座×预算』交互 story」。
- **T3** → 「λ 已标定的真实骨干上，锚预算方向不是有效自由度；R2 结论收窄为 MLP-only，与 E9 的 renorm 边界合并为架构依赖性证据」。
