# S5/CNN 预注册：段专门化审计与因果修复的跨架构检验

**写于 2026-10-25 13:45，先于任何 CNN-DeLP 数据落盘。**
架构选型见 `CNN_DeLP_架构设计.md`（方案 A：尾部 dendritic 层）。

## 主张（若存活）

S1（MLP-ADNet）发现的"反专门化 + 死神经元伪影 + own-bias 修复"
不是 MLP 特异：
- G1 修复泛化：own-bias 剂量响应在 ConvADNet 上复现
- G2 伪影复现：untrained naive 占比 >> active（同 S1 模式）
- G3 机制保持：无负向 signed-drift（尺度分离）+ 结构化容量
  （task-relevance：零相关任务全剂量 flat）

## 设计（固定）

- 模型：ConvADNet（4 conv + GN + GAP → nh=2048；dend/kWTA/out 与
  MLP-ADNet 逐接口同构）；gran=neuron，λ=1000
- 数据/任务：Permuted-MNIST、Permuted-Fashion（28×28，与 S1 同
  置换种子），T ∈ {10,20}
- 臂：b ∈ {0,1,3,10} × 2 datasets × 2 T × seeds 0–9 = 160 runs
- 超参：Adam lr 1e-3，epochs 15，bs 256，kWTA k=64（全部对齐 S1）
- 分片：4 GPU × i%4，每片 40 runs

## 预注册判据

**G1 存活**：20 个 (ds×T) 剂量格（b≥1）中 Δℓ 显著为正（|t|>2）
≥15/20，且符号一致 ≥18/20；Δavg 与 Δℓ 同时显著 ≥12/20。
**G1 判死**：显著格 <10/20 或符号一致 <14/20 → 撰写"修复架构特异"
分支论文（与 G2 联合判读）。
**饱和护栏**：若任一 (ds,T) 的 b=0 臂均值 ℓ>90，该格所有判据
标记 saturated，从分子分母同时剔除并在正文披露。

**G2 存活**：untrained probe 的 naive 段占比 > active 段占比
（3 seeds，差值 ≥0.05）；且 trained b=0 臂的 naive 占比回落至
active 水平附近（差值 <0.05）。
**G2 判死**：两者相当（<0.05）→ conv+GN 骨干下 kWTA 死亡率不足，
伪影不成立——本身可报（"伪影需要 MLP 的裸 Linear+kWTA 结构"）。

**G3**（仅 G1 存活后运行）：
- signed-drift：b=10 臂 own 段位移不显著为负（3 seeds × 10 phases）
- task-relevance：out 组（random 标签）全剂量 ℓ≈chance（|Δ|<1）
  且 high−low 差距随剂量不扩大

## 风险预告（诚实记录）

- Conv+GN 的特征死亡率可能远低于裸 Linear → G2 风险真实存在；
  若 G2 判死，G1 的"修复"可能只是容量效应无伪影背景，叙事降级。
- ℓ 天花板：permuted-MNIST 上 conv 可能学满 → 饱和护栏已设。
- 时间预算：160 runs 预计 ~2 GPU-天（4 卡并行 ~12h）；
  G2/G3 追加 ~0.5 天。

## 脚本与数据（计划）

- `bench/cnn_delp.py`：ConvADNet + BiasConvNet + run_gran 镜像
  （dend/kWTA/protect_grads 原样 import from exp4）
- 结果：`bench/results/cnnc2_{ds}_b{b}.json`（T10）、
  `cnnc2t20_{ds}_b{b}.json`（T20）
- 参照：无（首个 CNN 实验，全部自建臂）

## 收数后动作

- G1 存活 → 起草 CNN 论文（A+B 主张合体），S1 引为 MLP 基座
- G1 判死 + G2 存活 → "伪影泛化、修复特异"分支
- 双存活/双死 → 如实报告，架构定律叙事需重新设计
