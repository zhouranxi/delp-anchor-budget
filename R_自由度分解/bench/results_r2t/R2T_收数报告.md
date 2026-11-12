# R2T 收数报告：「保护对象」结论向 ResNet-18+GN 的迁移检验

**权威数据**：`r2t_grid.part{0,1}.json`（首轮 30 runs）+ `r2t_ext.part{0,1}.json`（扩样新臂 30 runs）
**共享对照**：`results_e9/e9_grid_s{0,1,2}.json` 的 `base`/`renorm` seeds 0–9（60 runs，复用不重跑）+ `r2t_ctrl_ext.part{0,1}.json`（E9 原脚本补跑 base/renorm seeds 10–19，40 runs，修订说明 §5）
**预注册**：`R2T_预注册.md`（2026-11-11 21:59:47 落盘，先于任何 R2T 数据）
**修订说明**：`R2T_修订说明.md`（§1–4：G5 检查器字段口径；§5：扩样条款执行与共享对照扩样）
**设计探针（预注册前，环境刻画非 R2T 数据）**：同卡同 seed 两次跑 E9 base，`Δell` 最大差 **6.83**——cuDNN 卷积非确定性坐实，conv 骨干上逐位门槛不可能成立，本预注册 §7 只写恒等式与运行时断言
**用途**：新论文素材（素材账本 §4 的硬缺口项）；不修改 TMLR 投稿稿 `main.tex`

---

## 0. 执行摘要

**T3「fail」（扩样终判）**：锚预算方向在 λ 已标定的真实骨干上不是有效自由度；首轮 n=10 的 T1（C1.ell +1.94，Holm 0.0411）被 n=20 扩样推翻（+0.72，p=0.335，方向 11/9），翻转如实披露。

终判口径（n=20）：C1 Δell +0.72（Holm 1）、Δbwt -0.47（Holm 0.13）；C2 Δell -0.81（Holm 1）、Δbwt +0.10。

§7 六个硬门槛全部 PASS（两轮均 PASS）；新跑 30+30 runs + 共享对照复用 E9 60 + 扩样对照 40 runs；主比较含跨运行非确定性（D 探针 6.83 点），以配对 t 为准。本报告不修改 `main.tex`，结论进新论文素材账本。

## 1. 数据与运行纪律

| 项 | 值 |
|---|---|
| 计算资源 | `10.10.52.210` GPU 2 / 3（共享卡；GPU 0/1 他人任务，**未触碰**） |
| 首轮 runs | **30** = 3 臂 × 10 seeds（22:57:55 / 22:58:07，rc=0） |
| 扩样 runs（触发条款 2） | **70** = 新臂 3 × 10 新 seeds（00:21/00:22，rc=0）+ E9 原脚本对照 2 × 10（rc=0） |
| 复用 runs | **60**（E9 网格 base/renorm seeds 0–9） |
| 协议 | 与 E9 逐字一致：ResNet-18+GN、Split-CIFAR-100 task-IL、SGD(0.05,0.9,5e-4)、5 ep、BS=128、λ=0.1、G=0.5749、kWTA k=128、masked softmax、clip 5.0 |
| 唯一干预 | `r2_recency.recency_weights` 逐字复用：`w_j=(1−α)+α·(v·e^{−g_j/τ}/Σe^{−g_k/τ})`，α=1.0、τ=T/4=2.5、总权重守恒（`Σw=v` 恒等式） |
| 代码复用 | `import e9_resnet` + `from r2_recency import recency_weights`（AST 验证无本地重定义；两模块本地/服务器 MD5 一致） |
| 数据校验 | 本地/服务器 MD5 全部一致（§10） |

## 2. 为什么要做（一段话版）

R2（MLP）证明「在 renorm 基座上把锚点预算向近期任务重分配」能同时保 gain 改 retention（T20 `Δℓ +8.08`、`Δbwt vs renorm +1.52`，Holm p=6.42e−05，anti 对照 20/20 全负）。但 E9 证明 renorm 前向在 ResNet-18+GN 上本身是最差臂（`−8.83`，p=0.001）。于是产生一个两者单独都无法回答的问题：**不动前向、只重分配锚预算（`base_rec10`），在 λ 已按架构标定（0.1）的真实骨干上还剩多少效应？** 预注册 §5 写死三层判读 T1/T2/T3，无事后择优。

## 3. §7 硬门槛结果（两轮均 6/6 PASS）

| 门槛 | 结果 | 细节 |
|---|---|---|
| `G1_weight_conservation` | **PASS** | bad_sum=0, bad_nonpositive=0, bad_length=0 |
| `G2_alpha_dispatch` | **PASS** | bad_alpha=0, bad_weight_direction=0 |
| `G3_reuse_completeness` | **PASS** | n_new=60, n_unique=60, expected_new=60 |
| `G4_runtime_constants` | **PASS** | bad_lam=0, bad_const=0 |
| `G5_completeness` | **PASS** | n_bad=0 |
| `G6_gate_band` | **PASS** | n_bad=0, window=[0.50,0.65], sd<0.05 |

**全部门槛**：PASS。门槛均为数学恒等式与运行时断言——D 探针（预注册 §7 总则）已证 conv 骨干逐位复现不可能，故无任何 bit-identity 条款。

## 4. 主比较与三层判读（n=10 → n=20 双轮并列）

| # | 臂 | 对照 | 问题 | Δell n10 → n20 | Δbwt n10 → n20 | Holm(ell/bwt) |
|---|---|---|---|---|---|---|
| C1 | `base_rec10` | `base` | 只改锚预算方向是否改善（主） | +1.94 (p=0.0103) → +0.72 (p=0.335) | -0.54 (p=0.057) → -0.47 (p=0.0325) | 0.0411 / 0.118 → 1 / 0.13 |
| C2 | `renorm_rec10` | `renorm` | R2 主候选在 ResNet 的边际（主） | -3.06 (p=0.0393) → -0.81 (p=0.432) | +0.17 (p=0.662) → +0.10 (p=0.666) | 0.118 / 0.662 → 1 / 1 |
| C3 | `base_anti10` | `base_rec10` | 方向特异性（探索） | -2.09 (p=0.0218) → -0.81 (p=0.209) | +0.43 (p=0.125) → +0.60 (p=0.00799) | — |
| C4 | `base_rec10` | `renorm` | 预算方向 vs 前向干预（探索） | +10.77 (p=0.000177) → +10.06 (p=6.87e-09) | -1.41 (p=0.00292) → -1.27 (p=1.97e-05) | — |

**三层判读（预注册 §5 数据前写死）**：**T3「fail」**（n=20（扩样后，终判））：锚预算方向在 λ 已标定的真实骨干上不是有效自由度。

**扩样翻转披露（修订说明 §5 承诺）**：首轮 n=10 曾判 T1（C1.ell +1.94，Holm 0.0411），n=20 后 C1.ell 缩水至 +0.72（p=0.335，方向 11/9）→ 终判 **T3**。n=10 的显著是幸运尾部，这正是预注册 §3.2 扩样本条款存在的理由（E8 前车之鉴）；两轮结果并列呈现，不隐藏首轮。

> Holm 家族 = {C1, C2} × {ell, bwt} 共 4 个配对 t；C3/C4 为探索性比较，报 p 不进家族。带 `*` 的 Holm 为扩样后仍显著。

## 5. 逐 seed 方向一致性

| 比较 | 指标 | n=10 | n=20 | 方向全同 |
|---|---|---|---|---|
| C1 | ell | +9/−1 | +11/−9 | **否（未达严格形式）** |
| C1 | bwt | +2/−8 | +6/−14 | **否（未达严格形式）** |
| C2 | ell | +2/−8 | +8/−12 | **否（未达严格形式）** |
| C2 | bwt | +5/−5 | +10/−10 | 是 |
| C3 | ell | +3/−7 | +9/−11 | **否（未达严格形式）** |
| C3 | bwt | +6/−4 | +15/−5 | **否（未达严格形式）** |
| C4 | ell | +10/−0 | +20/−0 | **否（未达严格形式）** |
| C4 | bwt | +0/−10 | +0/−19 | **否（未达严格形式）** |

> 与 p 值并列的纯描述量；混合符号如实记「未达严格形式」，不降级判通过。唯一方向全同的是 C4（ell 20/0、bwt 0/19+1零）——但 C4 的主体是 renorm 前向本身的差距（E9 −8.83）。

## 6. 水平表（均值）

| 臂 | 来源 | ell | avg | bwt | ell(n20) | avg(n20) | bwt(n20) |
|---|---|---|---|---|---|---|---|
| `base` | E9 复用 | 27.28 | 24.85 | -2.70 | 27.26 | 24.90 | -2.62 |
| `renorm` | E9 复用 | 18.46 | 16.80 | -1.84 | 17.91 | 16.27 | -1.82 |
| `base_rec10` | R2T 新跑 | 29.23 | 26.31 | -3.24 | 27.97 | 25.19 | -3.10 |
| `renorm_rec10` | R2T 新跑 | 15.39 | 13.89 | -1.67 | 17.10 | 15.54 | -1.73 |
| `base_anti10` | R2T 新跑 | 27.14 | 24.61 | -2.81 | 27.17 | 24.92 | -2.50 |

## 7. 探索性：方向对照与「预算方向 vs 前向干预」

- **C3 方向对照**（n=20）：`base_anti10` 的 `Δell = -0.81`（p=0.209，方向 9/11）、`Δbwt = +0.60`（p=0.00799，方向 15/5）。anti 在 ell 上不比 rec10 好、bwt 略好但符号混合 → **方向特异性在本骨干上未获支持**：与 R2 MLP 上的 20/20 强分离不同，预算方向在 ResNet 上近似对称。只保留现象结论，不写机制主张。
- **C4 预算 vs 前向**（n=20）：`base_rec10 − renorm` 的 `Δell = +10.06`（p=6.87e-09，方向 20/0 全同）。这个巨大差距的**主体是 renorm 前向本身有害**（E9：renorm−base −8.83），不是 base_rec10 的正贡献（扩样后 C1 ≈ 0）。解读为「在 ResNet 上，任何叠加 renorm 前向的方案都劣于不动前向」，与 E9 结论一致。
- 以上均为探索性：不进 Holm 家族、不改变三层判读。

## 7b. 条件触发判定（严格按预注册 §3 的三条）

| 预注册 §3 条款 | 判定 |
|---|---|
| 1 剂量轴（α∈{0.25,0.5}） | 触发条件为「非 T1 且 CI 跨零」；首轮即 T1 且扩样后翻为 T3，但扩样条款（2）先触发并已覆盖剂量不明的问题 → 不触发 |
| 2 n 扩到 20 | **触发并已执行**（首轮主比较 raw p 落带：0.0103, 0.0393, 0.057；扩样 30+40 runs 完成，终判以 n=20 为准，翻转已披露） |
| 3 toyCNN/GN-MLP 迁移 | 不属于本预注册，另议 |

诊断：首轮主比较 raw p = 0.0103, 0.057, 0.0393, 0.662。扩样后 raw p = 0.335, 0.0325, 0.432, 0.666。严格按写死条款判定，不新增事后触发条件。

## 8. 与 E9 / R2 的合并解读（按预注册 §5 红线）

**T3（扩样终判）**：R2 的「保护对象」结论**不迁移**到 ResNet-18+GN。与 E9 合并为架构依赖性证据链：小饱和 MLP 上预算方向是有效自由度（R2：T20 Δℓ +8.08、Δbwt vs renorm +1.52、anti 20/20 分离），λ 已按架构标定（0.1）的真实骨干上则不是（n=20：C1.ell +0.72 p=0.335 方向 11/9；C2.ell −0.81 p=0.432）。新论文叙事按素材账本 §4 收窄：MLP-only 机制发现 + 跨架构边界 + E9/E8/R1 的否证群。三条红线（不写通用改进、不改写 E9、不抹杀 R2）全部适用；**首轮 n=10 的 T1 与扩样翻转过程必须原样写进论文**——它本身是「小样本边缘显著不可信」的教科书案例，与 E8 的 +0.54 翻转互证。

## 9. 边界与诚实披露

- **不外推**：结论适用范围仅「本协议 ResNet-18+GN / Split-CIFAR-100 / task-IL / λ=0.1 / T=10 / n=20」；不写「通用持续学习改进」。
- **不改写 E9**：renorm 前向有害、门是承重墙仍是 E9 事实；T4 的 C4 大差距主体是 renorm 前向本身，不是 base_rec10 的正贡献。
- **不反向抹杀 R2**：MLP 上 H1 成立与 ResNet 上 T3 并不矛盾，并列陈述架构依赖性。
- **非确定性披露**：主比较含跨运行非确定性（D 探针同卡同 seed Δell 最大 6.83）；配对 t 为准，未因此追加或放宽任何门槛。
- **扩样翻转披露**：首轮 n=10 曾判 T1，n=20 终判 T3，两轮并列呈现（修订说明 §5 承诺），翻转过程本身写进新论文方法论。
- 任何偏离本预注册的操作写入独立 `R2T_修订说明.md`；预注册一字不改；原始 JSON/脚本/日志随包发布。

## 10. 复现命令

```bash
# pilot（已跑，留档 r2t_pilot.json）
CUDA_VISIBLE_DEVICES=2 <venv-gpu>/python r2t_resnet.py --task-id 0 --n-shards 1 \
    --arms base_rec10 --seeds 0 --out /tmp/r2t_pilot.json
# 首轮全网格（30 runs）
bash launch_r2t.sh 2 0 ; bash launch_r2t.sh 3 1
# 扩样（触发条款 2：新臂 30 + 对照 40）
bash launch_r2t_ext.sh 2 0 ; bash launch_r2t_ext.sh 3 1
# 门槛 + 判读
python bench/analyze_r2t.py --files "bench/results_r2t/r2t_grid.part*.json" \
    --e9 "bench/results_e9/e9_grid_s*.json" \
    --out bench/results_r2t/r2t_stats.json
python bench/analyze_r2t.py --files "bench/results_r2t/r2t_grid.part*.json" \
    --e9 "bench/results_e9/e9_grid_s*.json" \
    --ext "bench/results_r2t/r2t_ext.part*.json" \
    --e9-ext "bench/results_r2t/r2t_ctrl_ext.part*.json" \
    --out bench/results_r2t/r2t_stats_ext.json
# 报告填充（双轮对照）
python bench/summarize_r2t.py --stats bench/results_r2t/r2t_stats.json \
    --stats-ext bench/results_r2t/r2t_stats_ext.json \
    --report bench/results_r2t/R2T_收数报告.md --inplace
```

| 文件 | 留档位置 |
|---|---|
| `R2T_修订说明.md` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `R2T_收数报告.md` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `R2T_收数报告.模板.md` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `R2T_预注册.md` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_ctrl_ext.part0.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_ctrl_ext.part1.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_ext.part0.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_ext.part1.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu2.done` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu2.log` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu2_ext.done` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu2_ext.log` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu3.done` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu3.log` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu3_ext.done` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_gpu3_ext.log` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_grid.part0.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_grid.part1.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_pilot.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_stats.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |
| `r2t_stats_ext.json` | 本地 + `10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/results_r2t/` |

统计产物：`r2t_stats.json`（首轮）+ `r2t_stats_ext.json`（扩样终判）；共享对照存放于 `bench/results_e9/`（E9 原始包，未改动）。

## 11. 素材账本更新

素材账本 §4「ResNet 迁移检验」打勾，**结论 T3（不迁移，扩样终判）**——检验完成、结论为负，照常成文（负结果不隐藏）。R3 conv λ 重标定/披露的优先级上升（需排除「λ 标定方式掩盖预算效应」的替代解释，尽管 R2T 已用 S2 校准值 0.1）；剩余：复现包、口径红线复核。新论文叙事回到「三重否证 + 架构依赖性边界」，R2 的 MLP 阳性结果作为受限发现如实呈现；**n=10→20 的翻转过程写进方法论**（与 E8 互证小样本边缘显著的危险性）。
