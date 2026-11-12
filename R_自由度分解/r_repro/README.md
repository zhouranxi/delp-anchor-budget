# R 系列复现说明（R1 训练期 context noise / R2 保护对象 / R2T ResNet 迁移 / R3 conv λ 重标定）

R1–R3 是 E 系列（S1–S5、E1–E9，见仓库根 `README.md` 与 `S123_合并稿/paper_tmlr/`）
封口后的第二批实验，全部遵循同一纪律：**预注册先于数据落盘 → 收数报告 →
偏离写独立修订说明（预注册一字不改、判据不放宽）→ 原始 JSON/脚本/日志随包**。
本目录给出一键校验入口与数据地图。结论性叙事与成文门槛见
`bench/results_r2/新论文素材账本.md`。

## 0. 一键校验（无 GPU，本机即可）

从仓库根目录：

```bash
python r_repro/verify_package.py
```

它**从原始 JSON 独立重算**（不复用各 analyze_* 的中间产物）四个实验的头条数字，
并与收数报告中的数值逐项比对：

| 实验 | 复算内容 |
|---|---|
| R1 | v2 权威网格 280 runs：24 个 Δavg cell 的 Holm 最小 p；H1（ρ=0.2 的 `fwd−full`）四 cell；存活 B 四条 Δℓ>0 与保留率；v1/v2 等价性（214 个未改动 run 逐位相同） |
| R2 | 主网格 240 runs：`renorm_rec10` 四条 E7 阈值与 2 比较 Holm（应 6.42e−05）；α=0 哨兵 40 配对逐位相同；anti 20/20 方向；权重守恒 |
| R2T | 首轮 30 + 扩样 30 + 对照扩样 40 + E9 复用 60：C1/C2 两轮 Δell/Δbwt 与 n=20 终判 T3；首轮 T1 → 扩样翻转的数值链条 |
| R3 | 240 runs 的 gain(λ) 六点表（含 λ=1000 塌缩点）与 M1–M4 判读输入 |

退出码 0 当且仅当全部 PASS。文件→数据映射见 `r_repro/manifest.txt`。

## 1. 各实验一句话

| # | 问题 | 结论 | 权威数据 |
|---|---|---|---|
| R1 | 训练期任务上下文错标（前向/梯度掩码/usage+Fisher 三处）是否伤害精度 | **不伤害**（24 cell Holm 最小 p=0.585）；「context 通路训练期也不承重」，不得写成「对标签噪声鲁棒」 | `results_r1/r1_grid_v2.part*.json`（280 runs，GPU 2/3） |
| R2 | 总权重守恒地把锚点预算向近期任务重分配能否移动 gain–retention 前沿（MLP） | **能**（`renorm_rec10` 四条 E7 阈值全中，Holm 6.42e−05；anti 20/20）；限「renorm 基座上」 | `results_r2/r2_grid.part*.json`（240 runs）+ α=0 哨兵 80 + anti 40 + 同设备参照 80 |
| R2T | R2 的「保护对象」结论是否迁移到 ResNet-18+GN（λ=0.1 已标定） | **不迁移**（n=20 终判 T3：C1.ell +0.72 p=0.335 方向 11/9）；首轮 n=10 的 T1 被扩样翻转，两轮并列披露 | `results_r2t/r2t_grid+ext+ctrl_ext`（100 runs）+ E9 复用 60 |
| R3 | E5「conv 门效应消失」是否 λ=1000 错标定伪象 | **中间态**：gain(λ) 在 λ≤316 显著正（+0.8~+1.04，Holm 4/6）但量级为 MLP 的 19%；λ=1000 恰为塌缩点（+0.185 ns） | `results_r3/r3_grid.part*.json`（240 runs） |

## 2. 数据地图

数据都在 `bench/results_r{1,2,2t,3}/`；统计产物 `*_stats*.json` 可由
`bench/analyze_r*.py` 从原始 JSON 重生成；报告 `*_收数报告.md` 由
`bench/summarize_r*.py` 从 stats 重填充（`.模板.md` 为带占位符骨架，可重生成）。
逐文件映射与 MD5 见 `r_repro/manifest.txt`。

## 3. 训练重跑（需 GPU 服务器，非本机）

服务器：`xinguan@10.10.52.210:/mnt/mydisk/xinguan-dendritic/bench/`（venv-gpu，
torch 2.14.1+cu130）。共享卡 GPU 2/3（llama-server 常驻 ~19–21 GB、利用率 0%），
单进程显存增量实测 0.5–1.5 GB；**GPU 0/1 为他人任务，勿碰**。

| 实验 | 训练入口 | 网格 | 备注 |
|---|---|---|---|
| R1 | `r1_ctxnoise_train.py --shard/--n-shards/--seeds/--rhos/--biases` | 280 runs（v2，noise_rule=guaranteed） | 三版脚本留档（`.v1/.v2`）；标签规则校验 `verify_r1_labels.py` |
| R2 | `r2_recency.py --arms/--ntasks/--seeds` | 主 240 + 哨兵 80 + anti 40 | 权重函数被 R2T 逐字 import，改动会同时影响两者 |
| R2T | `r2t_resnet.py`（首轮）+ `launch_r2t_ext.sh`（扩样，含 E9 原脚本补对照） | 30 + 30 + 40 + E9 复用 60 | conv 骨干 cuDNN 非确定性（同 seed 两次 Δell 可差 6.8）——无逐位门槛 |
| R3 | `r3_lamsweep.py` | 240（λ∈{0…3160}） | `r3_lamsweep.py` 内含 `make_anchor_cnn` 设备补丁（E5 文件零改动，见 `R3_修订说明.md`） |

启动器：`bench/launch_r{1,1_v2,1_b3,2,2_ref,2_alpha0,2_anti,2t,2t_ext,3}.sh`。
所有 run 增量落盘；`.done` 文件记录 rc 与时间戳。

## 4. 运行纪律（六次门槛处置，全部留档）

| # | 事件 | 性质 | 处置文档 |
|---|---|---|---|
| 1 | R1 §7.3 G3 | 预注册数学冲突（`(1−ρ)^T`） | `results_r1/R1_修订说明.md` |
| 2 | R2 §7 G4 | 历史参照设备口径（CPU vs GPU） | `results_r2/R2_修订说明.md` |
| 3 | R2 §7 G2 | 门槛证据来源失效（pilot 作废连带） | `results_r2/R2_修订说明_2.md` |
| 4 | R2T §7 G5 | 检查器字段口径（`avg` vs `final_avg`） | `results_r2t/R2T_修订说明.md` §1–4 |
| — | R2T §3.2 | 条件触发执行（n 10→20，结论翻转） | `results_r2t/R2T_修订说明.md` §5 |
| 5 | R3 pilot | 上游潜伏设备 bug（E5 conv 仅 CPU 验证过） | `results_r3/R3_修订说明.md` |

通用规矩（写论文方法论可直接引用）：①随机抽样式门槛先算 `(1−ρ)^T`；
②跨数据对齐先确认设备/精度口径；③作废数据时检查其承载的其它门槛证据；
④跨协议复用 JSON 对齐被复用方 schema；⑤conv 骨干不写逐位门槛；
⑥复用到新设备时 pilot 必须覆盖全部依赖分支。

## 5. 与 Zenodo 的关系

本包是仓库内复现入口（对齐 `s1_repro/` 惯例）。**Zenodo 新版本另行决定**
（素材账本 §4：不阻塞成文）；若发布，按 `bench/package_zenodo_v8.py` 的
PLAN 式脚本另行打包，本目录整体纳入。
