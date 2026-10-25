# S1 复现说明（Branch-DeLP / 剂量响应审计）

S1（《Segment Specialization in Active-Dendrite Continual Learning: An Audit,
an Artifact, and a Causal Repair》）的全部论文产物都从**已归档的原始记录**
`bench/results/` 可推导。本目录给出一键复现入口与数据地图。

配套 S2 的复现入口见仓库根 `README.md`（`make_figs.py` / `make_fig5_cifar.py` /
`analyze_s2r6.py`，数据路径已做 `bench/` ↔ `bench/results/` 双布局回退）。

## 1. 分析 + 出图（无 GPU，本机即可）

从仓库根目录：

```bash
python s1_repro/run_s1_repro.py
```

它会依次：
1. `bench/analyze_s1.py` → `bench/results/s1_paired_stats.json`
   （branch / neuron / plain 三臂配对 t 检验，对应正文 Table「granularity ladder」
   与单锚参考 SI/MAS）；
2. `paper_s1/make_figs.py` → `paper_s1/figs/fig_{evolution,deadfrac,artifact,dose}.pdf`。

退出码 0 当且仅当两步都成功且产物存在。数据→脚本映射见 `s1_repro/manifest.txt`。

## 2. 数据地图（bench/results/ 里 S1 系文件）

| 用途 | 文件 | 被谁读 |
|---|---|---|
| 粒度阶梯 + 单锚参考 | `s1_mnist.json` `s1_fashion.json` `s1_refs.json` `s1_l1000_t10.json` `s1_l1000_t20.json` | `analyze_s1.py` |
| 演化 / 死神经元 / 假象 / 剂量图 | `gevo_mnist_t10` `gevo_mnist_t20` `gevo_fashion_t10` `untrained_probe.json` `c2_*` `diag_b*` | `paper_s1/make_figs.py` |
| 剂量-因果配对（正文 Table dose10/dose20） | `c2_mnist_b{0,1,3,10}.*` `c2_fashion_b{0,1,3,10}.*` | `paper_s1/make_figs.py` + 正文 |
| 签号漂移（§5.4 signed-drift） | `signeddrift.json` | 正文 §5.4（数据已归档，脚本 `bench/signed_drift.py` 重跑需 GPU） |
| learned-bias（Limitations #2） | `q8learned*_*.json` `q8learned2_*.json` | 正文 Limitations #2 |

## 3. 训练重跑（需 GPU 服务器，非本机）

`bench/branch_delp.py`（剂量-因果 c2_*）、`bench/signed_drift.py`（signeddrift.json）、
`bench/gating_evolution.py`（gevo_*）是重跑入口，都依赖共享基座
`bench/exp6_baselines.py` + `bench/exp4_branch_protection.py`（已从 companion
`delp-audit` vendor 进本仓库，import 已验证通过）。本机无 CUDA 时可 `--smoke`
冒烟验证，但正式 10-seed 结果需 GPU 盒子 + 4 卡分工（见各脚本 docstring）。
仓库里提交的是**已完成的原始记录**，重跑只用于抽检一致性，不是复现必需。

## 4. 已修的两个复现障碍（本次会话）

- `bench/analyze_s1.py` 原来写死 `BASE = D:/WorkBuddy/2026-10-17.../bench/results`，
  已改成相对解析（`bench/results` → CWD `results` → 原路径兜底）。
- `bench/exp6_baselines.py` / `exp4_branch_protection.py` 原在 companion 仓库、本仓库缺失，
  已 vendor 进 `bench/` 顶层，S1 与 S2 的训练脚本 flat import 均命中。
