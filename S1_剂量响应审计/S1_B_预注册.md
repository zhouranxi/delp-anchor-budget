# S1 选项 B 预注册：粒度 × 预算轴（branch-fifo vs neuron-fifo）

**写于 2026-10-24 20:10，先于任何 bfifo 数据落盘。** 沿用 S2 教训：判据先行。

## B 的主张（若存活）

在相同的锚预算 M 下，branch 粒度锚点（段归属漂移参考）比 neuron 粒度锚点
（单一参考）买到更多的学习侧精度：即细粒度降低了锚预算需求。

机制依据（P1 之前的假设）：neuron 锚的罚项对后续任务在**全部**共享权重上的
漂移都开火（包括后续任务自有段的合法漂移——spurious fight）；branch 锚通过
归因参考只追究自己段解释不了的漂移 → 同样的 M 之下，每个锚的"有效保护"
应该更干净。

## 实验设置（固定，不再改）

- λ=1000（S2 校准点）、fifo、M∈{2,5}、T∈{10,20}、n=10 seeds（0–9）
- 数据集：MNIST + Fashion-MNIST；granularities ∈ {neuron, branch}
- 配对分析：同 seed 同置换下 branch-fifo − neuron-fifo，配对 t 检验
- 共 160 jobs（mnist 80 / fashion 80），本地双等价 smoke 已 PASS
  （bfifo M≥T+1 不驱逐 ⇒ 与 gran 臂逐位一致；M=2 驱逐路径触发验证 evicted=[0,1]）

## 预注册判据

**B 存活**：主对比 branch vs neuron 的 Δℓ 在 **≥4/6 组合**（2 数据集 × 2 M × 2 T）
中显著为正（|t|>2 且方向 +），**且** BWT 无任何组合显著变差（|t|>2 且方向 −）。

**B 判死**：≥4/6 组合 |Δℓ| 不显著或为负，或任一组合 BWT 显著变差。
→ 转选项 C（门控弥散升格）。

**灰色地带**（2–3 组合显著）：只报事实不扩展主张，由用户裁决。

## 风险预告（诚实记录）

P1 诊断已显示 branch ≈ neuron（unbounded，归因自限：refs 追踪的漂移是罚项
自己产生的）。fifo 只截断锚栈、不改变罚项结构，**先验偏向 B 失败**。
本实验是给"细粒度省预算"假设一次干净的证伪机会，而非续命。

## 数据与脚本

- 脚本：`bench/branch_delp.py`（commit 820d8e3，`--budgets` + `--skip-gran`）
- 结果：`bench/results/s1b_{mnist,fashion}_{t10,t20}.json`（计划）
- 分析：同 P1 配对口径，另加 per-(M,T) 剂量对比 vs S2 neuron-fifo mgrid（context）
