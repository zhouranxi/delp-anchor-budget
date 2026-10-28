# ⚠️ 本目录部分文件已废弃（2026-10-27 数据一致性审计）

## 已废弃的文件（结论被推翻，勿引用）

| 文件 | 废弃原因 |
|---|---|
| `s4ctx_hard.part0/1` | λ=1000 从未生效（惩罚项漏乘 λ，实际有效 λ≈1），记录却写 lam=1000 |
| `s4ctx_soft.json` | 同上 |
| `s4nodend.json` | 同上（v1 无效版：另有多余 sigmoid 激活错误） |
| `s4nodend2.json` | 同上 + 无门臂硬编码 nh=2000，与树突臂（实际 256）宽差 7.8 倍 |
| `s4diag_mnist0.log` / `s4diag_mnist1.log` / `s4diag_fashion0.log` | λ 未生效 |

**这些文件的直接后果**：第一版论文的核心主张「MNIST 上 no-gate 对照
匹配修复臂（73.22 vs 72.63, n.s.）→ repair 通过删除门控起效」
**是宽度错配造成的假象**。修正后（宽度对齐 256、λ 真实生效），
no-gate 对照为 63.63 vs 修复 68.67（t=−13.6），两臂**可区分**，
bypass 结论被拒。

## 替代文件（修正版，论文所用）

| 文件 | 说明 |
|---|---|
| `s4ctx_hard_fixed.part0/1` | λ=1000 已应用，nh=256，40 runs |
| `s4ctx_soft_fixed.json` | 同上，20 runs |
| `s4nodend_fixed.json` | 无门对照，宽度对齐 256，λ 应用，20 runs |
| `s4diag_fixed_*.log` | 机制探针重跑（3 模型 × 2 bias） |
| `s4_constgate.json` | constant-gate 对照（h = kwta(t·0.565)），80 runs |

所有修正版记录带 `lam` 与 `nh` 字段，可机器校验。

## 为什么保留废弃文件

这是「数据一致性审计如何推翻一个已写好的结论」的完整证据链。
`prereg/S4_ctxnoise_预注册_修订说明_2026-10-27.md` 逐条记录了发现过程。
若你只想要正确数据，请直接使用 `*_fixed` 与 `s4_constgate` 文件。
