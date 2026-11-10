# Knowledge-Based Systems 投稿 — 提交前逐项核对清单

投稿系统：KBS 在 Editorial Manager（https://www.editorialmanager.com/kbs/ 或 ATS 转投链接直达）
本地文件目录：`S2_resubmit/upload/`
状态：**待用户在 ATS 点击 Transfer 或新建投稿**

> 前史：PR 稿 PR-D-26-16069 于 2026-11-10 编辑直接拒稿（未送外审，Area Editor: Cheng-Lin Liu）。
> 理由（逐字）："EWC has been shown low performance in comparison with later CL methods, and this
> paper does [not] compare with recent CL methods on various datasets. Such study does not provide
> valuable reference to the community."（[not] 为 HTML 渲染丢字，按语义补回）
> 新版已把该负面结果前置为头号贡献（Q4），并新增软硬保护律子节 + knowledge-based 框架。

---

## 一、Attach Files —— 每个文件的期望值

| 顺序 | Item Type | 文件 | 备注 |
|---|---|---|---|
| 1 | `Manuscript` | `main.pdf` | **33 页新版**（旧版 35 页）。判断方法：标题含 "Soft–Hard Protection Law" |
| 2 | `Supplementary material` | `supplementary.pdf` | 3 页：附录 A（统计理由+超参数）+ 附录 B（完整逐任务精度矩阵） |
| 3 | `Highlights` | `highlights.docx` | 5 条，每条 ≤85 字符（已换新 5 条） |
| 4 | `Declaration of competing interests` | `declarations.docx` | 沿用，无需改 |
| 5 | `Cover Letter` | `cover_letter.docx` | 新版：逐字引用 PR 编辑意见 + 5 条逐项回应 |
| 6–10 | `Figure` | `Figure_1.pdf` … `Figure_5.pdf` | 5 张分别传，均为纯矢量 |
| 11 | `Graphical Abstract` | `graphical_abstract.png` | 可选，建议传 |

---

## 二、关键字段速查（KBS）

| 步骤 | 答案 |
|---|---|
| Article Type | `Full Length Article` |
| Title（原文粘贴） | `The Anchor Accumulation Problem: Bounded Anchor Sets, a Soft-Hard Protection Law, and the Auditable Cost of Regularized Continual Learning` |
| Keywords | `continual learning, elastic weight consolidation, catastrophic forgetting, knowledge-based protection, stability-plasticity dilemma` |
| 行距 | `Double-spaced` |
| 引用格式 | `Numbered` |
| Highlights 数量 | `5` |
| 参考文献数 | `30`（旧版 35 → 附录移出后主文引用 30 条） |
| 正文字数（不含 refs） | ~5833 词 |
| 页数 | 33 页（双倍行距）+ 3 页 supplementary |
| 一稿多投 | `Yes` + 说明：PR 已于 2026-11-10 拒稿（PR-D-26-16069，编辑直拒未送审）；Neurocomputing 已于 2026-10-26 拒稿；Zenodo 预印本不算 prior publication |
| 前驱工作关系 | `Yes`（正文 Related work 的 scope 段保留） |
| GenAI 使用 | `Yes` + 披露保留 |
| Data availability | 沿用原文（GitHub + Zenodo DOI） |
| Suggested Reviewers | 跳过 |

## 三、摘要纯文本版（EM 文本框粘贴用）

```
Regularized continual-learning schemes that store one protection state per task pay a cost that grows with sequence length. In usage-localized elastic weight consolidation (DeLP), per-neuron elastic anchors persist at full weight as an elastic drag on later learning; a retention decomposition a_T = l + (T-1)/T BWT localizes the cost in the learning term l. We call this the anchor accumulation problem and study its repair: bounding the anchor stack at M entries. A FIFO budget recovers +4.5 and +9.0 points of final accuracy over unbounded DeLP at T = 10 and T = 20 (M = 5; n = 10 paired seeds; Wilcoxon p = 0.002), and the budget is a distinct knob from the elastic weight lambda: it controls the age structure of forgetting, lambda its amount. Eviction design is ruled by one constraint: any criterion keyed to current constraint mass grandfather-protects stale anchors and evicts new ones at birth (-34.5 BWT), whereas age-first criteria are immune. Sweeping online EWC's discount shows that discounting is itself a soft bound, and soft and hard bounds obey one log-linear law between the learning term and the protected-mass horizon (l = 84 - 11 ln h, R^2 = 0.99): bounding accumulated anchor mass, not eviction, is the active ingredient, and the explicit window retains what a scalar discount cannot - a discrete, named, policy-selectable protection set readable from the stack trace. We report the boundary honestly: on Split-CIFAR-100 under task-incremental evaluation, a no-protection baseline stays above every protection arm, so the budget result is internal to the protected family; the paper's contribution is a knowledge-based diagnosis and audit framework - when to protect, how to bound, which anchors survive - not a drop-in accuracy win.
```

## 四、最后的确认动作

1. 走完所有步骤 → `Build PDF for Approval` → 下载逐页检查：
   - 无 `??`
   - 标题/摘要为新版
   - 32 页正文
2. 勾选同意条款 → 提交
3. 提交后：记稿件编号、查确认邮件、勿重复提交

---

*本地文件：`S2_resubmit/upload/`　|　归档：Zenodo concept DOI `10.5281/zenodo.23668522`　|　仓库：github.com/zhouranxi/delp-anchor-budget*
