# Information Sciences 投稿 — 提交前逐项核对清单

**投稿入口**：https://submit.elsevier.com/INS （Elsevier 托管投稿系统，进入后由 Editorial Manager 跟踪）
**期刊官网**：https://www.sciencedirect.com/journal/information-sciences
**本地文件目录**：`S2_resubmit/upload/`
**状态**：材料已备齐，待用户提交

> 前史：PR 稿 PR-D-26-16069 于 2026-11-10 编辑直接拒稿（Area Editor: Cheng-Lin Liu）；
> Neurocomputing 稿 NEUCOM-D-26-24422 于 2026-10-26 拒稿。两稿均已正式关闭，非在审。
> 新版已把负面结果前置为头号贡献（Q4），并新增软硬保护律子节 + knowledge-centric 框架。

---

## 一、Attach Files —— 每个文件的期望值

| 顺序 | Item Type | 文件 | 规格核验 |
|---|---|---|---|
| 1 | `Manuscript` | `main.pdf` | **31 页双倍行距**（guideline ≤40 ✅）。标题含 "Soft–Hard Protection Law"；期刊字段已改为 Information Sciences |
| 2 | `Supplementary material` | `supplementary.pdf` | 5 页：附录 A（统计理由+超参数）+ 附录 B（完整逐任务精度矩阵）+ 附录 C（三张完整数值网格表） |
| 3 | `Highlights` | `highlights.docx` | 5 条，每条 ≤85 字符（含空格），文件名含 "highlights" ✅（官方为"鼓励提供"，非强制） |
| 4 | `Declaration of competing interest` | `declarations.docx` | **强制**；建议同时走 https://declarations.elsevier.com/ 生成 |
| 5 | `Vitae`（作者简历） | `vitae.docx` | **InfoSci 特有要求**。78 词（上限 100），可编辑 Word（非 PDF）✅ |
| 6 | `Figure` | `Figure_1.pdf` … `Figure_5.pdf` | 5 张分别传，均为纯矢量 PDF |
| 7 | `Figure` | **作者证件照** | **InfoSci 特有要求**：需你自行提供，传为独立 figure 文件 |
| 8 | `Cover Letter` | `cover_letter.docx` | InfoSci 版：逐字引用 PR 编辑意见 + 5 条逐项回应 |
| 9 | `Graphical Abstract` | `graphical_abstract.png` | 官方指南**完全未提及**图文摘要——可传可不传 |

**图表总数**：5 图 + 3 表 = **8**，正好等于官方 guideline 的 "8 figures + tables" 上限 ✅

**备用文本版**（EM 文本框直接粘贴用）：`cover_letter.txt`、`metadata_for_em.txt`

---

## 二、关键字段速查（Information Sciences）

| 步骤 / 字段 | 答案 | 依据 |
|---|---|---|
| Article Type | **Original research work** | 指南只列了这一个类型名 |
| Title（原文粘贴） | `The Anchor Accumulation Problem: Bounded Anchor Sets, a Soft–Hard Protection Law, and the Auditable Cost of Regularized Continual Learning` | 见 `metadata_for_em.txt` |
| Abstract | **197 词** ✅ | 官方两处冲突（250 / 200），取严按 200 做，本稿 197 词同时满足两者 |
| Conclusions section | 必须有 → 正文第 6 节 Conclusion ✅ | 官方明确要求 |
| Keywords | **6 个**（1–7）✅ | continual learning / elastic weight consolidation / catastrophic forgetting / knowledge management / protection-state curation / stability–plasticity dilemma |
| 评审模式 | **Single anonymized（单盲）** | 作者名保留在标题页，**无需匿名版**；也**不要求**单独拆标题页文件 |
| 引用格式 | 投稿时不限，录用后由期刊套格式 | 现有编号格式已一致，不需改 |
| 篇幅 | experimental ≤40 页 + ≤8 figures/tables（guideline） | 31 页 + 8 个图表，**均达标** |
| Highlights | 鼓励提供，3–5 条 × ≤85 字符 | 独立可编辑文件，文件名含 "highlights" |
| Graphical Abstract | 官方指南**未提及** | 传不传都不影响 |
| Artwork | 矢量图 EPS/PDF；位图 ≥300 dpi | 现有 Figure_1–5.pdf 均为矢量 |
| CRediT | **强制** | 已在 main.tex 末尾 + declarations.docx |
| Data availability | **Option C 强制**：deposit + 提交时给链接 | GitHub + Zenodo DOI，链接照填 |
| Competing interests | **强制**，.doc/.docx 上传 | 即使选 "I have nothing to declare" 也要生成文件 |
| Funding | 无 | 直接填 none |
| Vitae + 证件照 | **InfoSci 特有**：每作者 100 词内简历 + 证件照作独立 figure | vitae.docx 已备；**证件照需你提供** |
| GenAI 披露 | 保留 | main.tex 末尾已有专节 |
| 一稿多投 | `No`（当前无在审稿件） | 说明：PR 与 Neurocomputing 均已拒稿关闭；Zenodo 预印本不算 prior publication |
| Suggested Reviewers | 可留空 | |
| **费用** | **hybrid：订阅路线 0 费用** | OA 路线 $3,040，**不要选 OA** |

---

## 三、摘要纯文本版（197 词，EM 文本框直接粘贴）

```
Regularized continual learning stores one protection state per task, whose cost grows with sequence length. In usage-localized elastic weight consolidation (DeLP), full-weight anchors drag later learning; a retention decomposition a_T = l + (T-1)/T BWT localizes the cost in the learning term l. We call this the anchor accumulation problem, repairing it by bounding the stack at M entries. A FIFO budget recovers +4.5/+9.0 accuracy points over unbounded DeLP at T = 10/20 (M = 5; n = 10 paired seeds; Wilcoxon p = 0.002). The budget is distinct from the elastic weight lambda: it controls the age structure of forgetting, lambda its amount. Eviction design obeys one constraint: criteria keyed to constraint mass grandfather-protect stale anchors and evict new ones at birth (-34.5 BWT); age-first criteria are immune. Online EWC's Fisher discount is a soft bound, and both obey one log-linear law in protected-mass horizon (l = 84 - 11ln h, R^2 = 0.99): bounding anchor mass, not eviction, is the active ingredient. On Split-CIFAR-100 a no-protection baseline stays above every protection arm, so the budget result is internal to the protected family. The contribution is a knowledge-centric diagnosis and audit framework, not an accuracy gain.
```

---

## 四、最后确认动作

1. 走完所有步骤 → `Build PDF for Approval` → **下载 PDF 逐页检查**：
   - 无 `??`（未定义引用）
   - 首页标题含 "Soft–Hard Protection Law"、摘要为新版（197 词）
   - 参考文献 30 条、编号连续
   - 共 31 页
   - 末尾有 CRediT / competing interest / funding / data availability / GenAI 五段声明
   - 有独立的 Conclusion 节
2. 勾选同意条款 → `SUBMIT`
3. 提交后：**记下稿件编号**、查确认邮件、**勿重复提交**
4. 若录用：确认走 **Subscription 路线**（0 费用）而非 OA（$3,040）
5. 投稿成功后回到 GitHub 打 tag，并把 Zenodo 归档升到 v8

---

## 五、常见踩坑对照

| 坑 | 说明 |
|---|---|
| 稿号写错 | PR 稿号是 **PR-D-26-16069**，不是 PR-S-26-23277 |
| 摘要超限 | InfoSci 有 250 / 200 两个数，取严按 200；本稿已压到 197 |
| 忘了 Conclusion 节 | InfoSci 明确要求，本稿第 6 节即是 |
| 忘了 Vitae | InfoSci 特有要求：100 词内可编辑 Word，文件名建议含 "vitae" |
| 忘了证件照 | InfoSci 特有：护照规格照片，作为独立 Figure 上传 |
| 图表超 8 个 | 官方 guideline 是 "8 figures + tables" 合计；本轮已把 3 张数值网格表移入补充材料，现为 8 |
| 误以为要匿名 | 单盲，作者名照常写在标题页 |
| 误选 OA 路线 | OA 要 $3,040；订阅路线对作者 0 费用 |
| 误以为图摘要必传 | InfoSci 指南压根没提图文摘要，传不传都行 |

---

*本地文件：`S2_resubmit/upload/`　|　归档：Zenodo concept DOI `10.5281/zenodo.23668522`　|　仓库：github.com/zhouranxi/delp-anchor-budget*
