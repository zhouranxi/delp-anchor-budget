# Array 投稿 — 提交前逐项核对清单

**投稿入口**：https://submit.elsevier.com/ARRAY （Elsevier 托管投稿系统，进入后由 Editorial Manager 跟踪）
**期刊官网**：https://www.sciencedirect.com/journal/array
**本地文件目录**：`S2_resubmit/upload/`
**状态**：材料已备齐，待用户提交

> 前史：PR 稿 PR-D-26-16069 于 2026-11-10 编辑直接拒稿（Area Editor: Cheng-Lin Liu）；
> Neurocomputing 稿 NEUCOM-D-26-24422 于 2026-10-26 拒稿。两稿均已正式关闭，非在审。
> 新版已把负面结果前置为头号贡献（Q4），并新增软硬保护律子节 + knowledge-centric 框架。

---

## 一、Attach Files —— 每个文件的期望值

| 顺序 | Item Type | 文件 | 规格核验 |
|---|---|---|---|
| 1 | `Manuscript` | `main.pdf` | **33 页双倍行距**。标题含 "Soft–Hard Protection Law"；期刊字段已改为 Array |
| 2 | `Supplementary material` | `supplementary.pdf` | 3 页：附录 A（统计理由+超参数）+ 附录 B（完整逐任务精度矩阵） |
| 3 | `Highlights` | `highlights.docx` | 5 条，每条 ≤85 字符（含空格），文件名含 "highlights" ✅ |
| 4 | `Declaration of competing interest` | `declarations.docx` | Array **强制**；即使无冲突也须生成并上传 |
| 5 | `Cover Letter` | `cover_letter.docx` | Array 版：逐字引用 PR 编辑意见 + 5 条逐项回应 |
| 6 | `Graphical Abstract` | `graphical_abstract.png` | 可选（官方 "encouraged"）。**3900×2280 px，300 dpi，横向** ≥1328 px ✅ 合规 |
| 7–11 | `Figure` | `Figure_1.pdf` … `Figure_5.pdf` | 5 张分别传，均为纯矢量 PDF（Elsevier 接受 PDF 矢量图） |

**备用文本版**（EM 文本框直接粘贴用）：`cover_letter.txt`、`metadata_for_em.txt`

---

## 二、关键字段速查（Array）

| 步骤 / 字段 | 答案 | 依据 |
|---|---|---|
| Article Type | **Regular paper**（"research paper reporting new results"） | technical notes 限 10 页，本稿不适用 |
| Title（原文粘贴） | `The Anchor Accumulation Problem: Bounded Anchor Sets, a Soft–Hard Protection Law, and the Auditable Cost of Regularized Continual Learning` | 见 `metadata_for_em.txt` |
| Abstract | **247 词**（上限 250）✅ | 见第三节 |
| Keywords | **6 个**（1–7）✅ | continual learning / elastic weight consolidation / catastrophic forgetting / knowledge management / protection-state curation / stability–plasticity dilemma |
| 评审模式 | **Single anonymized（单盲）** | 作者名保留在标题页，**无需匿名版** |
| 引用格式 | **编号方括号**，按首次出现顺序编号 | 当前 `elsarticle-num.bst` 输出已符合，**不需转 APA** |
| 篇幅 | Regular paper **无页数上限**（仅 technical notes 限 10 页） | 33 页合规，**不需要压缩** |
| Highlights | 5 条 × ≤85 字符 | 独立可编辑文件，文件名含 "highlights" |
| Graphical Abstract | 可选，建议传 | ≥1328 px 宽、横版、13 cm 可读 |
| Artwork | 矢量图 EPS/PDF；位图 ≥300 dpi | 现有 Figure_1–5.pdf 均为矢量 |
| 行距 | Double-spaced | `elsarticle` review 模式 |
| 参考文献数 | 30 | 主文 30 条；补充材料另 5 条统计文献 |
| 页数 | 33 页 + 3 页 supplementary | |
| CRediT | **强制** | 已在 main.tex 末尾 + declarations.docx |
| Data availability | Option B（鼓励非强制）；已有 GitHub + Zenodo，链接照填 | |
| Competing interests | **强制**，须走 declarations tool | 即使选 "I have nothing to declare" 也要生成文件 |
| Funding | 无 | 直接填 none |
| GenAI 披露 | 保留 | main.tex 末尾已有专节 |
| 一稿多投 | `No`（当前无在审稿件） | 说明：PR 与 Neurocomputing 均已拒稿关闭；Zenodo 预印本不算 prior publication |
| Suggested Reviewers | 可留空 | |
| APC | **USD 2,770**（全 Gold OA，无订阅路线） | 投稿后若录用需支付；参与 GPOA 地理定价，实际可能更低 |

---

## 三、摘要纯文本版（247 词，EM 文本框直接粘贴）

```
Regularized continual-learning schemes storing one protection state per task pay a cost growing with sequence length. In usage-localized elastic weight consolidation (DeLP), per-neuron elastic anchors persist at full weight, dragging later learning; a retention decomposition a_T = l + (T-1)/T BWT localizes the cost in the learning term l. We call this the anchor accumulation problem and repair it by bounding the stack at M entries. A FIFO budget recovers +4.5 and +9.0 final-accuracy points over unbounded DeLP at T = 10 and 20 (M = 5; n = 10 paired seeds; Wilcoxon p = 0.002), and the budget is distinct from the elastic weight lambda: it controls the age structure of forgetting, lambda its amount. Eviction design is ruled by one constraint: criteria keyed to current constraint mass grandfather-protect stale anchors and evict new ones at birth (-34.5 BWT); age-first criteria are immune. Discounting online EWC's Fisher mass is itself a soft bound, and both obey one log-linear law between the learning term and the protected-mass horizon (l = 84 - 11ln h, R^2 = 0.99): bounding accumulated anchor mass, not eviction, is the active ingredient, while the explicit window retains what a scalar discount cannot: a discrete, named, policy-selectable protection set readable from the stack trace. We report the boundary honestly: on Split-CIFAR-100 a no-protection baseline stays above every protection arm, so the budget result is internal to the protected family. The contribution is a knowledge-centric diagnosis and audit framework, not a drop-in accuracy win.
```

---

## 四、最后确认动作

1. 走完所有步骤 → `Build PDF for Approval` → **下载 PDF 逐页检查**：
   - 无 `??`（未定义引用）
   - 首页标题含 "Soft–Hard Protection Law"、摘要为新版（247 词）
   - 参考文献 30 条、编号连续
   - 共 33 页
2. 勾选同意条款 → `SUBMIT`
3. 提交后：**记下稿件编号**、查确认邮件、**勿重复提交**
4. 若录用：Array 为全 Gold OA，需确认 APC 与 licence（CC BY / CC BY-NC / CC BY-NC-ND）
5. 投稿成功后回到 GitHub 打 tag，并把 Zenodo 归档升到 v8

---

## 五、常见踩坑对照

| 坑 | 说明 |
|---|---|
| 稿号写错 | PR 稿号是 **PR-D-26-16069**，不是 PR-S-26-23277 |
| 摘要超限 | Array 上限 250 词，本稿已压到 247 |
| 误选 technical note | 选 regular paper；technical note 限 10 页会被系统或编辑直接退回 |
| 忘记传 highlights | 文件名必须含 "highlights" 字样 |
| 忘记传 competing interests | Array 强制，即使无冲突也要用 declarations tool 生成 .docx 上传 |
| 误以为要匿名 | 单盲，作者名照常写在标题页 |
| APC 认知偏差 | 全 Gold OA，$2,770 必付，录用后才需处理，但应提前确认经费 |

---

*本地文件：`S2_resubmit/upload/`　|　归档：Zenodo concept DOI `10.5281/zenodo.23668522`　|　仓库：github.com/zhouranxi/delp-anchor-budget*
