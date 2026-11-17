# Knowledge-Based Systems 投稿 — 提交前逐项核对清单

**投稿入口**：https://submit.elsevier.com/KNOSYS （Elsevier 托管投稿系统，HTTP 200 已验证）
**期刊官网**：https://www.sciencedirect.com/journal/knowledge-based-systems
**指南**（所有依据均出自此页）：https://www.elsevier.com/journals/knowledge-based-systems/0950-7051/guide-for-authors
**本地文件目录**：`S2_resubmit/upload/`
**状态**：材料已备齐，待用户提交

> 前史：PR 稿 PR-D-26-16069 于 2026-11-10 编辑直接拒稿（Area Editor）；Neurocomputing 稿
> NEUCOM-D-26-24422 于 2026-10-26 拒稿。两稿均已关闭，非在审。
> 新版已把负面结果前置为头号贡献，并新增软硬保护律子节 + knowledge-based 框架。

---

## 一、Attach Files —— 每个文件的期望值

| 顺序 | Item Type | 文件 | 规格核验 |
|---|---|---|---|
| 1 | `Manuscript` | `main.pdf` | 双倍行距 26 页。期刊为**偏好性**建议 ≤20 页（"preferably no more than 20 double line spaced manuscript pages"），非硬上限；本稿以论证完整性优先，若编辑要求可再压缩 |
| 2 | `Supplementary material` | `supplementary.pdf` | **7 页**：附录 A（统计理由+超参数）+ 附录 B（完整逐任务精度矩阵）+ 附录 C（三张完整数值网格表）+ 附录 D（eviction 策略定义表）+ 2 张补充图 |
| 3 | `Highlights` | `highlights.docx` | 5 条，每条 ≤85 字符（含空格），**文件名含 "highlights"** ✅（官方措辞 "encouraged"，鼓励非强制） |
| 4 | `Declaration of competing interest` | `declarations.docx` | **强制**。官方要求走声明工具生成："The declarations tool should always be completed."；上传于 "attach/upload files" 步骤，.doc/.docx，无需签名 |
| 5 | `Cover Letter` | `cover_letter.docx` | KBS 指南**未提及** cover letter（非必需，属加分项）。本版为 KBS 定制：scope 锚定 KBS 的 knowledge-based 定位 |
| 6 | `Figure` | `Figure_1.pdf` … `Figure_3.pdf` | 正文 3 张，按出现顺序编号，均为纯矢量 PDF（age profile / criteria / stack trace） |
| 7 | `Figure`（补充） | `Figure_S1.pdf`、`Figure_S2.pdf` | 补充材料内 2 张，随 supplementary 一并提交 |
| 8 | `Graphical Abstract`（可选） | `graphical_abstract.png` | **KBS 指南完全没有 graphical abstract 章节**，传不传都不影响评审 |

**图表总数**：正文 **3 图 + 1 表**（tab:paired），补充材料另 2 图 5 表。KBS 指南**未规定图表数量上限**（NOT STATED）。

**不需要的文件**：作者 Vitae、作者照片 —— 这是 Pattern Recognition / Information Sciences 的要求，**KBS 指南无此要求**，故不传。

**备用文本版**（EM 文本框直接粘贴用）：`cover_letter.txt`、`metadata_for_em.txt`

> ⚠️ 每次重新编译后，务必运行 `python audit/sync_upload.py`，否则 upload/ 会残留旧版 PDF 或错序的图片。

---

## 二、关键字段速查（Knowledge-Based Systems）

| 步骤 / 字段 | 答案 | 依据 |
|---|---|---|
| Article Type | **Original high-quality research paper** | 指南仅列两类："Original high-quality research and review papers" 与 "Short communications"（后者限 10 页，不适用本稿） |
| Title（原文粘贴） | `The Anchor Accumulation Problem: Bounded Anchor Sets, a Soft–Hard Protection Law, and the Auditable Cost of Regularized Continual Learning` | 见 `metadata_for_em.txt` |
| Abstract | **197 词** ✅ | 官方上限 250 词："You are required to provide a concise and factual abstract that does not exceed 250 words." |
| Keywords | **5 个**（1–7）✅ | continual learning / elastic weight consolidation / catastrophic forgetting / knowledge-based protection / stability–plasticity dilemma |
| 评审模式 | **Single anonymized（单盲）** | 官方原文："This journal follows a single anonymized review process."；**无需**匿名稿，**不要求**单独拆标题页文件 |
| 引用格式 | 投稿时不限，只要一致 | 官方原文："This journal does not set strict requirements on reference formatting at submission... References can be in any style or format as long as the style is consistent." 期刊方括号数字风格在 proof 阶段套用；现稿编号格式已一致 |
| 篇幅 | 建议 ≤20 页（含表图），**偏好非硬限** | 官方原文含 "preferably"；Short communications 为 "no more than 10 double line spaced manuscript pages"（本稿不适用） |
| Highlights | 鼓励提供，3–5 条 × ≤85 字符 | 独立可编辑文件，文件名必须含 "highlights" |
| Graphical Abstract | 官方指南**完全未提及** | 无任何规格要求，传不传都行 |
| 图表数量上限 | **NOT STATED** | 指南无此限制 |
| Artwork | 矢量图 EPS/PDF（嵌入字体）；位图 ≥300 dpi，单栏 ≥1063 px，全页 ≥2244 px | 现有 Figure_1–3.pdf 均为矢量；每图须有 caption 并按出现顺序编号 |
| CRediT | **强制** | 官方："Corresponding authors are required to acknowledge co-author contributions using CRediT (Contributor Roles Taxonomy) roles" |
| Data availability | **Option C 强制** | 官方："For this journal, Option C instructions from our research data guidelines apply." → 必须 deposit + 提交时给链接，不能共享则须书面说明（该说明会随文发表） |
| Competing interests | **强制**，.doc/.docx 上传 | 即使选 "I have nothing to declare" 也要生成文件 |
| Funding | 无 | 官方无资助句式："This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors." |
| GenAI 披露 | **投稿时必须声明** | 官方："Authors must declare the use of generative AI tools in the manuscript preparation process upon submission of the paper."；放在参考文献前的专节。基础工具（语法/拼写/引用检查）免申报 |
| 一稿多投 | `No`（当前无在审稿件） | PR 与 Neurocomputing 均已拒稿关闭；Zenodo 预印本不算 prior publication |
| Suggested Reviewers | 可留空 | |
| **费用** | **hybrid：订阅路线 0 费用** | 官方原句："No publication fee charged to authors, and published articles are immediately available to subscribers." OA 路线 $3,240（full length）/ $700（short），**不要选 OA** |

---

## 三、摘要纯文本版（197 词，EM 文本框直接粘贴）

```
Regularized continual learning stores one protection state per task, whose cost grows with sequence length. In usage-localized elastic weight consolidation (DeLP), full-weight anchors drag later learning; a retention decomposition a_T = l + (T-1)/T BWT localizes the cost in the learning term l. We call this the anchor accumulation problem, repairing it by bounding the stack at M entries. A FIFO budget recovers +4.5/+9.0 accuracy points over unbounded DeLP at T = 10/20 (M = 5; n = 10 paired seeds; Wilcoxon p = 0.002). The budget is distinct from the elastic weight lambda: it controls the age structure of forgetting, lambda its amount. Eviction design obeys one constraint: criteria keyed to constraint mass grandfather-protect stale anchors and evict new ones at birth (-34.5 BWT); age-first criteria are immune. Online EWC's Fisher discount is a soft bound, and both obey one log-linear law in protected-mass horizon (l = 84 - 11 ln h, R^2 = 0.99): bounding anchor mass, not eviction, is the active ingredient. On Split-CIFAR-100 a no-protection baseline stays above every protection arm, so the budget result is internal to the protected family. The contribution is a knowledge-based diagnosis and audit framework, not an accuracy gain.
```

---

## 四、最后确认动作

1. 走完所有步骤 → `Build PDF for Approval` → **下载 PDF 逐页检查**：
   - 无 `??`（未定义引用）
   - 首页标题含 "Soft–Hard Protection Law"、摘要为新版（197 词）
   - 参考文献 30 条、编号连续
   - 正文 3 张图都嵌进去了
   - 有独立的 Conclusion 节
   - 末尾有 CRediT / competing interest / funding / data availability / GenAI 五段声明
2. 勾选同意条款 → `SUBMIT`
3. 提交后：**记下稿件编号**、查确认邮件、**勿重复提交**
4. 若录用：确认走 **Subscription 路线**（0 费用）而非 OA（$3,240）
5. 投稿成功后回到 GitHub 打 tag，并把 Zenodo 归档升到 v8

---

## 五、常见踩坑对照

| 坑 | 说明 |
|---|---|
| 稿号写错 | PR 稿号是 **PR-D-26-16069**；Neurocomputing 是 **NEUCOM-D-26-24422** |
| 摘要超限 | KBS 上限 **250 词**（不是 InfoSci 的 200/250 冲突），本稿 197 词安全 |
| 忘了 Conclusion 节 | 本稿有独立 Conclusion 节（Sec. 7） |
| 传了 Vitae / 证件照 | **KBS 无此要求**，传了反而多余；InfoSci / PR 才要 |
| 传了 5 张图 | 正文只留 3 张；另 2 张随 supplementary 提交即可，编号用 Figure_S1 / Figure_S2 |
| 误以为图表有 8 个上限 | KBS **无图表数量上限**，别照搬 InfoSci 的 "8 figures + tables" |
| 误以为要匿名 | 单盲，作者名照常写在标题页，无需匿名版 |
| 误以为要匿名版单独文件 | 指南**未要求**，不必拆 |
| 误选 OA 路线 | OA 要 $3,240；订阅路线对作者 0 费用 |
| 误以为图摘要必传 | KBS 指南压根没提图文摘要，传不传都行 |
| 忘了 GenAI 声明 | **投稿时必须声明**，在参考文献前的专节 |
| upload/ 里是旧 PDF | 每次编译后跑 `python audit/sync_upload.py` |

---

*本地文件：`S2_resubmit/upload/`　|　归档：Zenodo concept DOI `10.5281/zenodo.23668522`　|　仓库：github.com/zhouranxi/delp-anchor-budget*
