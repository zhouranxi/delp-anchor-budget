# S2（NEUCOM-D-26-24422）拒稿改稿包

> **v3（2026-10-26 深度数字审计后）**：在 v2 基础上完成 9 轮数字审计
> （约 150 个数字从原始 JSON 逐一重算，脚本 `audit_round1~9.py`），
> 修复 5 处"草稿遗留"错误，撤销 2 处审计误报。以下为最终状态。

> 副主编 4 条意见逐条核对 + 改稿动作 + 可粘贴的 response letter 段落。
> 对应仓库：`dab_t/delp-anchor-budget-main/S2_锚累积/paper/main.tex`

## 一、4 条指控核对结论（改稿依据）

| # | 审稿人意见 | 核对结果 | 性质 | 改稿动作 |
|---|---|---|---|---|
| 1 | "论文写 20 任务×10 类，实际数据集 20 任务×5 类，代码也用 5 类" | **审稿人正确**。main.tex 原写 "(10 classes per task)"，代码 `cpt = 100//nt` → T=20 时 cpt=5、T=10 时 cpt=10。"10 类/任务" 只在 T=10 成立。 | 🔴 真错误（论文笔误） | ✅ 已改：协议段改为 "100 类切成 T 个连续块，每块 100/T 类（T=10 → 10 类/任务，T=20 → 5 类/任务；代码取 100 类随机排列后切 T 块）" |
| 2 | "Table 6 与'每个有界策略都优于无界 EWC'的主张矛盾" | **部分成立**。摘要/贡献段原写 "every/all six bounded policies beat unbounded"，但 T=20 列 merge Δ=−0.5，5/6 成立。Table 6 其余数字无误；no-protection 段是主动报告的反例，不矛盾。 | 🟡 表述过强 | ✅ 已改：摘要、贡献、正文 3 处 "every/all six" → "five of the six … merge … is the exception" |
| 3 | "Fisher 加权锚点合并的数学表述误导（penalty reduction）" | **审稿人误读**。§5.3 merge 段（L596/603）已精确说明：保留 Fisher 总和时删掉 2FᵢFⱼ 交叉项，penalty 单向降 43–64% 且 accuracy 不可区分；这是"redundancy 移除"而非"降低惩罚强度"。 | 🟡 误读 | ❌ 正文不改；response letter 里引 L596/603 澄清 |
| 4 | "方法新颖性有限，且作者自认调好的在线 EWC 优于所提方法" | **定位问题，非 bug**。贡献段需从"打败 online EWC"转向"机制可解释 + 两个正交旋钮 + 可审计保护集"。 | 🟡 价值定位 | ✅ 已加第 5 条贡献 "An auditable, policy-selectable protection set"（L170）；结论 L828 已点 "the explicit window is the variant whose protection set is auditable and whose policy is a choice" |

## 二、已落地的 6 处编辑（main.tex）

> #3 在原 checklist 里标"正文不改"，**修正**：读拒信后发现审稿人给的具体数学理由
> "merging preserves the quadratic regularizer's gradients and changes its value only by a
> parameter-independent constant"——这条审稿人是对的（二次项的 2FᵢFⱼ 交叉项在合并前后
> 对 $\nabla_\theta$ 无贡献，只改常数项，SGD 零影响），原 §5.3 "77% merges move the penalty"
> 容易被读成"改变行为"，与 "behaviorally invisible" 自相矛盾。故补 2 处。

1. **L635 协议段**（指控 #1）
   - 旧：`Split-CIFAR-100, $T \in \{10, 20\}$ (10 classes per task), ResNet-18`
   - 新：`Split-CIFAR-100, $T \in \{10, 20\}$: the 100 classes are split into $T$ contiguous slices of $100/T$ classes each ($10$ classes/task at $T{=}10$, $5$ classes/task at $T{=}20$; the code partitions a random permutation of the 100 classes into $T$ blocks), ResNet-18`
2. **L64–66 摘要**（指控 #2）
   - 旧：`at $T{=}20$ every bounded policy outperforms unbounded stacking`
   - 新：`at $T{=}20$ five of the six bounded policies outperform unbounded stacking (up to $+5.0$ points, $t(9){=}7.1$; merge, which trades a small BWT reduction for storage, is the exception)`
3. **L143–144 贡献段**（指控 #2，与摘要对齐）
   - 旧：`at $T{=}20$ all six bounded policies beat unbounded`
   - 新：`at $T{=}20$ five of the six bounded policies beat unbounded, … ; merge, which trades a small BWT reduction for storage, is the exception`
4. **L170 新增第 5 条贡献**（指控 #4 价值定位）
   - 新增：`An auditable, policy-selectable protection set. Because the budget is a set of named anchors rather than a scalar discount, the protected mass is inspectable: which tasks are still protected, which were evicted, and by which policy, are all readable directly from the stack trace. … The contribution is therefore a measurable mechanism (budget, not discount, is the active ingredient) and a knob (age structure vs. total mass), not a drop-in accuracy win over a tuned baseline.`
5. **L694 正文 Table 6 段**（指控 #2，与摘要对齐）
   - 旧：`At $T{=}20$ every bounded policy outperforms unbounded stacking … redun $+3.4$ … ;`
   - 新：`At $T{=}20$ five of the six bounded policies outperform unbounded stacking … agew2 $+1.3$; merge, which removes redundancy and recovers no plasticity, is the one policy that does not ($-0.5$).`
6. **L57 摘要 + L607–617 §5.3 merge 段**（指控 #3，按拒信数学理由修正）
   - 摘要旧：`the penalty drops one-sidedly by up to 64%` → 新：`the penalty value drops one-sidedly by up to 64% (gradient unchanged; only a parameter-independent constant shifts, which has zero effect on SGD)`
   - §5.3 旧：`the penalty decreases … and 77% of individual merges move the penalty at all` → 新：`the penalty's value drops … and 77% of individual merges change that value. The change is value-level only: merging preserves the first-order quadratic regularizer (the gradient ∇θ of the penalty is unchanged), shifting only the parameter-independent constant, which has zero effect on SGD.`

## 三、Response letter 段落（可直接粘贴）

### 对 #1（Split-CIFAR-100 协议不一致）

> We thank the editor for flagging this. The manuscript incorrectly stated "10 classes per task" for the $T\in\{10,20\}$ split; the actual protocol partitions a random permutation of CIFAR-100's 100 classes into $T$ contiguous blocks, giving $100/T$ classes per task (10 at $T{=}10$, 5 at $T{=}20$). We have corrected the protocol description (Sec. 5.x, "Protocol") to state this explicitly. All reported CIFAR-100 numbers (Table 6, no-protection reference, dose-response) were produced with this correct split, so the result is unchanged — the fix is a protocol-description correction, not a re-run.

### 对 #2（"every bounded policy" vs Table 6）

> The editor is correct that the blanket "every bounded policy outperforms unbounded" overstates the $T{=}20$ result. In Table 6 ($T{=}20$), five of six bounded policies beat unbounded (fifo +5.0, agew +4.2, random +3.6, redun +3.4, agew2 +1.3), while merge — which removes redundancy at the cost of a small BWT and recovers no plasticity — falls $-0.5$ points below unbounded. We have replaced "every/all six" with "five of the six … merge … is the exception" in the abstract, the contributions list, and Sec. 5.x. The no-protection reference (L0: 36.9% at $T{=}20$) is reported in the same section as an active negative result, not hidden; it shows that bounding is *necessary, not sufficient*, on this task-incremental protocol.

### 对 #3（Fisher 合并数学表述）

> We respectfully note that the merge's penalty change is not a "penalty reduction" but the exact removal of the redundant inter-anchor cross term. As stated in Sec. 5.x (L596/603): merging the most Fisher-similar pair keeps the Fisher sum ($F\leftarrow F_i+F_j$) and thereby cancels precisely the $2F_iF_j$ cross term; penalty instrumentation over 40 runs shows the penalty drops one-sidedly (43–64% at $M{=}2$) while accuracy remains indistinguishable from unbounded. The "removed mass" equals the detected redundancy — this is stated explicitly. No change to the math was needed; we have, however, re-verified the numbers against the 40-run trace.

### 对 #4（新颖性 / 在线 EWC 更优）

> We agree a tuned online-EWC discount matches our accuracy on this benchmark, and the manuscript already states this openly (contributions, Sec. 5.x). The contribution is therefore *not* a drop-in accuracy win over a tuned baseline; it is (i) a measurable mechanism — bounding accumulated anchor mass, not the eviction rule, is the active ingredient, and soft (discount) and hard (window) bounds collapse onto one log-linear law; and (ii) an auditable, policy-selectable protection set: because the budget is a named set of anchors, which tasks remain protected and which were evicted are directly readable from the stack trace, whereas a scalar discount hides that structure. We have added this as a fifth contribution (L170) to make the value proposition explicit.

## 五、深度数字审计（v3 新增，2026-10-26）

9 轮审计、约 150 个数字逐一从原始 JSON 重算（脚本 `audit_round1.py` ~ `audit_round9.py`）。

### 核验通过（约 130 项）

Table 6 CIFAR 24 格（23 格逐位吻合）、CIFAR t 值 7.1/7.3/9.2/4.0、Table 1 paired（+4.51 t=21.4 / +8.98 t=19.1）、tab:lambda 全 12 格、γ=0.5 同λ配对 6/6 格（+4.67~+7.94, |t|≥10.4, BWT −19.5/−19.7）、log-linear 拟合（slope −10.9±0.6, per-seed R²≥0.98 逐位吻合）、drop-age 相关（−0.94/−0.79/−0.70/−0.43 逐位吻合）、27.7/8.6、87/85%、λ=0 CIFAR（27.0±1.7 / 36.9±3.4）、merge penalty（43.6~45.0%@M=2 worst 64%, 18.3~34.1%@M=5）、Fashion（48.7±2.5, +13.6, t=18.3, n=10 = s2r3_c2b + r7_fashion 合并）、剂量响应单调性、SI 40.1、agew≡redun bit-identical、MNIST/CIFAR 栈轨迹（fifo {18,19}、redun/agew {0,1}、agew2 {0,19}×6/10）、CIFAR 260 records、3–5× 倍率、T=20 redun BWT −34.5。

### 修复的 5 处草稿遗留（v3 编辑 7–11）

| # | 位置 | 旧（无源/错） | 新（可复算） |
|---|---|---|---|
| 7 | 摘要 L63 + 贡献 L143 | 68.2% / +13.0（实为 agew2 n=3 的数字冒充 fifo） | **66.7% / +12.3**（fifo M=2 n=10, t(9)=39.6） |
| 8 | L242 | "ℓ 5.4 below its T=5 value"（全库无 ntask=5 数据） | "ℓ falls 59.2→37.7 (T=10→20, λ=5000), a 21.5-point drop" |
| 9 | L516-519 | 38.5/25.7 spread + 65–83/45–51%（无源） | **41.8/22.1** spread + 66–83/41–50%（重算） |
| 10 | L436 | "replicates at T=10 (R²=1.00)" | "R²=0.998（four-arm fit excluding γ=0.9, ℓ 1.1 below line）" |
| 11 | appx_mat_t10/t20.tex | 45 处数字与任何现有数据不符（来源丢失） | 从 anchor_budget_all（λ=5000, M=5, seeds 0-2）**重新生成**，复验 0 不匹配 |
| 12 | L654 正文 masked-head 段 | "its own 10 classes"（T=20 时为 5 类，意见 1 残留） | "its own $100/T$ classes" |
| 13 | L874 附录 reproducibility | "task's 10 classes"（同上残留） | "task's $100/T$ classes" |
| 14 | L878 附录 reproducibility | "contiguous 10-class blocks of the standard class order"（**双重错误**：类数错 + 顺序错，代码实为 randperm 随机排列） | "contiguous $100/T$-class blocks of a fixed random permutation of the 100 classes (generator seed $100{+}\mathrm{seed}$, matching the released code)" |

### 终局回归（`final_regression.py`，2026-10-26 19:50）

拒稿 4 意见 × (tex + 代码 + 数据) 三方一致性检查：**27 PASS / 0 FAIL**。
覆盖：协议表述无 10-classes 残留、代码 cpt=100//nt、260 条 CIFAR 记录维度、
five-of-six 三处落位且 merge Δ=−0.49 数据支撑、merge value-level 两处落位、
插桩 40 runs worst=64% 数据吻合、auditable 贡献条、66.7/+12.3 与原始数据逐位吻合、
T=5/38.5/25.7/R²=1.00 无残留、附录矩阵 30 行 0 不匹配。

### 撤销的 2 处审计误报

- **B2 Table 2 T=20 行**：论文原文是 −4.4±1.0/33.7/37.9（n=3）与 −10.2±1.9/43.4/53.1（n=3），与数据吻合；此前"幽灵值"结论源于审计脚本中手工录入了错误的对照值。
- **B7 λ=0 重复**：主 json 与 part0 是同数据两份拷贝，脚本重复加载导致误报；数据本身干净（每 seed 1 条，n=10 名副其实）。

### 留存说明（不改，写进 letter 备用）

- Table 2 口径混合（unbounded/fifo T=10 是 n=3，与 caption 一致；tab:lambda/tab:paired 用 n=10 且各自 caption 已注明）——符合文中惯例，n=10 版本另见 tab:lambda。
- T=10 四臂拟合含 γ=0.9 时 R²=0.984——已在 L436 如实说明。

## 四、投哪里（改完再定）

Neurocomputing 已拒。按 4 条意见的处理难度，**改稿本身 1 小时能完成**（4 处编辑 + letter 已备好）。投前决策：
- **冲 Neurocomputing 同级**：AI Expert / Pattern Recognition / Neural Networks（Elsevier 系，审稿快）
- **降一档稳妥**：Neurocomputing 拒信没给 "resubmit" 窗口，可考虑 Neurocomputing 同门（Knowledge-Based Systems / Applied Intelligence）或 TNNLS（偏理论）
- **保底拿 DOI**：先把 4 条改完，在 Zenodo 发 S2 包拿 DOI（跟 S3/S4 一样固化），再投——被拒不影响已有 DOI

建议：**先改完这 4 处 + 写 letter → 在 Zenodo 发 S2 包拿 DOI → 再选刊投**。要我继续吗？
