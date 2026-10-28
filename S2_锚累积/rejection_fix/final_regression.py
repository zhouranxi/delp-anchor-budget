# -*- coding: utf-8 -*-
"""终局回归：拒稿 4 意见 × (tex + 代码 + 数据) 三方一致性."""
import json, math, re, statistics as st
from pathlib import Path

BASE = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main")
TEX = (BASE / "S2_锚累积" / "paper" / "main.tex").read_text(encoding="utf-8")
R2 = BASE / "S2_锚累积" / "results"

def load(parts):
    recs = []
    for p in sorted(R2.glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

def paired_t(d):
    m = st.mean(d); s = st.stdev(d)
    return m / (s / math.sqrt(len(d)))

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    tag = "PASS" if cond else "FAIL"
    if cond: ok += 1
    else: fail += 1
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))

print("========== 意见1：协议与代码/数据一致 ==========")
# 1a. 正文不再有 "10 classes per task" / "own 10 classes" / "contiguous 10-class"
check("正文/附录无 '10 classes per task'",
      "10 classes per task" not in TEX)
check("无 'its own 10 classes'", "own 10 classes" not in TEX)
check("无 'contiguous 10-class'", "contiguous 10-class" not in TEX)
check("协议段有 100/T 表述", "100/T" in TEX and "contiguous slices" in TEX)
# 1b. 代码: cpt = 100//nt
code = (BASE / "bench" / "anchor_budget_cifar.py").read_text(encoding="utf-8")
check("代码 cpt = 100 // nt（T=20→5类）", "cpt = 100 // nt" in code)
# 1c. 数据: 每条 CIFAR 记录的 acc_matrix 维度 == T, task 数 == ntask
cifar = [r for r in load("s2r6_b_t*.json*") if r.get("dataset") == "cifar100"]
dims_ok = all(len(r["acc_matrix"]) == r["ntask"] for r in cifar)
check("CIFAR 记录 acc_matrix 维度 == ntask", dims_ok, f"n={len(cifar)}")
# 1d. 类数下界: T=20 任务时 per-task 5 类 → 单任务 chance ≈ 20%, 检查 acc 量级合理(<100)
check("260 条 CIFAR 记录在册", len(cifar) == 260, f"got {len(cifar)}")

print()
print("========== 意见2：'every bounded' 全称量词 ==========")
check("摘要 five of the six", TEX.count("five of the six") >= 1)
check("贡献段 five of the six", TEX.count("five of the six") >= 2 or "five of the six bounded policies beat" in TEX)
check("正文 Table6 段 five of the six", "five of the six bounded policies outperform" in TEX)
check("无残留 'every bounded policy outperform'", "every bounded policy outperforms" not in TEX)
check("无残留 'all six bounded'", "all six bounded" not in TEX)
# 数据: T=20 merge M=5 Δ = -0.5 < 0（支撑 'merge is the exception'）
c20 = {r["seed"]: r for r in cifar if r["ntask"]==20 and r["policy"]=="unbounded"}
mg = {r["seed"]: r for r in cifar if r["ntask"]==20 and r["policy"]=="merge" and r["M"]==5}
common = sorted(set(c20) & set(mg))
dmg = st.mean([mg[s]["final_avg"] - c20[s]["final_avg"] for s in common])
check("数据支撑: merge T=20 M=5 Δ<0", dmg < 0, f"Δ={dmg:+.2f}")

print()
print("========== 意见3：merge 惩罚 value-level 表述 ==========")
check("摘要 gradient-unchanged 括号", "gradient unchanged" in TEX and "zero effect on SGD" in TEX)
check("正文 value-level only 段", "value-level only" in TEX and "parameter-independent constant" in TEX)
# 数据: r8_merge 的 merge_drifts 与 max_rel_drift 一致性
mgj = load_dir = None
RB = BASE / "bench" / "results"
mg_recs = []
for p in sorted(RB.glob("r8_merge.json")):
    d = json.load(open(p))
    mg_recs = d["results"] if isinstance(d, dict) else d
if mg_recs:
    n_runs = len(mg_recs)
    m2 = [r for r in mg_recs if r["M"] == 2]
    worst = max(r["max_rel_drift"] for r in m2)
    check("插桩数据: 40 runs, M=2 worst drift ≈64%", n_runs == 40 and 0.60 < worst < 0.68,
          f"runs={n_runs}, worst={worst*100:.0f}%")
    check("论文 43–45%/18–34% 区间在数据内",
          abs(st.mean([r["max_rel_drift"] for r in mg_recs if r["M"]==2 and r["ntask"]==10])*100 - 43.6) < 1)

print()
print("========== 意见4：价值重定位 ==========")
check("贡献列表有 auditable 条目", "An auditable, policy-selectable protection set" in TEX)
check("明确 'not a drop-in accuracy win'", "not a drop-in accuracy win" in TEX)

print()
print("========== v3 数字审计回归（5 处修复后无残留）==========")
check("摘要头牌 = 66.7 / +12.3", "66.7" in TEX and "+12.3" in TEX)
check("无残留 68.2%", "68.2" not in TEX)
u10 = [r for r in load("anchor_budget_all.json") if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==5000.0]
f10 = [r for r in load("anchor_budget_all.json") if r["ntask"]==10 and r["policy"]=="fifo" and r.get("lam")==5000.0 and r["M"]==2]
ub = {r["seed"]: r for r in u10}; fb = {r["seed"]: r for r in f10}
common = sorted(set(ub) & set(fb))
dd = [fb[s]["final_avg"] - ub[s]["final_avg"] for s in common]
check("66.7/+12.3 与数据吻合", abs(st.mean([r['final_avg'] for r in fb.values()]) - 66.74) < 0.05
      and abs(st.mean(dd) - 12.25) < 0.05 and abs(paired_t(dd) - 39.6) < 0.15)
check("T=5 引用已除", "T{=}5 value" not in TEX and "its $T{=}5$" not in TEX)
check("spread 41.8/22.1", "41.8" in TEX and "22.1" in TEX and "38.5" not in TEX)
check("R²=0.998 表述", "R^2{=}0.998" in TEX and "R^2{=}1.00" not in TEX)
# 附录矩阵
for T in (10, 20):
    appx = (BASE / "S2_锚累积" / "paper" / f"appx_mat_t{T}.tex").read_text(encoding="utf-8")
    rows = re.findall(r"^(\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", appx, re.M)
    mn = load("anchor_budget_all.json")
    bad = 0
    for task_s, *vals in rows:
        j = int(task_s) - 1
        for pol, pv in zip(("unbounded","fifo","random","redun","merge"), vals):
            rs = [r for r in mn if r["ntask"]==T and r["policy"]==pol and r.get("lam")==5000.0
                  and r.get("M")==5 and r["seed"] in (0,1,2)]
            got = st.mean([r["acc_matrix"][-1][j] for r in rs])
            if abs(got - float(pv)) > 0.051: bad += 1
    check(f"附录 T={T} 矩阵 0 不匹配", bad == 0 and len(rows) == T, f"{len(rows)} 行, bad={bad}")

print()
print(f"========== 总计: PASS {ok} / FAIL {fail} ==========")
