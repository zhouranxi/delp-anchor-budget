# -*- coding: utf-8 -*-
"""S2 修复：① λ=0 CIFAR JSON 去重（每 seed 留 1 条）② 从 anchor_budget_all 重新生成附录矩阵 tex."""
import json, statistics as st
from pathlib import Path
from collections import Counter

BASE = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main")
R2 = BASE / "S2_锚累积" / "results"

def load_json(p):
    d = json.load(open(p))
    return d["results"] if isinstance(d, dict) and "results" in d else d

# ---------- 1. λ=0 去重 ----------
print("== 1. λ=0 去重 ==")
for T in (10, 20):
    fn = R2 / f"r7_cifar_lam0_t{T}.json"
    if not fn.exists():
        print(f"  {fn.name} 不存在（只有 part 文件），跳过主文件处理")
        continue
    recs = load_json(fn)
    seen, out = set(), []
    for r in recs:
        key = (r["ntask"], r["seed"], r.get("policy"), r.get("M"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    print(f"  {fn.name}: {len(recs)} -> {len(out)} 条")
    json.dump({"results": out}, open(fn, "w"), indent=1)

# part 文件也去重（发布数据时合并的是这些）
for p in sorted(R2.glob("r7_cifar_lam0_*.json.part*")):
    recs = load_json(p)
    seen, out = set(), []
    for r in recs:
        key = (r["ntask"], r["seed"], r.get("policy"), r.get("M"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if len(out) != len(recs):
        print(f"  {p.name}: {len(recs)} -> {len(out)} 条")
        json.dump({"results": out}, open(p, "w"), indent=1)
    else:
        print(f"  {p.name}: 无重复")

# ---------- 2. 附录矩阵重生成 ----------
print("\n== 2. 附录矩阵重生成（λ=5000, M=5, seeds 0-2, n=3）==")
recs = load_json(R2 / "anchor_budget_all.json")
POLICIES = ["unbounded", "fifo", "random", "redun", "merge"]

for T in (10, 20):
    cols = {}
    for pol in POLICIES:
        rs = [r for r in recs if r["ntask"] == T and r["policy"] == pol
              and r.get("lam") == 5000.0 and r.get("M") == 5 and r["seed"] in (0, 1, 2)]
        assert len(rs) == 3, f"{pol} T={T}: 只有 {len(rs)} 条"
        cols[pol] = [st.mean([r["acc_matrix"][-1][j] for r in rs]) for j in range(T)]
    lines = [
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"task & unbounded & fifo & random & redun & merge \\",
        r"\midrule",
    ]
    for j in range(T):
        vals = " & ".join(f"{cols[pol][j]:.1f}" for pol in POLICIES)
        lines.append(f"{j+1} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = BASE / "S2_锚累积" / "paper" / f"appx_mat_t{T}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  写出 {out.name} ({T} 行)")

# 验证重生成后的表格与数据逐位一致
print("\n== 3. 复验 ==")
for T in (10, 20):
    fn = BASE / "S2_锚累积" / "paper" / f"appx_mat_t{T}.tex"
    import re
    rows = re.findall(r"^(\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", fn.read_text(encoding="utf-8"), re.M)
    bad = 0
    for task_s, *vals in rows:
        j = int(task_s) - 1
        for pol, pv in zip(POLICIES, vals):
            rs = [r for r in recs if r["ntask"] == T and r["policy"] == pol
                  and r.get("lam") == 5000.0 and r.get("M") == 5 and r["seed"] in (0, 1, 2)]
            got = st.mean([r["acc_matrix"][-1][j] for r in rs])
            if abs(got - float(pv)) > 0.051:
                bad += 1
    print(f"  T={T}: {len(rows)} 行, 不匹配 {bad}")
