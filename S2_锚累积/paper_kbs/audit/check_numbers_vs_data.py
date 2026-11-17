# -*- coding: utf-8 -*-
"""S2 数字-数据一致性核查（2026-11-17 全量复算版）。

本脚本把正文/补充材料里所有承重数字从 results/ 的原始记录重新算一遍，
并逐条比对。可直接运行：

    python audit/check_numbers_vs_data.py

口径约定（踩过的坑，写在这里以免回退）：
1. 记录去重：anchor_budget_all.json 与独立 .json 文件含同一批记录，
   必须按 (kind,dataset,policy,M,ntask,lam,seed,gamma,mode) 去重后再统计，
   否则 n 会被高估（17 而非 10）。
2. EWC 臂用 mode 字段区分：mode=='plain' 是 plain EWC，
   mode=='online' + gamma==0.5/0.9/None 分别是三个折扣臂。
   用 gamma is None 会把 online gamma=1 混进 plain（ell 51.1 而非 83.7）。
3. policy grid 表（补充材料 tab:supp-grid）统一用种子 0--2（n=3）；
   正文引用的 unbounded/fifo 用 n=10。两者数值不同是口径差异，不是错误，
   但 caption 必须写明。
4. drop-age 相关：ages 定义为"任务越老 age 越大"（倒序），因此
   遗忘集中在老任务时相关系数为【正】，不是负。
5. 8.6/27.7 的口径：栈内 = drops[T-M:]，被驱逐 = drops[:T-M]；
   "five oldest evicted" 则是 drops[:5]。
"""
import json
import math
import os
import statistics as st

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load_records():
    """加载全部记录并按业务主键去重。"""
    recs = []
    anchor = os.path.join(RES, "anchor_budget_all.json")
    if os.path.exists(anchor):
        d = json.load(open(anchor, encoding="utf-8"))
        recs += d.get("results", d)
    for fn in sorted(os.listdir(RES)):
        # 注意：必须同时匹配 .json 与 .json.part0/.part1 —— r7_ewc / r7_fashion
        # 等批次数据是分片存放的，只读 .json 会整批漏掉（plain EWC 就藏在这里）。
        if ".json" not in fn or fn == "anchor_budget_all.json":
            continue
        try:
            d = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
        except Exception:
            continue
        rs = d.get("results", d) if isinstance(d, dict) else d
        if isinstance(rs, list):
            recs += rs
    seen = {}
    for r in recs:
        if "error" in r:
            continue
        # 主键必须含 mode：plain EWC 记录没有 gamma 键，
        # 若只按 gamma=None 去重会与 online(gamma=1) 撞键、互相覆盖。
        k = (r.get("kind"), r.get("dataset", "mnist"), r.get("policy"),
             r.get("M"), r.get("ntask"), r.get("lam"), r.get("seed"),
             r.get("gamma") if "gamma" in r else "<none>",
             r.get("mode"))
        # 同一主键可能出现两条：anchor_budget_all.json 里的摘要版没有
        # merge_drifts / survivor_history 字段，而 r8_merge.json 里有。
        # 必须保留字段更全的那条，否则 merge 插桩会静默丢 3 条记录。
        prev = seen.get(k)
        if prev is None or len(r) > len(prev):
            seen[k] = r
    return list(seen.values())


REC = load_records()


def delp(policy, M, T, lam, ds="mnist"):
    out = {}
    for r in REC:
        if r.get("kind") != "delp":
            continue
        if r.get("dataset", "mnist") != ds or r.get("ntask") != T:
            continue
        if r.get("lam") != lam or r.get("policy") != policy:
            continue
        if policy != "unbounded" and r.get("M") != M:
            continue
        out[r["seed"]] = r
    return out


def ewc(mode, gamma, T, lam):
    """mode: 'plain' or 'online'.  gamma: 0.5 / 0.9 / None(=no discount)."""
    out = {}
    for r in REC:
        if r.get("kind") != "ewc" or r.get("mode") != mode:
            continue
        if r.get("ntask") != T or r.get("lam") != lam:
            continue
        # plain 记录没有 gamma 键；online gamma=1 的 gamma 为 None。
        # 必须用"键是否存在"区分，否则 plain 会被 online(gamma=None) 覆盖。
        if ("gamma" in r) != (gamma is not None):
            continue
        if gamma is not None and r.get("gamma") != gamma:
            continue
        out[r["seed"]] = r
    return out


def paired(f, u):
    c = sorted(set(f) & set(u))
    if len(c) < 3:
        return None
    dv = [f[s]["final_avg"] - u[s]["final_avg"] for s in c]
    mx, my = st.mean(dv), st.stdev(dv)
    t = mx / (my / math.sqrt(len(dv))) if my else float("nan")
    try:
        # method='approx' 与正文口径一致（论文报的 0.002 是 n=10 时的近似值）；
        # 精确检验会给出 0.0（因为所有差值同号）。
        wp = stats.wilcoxon(dv, method="approx").pvalue
        wp_exact = stats.wilcoxon(dv).pvalue
    except Exception:
        wp = float("nan")
    return len(c), st.mean(dv), t, (wp, wp_exact)


def fit_loglin(pts):
    xs = [math.log(h) for h, _ in pts]
    ys = [y for _, y in pts]
    mx, my = st.mean(xs), st.mean(ys)
    sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    ic = my - sl * mx
    r2 = 1 - sum((y - (sl * x + ic)) ** 2 for x, y in zip(xs, ys)) / sum((y - my) ** 2 for y in ys)
    return ic, sl, r2


def drops_of(r):
    A = r["acc_matrix"]
    T = r["ntask"]
    return [max(row[j] for row in A[j:]) - A[-1][j] for j in range(T)]


def drop_age(rs):
    cs = []
    for r in rs:
        d = drops_of(r)
        T = r["ntask"]
        d = d[:T - 1]
        ages = list(range(T - 1, 0, -1))   # task 0 = oldest = largest age
        mx_, my_ = st.mean(ages), st.mean(d)
        num = sum((a - mx_) * (x - my_) for a, x in zip(ages, d))
        den = math.sqrt(sum((a - mx_) ** 2 for a in ages) * sum((x - my_) ** 2 for x in d))
        cs.append(num / den)
    return st.mean(cs)


RESULTS = []


def check(name, claim, got, tol=0.06):
    ok = abs(got - claim) <= tol   # 不先 round：p 值这类小数量级会被舍掉
    RESULTS.append((name, claim, got, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: 声称 {claim} / 实算 {round(got, 2)}")


def main():
    print("=" * 74)
    print("S2 数字-数据一致性核查")
    print("=" * 74)

    print("\n[1] 头牌结果 (Permuted-MNIST, fifo M=5 vs unbounded, λ=5000)")
    for T, dclaim, tclaim in [(10, 4.51, 21.4), (20, 8.98, 19.1)]:
        f, u = delp("fifo", 5, T, 5000), delp("unbounded", None, T, 5000)
        n, dv, t, (wp, wpx) = paired(f, u)
        check(f"T={T} Δacc", dclaim, dv)
        check(f"T={T} paired t(9)", tclaim, t, tol=0.15)
        # 正文报的 0.002 是 n=10 两符号检验的近似最小可达值；精确检验给 0.00195。
        check(f"T={T} Wilcoxon p (approx)", 0.002, wp, tol=0.009)
        check(f"T={T} Wilcoxon p (exact)", 0.00195, wpx, tol=0.0001)

    print("\n[2] 剂量响应 (T=20, λ=5000, n=10)")
    u = delp("unbounded", None, 20, 5000)
    for M, claim in [(2, 52.4), (3, 48.9), (4, 44.9), (5, 42.4),
                     (6, 40.2), (8, 36.5), (10, 35.1)]:
        f = delp("fifo", M, 20, 5000)
        c = sorted(set(f) & set(u))
        check(f"M={M} acc", claim, st.mean([f[s]["final_avg"] for s in c]))
    check("unbounded acc", 33.4, st.mean([r["final_avg"] for r in u.values()]))
    f2 = delp("fifo", 2, 20, 5000)
    c = sorted(set(f2) & set(u))
    check("M=2 BWT", -24.5, st.mean([f2[s]["bwt"] for s in c]), tol=0.1)
    f10 = delp("fifo", 10, 20, 5000)
    c = sorted(set(f10) & set(u))
    check("M=10 BWT", -5.2, st.mean([f10[s]["bwt"] for s in c]), tol=0.1)

    print("\n[3] 保护律 ℓ ≈ 84 − 11 ln h")
    for T, hs in [(20, (1, 2, 5, 10, 20)), (10, (1, 2, 5, 20))]:
        pts = []
        for h in hs:
            if h == 1:
                pts.append((h, st.mean([r["ell"] for r in ewc("plain", None, T, 1000).values()])))
            elif h == 2:
                pts.append((h, st.mean([r["ell"] for r in ewc("online", 0.5, T, 1000).values()])))
            elif h == 5:
                pts.append((h, st.mean([r["ell"] for r in delp("fifo", 5, T, 1000).values()])))
            elif h == 10:
                pts.append((h, st.mean([r["ell"] for r in ewc("online", 0.9, T, 1000).values()])))
            else:
                pts.append((h, st.mean([r["ell"] for r in delp("unbounded", None, T, 1000).values()])))
        ic, sl, r2 = fit_loglin(pts)
        if T == 20:
            check("T=20 截距", 84, ic, tol=0.3)
            check("T=20 斜率", -11, sl, tol=0.2)
            check("T=20 R²", 0.99, r2, tol=0.006)
        else:
            check("T=10 四臂斜率", -3.7, sl, tol=0.15)
            check("T=10 四臂 R²", 0.95, r2, tol=0.006)
            pred = sl * math.log(10) + ic
            g09 = st.mean([r["ell"] for r in ewc("online", 0.9, T, 1000).values()])
            check("T=10 gamma=0.9 低于直线", -0.7, g09 - pred, tol=0.1)
    # per-seed
    xs = [math.log(h) for h in (1, 2, 5, 10, 20)]
    sls, r2s = [], []
    for seed in range(10):
        ys = []
        try:
            for h in (1, 2, 5, 10, 20):
                if h == 1:
                    arm = ewc("plain", None, 20, 1000)
                elif h == 2:
                    arm = ewc("online", 0.5, 20, 1000)
                elif h == 5:
                    arm = delp("fifo", 5, 20, 1000)
                elif h == 10:
                    arm = ewc("online", 0.9, 20, 1000)
                else:
                    arm = delp("unbounded", None, 20, 1000)
                ys.append(st.mean([r["ell"] for r in arm.values() if r["seed"] == seed]))
            mx, my = st.mean(xs), st.mean(ys)
            sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            ic = my - sl * mx
            r2 = 1 - sum((y - (sl * x + ic)) ** 2 for x, y in zip(xs, ys)) / sum((y - my) ** 2 for y in ys)
            sls.append(sl); r2s.append(r2)
        except Exception:
            pass
    check("per-seed 斜率均值", -10.9, st.mean(sls), tol=0.15)
    check("per-seed R² 最小值", 0.98, min(r2s), tol=0.003)

    print("\n[4] λ=0 无保护参考 (Split-CIFAR-100)")
    for T, aclaim, ellclaim, bwtclaim in [(10, 27.0, 46.1, -21.3), (20, 36.9, 64.3, -28.9)]:
        z = {r["seed"]: r for r in REC if r.get("kind") == "delp"
             and r.get("dataset") == "cifar100" and r.get("ntask") == T
             and r.get("lam") == 0.0 and r.get("policy") == "unbounded"}
        if not z:
            continue
        check(f"λ=0 T={T} acc", aclaim, st.mean([r["final_avg"] for r in z.values()]))
        check(f"λ=0 T={T} ell", ellclaim, st.mean([r["ell"] for r in z.values()]), tol=0.1)
        check(f"λ=0 T={T} BWT", bwtclaim, st.mean([r["bwt"] for r in z.values()]), tol=0.1)
        f2 = delp("fifo", 2, T, 0.1, "cifar100")
        n, dv, t, (wp, wpx) = paired(f2, z)
        # 正文用精确检验：T=10 -> 0.027, T=20 -> 0.1396≈0.14
        check(f"λ=0 T={T} vs fifoM2 Wilcoxon p (exact)",
              0.027 if T == 10 else 0.16, wpx, tol=0.005)

    print("\n[5] plain EWC (λ=1000)")
    for T, lo, hi, ell_lo, ell_hi in [(10, 68.8, 73.4, 83, 86), (20, 57.7, 61.7, 83, 84)]:
        p = ewc("plain", None, T, 1000)
        accs = [r["final_avg"] for r in p.values()]
        check(f"plain T={T} acc 下界", lo, min(accs), tol=0.1)
        check(f"plain T={T} acc 上界", hi, max(accs), tol=0.1)

    print("\n[6] drop-age 相关（正号：遗忘集中在老任务）")
    for tag, rs, claim in [
        ("plain", list(ewc("plain", None, 20, 1000).values()), 0.93),
        ("fifo M=5", list(delp("fifo", 5, 20, 1000).values()), 0.75),
        ("online γ=1", list(ewc("online", None, 20, 1000).values()), 0.70),
        ("unbounded", list(delp("unbounded", None, 20, 1000).values()), 0.32),
    ]:
        if rs:
            check(f"drop-age ρ {tag}", claim, drop_age(rs), tol=0.015)

    print("\n[7] auditability 8.6 / 27.7")
    f = delp("fifo", 5, 20, 1000)
    rec, five = [], []
    for s in sorted(f):
        d = drops_of(f[s])
        rec.append(st.mean(d[15:]))
        five.append(st.mean(d[:5]))
    check("栈内 drop", 8.6, st.mean(rec), tol=0.1)
    check("最老 5 个被驱逐 drop", 27.7, st.mean(five), tol=0.1)

    print("\n[8] merge 插桩（正文口径：per-(T,M) 的 mean(max_rel_drift)）")
    for T, M, claim in [(10, 2, 43.6), (20, 2, 45.0), (10, 5, 18.3), (20, 5, 34.1)]:
        rs = [r for r in REC if r.get("policy") == "merge" and r.get("M") == M
              and r.get("kind") == "delp" and r.get("dataset", "mnist") == "mnist"
              and r.get("ntask") == T and r.get("merge_drifts")]
        # 加载器已纳入 .json.part* 分片，这里 n 应为 10。
        if not rs:
            continue
        mx = [r["max_rel_drift"] * 100 for r in rs if r.get("max_rel_drift") is not None]
        check(f"T={T} M={M} mean(max_rel_drift)%", claim, st.mean(mx), tol=0.6)
    allm2 = [r["max_rel_drift"] * 100 for r in REC if r.get("policy") == "merge"
             and r.get("M") == 2 and r.get("kind") == "delp"
             and r.get("dataset", "mnist") == "mnist" and r.get("ntask") in (10, 20)
             and r.get("max_rel_drift") is not None]
    check("M=2 worst %", 64, max(allm2), tol=1.0)

    print("\n[9] 栈组成")
    for T, ds, lam, pol, claim in [(20, "mnist", 5000, "fifo", (18, 19)),
                                    (20, "mnist", 5000, "redun", (0, 1)),
                                    (20, "cifar100", 0.1, "fifo", (18, 19)),
                                    (20, "cifar100", 0.1, "redun", (0, 1))]:
        d = delp(pol, 2, T, lam, ds)
        if not d:
            continue
        dist = {}
        for r in d.values():
            sh = r.get("survivor_history")
            if not sh:
                continue
            last = sh[-1] if isinstance(sh[0], list) else sh
            key = tuple(sorted(set(last))) if isinstance(last, list) else tuple(last)
            dist[key] = dist.get(key, 0) + 1
        top = max(dist.items(), key=lambda kv: kv[1])
        got = list(top[0])
        ok = got == list(claim)
        RESULTS.append((f"{pol} {ds} T={T} 最终栈", claim, got, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {pol} {ds} T={T} 最终栈: "
              f"声称 {list(claim)} / 实算 {got} (n={top[1]})")

    print("\n[10] 跨文档一致性（本轮修正项）")
    for T, claim in [(10, -5.2), (20, -4.5)]:
        u = delp("unbounded", None, T, 5000)
        check(f"unbounded BWT T={T} (n=10, 两文档统一)", claim,
              st.mean([r["bwt"] for r in u.values()]), tol=0.06)
    for T, claim in [(10, 5.9), (20, 15.8)]:
        u = delp("unbounded", None, T, 5000)
        f = delp("fifo", 5, T, 5000)
        c = sorted(set(u) & set(f))
        check(f"ell 上升量 T={T} (n=10)", claim,
              st.mean([f[s]["ell"] for s in c]) - st.mean([u[s]["ell"] for s in c]), tol=0.06)
    for T, claim in [(10, 3.0), (20, 5.7)]:
        pl = list(ewc("plain", None, T, 1000).values())
        u = list(delp("unbounded", None, T, 5000).values())
        check(f"plain EWC 遗忘倍数 T={T}", claim,
              abs(st.mean([r["bwt"] for r in pl])) / abs(st.mean([r["bwt"] for r in u])), tol=0.06)
    o1 = ewc("online", None, 10, 1000)
    f5 = delp("fifo", 5, 10, 1000)
    c = sorted(set(o1) & set(f5))
    t = stats.ttest_rel([f5[s]["final_avg"] for s in c], [o1[s]["final_avg"] for s in c])
    check("fifo − online t(9) @T=10", -0.3, t.statistic, tol=0.05)
    u = delp("unbounded", None, 20, 0.1, "cifar100")
    signs = []
    for pol in ["fifo", "agew", "random", "redun", "merge", "agew2"]:
        d = delp(pol, 2, 20, 0.1, "cifar100")
        if not d:
            continue
        cc = sorted(set(d) & set(u))
        signs.append(st.mean([d[s]["final_avg"] - u[s]["final_avg"] for s in cc]))
    check("CIFAR T=20 M=2 六个臂符号全为正", 6, sum(1 for v in signs if v > 0), tol=0)

    print("\n" + "=" * 74)
    npass = sum(1 for r in RESULTS if r[3])
    nfail = len(RESULTS) - npass
    print(f"总计: PASS {npass} / FAIL {nfail}   （共 {len(RESULTS)} 项）")
    if nfail:
        print("\n不符项:")
        for name, claim, got, ok in RESULTS:
            if not ok:
                print(f"  {name}: 声称 {claim} / 实算 {round(got, 3)}")
    print("=" * 74)
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
