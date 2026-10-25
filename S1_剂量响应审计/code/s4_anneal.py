#!/usr/bin/env python
"""S4/E3 退火偏置（anneal/ramp schedule）。

问题（S1 Limitations #2）：constant bias 在 T20 有小而真实的 BWT 代价
（3/8 cells -0.66~-0.95）。退火/渐进 schedule 能否解耦"学习增益"与
"遗忘代价"？

设计（预注册，先于数据落盘）：
  schedule:
    anneal: b_p = b_max * (1 - p/ntask)      # 训练早期强、晚期撤除
    ramp:   b_p = b_max * p / (ntask - 1)    # 早期无干扰、晚期补
  b_max = 10（constant b=10 参照臂已有数据：c2/c2t20）
  T ∈ {10, 20}, datasets ∈ {mnist, fashion}, gran=neuron, seeds 0-9
  共 2 sched × 2 T × 2 ds × 10 seeds = 80 runs。

机制注意：bias 在 forward 中生效（train/eval/anchor 皆读 model.own_bias
当前值），per-phase 更新即可；eval 在 phase 末用当期 b_p。

预注册判据：
  E3 存活：T20 上 anneal 相对 constant(b=10) 的 BWT 显著改善（配对
  ΔΔBWT>0, t>2），且 Δℓ（vs b=0）保留 constant 增益的 ≥50%。
  E3 判死：BWT 无改善 或 Δℓ 保留 <50% → 报"退火不能解开 trade-off"。
"""
import argparse, json, os, sys, time
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

import exp4_branch_protection as e4
import anchor_budget
import exp6_baselines as eb
from branch_delp import (BiasADNet, flat_params,
                        make_anchor, anchor_penalty, _clone)
from exp6_baselines import eval_model, finish, get_perms
from anchor_budget import load_data

B_MAX = 10.0
LAM = 1000.0
SEEDS = list(range(10))


def bias_at(schedule, phase, ntask):
    if schedule == "anneal":
        return B_MAX * (1.0 - phase / ntask)
    if schedule == "ramp":
        return B_MAX * phase / max(ntask - 1, 1)
    raise ValueError(schedule)


def run_anneal(seed, data, perms, ntask, lam, schedule):
    """镜像 branch_delp.run_gran，仅一处不同：每 phase 开始更新
    model.own_bias = bias_at(schedule, phase, ntask)。"""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = BiasADNet(ntask, ntask, own_bias=0.0,
                      bias_mode="fixed").to(eb.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=eb.LR)
    anchors, acc = [], {}
    dom_all = []
    for phase in range(ntask):
        model.own_bias = bias_at(schedule, phase, ntask)   # <-- the only delta
        xs, ys = e4.permute(xtr[:eb.NTRAIN], perms[phase]), ytr[:eb.NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=eb.BS, shuffle=True)
        model.train()
        for _ep in range(eb.EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(eb.DEV), yb.to(eb.DEV)
                ctx = F.one_hot(
                    torch.full((xb.size(0),), phase, device=eb.DEV),
                    ntask).float()
                opt.zero_grad()
                out, seg = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, a["gran"])
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, ntask, "neuron"))
        dom_all.append({k: anchors[-1][k] for k in
                        ("dom_hist", "dom_hist_active", "dead_frac")})
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = eb.finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "schedule": schedule, "gran": "neuron",
            "b_max": B_MAX, "ntask": ntask, "lam": lam,
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)],
            "dom_per_phase": dom_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s4anneal.json")
    ap.add_argument("--ntasks", default="10,20")
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()

    nt_list = [int(s) for s in a.ntasks.split(",")]
    ds_list = a.datasets.split(",")
    jobs = []
    for ds in ds_list:
        for nt in nt_list:
            for sched in ("anneal", "ramp"):
                for seed in SEEDS:
                    jobs.append((ds, nt, sched, seed))
    mine = [j for i, j in enumerate(jobs) if i % a.n_shards == a.shard]
    print(f"[shard {a.shard}/{a.n_shards}] {len(mine)}/{len(jobs)} jobs",
          flush=True)

    results, t0 = [], time.time()
    for ds, nt, sched, seed in mine:
        rec = {"kind": "anneal", "dataset": ds, "ntask": nt,
               "schedule": sched, "seed": seed}
        try:
            data = load_data(ds)
            perms = get_perms(seed, nt)
            rec.update(run_anneal(seed, data, perms, nt, LAM, sched))
        except Exception as exc:                            # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results.append(rec)
        print(f"  {ds} T{nt} {sched} s{seed}: ell={rec.get('ell')} "
              f"avg={rec.get('final_avg')} bwt={rec.get('bwt')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(a.out, "w") as f:
        json.dump({"results": results, "elapsed": time.time() - t0},
                  f, indent=1)
    print(f"\nDONE: {len(results)} runs in {time.time()-t0:.0f}s -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
