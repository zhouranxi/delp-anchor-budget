#!/usr/bin/env python
"""Identify the exact statistic behind the paper's \\own column."""
import glob
import json
import os
import statistics as st
from collections import defaultdict

RES = "C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results"
RES2 = "C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit/s1_dose_base"
PAPER = {
    ("mnist", 10, "neuron", 1.0): 0.19, ("mnist", 10, "neuron", 3.0): 0.17,
    ("mnist", 10, "neuron", 10.0): 0.16, ("mnist", 10, "branch", 10.0): 0.16,
    ("fashion", 10, "neuron", 1.0): 0.15, ("fashion", 10, "neuron", 3.0): 0.14,
    ("fashion", 10, "neuron", 10.0): 0.13, ("fashion", 10, "branch", 3.0): 0.14,
    ("mnist", 20, "neuron", 1.0): 0.14, ("mnist", 20, "neuron", 3.0): 0.14,
    ("mnist", 20, "neuron", 10.0): 0.13, ("mnist", 20, "branch", 3.0): 0.14,
    ("mnist", 20, "branch", 10.0): 0.13, ("fashion", 20, "neuron", 1.0): 0.14,
    ("fashion", 20, "neuron", 3.0): 0.14, ("fashion", 20, "neuron", 10.0): 0.13,
    ("fashion", 20, "branch", 3.0): 0.13, ("fashion", 20, "branch", 10.0): 0.13,
}


def load():
    out = defaultdict(dict)
    for base in (RES, RES2):
        for pat in ("c2_*.json.part*", "c2t20_*.json.part*", "s3e2_t20_*.json",
                    "c2_*.json", "c2t20_*.json"):
            for f in sorted(glob.glob(os.path.join(base, pat))):
                fn = os.path.basename(f)
                if ".part" in fn and fn.endswith(".json"):
                    continue
                try:
                    d = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                rs = d.get("results", []) if isinstance(d, dict) else (
                    d if isinstance(d, list) else [])
                if not isinstance(rs, list):
                    continue
                for r in rs:
                    if not isinstance(r, dict) or r.get("error"):
                        continue
                    ob = r.get("own_bias")
                    if ob is None:
                        if "_b0" not in fn:
                            continue
                        ob = 0.0
                    else:
                        ob = float(ob)
                    key = (r.get("dataset"), r.get("ntask"), r.get("gran"), ob)
                    out[key].setdefault(r.get("seed"), r)
    return out


def phase_share(rec, idx):
    """own-segment share at phase idx (0 = first, -1 = last)."""
    dp = rec.get("dom_per_phase") or []
    if not dp:
        return None
    d = dp[idx]
    for k in ("dom_hist_active", "dom_hist"):
        h = d.get(k)
        if h and sum(h):
            return h[0] / sum(h)
    return None


def phase_share_diag(rec, idx, use_active=True):
    """p_own at phase idx: dom_hist[p][p] / sum(dom_hist[p])  (own task = phase p)."""
    dp = rec.get("dom_per_phase") or []
    if not dp or idx >= len(dp):
        return None
    d = dp[idx]
    for k in (("dom_hist_active", "dom_hist") if use_active else ("dom_hist", "dom_hist_active")):
        h = d.get(k)
        if h and sum(h):
            return h[idx] / sum(h)
    return None


def phase0_all(rec):
    dp = rec.get("dom_per_phase") or []
    if not dp:
        return None
    h = dp[0].get("dom_hist") or []
    if not h or not sum(h):
        return None
    return h[0] / sum(h)


def main():
    data = load()
    cands = {
        "phase0": lambda r: phase_share(r, 0),
        "phase-1(last)": lambda r: phase_share(r, -1),
        "mean(all phases)": lambda r: st.mean(
            [phase_share(r, i) for i in range(len(r.get("dom_per_phase") or []))
             if phase_share(r, i) is not None]) if r.get("dom_per_phase") else None,
        "max over phases": lambda r: max(
            [phase_share(r, i) for i in range(len(r.get("dom_per_phase") or []))
             if phase_share(r, i) is not None]) if r.get("dom_per_phase") else None,
        "phase0 of dom_hist(all)": lambda r: phase0_all(r),
        "diag phase0 (active)": lambda r: phase_share_diag(r, 0, True),
        "diag mean(all phases)": lambda r: st.mean(
            [x for x in (phase_share_diag(r, i, True)
                         for i in range(len(r.get("dom_per_phase") or [])))
             if x is not None]) if r.get("dom_per_phase") else None,
        "diag max over phases": lambda r: max(
            [x for x in (phase_share_diag(r, i, True)
                         for i in range(len(r.get("dom_per_phase") or [])))
             if x is not None]) if r.get("dom_per_phase") else None,
        "diag phase0 (all neurons)": lambda r: phase_share_diag(r, 0, False),
    }
    print("=== 各口径对 18 格的吻合数 ===")
    for nm, fn in cands.items():
        exact = 0
        near = 0
        for key, pw in PAPER.items():
            arm = data.get(key, {})
            if not arm:
                continue
            vals = [fn(r) for r in arm.values()]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m = st.mean(vals)
            if abs(round(m, 2) - pw) < 0.006:
                exact += 1
            if abs(m - pw) <= 0.021:
                near += 1
        print(f"  {nm:22} 精确(±0.005) {exact:2}/18   近似(±0.02) {near:2}/18")
    print()
    print("=== phase0 口径的逐格偏差 ===")
    fn = cands["phase0"]
    for key, pw in sorted(PAPER.items()):
        arm = data.get(key, {})
        if not arm:
            continue
        vals = [fn(r) for r in arm.values()]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        m = st.mean(vals)
        flag = "OK" if abs(round(m, 2) - pw) < 0.006 else f"(+{m-pw:+.3f})"
        print(f"  {str(key):34} paper={pw:.2f}  phase0={m:.3f}  {flag}")


if __name__ == "__main__":
    main()
