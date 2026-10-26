#!/usr/bin/env python
"""Check the significance stars in S1 Tables 5/6 against the recomputed t(9).

The paper marks a cell with * when |t(9)| >= 2.262 (df=9, two-sided 0.05).
The dose tables are located by their header row ("arm ($b$)").
"""
import io
import re

TEX = io.open("C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/"
              "delp-anchor-budget-main/S1_剂量响应审计/paper/main.tex",
              encoding="utf-8").read()
CRIT = 2.0     # the paper's caption states: *: |t| > 2

# recomputed paired t(9): (d_ell, d_avg, d_bwt)
TT = {
    ("mnist", 10, "neuron", 1): (8.1, 11.0, 9.4),
    ("mnist", 10, "neuron", 3): (13.3, 18.4, 10.0),
    ("mnist", 10, "neuron", 10): (14.7, 17.0, 6.7),
    ("mnist", 10, "branch", 10): (10.5, 13.0, 5.6),
    ("mnist", 10, "branch", 3): (11.8, 16.2, 10.7),
    ("fashion", 10, "branch", 1): (2.1, 2.5, 1.6),
    ("fashion", 10, "branch", 10): (2.0, 2.4, 0.2),
    ("fashion", 10, "neuron", 1): (1.7, 1.6, 0.2),
    ("fashion", 10, "neuron", 3): (2.7, 3.3, 2.1),
    ("fashion", 10, "neuron", 10): (2.4, 1.5, -1.1),
    ("fashion", 10, "branch", 3): (1.7, 2.4, 2.1),
    ("mnist", 20, "neuron", 1): (3.9, 7.3, 2.6),
    ("mnist", 20, "neuron", 3): (7.5, 9.2, -1.4),
    ("mnist", 20, "neuron", 10): (8.1, 10.6, -2.8),
    ("mnist", 20, "branch", 3): (9.3, 12.9, -2.045),
    ("mnist", 20, "branch", 10): (7.8, 8.1, -1.4),
    ("fashion", 20, "neuron", 1): (4.1, 3.8, -0.5),
    ("fashion", 20, "neuron", 3): (5.8, 6.6, -0.4),
    ("fashion", 20, "neuron", 10): (5.4, 3.7, -2.7),
    ("fashion", 20, "branch", 3): (3.6, 6.4, 1.6),
    ("fashion", 20, "branch", 10): (5.5, 7.4, -0.5),
}

TAB_RE = re.compile(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", re.S)
ROW_RE = re.compile(r"(\w+)\s*\(\$(\d+)\$\)")


def dose_tables():
    """-> [(T, rows)] for every tabular whose header mentions the dose arm."""
    out = []
    for m in TAB_RE.finditer(TEX):
        seg = m.group(1)
        if "arm ($b$)" not in seg:
            continue
        # the two dose tables appear in manuscript order: T=10 first
        T = 10 if not out else 20
        rows = []
        for l in seg.split("\n"):
            if l.startswith("MNIST") or l.startswith("Fashion"):
                cells = [c.strip() for c in l.replace("\\\\", "").split("&")]
                if len(cells) >= 6:
                    rows.append(cells)
        out.append((T, rows))
    return out


def main():
    print(f"significance rule: |t(9)| > {CRIT}  ->  *   (paper caption)\n")
    print(f"{'cell':44} {'paper':6} {'recalc':6} verdict")
    mism = []
    for Tb, rows in dose_tables():
        for cells in rows:
            ds = "mnist" if cells[0].startswith("MNIST") else "fashion"
            mm = ROW_RE.search(cells[1])
            if not mm:
                print("  UNPARSED:", cells[:2])
                continue
            gran, b = mm.group(1), int(mm.group(2))
            pe = "*" in cells[2]
            pa = "*" in cells[3]
            pb = "*" in cells[4]
            te, ta, tb = TT[(ds, Tb, gran, b)]
            re_ = abs(te) > CRIT
            ra = abs(ta) > CRIT
            rb = abs(tb) > CRIT
            ok = (pe == re_ and pa == ra and pb == rb)
            mk = lambda x: "*" if x else "."
            if not ok:
                mism.append((ds, Tb, gran, b, mk(pe) + mk(pa) + mk(pb),
                             mk(re_) + mk(ra) + mk(rb), te, ta, tb))
            print(f"{ds:8} T={Tb:2} {gran:7} b={b:3}   "
                  f"{mk(pe)+mk(pa)+mk(pb):6} {mk(re_)+mk(ra)+mk(rb):6} "
                  f"{'OK' if ok else '<<< MISMATCH'}")
    print(f"\nmismatched cells: {len(mism)}")
    for r in mism:
        print(f"  {r[0]:8} T={r[1]:2} {r[2]:7} b={r[3]:3}  paper {r[4]} vs "
              f"recalc {r[5]}  (t={r[6]:+.1f},{r[7]:+.1f},{r[8]:+.1f})")


if __name__ == "__main__":
    main()
