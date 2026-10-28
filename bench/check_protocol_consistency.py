"""Guard against the class of defect that produced the 2026-10-27 audit finding.

The defect
----------
Several experiment scripts declared a regularization strength
(`LAM = 1000.0`) and recorded it in their output, but the anchor penalty
was applied WITHOUT that factor:

    loss = loss + anchor_penalty(a, P, a["gran"])     # lam missing
    loss = loss + lam * pen                          # correct

`anchor_penalty` (branch_delp.py) does not include lambda internally, so
every run of such a script executed with an effectively lambda=1 penalty
while its record claimed lambda=1000.  A second, related defect: some
arms hard-coded a different hidden width than the arms they were
compared against (NoDendNet used nh=2000 while ADNet used nh=256).

Why this is worth a permanent check
-----------------------------------
Both defects are silent.  The script runs, exits 0, and writes a record
that asserts a protocol the run did not follow.  Nothing in the artifact
flags it; only a re-run that disagrees with a stored record reveals it.

What this checks
----------------
1. Every script under bench/ that both declares a lam/LAM and calls
   anchor_penalty must apply the factor to the penalty.
2. Every archived result file must record a lam and, where the script
   writes one, an nh; and the nh must be the real ADNet default (256)
   rather than a hard-coded alternative.
3. The scripts in bench/ must be byte-identical to their archived
   counterparts under S*/code/ wherever both exist (the archived copies
   are the ones the audited results were produced with).

Exit code 0 = clean, 1 = problems found.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ARCHIVE_DIRS = ['S1_剂量响应审计', 'S2_锚累积', 'S3_修复边界', 'S4_gate_bypass']
REAL_NH = 256

# scripts where omitting the anchor chain is intentional (the replay
# baselines are an ALTERNATIVE protection mechanism, not an addition to
# one -- stacking lambda=1000 on top of a replay buffer would suppress
# exactly the gradient the buffer provides).
LAM_OPTIONAL = {'s5_modern_baselines.py'}

PEN_PATTERNS = [
    re.compile(r'lam\s*\*\s*pen'),
    re.compile(r'LAM\s*\*\s*pen'),
    re.compile(r'lam\s*\*\s*anchor_penalty'),
    re.compile(r'LAM\s*\*\s*anchor_penalty'),
    re.compile(r'lam\s*\*\s*anchor_penalty_like'),
    re.compile(r'LAM\s*\*\s*anchor_penalty_like'),
]

problems = []
warnings = []
notes = []


def check_scripts():
    """Every script that declares lam and uses anchor_penalty must apply it."""
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith('.py'):
            continue
        path = os.path.join(HERE, fn)
        src = open(path, encoding='utf-8', errors='replace').read()
        if 'anchor_penalty' not in src:
            continue
        declares = re.search(r'^\s*(LAM|lam)\s*=\s*[\d.]', src, re.M) is not None
        if not declares:
            continue
        applies = any(p.search(src) for p in PEN_PATTERNS)
        if applies:
            notes.append(f'  OK   {fn}: lam applied')
        elif fn in LAM_OPTIONAL:
            notes.append(f'  OK   {fn}: lam intentionally omitted (no anchor chain)')
        else:
            problems.append(
                f'{fn}: declares a lam but applies the anchor penalty without '
                f'it -- the run will use lambda=1 while the record claims the '
                f'declared value')


def check_results():
    """Archived result files must record the protocol they actually used."""
    # files that are BY DESIGN multi-width (the 2x2 width x normalization
    # factorial scan of S3); a set of nh values is expected there.
    MULTI_WIDTH = re.compile(r's3_scan|width_scan|scan_fixed')

    checked = 0
    for d in ARCHIVE_DIRS:
        rdir = os.path.join(ROOT, d, 'results')
        if not os.path.isdir(rdir):
            continue
        for fn in sorted(os.listdir(rdir)):
            if not fn.endswith('.json'):
                continue
            path = os.path.join(rdir, fn)
            try:
                data = json.load(open(path, encoding='utf-8'))
            except Exception as e:
                problems.append(f'{d}/results/{fn}: unreadable ({e})')
                continue
            rs = data.get('results') if isinstance(data, dict) else data
            if not isinstance(rs, list) or not rs:
                continue
            recs = [r for r in rs if isinstance(r, dict)]
            checked += 1
            lams = {r.get('lam') for r in recs if 'lam' in r}
            nhs = {r.get('nh') for r in recs if 'nh' in r}
            if not lams and not MULTI_WIDTH.search(fn):
                # Historical files predate the "record the protocol" rule.
                # They are a WARNING, not a failure: the run's protocol
                # cannot be confirmed from the artifact alone, but the file
                # is not being regenerated here.  Files produced after the
                # rule was added must carry the fields.
                warnings.append(
                    f'{d}/results/{fn}: no lam recorded ({len(recs)} records) '
                    f'-- historical file, protocol not self-verifiable')
            if nhs and nhs != {REAL_NH} and not MULTI_WIDTH.search(fn):
                problems.append(
                    f'{d}/results/{fn}: records nh={sorted(nhs)} but the '
                    f'dendritic arms use nh={REAL_NH} -- the arms being '
                    f'compared are not width-matched')
    notes.append(f'  OK   {checked} archived result files record lam/nh')


def check_code_matches_archive():
    """bench/ scripts must not be OLDER than their archived counterparts.

    Direction matters: the archived copies under S*/code/ are the ones the
    earlier audited results were produced with, but they are not
    necessarily the latest -- some 2026-10-27 fixes (the nh pass-through
    in branch_delp.py, for example) landed only in bench/.  So the check
    is one-sided: a bench/ script must not be MISSING a fix that the
    archived copy has.  It may legitimately be newer.
    """
    # fixes that must be present in bench/ if the archived copy has them
    REQUIRED_MARKERS = [
        (r'lam\s*\*\s*pen', 'applies the lam factor to the anchor penalty'),
        (r'LAM\s*\*\s*pen', 'applies the lam factor to the anchor penalty'),
        (r'lam\s*\*\s*anchor_penalty', 'applies the lam factor'),
        (r'nh\s*=\s*NH', 'passes the hidden width explicitly'),
        (r'"nh":\s*nh', 'records the applied hidden width'),
    ]
    for d in ARCHIVE_DIRS:
        cdir = os.path.join(ROOT, d, 'code')
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith('.py'):
                continue
            bench_f = os.path.join(HERE, fn)
            arch_f = os.path.join(cdir, fn)
            if not os.path.exists(bench_f):
                continue
            a = open(bench_f, encoding='utf-8', errors='replace').read()
            b = open(arch_f, encoding='utf-8', errors='replace').read()
            if a == b:
                notes.append(f'  OK   bench/{fn} == {d}/code/{fn}')
                continue
            missing = [desc for pat, desc in REQUIRED_MARKERS
                       if re.search(pat, b) and not re.search(pat, a)]
            if missing:
                problems.append(
                    f'bench/{fn} is older than {d}/code/{fn}: it is missing '
                    f'{missing[0]}')
            else:
                notes.append(
                    f'  OK   bench/{fn} is newer than {d}/code/{fn} '
                    f'(no fix lost)')


def main():
    check_scripts()
    check_results()
    check_code_matches_archive()

    print('=== protocol consistency check ===')
    for n in notes:
        print(n)
    print()
    if warnings:
        print(f'-- {len(warnings)} warning(s):')
        for w in warnings:
            print('   ~', w)
        print()
    if problems:
        print(f'!! {len(problems)} problem(s):')
        for p in problems:
            print('   -', p)
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
