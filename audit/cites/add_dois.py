# -*- coding: utf-8 -*-
"""给 refs.bib 补上已核验的 DOI（逐行处理，稳妥）。"""
import re, os

DOIS = {
    'boahen2022': '10.1038/s41586-022-05340-6',
    'poirazi2020': '10.1038/s41583-020-0301-7',
    'masse2018': '10.1073/pnas.1803839115',
    'kirkpatrick2017': '10.1073/pnas.1611835114',
    'parisi2019': '10.1016/j.neunet.2019.01.012',
}

PATHS = [
    r'dab_t/delp-anchor-budget-main/S3_修复边界/paper/refs.bib',
    r'dab_t/delp-anchor-budget-main/S4_gate_bypass/paper/refs.bib',
    r'papers/s3_zenodo/S3_repair_headroom/refs.bib',
    r'papers/s4_zenodo/S4_repair_bypasses_gate/refs.bib',
]

for path in PATHS:
    if not os.path.exists(path):
        print('SKIP missing', path); continue
    lines = open(path, encoding='utf-8').read().split('\n')
    out = []
    cur = None
    added = 0
    for ln in lines:
        m = re.match(r'@article\{([^,]+),', ln)
        if m:
            cur = m.group(1)
        out.append(ln)
        if cur in DOIS and re.match(r'\s*year\s*=', ln):
            # 下一行是否已有 note
            nxt = lines[len(out)] if len(out) < len(lines) else ''
            if 'note' not in nxt:
                out.append(f'  note    = {{doi:{DOIS[cur]}}}')
                added += 1
    open(path, 'w', encoding='utf-8').write('\n'.join(out))
    print(f'{path}: 补 {added} 条 DOI')
