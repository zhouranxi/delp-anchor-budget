# -*- coding: utf-8 -*-
"""Rebuild the Zenodo relation graph for the v8 chain, with the correct
relation_type vocabulary.

The relation_type ids are DataCite terms and must come from
GET /vocabularies/relationtypes.  The relevant ones are:

    isnewversionof        this record is a new version of the target
    ispreviousversionof    this record is the previous version of the target
    ispartof              the target is a part of this record
    references

An earlier round used `isversionof`, which is NOT in the vocabulary.  Zenodo
accepts the PUT without complaint but silently downgrades every such entry
to `references`, so the chain looked fine in the request and was wrong on
the server.  The vocabulary must be consulted before writing, never guessed.
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE = 'https://zenodo.org/api'
TOK = json.load(open(os.path.expanduser('~/.config/zenodo.json')))['token']

NEW_VERSION = 'isnewversionof'          # NOT "isversionof"
# The repository is a part of the record, not a supplement of it: there is no
# 'issupplementto' in the relationtypes vocabulary.
SUPPLEMENT = 'ispartof'

V8 = '10.5281/zenodo.24017966'
V7 = '10.5281/zenodo.23730915'
GITHUB = 'https://github.com/zhouranxi/delp-anchor-budget'

OLD_DOIS = [
    '10.5281/zenodo.23685865',  # S1
    '10.5281/zenodo.23685385',  # S3
    '10.5281/zenodo.23686689',  # S4
    '10.5281/zenodo.23685906',  # S2
    '10.5281/zenodo.23685788',  # pre-registration audit
    '10.5281/zenodo.23685822',  # data & code
]

OLD_RECORDS = {
    23685788: ("Where Do the Gains Come From? An Audit of Dendritic "
               "Context-Gated Protection for Continual Learning", '2'),
    23685865: ("Segment Specialization in Active-Dendrite Continual Learning: "
               "An Audit, an Artifact, and a Causal Repair", '3'),
    23685906: ("The Anchor Accumulation Problem: Bounded Anchor Sets Recover "
               "Plasticity in Usage-Localized Elastic Weight Consolidation", '2'),
    23685385: ("Repair Headroom, Not Dying Gates: Where the Own-Segment Bias "
               "Transfers, and What Actually Tracks the Gain", '1'),
    23686689: ("The Repair Saturates the Gate: Context Noise as a Degeneracy "
               "Diagnostic", '3'),
    23685822: ("DeLP continual-learning data & code (per-seed records, "
               "pre-registrations, figures)", '3'),
}

# merged-manuscript chain: record id -> (version, next version DOI or None)
CHAIN = [
    (23719801, '1', '10.5281/zenodo.23726912'),
    (23726912, '2', '10.5281/zenodo.23728217'),
    (23728217, '3', '10.5281/zenodo.23729840'),
    (23729840, '4', '10.5281/zenodo.23730592'),
    (23730357, '5', '10.5281/zenodo.23730592'),
    (23730592, '6', V7),
    (23730915, '7', V8),
]

TITLE = ("What an Active-Dendrite Continual Learner Actually Learns: "
         "A Dead-Neuron Artifact, a Granularity Result That Did Not Survive "
         "Its Own Controls, and a Repair That Is a Damping Factor")

KEYWORDS = [
    'continual learning', 'elastic weight consolidation', 'active dendrites',
    'dendritic segments', 'catastrophic forgetting', 'interpretability',
    'measurement validity', 'negative results', 'pre-registration',
    'experience replay', 'dark experience replay', 'equivalence testing',
]


def call(m, p, body=None):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    r = urllib.request.Request(BASE + p, data=data, method=m)
    if data is not None:
        r.add_header('Content-Type', 'application/json')
    r.add_header('Authorization', 'Bearer ' + TOK)
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            t = resp.read().decode('utf-8', 'replace')
            try:
                return resp.status, json.loads(t)
            except Exception:
                return resp.status, t[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace')[:400]
    except Exception as e:
        return -1, str(e)


def valid_relation_ids():
    s, v = call('GET', '/vocabularies/relationtypes')
    if s != 200:
        raise SystemExit('无法读取 relationtypes 词表')
    return {t['id'] for t in v['hits']['hits']}


def rel(ident, relation, scheme='doi'):
    return {'identifier': ident, 'scheme': scheme,
            'relation_type': {'id': relation}}


def public_relations(rec):
    """Read relations through the PUBLIC api and convert them to the
    write schema.

    Two traps here, both of which silently produce a wrong result:
      * the public api returns the relation as a camelCase string
        ('isNewVersionOf'); the write schema wants lower case.
      * the draft endpoint returns a different schema entirely
        ('relation' string), which the PUT rejects with 400.
    """
    s, d = call('GET', f'/records/{rec}')
    if s != 200:
        return []
    out = []
    seen = set()
    for e in ((d.get('metadata') or {}).get('related_identifiers') or []):
        ident = e.get('identifier')
        relation = e.get('relation')
        if isinstance(relation, dict):
            relation = relation.get('id')
        if not ident or ident in seen:
            continue
        seen.add(ident)
        out.append({'identifier': ident,
                    'scheme': e.get('scheme') or 'doi',
                    'relation_type': {'id': (relation or 'references').lower()}})
    return out


def build(entries):
    """De-duplicate by identifier, keeping the LAST relation given."""
    out = {}
    for e in entries:
        out[e['identifier']] = e
    return list(out.values())


def update(rid, title, version, entries):
    call('POST', f'/records/{rid}/draft')
    time.sleep(2)
    s, r = call('GET', f'/records/{rid}/draft')
    if s != 200:
        return f'draft 失败 ({s})'
    old = r.get('metadata') or {}
    desc = old.get('description') or ''
    if isinstance(desc, list):
        desc = '\n'.join(desc)

    payload = {
        'title': title,
        'description': desc,
        'publication_date': old.get('publication_date') or '2026-10-28',
        'publisher': 'Zenodo',
        'resource_type': {'id': 'publication-preprint'},
        'creators': [{'person_or_org': {'type': 'personal',
                                        'family_name': 'Zhou',
                                        'given_name': 'Ranxi'},
                      'role': {'id': 'researcher'},
                      'affiliations': [{'name': 'Hunan University of Chinese '
                                          'Medicine, Changsha, China'}]}],
        'rights': [{'id': 'cc-by-4.0'}],
        # 'subjects' is the field to WRITE; Zenodo normalizes it and the
        # record then reads back under 'keywords'.  Writing 'keywords'
        # directly is accepted but comes back empty.
        'subjects': [{'subject': k} for k in KEYWORDS],
        'related_identifiers': build(entries),
        'version': version,
    }
    s2, v2 = call('PUT', f'/records/{rid}/draft', {'metadata': payload})
    if s2 != 200:
        return f'PUT 失败 ({s2}) {str(v2)[:200]}'
    s3, v3 = call('POST', f'/records/{rid}/draft/actions/publish')
    if s3 not in (200, 201, 202):
        return f'publish 失败 ({s3}) {str(v3)[:200]}'
    return 'OK'


def main():
    valid = valid_relation_ids()
    # 'references' is Zenodo's default relation and is accepted on write even
    # though it is not published in the relationtypes vocabulary, so only the
    # two non-default ids are checked against the vocabulary.
    for r in (NEW_VERSION, SUPPLEMENT):
        if r not in valid:
            raise SystemExit(f'relation id 不在词表里: {r}')
    print(f'词表校验通过: {NEW_VERSION} / {SUPPLEMENT}'
          f'（references 为 Zenodo 默认关系，不在公开词表中但可写入）\n')

    # ---- 1. the merged-manuscript chain -----------------------------------
    print('=== 合并稿版本链 ===')
    for rid, ver, nxt in CHAIN:
        entries = [rel(d, 'references') for d in OLD_DOIS]
        entries.append(rel(GITHUB, SUPPLEMENT, 'url'))
        if nxt:
            entries.append(rel(nxt, NEW_VERSION))
        print(f'  {rid} v={ver}: {update(rid, TITLE, ver, entries)}')
        time.sleep(4)

    # ---- 2. v8 itself -----------------------------------------------------
    print('\n=== v8 ===')
    entries = [rel(V7, NEW_VERSION)]
    entries += [rel(d, 'references') for d in OLD_DOIS]
    entries.append(rel(GITHUB, SUPPLEMENT, 'url'))
    print(f'  24017966 v=8: {update(24017966, TITLE, "8", entries)}')
    time.sleep(4)

    # ---- 3. the six earlier records ---------------------------------------
    print('\n=== 旧六条 -> v8 ===')
    for rid, (title, ver) in OLD_RECORDS.items():
        entries = [rel(d, 'references') for d in OLD_DOIS]
        entries.append(rel(GITHUB, SUPPLEMENT, 'url'))
        entries.append(rel(V7, 'references'))
        entries.append(rel(V8, NEW_VERSION))
        print(f'  {rid} v={ver}: {update(rid, title, ver, entries)}')
        time.sleep(4)


if __name__ == '__main__':
    main()
