# -*- coding: utf-8 -*-
"""Publish the v8 package to Zenodo as a new record and chain the versions.

Lessons from the earlier rounds, encoded here:

  * A published Zenodo record cannot gain files.  Adding a file to an
    existing record returns 404 ("has no file"), then 400 ("Invalid input
    type" -- the /files endpoint wants a LIST), then 403 ("Bucket is
    locked").  The only way to change the content is to publish a NEW
    record.  So v8 is a fresh record, not an edit of v7.
  * POST /records/{concept}/versions does NOT reuse the concept; each call
    creates a fresh one, so the version chain has to be stitched by hand
    through related_identifiers.
  * The package directory must come from the environment.  An earlier
    script had it hard-coded, so a freshly prepared package was silently
    ignored and the previous contents were uploaded instead.

The upload itself is: init all file entries in one POST, then PUT +
commit each file, verifying the md5 that Zenodo reports back.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://zenodo.org/api'
TOK = json.load(open(os.path.expanduser('~/.config/zenodo.json')))['token']
PKG = os.environ.get(
    'ZENODO_PKG',
    r'C:\Users\Lance\WorkBuddy\2026-10-25-10-08-27\_zenodo_v8')
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def call(m, p, body=None, raw=None, ctype='application/json'):
    data = raw if raw is not None else (
        json.dumps(body).encode('utf-8') if body is not None else None)
    r = urllib.request.Request(BASE + p, data=data, method=m)
    if data is not None:
        r.add_header('Content-Type', ctype)
    r.add_header('Authorization', 'Bearer ' + TOK)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            t = resp.read().decode('utf-8', 'replace')
            try:
                return resp.status, json.loads(t)
            except Exception:
                return resp.status, t[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace')[:400]
    except Exception as e:
        return -1, str(e)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return 'md5:' + h.hexdigest()


def walk(root):
    out = []
    for base, _, fs in os.walk(root):
        for f in sorted(fs):
            full = os.path.join(base, f)
            rel = os.path.relpath(full, root).replace('\\', '/')
            if rel in ('log.txt', 'result.json'):
                continue
            out.append((rel, full))
    return sorted(out)


def to_html(text):
    paras = [x.strip() for x in (text or '').split('\n\n') if x.strip()]
    return ''.join('<p>' + x.replace('\n', ' ') + '</p>' for x in paras)


def main():
    meta = json.load(open(os.path.join(PKG, 'paper', 'zenodo_metadata.json'),
                          encoding='utf-8'))
    version = str(meta.get('version', '8'))

    creators = [{
        'person_or_org': {'type': 'personal',
                           'family_name': c['family_name'],
                           'given_name': c['given_name']},
        'role': {'id': 'researcher'},
        'affiliations': [{'name': c['affiliation']}]}
        for c in meta['creators']]

    rels = [{'identifier': e['identifier'], 'scheme': e['scheme'],
             'relation_type': {'id': e['relation']}}
            for e in meta['related_identifiers']]

    payload = {
        'title': meta['title'],
        'description': to_html(meta['description']),
        'publication_date': meta['publication_date'],
        'publisher': 'Zenodo',
        'resource_type': {'id': 'publication-preprint'},
        'creators': creators,
        'rights': [{'id': meta['license']['id']}],
        'subjects': [{'subject': k} for k in meta['keywords']],
        'related_identifiers': rels,
        'version': version,
    }

    files = walk(PKG)
    log(f'package   : {PKG}')
    log(f'files     : {len(files)}')
    log(f'version   : {version}')
    log(f'title     : {meta["title"][:70]}')
    if len(files) < 50:
        log('⚠️ 文件数异常少，检查打包是否完整')

    s, d = call('POST', '/records', body={'metadata': payload})
    if s not in (200, 201):
        log(f'❌ POST /records 失败: {s} {str(d)[:400]}')
        return
    new_id = d['id']
    log(f'新记录 id = {new_id}')

    time.sleep(2)

    # init all file entries in ONE request (the endpoint takes a list)
    for a in range(3):
        s, f = call('POST', f'/records/{new_id}/draft/files',
                    body=[{'key': rel, 'size': os.path.getsize(full)}
                          for rel, full in files])
        if s in (200, 201):
            log('init 成功')
            break
        if 'already exists' in str(f):
            log('init: 条目已存在，跳过')
            break
        log(f'  init -> {s} {str(f)[:150]}')
        time.sleep(4)
    time.sleep(3)

    s, fl = call('GET', f'/records/{new_id}/draft/files?size=250')
    ent = {e.get('key'): e for e in
           ((fl.get('entries') or []) if isinstance(fl, dict) else [])}

    ok = 0
    failed = []
    for rel, full in files:
        want = md5(full)
        if ent.get(rel, {}).get('checksum') == want:
            log(f'  SKIP {rel}')
            ok += 1
            continue
        data = open(full, 'rb').read()
        done = False
        for a in range(5):
            s2, _ = call(
                'PUT',
                f'/records/{new_id}/draft/files/{urllib.parse.quote(rel)}/content',
                raw=data, ctype='application/octet-stream')
            if s2 != 200:
                log(f'  PUT {rel} -> {s2}')
                time.sleep(3)
                continue
            time.sleep(2)
            s3, _ = call(
                'POST', f'/records/{new_id}/draft/files/{urllib.parse.quote(rel)}/commit')
            if s3 == 200:
                time.sleep(1)
                s4, fl2 = call('GET', f'/records/{new_id}/draft/files?size=250')
                e = {x.get('key'): x for x in
                     ((fl2.get('entries') or []) if isinstance(fl2, dict) else [])}
                if e.get(rel, {}).get('checksum') == want:
                    log(f'  OK   {rel} ({len(data)}B)')
                    ok += 1
                    done = True
                    break
            else:
                log(f'  commit {rel} -> {s3}')
                time.sleep(3)
        if not done:
            failed.append(rel)
    log(f'上传 {ok}/{len(files)}')
    if failed:
        log(f'❌ 失败文件: {failed}')

    s, v = call('POST', f'/records/{new_id}/draft/actions/publish')
    log(f'publish -> {s}')
    if s not in (200, 201, 202):
        log(str(v)[:400])
        return
    time.sleep(12)

    s, rv = call('GET', f'/records/{new_id}')
    m = rv.get('metadata') or {}
    log(f'✅ 回读: id={rv.get("id")} version={m.get("version")} '
        f'doi={rv.get("doi")} concept={rv.get("conceptrecid")} '
        f'files={len(rv.get("files") or [])}')

    out = {'id': rv.get('id'), 'doi': rv.get('doi'),
           'concept': rv.get('conceptrecid'),
           'version': m.get('version'), 'files': len(rv.get('files') or []),
           'failed': failed}
    with open(os.path.join(PKG, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(PKG, 'log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
