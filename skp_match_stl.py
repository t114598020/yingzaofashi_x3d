"""Models whose SKP is not the STL the page shows: make the SKP data match the STL, then convert.

Work is done on the exported SKP data only (the .skp files are not changed):
  1. Both are split into parts (connected pieces, welded). An SKP part matches an STL part with the same
     shape at the same place, after the one offset that explains the most of the STL.
  2. Top-level groups of the SKP that hold any triangle of an unmatched part are left out (repeated until
     no kept group touches a left-out part), so groups the STL does not have, or has in another shape, go.
  3. STL parts not covered by a kept group are added from the STL as one more merged mesh
     (normals from the triangle winding; the STL's own normals are often wrong).
Kept groups keep their DEF/USE. Then the result is checked against the STL (vertices both ways, area).
usage: python convert_fix.py 01-2 [14-1 ...]
"""
import collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r'F:/wang/yingzaofashi_x3d'); sys.path.insert(0, HERE)
from skp2x3d import load, flatten, write
from stl2x3d import read_stl
from stl2x3d_def import components
exec(open(HERE + '/../compare_orig.py', encoding='utf-8').read().split('cfg = json.loads')[0])   # x3d_tris, stl_tris, stats

X3D = r'F:/wang/yingzaofashi_x3d/x3d1'
names = json.load(open(HERE + '/batch_map.json', encoding='utf-8'))
match = json.load(open(HERE + '/match.json', encoding='utf-8'))
cfg = json.loads(open(X3D + '/model_config.js', encoding='utf-8').read().split('=', 1)[1].strip().rstrip(';'))
TOL = 0.2                                                  # inches


def split(t):
    uniq, inv = np.unique(np.round(t.reshape(-1, 3), 3), axis=0, return_inverse=True)
    comp = components(inv.reshape(-1, 3), len(uniq))
    g = collections.defaultdict(list)
    for i, c in enumerate(comp): g[c].append(i)
    out = []
    for idx in g.values():
        idx = np.array(idx); p = t[idx].reshape(-1, 3)
        out.append((bytes(np.unique(np.round(p - p.min(0), 1), axis=0).astype(np.float32)), p.min(0), idx))
    return out


def area(t):
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def covered(a, b, Q=1e-4):
    OFF = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    enc = lambda c: ((c[:, 0] + (1 << 20)) << 42) | ((c[:, 1] + (1 << 20)) << 21) | (c[:, 2] + (1 << 20))
    kb = np.unique(enc(np.floor(b / Q).astype(np.int64))); ca = np.floor(a / Q).astype(np.int64)
    hit = np.zeros(len(a), bool)
    for d in OFF: hit |= np.isin(enc(ca + d), kb)
    return float(hit.mean())


def fix(model):
    k = [k for k, n in names.items() if n.startswith('skp_模型/') and os.path.basename(n).startswith(model + ' ')][0]
    zip_path = f"F:/wang/yingzaofashi/{os.path.dirname(cfg[model]['stlPage'])}/stl_zip/{match[model]['stl']}"
    s = read_stl(zip_path)[1].reshape(-1, 3, 3)
    s = s[np.linalg.norm(np.cross(s[:, 1] - s[:, 0], s[:, 2] - s[:, 0]), axis=1) > 0]
    defs = load(f'{HERE}/dumps/{k}.skpdump')
    B = split(s)
    pseudo = {}                                  # component -> definition holding only its own faces
    expanded = 0
    while True:
        # SKP triangles with the top-level group they belong to (-1 = faces of the model itself)
        tris, group = [], []
        if len(defs[0]['tris']): tris.append(defs[0]['tris']); group += [-1] * len(defs[0]['tris'])
        for gi, (c, M) in enumerate(defs[0]['children']):
            t, _ = flatten(defs, c, M); tris.append(t); group += [gi] * len(t)
        t = np.concatenate(tris); group = np.array(group)
        A = split(t)
        bad, dropped, partner, off = match_parts(A, B, group)
        # a left-out group that has sub-groups is opened one level, so only the parts that really differ go
        openable = [g for g in bad if g >= 0 and defs[defs[0]['children'][g][0]]['children']]
        if not openable: break
        new_children = []
        for gi, (c, M) in enumerate(defs[0]['children']):
            if gi not in openable:
                new_children.append((c, M)); continue
            expanded += 1
            new_children += [(cc, M @ Mc) for cc, Mc in defs[c]['children']]
            if len(defs[c]['tris']):
                if c not in pseudo:
                    pid = max(defs) + 1
                    defs[pid] = {'name': defs[c]['name'] + ' (faces)', 'children': [], 'tris': defs[c]['tris'], 'normals': defs[c]['normals']}
                    pseudo[c] = pid
                new_children.append((pseudo[c], M))
        defs[0]['children'] = new_children
    return finish(model, k, zip_path, s, defs, t, A, B, bad, dropped, partner, off, expanded)


def match_parts(A, B, group):
    # the offset STL - SKP that explains the most STL triangles
    skp_by = collections.defaultdict(list)
    for key, mn, idx in A: skp_by[key].append(mn)
    cand = collections.Counter()
    for key, mn, idx in B:
        for am in skp_by.get(key, []): cand[tuple(np.round(mn - am, 1))] += len(idx)
    off = np.array(cand.most_common(1)[0][0]) if cand else np.zeros(3)
    # one-to-one matching of parts
    stl_free = collections.defaultdict(list)
    for bi, (key, mn, idx) in enumerate(B): stl_free[key].append(bi)
    partner = {}
    for ai, (key, mn, idx) in enumerate(A):
        for bi in stl_free.get(key, []):
            if np.abs(B[bi][1] - (mn + off)).max() < TOL:
                partner[ai] = bi; stl_free[key].remove(bi); break

    # exact offset: median over the matched parts (the search above rounds to 0.1 inch)
    if partner:
        off = np.median(np.array([B[bi][1] - A[ai][1] for ai, bi in partner.items()]), axis=0)

    # leave out groups touching unmatched parts, until stable
    bad = set()
    dropped = set(ai for ai in range(len(A)) if ai not in partner)
    while True:
        new_bad = set(group[i] for ai in dropped for i in A[ai][2]) - bad
        if not new_bad: break
        bad |= new_bad
        dropped |= set(ai for ai in range(len(A)) if any(group[i] in bad for i in A[ai][2]))
    return bad, dropped, partner, off


def finish(model, k, zip_path, s, defs, t, A, B, bad, dropped, partner, off, expanded):
    kept_parts = [ai for ai in range(len(A)) if ai not in dropped]
    covered_stl = set(partner[ai] for ai in kept_parts)
    extra_idx = np.concatenate([B[bi][2] for bi in range(len(B)) if bi not in covered_stl] or [np.zeros(0, int)])
    extra_t = s[extra_idx] - off                            # back into the SKP's coordinates
    extra_n = np.cross(extra_t[:, 1] - extra_t[:, 0], extra_t[:, 2] - extra_t[:, 0])
    extra_n = extra_n / np.maximum(np.linalg.norm(extra_n, axis=1, keepdims=True), 1e-12)

    if -1 in bad: defs[0]['tris'] = np.zeros((0, 3, 3)); defs[0]['normals'] = np.zeros((0, 3))
    n_groups = len(defs[0]['children'])
    defs[0]['children'] = [cm for gi, cm in enumerate(defs[0]['children']) if gi not in bad]
    out = f'{X3D}/model_skp/{model}.x3d'
    st = write(defs, out, model, 'assembly', shift=off, source=names[k] + ' (made to match the STL)', extra=(extra_t, extra_n))

    o = x3d_tris(open(out, 'rb').read()); r = stl_tris(zip_path)
    r = r[np.linalg.norm(np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]), axis=1) > 0]
    va = np.unique(np.round(o.reshape(-1, 3), 6), axis=0); vb = np.unique(np.round(r.reshape(-1, 3), 6), axis=0)
    res = dict(st, skp=k, source=names[k] + ' (made to match the STL)', shift=off.tolist(), ref='stl:' + match[model]['stl'],
               groupsOpened=expanded, groupsTotal=n_groups, groupsLeftOut=len([g for g in bad if g >= 0]), skpTris=len(t), stlTris=len(s),
               trisFromSkp=len(t) - sum(len(A[ai][2]) for ai in dropped), trisFromStl=len(extra_idx),
               vertsOutInRef=round(covered(va, vb), 5), vertsRefInOut=round(covered(vb, va), 5),
               areaRatio=round(stats(o)[2] / stats(r)[2], 6), bytes=os.path.getsize(out), bytesNow=os.path.getsize(f'{X3D}/model/{model}.x3d'))
    res['accepted'] = min(res['vertsOutInRef'], res['vertsRefInOut']) > 0.999 and abs(res['areaRatio'] - 1) < 1e-3
    res['why'] = 'SKP made to match the STL: %d groups opened, %d of %d groups left out, %d triangles added from the STL' % (
        expanded, res['groupsLeftOut'], n_groups, res['trisFromStl'])
    if not res['accepted']: os.remove(out)
    return res


if __name__ == '__main__':
    report = json.load(open(HERE + '/convert_report.json', encoding='utf-8'))
    for model in sys.argv[1:]:
        r = fix(model)
        print(model, {x: r[x] for x in ('accepted', 'groupsOpened', 'groupsTotal', 'groupsLeftOut', 'skpTris', 'stlTris', 'trisFromSkp', 'trisFromStl',
                                       'draw_calls', 'vertsOutInRef', 'vertsRefInOut', 'areaRatio', 'bytes', 'bytesNow')}, flush=True)
        if r['accepted']:
            report[model] = r
    listing = {m: j['source'] for m, j in report.items() if j.get('accepted')}
    with open(X3D + '/model_skp/skp_models.js', 'w', encoding='utf-8', newline='\n') as f:
        f.write('// Models that have a DEF/USE version converted from the SketchUp file (skp2x3d.py, assembly mode)\n'
                '// and the SKP each one came from (the A version where there is one). Generated; regenerate rather\n'
                '// than edit by hand.\n'
                'window.SKP_MODELS = ' + json.dumps(listing, ensure_ascii=False, indent=1) + ';\n')
    json.dump(report, open(HERE + '/convert_report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print('SKP versions', len(listing))
