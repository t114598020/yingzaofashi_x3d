"""SketchUp component tree (from skp_dump.rb) -> X3D with DEF/USE taken from the SketchUp components.

  --mode tree      every component placed more than once is a DEF (nested, like the SKP);
                   each placement of a component that has faces is one Shape -> one draw call
  --mode assembly  the highest component on each path that is placed more than once is merged
                   (with everything inside it) into one Shape DEF; its other placements are USE.
                   Faces that are not inside a repeated component go into one more Shape.
  --handed N       (assembly) only cut at a component that has at least two placements of the same
                   handedness (mirrored copies need their own geometry in x3dom, so one plain plus one
                   mirrored placement saves nothing) and at least N triangles; smaller repeated parts
                   are merged into the one extra Shape instead of being drawn one by one.

Same wrapper as stl2x3d.py (SketchUp_TRANSFORM: inch -> m, Z-up -> Y-up), facet normals, solid="true".
usage: python skp2x3d.py <dump> <out.x3d> <title> --mode tree|assembly [--handed N]
"""
import sys, math, collections
import numpy as np

sys.path.insert(0, r'F:/wang/yingzaofashi_x3d')
from stl2x3d import fmt


def load(path):
    defs = {}
    cur = None
    tris = collections.defaultdict(list)
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('F '):
            tris[cur].append(line[2:].split())
        elif line.startswith('D\t'):
            _, i, name = line.rstrip('\n').split('\t', 2)
            cur = int(i); defs[cur] = {'name': name, 'children': []}
        elif line.startswith('C\t'):
            _, i, m = line.rstrip('\n').split('\t')
            defs[cur]['children'].append((int(i), np.array(m.split(), float).reshape(4, 4).T))
    for i, d in defs.items():
        a = np.array(tris[i], float).reshape(-1, 12) if tris[i] else np.zeros((0, 12))
        t, n = a[:, :9].reshape(-1, 3, 3), a[:, 9:]
        # keep the winding consistent with the face normal (front face = counter-clockwise)
        cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        flip = (cross * n).sum(1) < 0
        t[flip] = t[flip][:, ::-1]
        keep = np.linalg.norm(cross, axis=1) > 0
        d['tris'], d['normals'] = t[keep], n[keep]
    return defs


def placements(defs):
    count = collections.Counter()

    def walk(i):
        count[i] += 1
        for c, _ in defs[i]['children']:
            walk(c)
    walk(0)
    return count


def flatten(defs, i, M=np.eye(4)):
    """all triangles of definition i under matrix M"""
    out_t, out_n = [], []
    d = defs[i]
    if len(d['tris']):
        t = d['tris'] @ M[:3, :3].T + M[:3, 3]
        n = d['normals'] @ np.linalg.inv(M[:3, :3])        # normal matrix = inverse transpose
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        if np.linalg.det(M[:3, :3]) < 0:
            t = t[:, ::-1]                                  # mirrored: keep the front side outward
        out_t.append(t); out_n.append(n)
    for c, Mc in d['children']:
        t, n = flatten(defs, c, M @ Mc)
        out_t.append(t); out_n.append(n)
    if not out_t:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.concatenate(out_t), np.concatenate(out_n)


# ---- X3D writing -------------------------------------------------------------------------------

APPEARANCE = {'written': False}


def shape_xml(tris, normals, indent, def_name=None):
    uniq, inv = np.unique(np.round(tris.reshape(-1, 3), 4), axis=0, return_inverse=True)
    nuniq, ninv = np.unique(np.round(normals, 4), axis=0, return_inverse=True)
    i = indent
    head = '<Shape DEF="%s">' % def_name if def_name else '<Shape>'
    # one <Appearance DEF="wood"> for the whole file, every other shape USEs it: x3d.html adds a
    # shadow shader per Appearance, and x3dom compiles one shader program per Appearance
    if APPEARANCE['written']:
        app = f'{i}\t<Appearance USE="wood"/>\n'
    else:
        APPEARANCE['written'] = True
        app = f'{i}\t<Appearance DEF="wood">\n{i}\t\t<Material diffuseColor="0.784 0.659 0.502" />\n{i}\t</Appearance>\n'
    return (f'{i}{head}\n' + app +
            f'{i}\t<IndexedFaceSet solid="true" normalPerVertex="false"\n'
            f'{i}\t                coordIndex="' + ' '.join('%d %d %d -1' % tuple(t) for t in inv.reshape(-1, 3)) + '"\n'
            f'{i}\t                normalIndex="' + ' '.join(str(int(n)) for n in ninv.reshape(-1)) + '">\n'
            f'{i}\t\t<Coordinate point="' + ' '.join(' '.join(fmt(c) for c in p) for p in uniq) + '" />\n'
            f'{i}\t\t<Normal vector="' + ' '.join(' '.join(fmt(c) for c in p) for p in nuniq) + '" />\n'
            f'{i}\t</IndexedFaceSet>\n{i}</Shape>\n')


def num(x):
    s = '%.6f' % x
    s = s.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def axis_angle(R):
    q_w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    angle = 2 * math.acos(min(1.0, q_w))
    if angle < 1e-9:
        return None
    if abs(angle - math.pi) < 1e-4:                      # 180 deg: axis from the symmetric part
        B = (R + np.eye(3)) / 2
        axis = B[np.argmax(np.diag(B))]
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / np.linalg.norm(axis)
    return ' '.join(num(c) for c in axis) + ' ' + num(angle)


def transform_attrs(M):
    """4x4 affine -> X3D translation / rotation / scale / scaleOrientation (exact, mirrors allowed)"""
    U, S, Vt = np.linalg.svd(M[:3, :3])
    V = Vt.T
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1; U[:, 2] *= -1
    R = U @ V.T
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1; S = S.copy(); S[2] *= -1
        R = U @ V.T
    attrs = []
    if np.abs(M[:3, 3]).max() > 1e-9:
        attrs.append('translation="%s"' % ' '.join(num(c) for c in M[:3, 3]))
    r = axis_angle(R)
    if r:
        attrs.append('rotation="%s"' % r)
    if np.abs(S - 1).max() > 1e-7:
        attrs.append('scale="%s"' % ' '.join(num(c) for c in S))
        if np.abs(S - S[0]).max() > 1e-7:
            so = axis_angle(V)
            if so:
                attrs.append('scaleOrientation="%s"' % so)
    assert np.allclose(U @ np.diag(S) @ V.T, M[:3, :3], atol=1e-6)
    return ' '.join(attrs)


def handed_placements(defs):
    """definition -> [plain placements, mirrored placements] over the whole model"""
    count = collections.defaultdict(lambda: [0, 0])

    def walk(i, M):
        for c, Mc in defs[i]['children']:
            W = M @ Mc
            count[c][int(np.linalg.det(W[:3, :3]) < 0)] += 1
            walk(c, W)
    walk(0, np.eye(4))
    return count


def write(defs, out, title, mode, shift=None, source='', handed=None, extra=None):
    """shift: move the model by this many inches (to where the three.js page's STL has it)
    handed: None = cut at the highest component placed more than once; a number N = see --handed
    extra: (triangles, normals) in the SKP's coordinates added to the one merged Shape (assembly mode),
           e.g. pieces taken from the STL where the SKP differs from it"""
    APPEARANCE['written'] = False
    M0 = np.eye(4)
    if shift is not None:
        M0[:3, 3] = shift
    count = placements(defs)
    hand = handed_placements(defs) if handed is not None else None
    body, stats = [], collections.Counter()
    written = set()

    def def_name(i):
        return 'c%d' % i

    if mode == 'tree':
        def node(i, M, indent):
            d = defs[i]
            s = indent + ('<Transform %s>\n' % transform_attrs(M))
            if count[i] > 1 and i in written:
                s += indent + '\t<Group USE="%s"/>\n' % def_name(i)
                stats['draw_calls'] += subtree_shapes[i]
            else:
                inner = indent + '\t'
                if count[i] > 1:
                    written.add(i)
                    s += inner + '<Group DEF="%s">\n' % def_name(i); inner += '\t'
                if len(d['tris']):
                    s += shape_xml(d['tris'], d['normals'], inner)
                    stats['draw_calls'] += 1; stats['tris_written'] += len(d['tris'])
                for c, Mc in d['children']:
                    s += node(c, Mc, inner)
                if count[i] > 1:
                    s += indent + '\t</Group>\n'
            return s + indent + '</Transform>\n'

        subtree_shapes = {}

        def shapes(i):
            if i not in subtree_shapes:
                subtree_shapes[i] = (1 if len(defs[i]['tris']) else 0) + sum(shapes(c) for c, _ in defs[i]['children'])
            return subtree_shapes[i]
        shapes(0)
        body.append(node(0, M0, '\t\t\t'))

    else:  # assembly
        static_t, static_n = [], []

        # x3dom picks the front face from ccw only and ignores mirroring Transforms, so a mirrored
        # placement uses its own copy of the geometry with the mirror (MIR) already applied, and
        # the Transform itself is no longer a mirror: M = (M MIR) MIR
        MIR = np.diag([-1.0, 1, 1, 1])

        def walk(i, M):
            d = defs[i]
            if handed is None:
                cut = count[i] > 1 and shapes_tris(i) > 0
            else:
                cut = max(hand[i]) >= 2 and shapes_tris(i) >= max(handed, 1)
            if i != 0 and cut:
                mirrored = np.linalg.det(M[:3, :3]) < 0
                name = def_name(i) + ('m' if mirrored else '')
                if mirrored:
                    M = M @ MIR
                    stats['mirrored'] += 1
                s = '\t\t\t<Transform %s>\n' % transform_attrs(M)
                if name in written:
                    s += '\t\t\t\t<Shape USE="%s"/>\n' % name
                else:
                    written.add(name)
                    t, n = flatten(defs, i, MIR if mirrored else np.eye(4))
                    s += shape_xml(t, n, '\t\t\t\t', name)
                    stats['tris_written'] += len(t)
                body.append(s + '\t\t\t</Transform>\n')
                stats['draw_calls'] += 1
                return
            if len(d['tris']):
                t = d['tris'] @ M[:3, :3].T + M[:3, 3]
                n = d['normals'] @ np.linalg.inv(M[:3, :3]); n /= np.linalg.norm(n, axis=1, keepdims=True)
                if np.linalg.det(M[:3, :3]) < 0: t = t[:, ::-1]
                static_t.append(t); static_n.append(n)
            for c, Mc in d['children']:
                walk(c, M @ Mc)

        sub = {}

        def shapes_tris(i):
            if i not in sub:
                sub[i] = len(defs[i]['tris']) + sum(shapes_tris(c) for c, _ in defs[i]['children'])
            return sub[i]
        walk(0, M0)
        if extra is not None and len(extra[0]):
            static_t.append(extra[0] + M0[:3, 3]); static_n.append(extra[1])
            stats['extra_tris'] = len(extra[0])
        if static_t:
            t, n = np.concatenate(static_t), np.concatenate(static_n)
            body.append(shape_xml(t, n, '\t\t\t'))      # last: it USEs the Appearance DEFed above
            stats['draw_calls'] += 1; stats['tris_written'] += len(t)

    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.0//EN" "http://www.web3d.org/specifications/x3d-3.0.dtd">
<X3D version="3.0" profile="Immersive" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.0.xsd">
	<head>
		<meta name="filename" content="%s.x3d" />
		<meta name="generator" content="skp2x3d.py --mode %s (DEF/USE from the SketchUp components)" />
		<meta name="source" content="%s" />
	</head>
	<Scene>
		<Transform DEF="SketchUp_TRANSFORM"
		           translation="0.000000 0.000000 0.000000"
		           scale="0.025400 0.025400 0.025400"
		           rotation="-1.000000 0.000000 0.000000 1.570796"
		           >
''' % (title, mode, source.replace('&', '&amp;').replace('"', '&quot;')))
        f.write(''.join(body))
        f.write('''		</Transform>
	</Scene>
</X3D>
''')
    return dict(stats)


if __name__ == '__main__':
    src, out, title = sys.argv[1:4]
    mode = sys.argv[sys.argv.index('--mode') + 1] if '--mode' in sys.argv else 'assembly'
    handed = int(sys.argv[sys.argv.index('--handed') + 1]) if '--handed' in sys.argv else None
    print(out, write(load(src), out, title, mode, handed=handed))
