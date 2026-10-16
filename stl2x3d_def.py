"""STL -> X3D like stl2x3d.py, but repeated parts are stored once (DEF) and reused (USE).

The STL is one triangle soup. It is split into connected parts (triangles sharing welded
vertices). Parts that are the same shape up to a translation and a rotation of k*90deg about the
STL's vertical (Z) axis share one geometry: the first one is written as <Shape DEF="p<n>"> inside a
Transform, the others as <Transform ...><Shape USE="p<n>"/></Transform>. Parts that are not
repeated (or too small to be worth a separate draw call) go into one merged mesh.

Geometry, facet normals and winding are kept exactly, so the result renders the same as the
single-mesh file from stl2x3d.py.

usage: python stl2x3d_def.py <stl or zip> <out.x3d> <title> [min_tris]
"""
import sys
import numpy as np
from stl2x3d import read_stl, fmt

DECIMALS = 4            # vertex welding / output precision (inches)
KEY_DECIMALS = 3        # tolerance when deciding two parts are the same shape


def rot_z(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def components(tri_idx, nverts):
    parent = np.arange(nverts)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b, c in tri_idx:
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb: parent[rb] = ra
        rc = find(c)
        if rc != ra: parent[rc] = ra
    roots = np.array([find(v) for v in range(nverts)])
    return roots[tri_idx[:, 0]]


def canonical(tris, normals):
    """tris [n,3,3] already moved to min corner 0 -> key (bytes) that is the same for equal shapes"""
    q = np.round(tris * 10 ** KEY_DECIMALS).astype(np.int64)          # >= 0 after the min-corner shift
    K = np.int64(1 << 21)
    vkey = (q[:, :, 0] * K + q[:, :, 1]) * K + q[:, :, 2]            # one int per vertex, lexicographic
    # start each triangle at its smallest vertex, keeping the winding
    first = vkey.argmin(1)
    idx = (first[:, None] + np.arange(3)[None, :]) % 3
    vk = np.take_along_axis(vkey, idx, axis=1)
    nq = np.round(normals * 100).astype(np.int64)
    rows = np.concatenate([vk, nq], axis=1)
    rows = rows[np.lexsort(rows.T[::-1])]
    return rows.tobytes()


def analyse(normals, pts, min_tris):
    tri = pts.reshape(-1, 3, 3)
    keep = ~((tri[:, 0] == tri[:, 1]).all(1) | (tri[:, 1] == tri[:, 2]).all(1) | (tri[:, 0] == tri[:, 2]).all(1))
    tri, normals = tri[keep], normals[keep]
    uniq, inv = np.unique(np.round(tri.reshape(-1, 3), DECIMALS), axis=0, return_inverse=True)
    comp = components(inv.reshape(-1, 3), len(uniq))
    order = np.argsort(comp, kind='stable')
    bounds = np.flatnonzero(np.diff(comp[order])) + 1
    parts = np.split(order, bounds)

    groups = {}          # key -> list of (part triangle indices, k, translation m)
    rest = []
    for p in parts:
        if len(p) < min_tris:
            rest.append(p); continue
        best = None
        for k in range(4):
            R = rot_z(k)
            v = tri[p] @ R.T
            m = v.reshape(-1, 3).min(0)
            key = canonical(v - m, normals[p] @ R.T)
            if best is None or key < best[0]:
                best = (key, k, m)
        groups.setdefault(best[0], []).append((p, best[1], best[2]))

    shared, single = [], []
    for key, inst in groups.items():
        (shared if len(inst) > 1 else single).append(inst)
    for inst in single:
        rest.append(inst[0][0])
    return tri, normals, shared, (np.concatenate(rest) if rest else np.zeros(0, int))


def mesh_xml(tris, normals, indent):
    uniq, inv = np.unique(np.round(tris.reshape(-1, 3), DECIMALS), axis=0, return_inverse=True)
    nuniq, ninv = np.unique(np.round(normals, DECIMALS), axis=0, return_inverse=True)
    i = indent
    return (f'{i}<Appearance>\n{i}\t<Material diffuseColor="0.784 0.659 0.502" />\n{i}</Appearance>\n'
            f'{i}<IndexedFaceSet solid="true" normalPerVertex="false"\n'
            f'{i}                coordIndex="' + ' '.join('%d %d %d -1' % tuple(t) for t in inv.reshape(-1, 3)) + '"\n'
            f'{i}                normalIndex="' + ' '.join(str(int(n)) for n in ninv.reshape(-1)) + '"\n'
            f'{i}                >\n'
            f'{i}\t<Coordinate point="' + ' '.join(' '.join(fmt(c) for c in p) for p in uniq) + '" />\n'
            f'{i}\t<Normal vector="' + ' '.join(' '.join(fmt(c) for c in p) for p in nuniq) + '" />\n'
            f'{i}</IndexedFaceSet>\n')


def axis_angle_z(k):
    return '0 0 1 %.7f' % (-k * np.pi / 2)     # instance rotation = R_k^-1


def write_x3d(stl, out, title, min_tris=1):
    normals, pts = stl
    tri, normals, shared, rest = analyse(normals, pts, min_tris)
    body = []
    if len(rest):
        body.append('\t\t\t<Shape>\n' + mesh_xml(tri[rest], normals[rest], '\t\t\t\t') + '\t\t\t</Shape>\n')
    n_inst = 0
    for gi, inst in enumerate(shared):
        p0, k0, m0 = inst[0]
        R = rot_z(k0)
        local = tri[p0] @ R.T - m0                     # canonical geometry, min corner at 0
        local_n = normals[p0] @ R.T
        for j, (p, k, m) in enumerate(inst):
            Rinv = rot_z(-k)
            t = Rinv @ m
            head = '\t\t\t<Transform translation="%s" rotation="%s">\n' % (' '.join(fmt(c) for c in t), axis_angle_z(k))
            if j == 0:
                shape = '\t\t\t\t<Shape DEF="p%d">\n' % gi + mesh_xml(local, local_n, '\t\t\t\t\t') + '\t\t\t\t</Shape>\n'
            else:
                shape = '\t\t\t\t<Shape USE="p%d"/>\n' % gi
            body.append(head + shape + '\t\t\t</Transform>\n')
            n_inst += 1
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.0//EN" "http://www.web3d.org/specifications/x3d-3.0.dtd">
<X3D version="3.0" profile="Immersive" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.0.xsd">
	<head>
		<meta name="filename" content="%s.x3d" />
		<meta name="generator" content="stl2x3d_def.py (converted from the STL used by the three.js page, branch main; repeated parts use DEF/USE)" />
	</head>
	<Scene>
		<Transform DEF="SketchUp_TRANSFORM"
		           translation="0.000000 0.000000 0.000000"
		           scale="0.025400 0.025400 0.025400"
		           rotation="-1.000000 0.000000 0.000000 1.570796"
		           >
''' % title)
        f.write(''.join(body))
        f.write('''		</Transform>
	</Scene>
</X3D>
''')
    return {'triangles': len(tri), 'shared_shapes': len(shared), 'instances': n_inst,
            'merged_triangles': int(len(rest)), 'draw_calls': n_inst + (1 if len(rest) else 0)}


if __name__ == '__main__':
    src, out, title = sys.argv[1:4]
    min_tris = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(src, '->', out, write_x3d(read_stl(src), out, title, min_tris))
