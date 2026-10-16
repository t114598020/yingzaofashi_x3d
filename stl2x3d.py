"""Convert a (zipped) binary/ASCII STL from the three.js pages (branch main) into an X3D file that
follows the same convention as the SketchUp exports in x3d1/model/*.x3d:

    SketchUp_TRANSFORM: scale 0.0254 (inch -> m), rotation -90deg about X (Z-up -> Y-up)

so x3d1/x3d.html can undo that one convention and then apply the three.js page's own
mesh.position / rotation / scale, exactly like the STL page does.

To render like THREE.STLLoader + MeshPhongMaterial (r66):
  * each triangle keeps the facet normal stored in the STL (STLLoader lights with those, and many
    of these SketchUp STLs have normals that do not match the triangle winding);
  * the winding is kept and solid="true", i.e. back faces are culled like three.js FrontSide.

usage: python stl2x3d.py <stl or zip> <out.x3d> <title>
"""
import sys, re, struct, zipfile
import numpy as np


def read_stl(path):
    """-> (normals [n,3], vertices [n*3,3])"""
    if path.lower().endswith('.zip'):
        zf = zipfile.ZipFile(path)
        data = zf.read([n for n in zf.namelist() if n.lower().endswith('.stl')][0])
    else:
        data = open(path, 'rb').read()
    if data[:5] == b'solid' and b'facet' in data[:2000]:
        num = rb'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        nrm = np.array(re.findall(rb'facet\s+normal\s+' + num + rb'\s+' + num + rb'\s+' + num, data), dtype=np.float64)
        pts = np.array(re.findall(rb'vertex\s+' + num + rb'\s+' + num + rb'\s+' + num, data), dtype=np.float64)
        return nrm.reshape(-1, 3), pts.reshape(-1, 3)
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data, dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]), count=n, offset=84)
    return rec['n'].astype(np.float64), rec['v'].reshape(-1, 3).astype(np.float64)


def fmt(x):
    s = '%.4f' % x
    s = s.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def write_x3d(stl, out, title):
    normals, pts = stl
    tri = pts.reshape(-1, 3, 3)
    keep = ~((tri[:, 0] == tri[:, 1]).all(1) | (tri[:, 1] == tri[:, 2]).all(1) | (tri[:, 0] == tri[:, 2]).all(1))
    tri, normals = tri[keep], normals[keep]
    # shared vertices / shared normals keep the file small
    uniq, inv = np.unique(np.round(tri.reshape(-1, 3), 4), axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    nuniq, ninv = np.unique(np.round(normals, 4), axis=0, return_inverse=True)
    coord_index = ' '.join('%d %d %d -1' % tuple(t) for t in inv)
    normal_index = ' '.join(str(int(i)) for i in ninv.reshape(-1))
    points = ' '.join(' '.join(fmt(c) for c in p) for p in uniq)
    vectors = ' '.join(' '.join(fmt(c) for c in p) for p in nuniq)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.0//EN" "http://www.web3d.org/specifications/x3d-3.0.dtd">
<X3D version="3.0" profile="Immersive" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.0.xsd">
	<head>
		<meta name="filename" content="%s.x3d" />
		<meta name="generator" content="stl2x3d.py (converted from the STL used by the three.js page, branch main)" />
	</head>
	<Scene>
		<Transform DEF="SketchUp_TRANSFORM"
		           translation="0.000000 0.000000 0.000000"
		           scale="0.025400 0.025400 0.025400"
		           rotation="-1.000000 0.000000 0.000000 1.570796"
		           >
			<Shape>
				<Appearance>
					<Material diffuseColor="0.784 0.659 0.502" />
				</Appearance>
				<IndexedFaceSet solid="true" normalPerVertex="false"
				                coordIndex="%s"
				                normalIndex="%s"
				                >
					<Coordinate point="%s" />
					<Normal vector="%s" />
				</IndexedFaceSet>
			</Shape>
		</Transform>
	</Scene>
</X3D>
''' % (title, coord_index, normal_index, points, vectors))
    return len(tri), len(uniq)


if __name__ == '__main__':
    src, out, title = sys.argv[1:4]
    print(src, '->', out, write_x3d(read_stl(src), out, title))
