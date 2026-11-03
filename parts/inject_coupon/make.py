"""Resin injection coupon (Spark 3 Nov: every node socket resin-injectable after the rod is in; print and test this before every part
is recut). Drawn only, no topology run: python parts/inject_coupon/make.py writes part.stl. Two sockets exactly as the parts would
cut them (parts/socket_inject.py): a 6/4 tube socket O6.2 x 15 and a 3 mm rod socket O3.2 x 10, 3.0 floors,
bores horizontal (the hard case: teardrop crown up), the fill port (O2.0) and the vent (O1.5) on top. Test: rod home, an 18 G blunt
needle in the port, thin epoxy until it shows at the vent and the mouth; cure, saw it lengthwise, look for voids; pull one."""
import os, sys, numpy as np, trimesh, manifold3d as mf
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from parts.cage_node_lib import KIND, FLOOR, WALL
from parts.socket_inject import injectable_socket_cuts
from topo.part import man

def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

solid, cuts, y = None, [], 0.0
for kind in ("tube", "link"):
    r, depth = KIND[kind]["bore_r"], KIND[kind]["depth"]; ro = r + WALL
    c = np.array([0.0, y + ro, ro]); u = np.array([1.0, 0, 0])                # bore along +x, centre ro above the bed
    b = box(-FLOOR, depth, y, y + 2 * ro, 0.0, ro + r * np.sqrt(2.0) + WALL)    # the crown's wall above the teardrop apex
    solid = b if solid is None else solid + b
    cuts += injectable_socket_cuts(c, u, 0.0, depth, r, port=(0, 0, 1.0))
    y += 2 * ro + 4.0
solid = solid + box(-FLOOR, 10.0, 0.0, y - 4.0, 0.0, 3.0)                       # web joining the two (one body)
for k in cuts:
    solid = solid - k
g = solid.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts)
t.export(os.path.join(os.path.dirname(os.path.abspath(__file__)), "part.stl"))
print(f"inject_coupon: {abs(t.volume) * 1.27e-3:.1f} g PETG, {np.round(t.extents, 1)} mm, watertight {t.is_watertight}")
