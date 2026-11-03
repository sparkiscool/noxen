"""The Scout's belly skirt (3 Nov, drawn, E): the Scout's seat in the shared funnel. The funnel seats the Overseer receiver's
outline, 184 x 66.8 (parts/box_funnel/make.py); the Scout's own receiver is 105.6 x 65.5, so two runners extend its outline fore
and aft to the same 184 at the same bottom level (z -44.3), 20 tall, joined to the belly mount above. The runners' bottom outer
edges carry the 55 deg lead-in the funnel faces meet. Drone frame (x nose, y left, z up). Nothing else of the Scout may be below
z 7.7 (the funnel mouth) outside the funnel's faces: tools/seat_check.py checks it.
Run: python parts/scout_belly_skirt/make.py
"""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
CX, L, W, Z0, HGT = 1.5, 184.0, 66.8, -44.3, 20.0
RX0, RX1 = -51.3, 54.3                            # the receiver v3 ends: the runners start 1.0 clear of them


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def runner(x0, x1):
    """a plain block; the 55 deg lead-in on its bottom outer edges is the receiver's own (E) and is not drawn"""
    return box(x0, x1, -W / 2, W / 2, Z0, Z0 + HGT)


if __name__ == "__main__":
    nose = runner(RX1 + 1.0, CX + L / 2); tail = runner(CX - L / 2, RX0 - 1.0)
    bridge = [box(CX - L / 2, CX + L / 2, min(s * (W / 2 - 4), s * W / 2), max(s * (W / 2 - 4), s * W / 2), Z0 + HGT - 4, Z0 + HGT) for s in (-1, 1)]   # rails along the receiver's sides
    m = mf.Manifold.batch_boolean([nose, tail] + bridge, mf.OpType.Add).to_mesh()
    t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    t.export(os.path.join(HERE, "part.stl"))
    print(f"skirt: {t.extents.round(1).tolist()} min {t.bounds[0].round(1).tolist()} max {t.bounds[1].round(1).tolist()}, {t.volume * 1.27e-3:.0f} g PETG, watertight {t.is_watertight}")
