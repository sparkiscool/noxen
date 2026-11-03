"""The box's top deck (3 Nov): the enclosure ends here and the funnel stands on it. Box W frame. The Overseer's cage hangs round the
funnel when it is seated, its loop feet at W Z 71.6 (seat z -83.7), so nothing of the box may stand above the deck but the funnel:
deck top W Z 66, 3 thick, the enclosure footprint (X +-209, Y -69..340, the v2 layout), with the swap opening for the heads
and both cartridges centred on the catch centre (0, 167.1): 220 along Y (the axle) x 140 across (the head body 126 x 15 plus clearance; concept F had 210 x 106).
Run: python parts/box_deck/make.py
"""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
Z_TOP, T = 66.0, 3.0
X0, X1, Y0, Y1 = -209.0, 209.0, -69.0, 340.0
YC, HOLE_Y, HOLE_X = 167.1, 220.0, 140.0           # 3 Nov: the swap head's body (126 along X) passes up to the funnel's under-seat skirt


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


if __name__ == "__main__":
    deck = box(X0, X1, Y0, Y1, Z_TOP - T, Z_TOP) - box(-HOLE_X / 2, HOLE_X / 2, YC - HOLE_Y / 2, YC + HOLE_Y / 2, Z_TOP - T - 1, Z_TOP + 1)
    m = deck.to_mesh(); t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    t.export(os.path.join(HERE, "part.stl"))
    print(f"deck: {t.extents.round(1).tolist()} at W Z {Z_TOP - T}..{Z_TOP}, hole {HOLE_Y} x {HOLE_X} at Y {YC}, {t.volume * 1.27e-3:.0f} g PETG")
