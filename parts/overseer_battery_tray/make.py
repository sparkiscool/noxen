"""Overseer battery tray (10 Nov, Spark: "a battery holder that holds the battery as far front as possible, it's back heavy;
bolts to the 4 bolts; don't generate anything, zip ties"). Drawn only, no topology run: python parts/overseer_battery_tray/make.py
writes part.stl (drone frame, as mounted) and print/overseer_battery_tray.stl (top face on the bed).

One PCTG plate on the base's 4 M3 x 5.7 inserts (dimensions.md: (-60, +-25), (20, +-25), faces on z -8.1), top flat on z -8.1.
The pack (Zeee 4S 5200, 138 x 47 x 48 E, 475 g E) sits under it in a tray: side rails 10 deep, a 6 thick front lip 20 deep,
a rear lip 12 deep with a 28 wide notch for the leads; 1 mm clearance a side and at each end (line it with foam tape).
Held up by 3 zip ties (up to 5 x 1.5): each runs through a slot at the rail, over the plate top in a 2 deep groove (under the
base) and round the pack. Bolts M3 x 8 SHCS up from below, heads in O6.5 x 3 counterbores (3 of grip, 5 into the insert).

How far forward: the room under the drone is free to x ~157 at the pack's depth, but the MTF-02P flow sensor (x 80..108, lens on
z -11.1, looking down; its flow FOV 42 deg full) must see past the pack: the pack's front-bottom corner on a 26 deg half-angle line
from the lens's rear edge (x 88) puts the pack front on x 60 (pack x -80..60, centre -10: 10 ahead of the base's design point -20,
inside the bolts). The assembly's 40 deg opening rule would put it at x 43; the old slot ended at 61.9."""
import os, sys
import numpy as np
import manifold3d as mf
import trimesh
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))

Z_TOP = -8.1
T = 6.0                                   # plate
BOLTS = [(-60.0, 25.0), (-60.0, -25.0), (20.0, 25.0), (20.0, -25.0)]
PACK = (138.0, 47.0, 48.0); CLR = 1.0
X1 = 60.0 + CLR                            # cavity front (pack front on x 60)
X0 = X1 - PACK[0] - 2 * CLR                # cavity rear
YW = PACK[1] / 2 + CLR                     # cavity half-width 24.5
ZP = Z_TOP - T                             # plate underside = the pack's top
RAIL_T, RAIL_D = 3.0, 10.0
LIP_T, FRONT_D, REAR_D, NOTCH = 6.0, 20.0, 12.0, 14.0
YP = 31.5                                  # plate half-width (the base drops below z -9 past |y| 33.5)
TIES = (-75.0, -35.0, 40.0)                # tie x (clear of the bolt pads at x -60, 20)
TIE_W, TIE_GROOVE = 6.5, 2.0
CB_R, CB_D, SHANK_R = 3.25, 3.0, 1.7


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def rbox(x0, x1, y0, y1, z0, z1, r):
    cs = mf.CrossSection.square([x1 - x0 - 2 * r, y1 - y0 - 2 * r]).translate([x0 + r, y0 + r]).offset(r, mf.JoinType.Round, circular_segments=32)
    return mf.Manifold.extrude(cs, z1 - z0).translate([0, 0, z0])


def cyl(x, y, z0, z1, r, n=48):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def build():
    xa, xb = X0 - LIP_T, X1 + LIP_T
    plate = rbox(xa, xb, -YP, YP, ZP, Z_TOP, 4.0)
    rails = [box(xa, xb, s * YW if s > 0 else -YW - RAIL_T, s * YW + RAIL_T if s > 0 else -YW, ZP - RAIL_D, ZP + 0.01) for s in (1, -1)]
    front = box(X1, xb, -YW - RAIL_T, YW + RAIL_T, ZP - FRONT_D, ZP + 0.01)
    rear = [box(xa, X0, NOTCH, YW + RAIL_T, ZP - REAR_D, ZP + 0.01), box(xa, X0, -YW - RAIL_T, -NOTCH, ZP - REAR_D, ZP + 0.01)]
    gus = [box(X1 - 0.01, X1 + LIP_T, s * (YW + RAIL_T) - (RAIL_T if s > 0 else 0), s * (YW + RAIL_T) + (0 if s > 0 else RAIL_T), ZP - FRONT_D, ZP) for s in (1, -1)]
    body = mf.Manifold.batch_boolean([plate, front] + rails + rear + gus, mf.OpType.Add)
    cuts = []
    for x, y in BOLTS:
        cuts += [cyl(x, y, ZP - 30, Z_TOP + 1, SHANK_R), cyl(x, y, ZP - 30, ZP + CB_D, CB_R)]
    for x in TIES:
        for s in (1, -1):
            y0, y1 = (YW, YW + RAIL_T) if s > 0 else (-YW - RAIL_T, -YW)
            cuts.append(rbox(x - TIE_W / 2, x + TIE_W / 2, y0 - 0.01, y1 + 0.01, ZP - RAIL_D - 1, Z_TOP + 1, 0.8))   # slot through plate and rail
        cuts.append(box(x - TIE_W / 2, x + TIE_W / 2, -YW - RAIL_T, YW + RAIL_T, Z_TOP - TIE_GROOVE, Z_TOP + 1))     # groove over the top
    for wa, wb in ((-69.0, -42.0), (-28.0, 33.0), (47.0, 56.0)):                                                    # lightening windows
        cuts.append(rbox(wa, wb, -17.0, 17.0, ZP - 1, Z_TOP + 1, 4.0))
    return body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)


def to_tri(m):
    g = m.to_mesh(); return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts), process=True)


if __name__ == "__main__":
    b = build()
    t = to_tri(b)
    t.export(os.path.join(HERE, "part.stl"))
    p = t.copy(); p.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))   # top face (z -8.1) on the bed
    p.apply_translation(-p.bounds[0]); os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
    p.export(os.path.join(ROOT, "print", "overseer_battery_tray.stl"))
    print(dict(volume_cm3=round(b.volume() / 1000, 1), mass_g_pctg=round(b.volume() * 1.27e-3, 1), genus=b.genus(), watertight=bool(t.is_watertight),
               bounds=t.bounds.round(1).tolist(), print_extents=p.extents.round(1).tolist(), cavity_x=(round(X0, 1), round(X1, 1)), pack_top_z=ZP))
