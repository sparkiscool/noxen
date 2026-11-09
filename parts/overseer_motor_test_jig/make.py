"""Motor test jig (8 Nov, Spark: not sure the arms will not crack; a mount to run a motor on and see). It holds a REAL printed rear
arm (parts/overseer_arm_rear) exactly as the drone does: the root block clamped between two slabs 45 apart on the arm's own 4
through-bolts (A at u 63.17 v +-15, B at u 63.74 v +-10) and 2 pegs (u 57.66 v +-11), so the motor end that cracked is tested on
the part itself, at full thrust, with the prop on. Drawn only: python parts/overseer_motor_test_jig/make.py writes part.stl.
* A C-block in the arm's own (u, v, z) frame: base slab z -8..0 (u 30..92, |v| <= 36), back wall u 30..40 (z -8..53), top slab
  z 45..53 (u 40..73, ending where the root leaves the plate). O3.5 bolt holes through both slabs at A and B, O6.2 x 2 peg
  pockets in the top slab's underside, 4 x O4.5 screw-down holes at u 80 / 88, v +-28 (clear of the top slab) for a plank.
* Print it on its side (the open side of the C up): the layers run along the C, so the moment from the thrust is in-layer.
  PCTG (the drone's material), 100 % infill. ~160 g.
* Test: plank clamped to a bench, motor + prop, throttle steps to full for 30 s each, watch the pad and the boss; a thermal
  camera or finger test on the pad after a 2 min hover-throttle run tells whether the motor heat is the problem."""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
T_SLAB, T_WALL, GAP = 8.0, 10.0, 45.0
A, B, PEG = (63.17, 15.0), (63.74, 10.0), (57.66, 11.0)
HOLE_R, PEG_R, PEG_D, SCREW_R = 1.75, 3.1, 2.0, 2.25
U0, U_TOP, U_END, V = 30.0, 73.0, 92.0, 36.0


def box(u0, u1, v0, v1, z0, z1):
    return mf.Manifold.cube([u1 - u0, v1 - v0, z1 - z0]).translate([u0, v0, z0])


def cyl(u, v, z0, z1, r):
    return mf.Manifold.cylinder(z1 - z0, r, r, 48).translate([u, v, z0])


m = box(U0, U_END, -V, V, -T_SLAB, 0.0) + box(U0, U0 + T_WALL, -V, V, -T_SLAB, GAP + T_SLAB) + box(U0, U_TOP, -V, V, GAP, GAP + T_SLAB)
cuts = []
for s in (1, -1):
    for (u, v) in (A, B):
        cuts.append(cyl(u, s * v, -T_SLAB - 1, GAP + T_SLAB + 1, HOLE_R))
    cuts.append(cyl(PEG[0], s * PEG[1], GAP - 0.01, GAP + PEG_D, PEG_R))
    for u in (80.0, 88.0):
        cuts.append(cyl(u, s * 28.0, -T_SLAB - 1, 1.0, SCREW_R))
m = m - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts); t.export(os.path.join(HERE, "part.stl"))
print(f"{abs(t.volume) / 1000:.1f} cm3 = {abs(t.volume) * 1.23e-3:.0f} g PCTG, watertight {t.is_watertight}, box {np.round(t.bounds[1] - t.bounds[0], 1)}")
