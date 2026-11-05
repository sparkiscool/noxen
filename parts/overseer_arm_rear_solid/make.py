"""Overseer rear arm, solid (5 Nov, Spark: the printed rear arms snapped at the motor base, printed flat with the layers across the
thrust; demo tomorrow, be safe and fast: no topology, a drawn solid part). Arm 2 (+y); arm 3 is the y-mirror (mirror in the slicer).
python parts/overseer_arm_rear_solid/make.py writes part.stl, in the drone frame.
* Bolts to the plates as before (parts/overseer_arm_lib.py: root block u 57.5..72 |v| <= 22 between the plates, the 4 through-bolts,
  nut pockets, pegs), motor at motors_xy[1], pad top z 45, thrust plane z 82.
* Installs with core tie left.2 (right.2 on arm 3) in place: its tunnel through the arm is swept straight down and out, so the arm
  lowers onto the bottom plate over the rod with the top plate off.
* Solid: the root block (14.5 x 44 x 45), a 14 x 33 beam along the arm (z 12..45) to the motor, the motor disc O36 x 6 with a O36 x 8
  boss under it, the lib's lugs, slots, head seats, driver sweeps and O8.6 shaft bore, the cage members near it cut clear.
* Print on its side (a flat side of the beam on the bed, the motor disc's axis horizontal): the layers run along the arm and through
  the pad, so thrust and bending are in-layer. PETG, 100 % infill, 5 walls, 0.2 layers. Fits the A1 mini (~125 x 45, 44 tall).
Mass ~90 g (E, 1.27 g/cm3): the optimised arm was 54 g; the weight is the price of no lace and no wait."""
import os, sys, numpy as np, trimesh, manifold3d as mf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from parts.overseer_arm_lib import Arm, CAGE, Z_PAD, PAD_T, cyl, along, U

INSTALL_OVER = "core tie left.2"
SLOT_R, SLOT_DROP = 4.0, 80.0
BEAM_V, BEAM_Z0 = 7.0, 12.0

ARM = Arm("rear", 135.0, 1, [], [-73.03, 139.5, 25.0])
drawn, cuts, env = ARM.drawn_cuts_env()
a3 = np.array([ARM.a[0], ARM.a[1], 0.0]); n3 = np.array([-a3[1], a3[0], 0.0]); R3 = np.array([ARM.R[0], ARM.R[1], 0.0]); L = ARM.len; M = ARM.M
ud3 = np.array([ARM.ud[0], ARM.ud[1], 0.0]); vd3 = np.array([ARM.vd[0], ARM.vd[1], 0.0])


def beam(s0, s1, v0, v1, z0, z1, A=a3, N=n3, O=R3):
    return mf.Manifold.cube([s1 - s0, v1 - v0, z1 - z0]).translate([s0, v0, z0]).transform(np.c_[A, N, [0, 0, 1.0], O])


root = beam(57.5, 72.0, -22.0, 22.0, 0.0, Z_PAD, ud3, vd3, np.zeros(3))                        # the plate block, in the diagonal frame
body = [root, beam(4.0, L - 8.0, -BEAM_V, BEAM_V, BEAM_Z0, Z_PAD), cyl(M[0], M[1], Z_PAD - PAD_T, Z_PAD, 18.0, 64),
        cyl(M[0], M[1], Z_PAD - PAD_T - 8.0, Z_PAD - PAD_T + 0.01, 18.0, 64)]
p, q = (np.array(v, float) for v in CAGE["links"][INSTALL_OVER]); u = (q - p) / np.linalg.norm(q - p)
tube = along(p, u, 0.0, float(np.linalg.norm(q - p)), SLOT_R)
slot = mf.Manifold.batch_hull([tube, tube.translate([0.0, 0.0, -SLOT_DROP])])
m = (U(body) + drawn) - cuts - slot
g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts); t.merge_vertices()
parts = t.split(only_watertight=False); t = max(parts, key=lambda q_: abs(q_.volume)) if len(parts) > 1 else t
t.export(os.path.join(HERE, "part.stl"))
b = t.bounds
print(f"{abs(t.volume) / 1000:.1f} cm3 = {abs(t.volume) * 1.27e-3:.0f} g, bodies {len(parts)}, watertight {t.is_watertight}, box {np.round(b[1] - b[0], 1)}")
