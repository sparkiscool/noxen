"""Overseer rear arm, solid v2 (5 Nov, Spark: the printed rear arms snapped at the motor base, printed flat with the layers across the
thrust; demo tomorrow: no topology, a drawn solid, smooth not geometric, even more support round the motor plate). Arm 2 (+y);
arm 3 is the y-mirror (mirror in the slicer). python parts/overseer_arm_rear_solid/make.py writes part.stl, in the drone frame
(parts/arm_solid_lib.py: a blurred distance field meshed, then the arm lib's exact bosses added and its cuts subtracted).
* Bolts to the plates as before (parts/overseer_arm_lib.py: root block u 57.5..72 |v| <= 22 between the plates with its flat faces,
  the 4 through-bolts, nut pockets, pegs); motor at motors_xy[1], pad top z 45, thrust plane z 82.
* Installs with core tie left.2 (right.2 on arm 3) in place: its tunnel is swept straight down and out, so the arm lowers onto the
  bottom plate over the rod with the top plate off.
* Shape: the root block, a rounded 16 x 33 beam along the arm (z 12..45, 5 mm corners), the motor disc O36 x 8 on a bell under
  it (O36 at the pad down to O32 at z 14, 23 deep, outside the driver sweeps) blended into the beam: all one smooth body; the lugs, slots (r 8..9.5), head
  seats, driver sweeps and O8.6 shaft bore cut as the lib draws them, the cage members near it cut clear.
* Print on its side (a flat side of the beam on the bed, the motor disc's axis horizontal): layers along the arm and through the
  pad, thrust and bending in-layer. PETG, 5 walls, 50 % gyroid, 0.2 layers (half the time of solid, in-layer loads). Fits the A1 mini
  (~135 x 48, 47 tall)."""
import os, sys, numpy as np, manifold3d as mf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from parts.overseer_arm_lib import Arm, CAGE, Z_PAD, cyl, along, U
from parts.arm_solid_lib import grid, rbox, cylinder, bell, softmax, mesh, export

INSTALL_OVER = "core tie left.2"
SLOT_R, SLOT_DROP = 4.0, 80.0
PAD_T, BELL_Z0, BELL_R0 = 8.0, 14.0, 16.0        # the bell stays outside the 4 driver sweeps (r 13 from the motor axis) with a 3 mm wall
BEAM_HALF = (7.0, 16.5)

ARM = Arm("rear", 135.0, 1, [], [-73.03, 139.5, 25.0])
drawn, cuts, env = ARM.drawn_cuts_env()
a3 = np.array([ARM.a[0], ARM.a[1], 0.0]); n3 = np.array([-a3[1], a3[0], 0.0]); R3 = np.array([ARM.R[0], ARM.R[1], 0.0]); L = ARM.len; M = ARM.M
ud3 = np.array([ARM.ud[0], ARM.ud[1], 0.0]); vd3 = np.array([ARM.vd[0], ARM.vd[1], 0.0])

lo = np.minimum(R3 + a3 * 0 - 50, [M[0] - 25, M[1] - 25, -5]); hi = np.maximum(R3 + 50, [M[0] + 25, M[1] + 25, Z_PAD + 2])
ax, P = grid(lo, hi)
beam_c = R3 + a3 * (4.0 + L - 8.0) / 2 + [0, 0, 12.0 + BEAM_HALF[1]]
F = softmax([
    rbox(P, ud3 * 64.75 + [0, 0, Z_PAD / 2], ud3, vd3, (7.25, 22.0, Z_PAD / 2), r=1.0),      # the root block, near-sharp
    rbox(P, beam_c, a3, n3, ((L - 12.0) / 2, BEAM_HALF[0], BEAM_HALF[1]), r=5.0),           # the beam
    cylinder(P, M, Z_PAD - PAD_T, Z_PAD, 18.0),                                             # the motor disc
    bell(P, M, BELL_Z0, BELL_R0, Z_PAD - PAD_T + 0.5, 18.0),                                # the bell under it
])
body = mesh(ax, F)
p, q = (np.array(v, float) for v in CAGE["links"][INSTALL_OVER]); u = (q - p) / np.linalg.norm(q - p)
tube = along(p, u, 0.0, float(np.linalg.norm(q - p)), SLOT_R)
slot = mf.Manifold.batch_hull([tube, tube.translate([0.0, 0.0, -SLOT_DROP])])
m = (body + drawn) - cuts - slot
export(m, os.path.join(HERE, "part.stl"))
