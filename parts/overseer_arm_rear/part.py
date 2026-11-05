"""Overseer rear arm (arm 2 at +y, arm 3 is its y-mirror), v10 setup (5 Nov, Spark: the printed rear arms snapped off at the motor
end, the arm itself held; reprint them so they bolt on without taking out the rod through them, with better support where the motor
mounts). v9 base (29 Oct): one printed PETG piece from the plates to the motor, no tubes, no spokes; geometry and cases
parts/overseer_arm_lib.py (flight 4 g + 21 N thrust +-0.5 N m at the plates alone, landing 6 g, the frame crash set on the motor's
inertia, HANDLING). Motor pad at motors_xy[1] (-73.03, 139.5), arm top z 45, thrust plane z 82.
v10:
* Installs over core tie left.2 (right.2 on arm 3): that link crosses the arm 58 deg off its axis at z ~14, 67 from the motor,
  and v9 grew a closed tunnel round it. Its tunnel (r 3.5 + 0.5) is now swept straight down out of the envelope, so the arm is
  lowered onto the bottom plate over the rod with the top plate off (the assembly order) and the rod stays in.
* The motor end: a drawn skirt 8 deep under the pad's rim (r 12.5..18, between the 4 motor-screw driver sweeps, which still cut
  through it) gussets the 6 mm pad into the arm, and the material rises 32 -> 46 cm3; the strength limit is 0.3 (v9 broke at 0.30). v9's worst was 0.30 at the pad in flight;
  it broke under thrust alone (Spark), so the layers, not the load, are the cause (below).
* Printed on its side (build +x): v9 broke under thrust alone at 0.30 printed flat, so the joint under the pad fails across the
  layers well below the model; on its side thrust and bending are in-layer. Fits the A1 mini (about 120 x 45 on the bed, 60 tall).
* h 2.0 as v9 (fastest that keeps the pad's features; Spark 5 Nov: as fast as possible)."""
import numpy as np, manifold3d as mf
from parts.overseer_arm_lib import Arm, build, CAGE, Z_PAD, PAD_D, PAD_T, cyl, along

INSTALL_OVER = "core tie left.2"                 # the link the arm is lowered over (arm 3, the mirror: right.2)
SLOT_R, SLOT_DROP = 4.0, 80.0
SKIRT_R0, SKIRT_D = 12.5, 8.0


class RearArm(Arm):
    def drawn_cuts_env(self):
        drawn, cuts, env = super().drawn_cuts_env()
        p, q = (np.array(v, float) for v in CAGE["links"][INSTALL_OVER])
        u = (q - p) / np.linalg.norm(q - p); L = float(np.linalg.norm(q - p))
        tube = along(p, u, 0.0, L, SLOT_R)
        slot = mf.Manifold.batch_hull([tube, tube.translate([0.0, 0.0, -SLOT_DROP])])       # the rod's tunnel swept down: open below
        M = self.M
        skirt = cyl(M[0], M[1], Z_PAD - PAD_T - SKIRT_D, Z_PAD - PAD_T + 0.01, PAD_D / 2, 64) - \
            cyl(M[0], M[1], Z_PAD - PAD_T - SKIRT_D - 1.0, Z_PAD - PAD_T + 1.0, SKIRT_R0, 64)
        return drawn + skirt, cuts + slot, env


ARM = RearArm("rear", 135.0, 1, [], [-73.03, 139.5, 25.0])
SPEC = dict(h=2.0, gen_cm3=46.0, max_worst=0.3)   # v10 (5 Nov): 32 -> 46 for the motor end, limit 0.5 -> 0.3 (v9 broke at 0.30 under thrust: 0.3 is the real margin); h 2.0 (29 Oct: at 1.2 the one-piece grid was 1.0 M cells)


def setup(h):
    p = build(ARM, h)
    p.build = "+x"          # printed on its side (5 Nov, Spark: v9 snapped at the pad under motor thrust alone, index 0.30 printed flat):
    return p                # thrust pulls the pad up across flat layers; on its side thrust and the arm's bending run in-layer
