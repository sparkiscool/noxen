"""Overseer front arm, PAD: the printed motor pad at the end of the two 6/4 tubes (29 Oct, Spark: tube arms, as light as possible,
no clamps): motor slots, shaft bore, the two 3 mm spoke sockets and two tube sockets facing the root. Geometry, cases and numbers:
parts/overseer_tube_arm_lib.py on parts/overseer_arm_lib.py. Motor at motors_xy[0] (205.97, 139.5), pad top z 45, thrust plane z 82;
spokes 1.1 / 1.3 end at (178.94, 146.7, 8). Arm 4 is the y-mirror."""
from parts.overseer_tube_arm_lib import TubeArm, build_piece
from parts.overseer_arm_lib import Z_PAD, PAD_D, PAD_T, cyl

SKIRT_R0, SKIRT_D = 12.5, 8.0      # v2 (5 Nov, Spark: the printed front pads cracked at the motor, printed flat): a drawn skirt under the
                                   # pad rim gussets the 6 mm pad into the arm, between the 4 motor-screw driver sweeps (they still cut it)


class FrontPad(TubeArm):
    def drawn_cuts_env(self):
        drawn, cuts, env = super().drawn_cuts_env()
        M = self.M
        skirt = cyl(M[0], M[1], Z_PAD - PAD_T - SKIRT_D, Z_PAD - PAD_T + 0.01, PAD_D / 2, 64) - \
            cyl(M[0], M[1], Z_PAD - PAD_T - SKIRT_D - 1.0, Z_PAD - PAD_T + 1.0, SKIRT_R0, 64)
        return drawn + skirt, cuts, env


ARM = FrontPad("front", 45.0, 0, ["spoke 1.1", "spoke 1.3"], [178.94, 146.7, 8.0], "pad")
SPEC = dict(h=1.5, gen_cm3=18.0, max_worst=0.3)     # v2 (5 Nov): 12 -> 18 and the limit 0.5 -> 0.3: the rear arm broke at 0.30 under thrust, so 0.3 is the real margin; 12 (29 Oct): 7 gave 17.1 g at index 0.70 in the -z cage crash (spokes 1.1 / 1.3 pushing and pulling), and the spoke loads rose 8..15 % with the rear spokes gone; the tip piece passed at 12 (E)


def setup(h):
    p = build_piece(ARM, h)
    p.build = "+y"          # printed on its side (5 Nov): the rear arm v9 snapped at its pad under thrust alone at 0.30 printed flat; this
    return p                # pad sat at 0.28 in-layer / 0.17 across in the same case. On its side (layers in xz) thrust is in-layer
