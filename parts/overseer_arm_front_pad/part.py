"""Overseer front arm, PAD: the printed motor pad at the end of the two 6/4 tubes (29 Oct, Spark: tube arms, as light as possible,
no clamps): motor slots, shaft bore, the two 3 mm spoke sockets and two tube sockets facing the root. Geometry, cases and numbers:
parts/overseer_tube_arm_lib.py on parts/overseer_arm_lib.py. Motor at motors_xy[0] (205.97, 139.5), pad top z 45, thrust plane z 82;
spokes 1.1 / 1.3 end at (178.94, 146.7, 8). Arm 4 is the y-mirror."""
from parts.overseer_tube_arm_lib import TubeArm, build_piece
ARM = TubeArm("front", 45.0, 0, ["spoke 1.1", "spoke 1.3"], [178.94, 146.7, 8.0], "pad")
SPEC = dict(h=1.5, gen_cm3=12.0, max_worst=0.5)     # 12 (29 Oct): 7 gave 17.1 g at index 0.70 in the -z cage crash (spokes 1.1 / 1.3 pushing and pulling), and the spoke loads rose 8..15 % with the rear spokes gone; the tip piece passed at 12 (E)


def setup(h):
    return build_piece(ARM, h)
